"""v4 cross-modality code on CPU (DESIGN_v4.md): label map, CT reduction, CTNet and distillation, the s4/xm.py stages chained with NIH
fine-tuning, cross-modal maps, hallucination tests and tools/xm_stats.py, on tiny synthetic data."""
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from s4 import ct_prep, xm, xm_halluc, xm_xai
from s4.prep import NIH
from s4.xm_labels import SHARED, columns, from_meta, to_shared

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
S, K = 64, 3
# The CT-RATE volume lists of s4/data stay out of the public repository (DESIGN_v4.md section 2), so the tests that read them skip there
needs_ct_lists = pytest.mark.skipif(not ct_prep.SKIP.exists(), reason="CT-RATE volume lists are not in the public repository")


def test_label_map():
    y = np.zeros((3, 14), np.float32)
    y[0, NIH.index("Nodule")] = y[1, NIH.index("Infiltration")] = y[2, NIH.index("Effusion")] = 1
    assert to_shared(y, "nih").tolist() == [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
    assert columns("vin")[5] == [8] and columns("nih")[5] == [NIH.index("Mass"), NIH.index("Nodule")]
    assert from_meta([{k: "1" if k == "cardiomegaly" else "0" for k in SHARED}]).tolist() == [[0, 0, 0, 1, 0, 0]]


def phantom(nx=64, ny=56, nz=40):
    """HU volume in CT-RATE order (axis 0 toward the patient's left, 1 toward posterior, 2 toward the head): air, an elliptic body, two
    lungs from z = 8 to 32, and a bone marker in front on the patient's right at the top of the lungs."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    hu = np.full((nx, ny, nz), -1000.0, np.float32)
    hu[((x - nx / 2) / (nx * 0.42)) ** 2 + ((y - ny / 2) / (ny * 0.40)) ** 2 < 1] = 40.0
    for cx in (0.3 * nx, 0.7 * nx):
        hu[((x - cx) / (nx * 0.13)) ** 2 + ((y - ny / 2) / (ny * 0.25)) ** 2 < 1, 8:33] = -800.0
    hu[10:14, 12:16, 30:33] = 1000.0
    return hu


def test_ct_reduction_orientation_and_order():
    slices, ap, box = ct_prep.process(phantom(), k=6, size=32)
    assert slices.shape == (6, 3, 32, 32) and slices.dtype == np.uint8 and ap.shape == (6, 32) and ap.dtype == np.float16
    bone = slices[:, 2]
    row, col = np.unravel_index(bone[0].argmax(), bone[0].shape)
    assert row < 16 and col < 16  # radiological view: anterior at the top, the patient's right on the image left
    assert bone[0].max() > bone[-1].max() + 50  # superior slice first: the marker sits at the top of the lungs
    assert 0 < box[0] < box[1] < 31
    with pytest.raises(ValueError):
        ct_prep.check_hu(np.zeros((8, 8, 8), np.float32))  # no air around a body: not HU


@needs_ct_lists
def test_select_rescale_and_paths(tmp_path):
    rows = [dict(volume_name=f"train_{p}_{r}_1", patient=f"train_{p}", split=s) for p, s in [(1, "train"), (2, "train"), (3, "val"), (4, "test"), (5, "train")]
            for r in ("b", "a")]
    sel = ct_prep.select(rows, n_train=1)
    assert len(sel) == 3 and {x["split"] for x in sel} == {"train", "val", "test"} and all(x["volume_name"].endswith("_a_1") for x in sel)
    sel = ct_prep.select(rows, n_train=1, skip={"train_3_a_1", "train_4_a_1", "train_4_b_1"})
    assert {x["patient"]: x["volume_name"] for x in sel if x["split"] != "train"} == {"train_3": "train_3_b_1"}  # next scan; patient 4 has none
    assert "train_10712_a_1" in ct_prep.skip_set()  # CT-RATE no-chest list
    assert ct_prep.hf_path("valid_12_b_2") == "dataset/valid/valid_12/valid_12_b/valid_12_b_2.nii.gz"
    (tmp_path / "a.csv").write_text("VolumeName,RescaleSlope,RescaleIntercept\ntrain_1_a_1.nii.gz,1,-8192\n")  # CT-RATE names end in .nii.gz
    (tmp_path / "b.csv").write_text("volume_name,rescale_slope,rescale_intercept\nvalid_1_a_1,2,0\n")
    assert ct_prep.rescale([tmp_path / "a.csv", tmp_path / "b.csv"]) == {"train_1_a_1": (1.0, -8192.0), "valid_1_a_1": (2.0, 0.0)}


def test_build_marks_failures_and_resumes(tmp_path, monkeypatch):
    np.save(tmp_path / "v.npy", phantom())
    monkeypatch.setattr(ct_prep, "load_hu", lambda p, slope=1.0, intercept=0.0: np.load(p) * slope + intercept)  # nibabel only on Kaggle
    rows = [{**{s: "0" for s in SHARED}, "volume_name": n, "patient": n[:-4], "split": "test"} for n in ("train_1_a_1", "train_2_a_1")]
    rows[1]["atelectasis"] = "1"
    subset = [dict(row=i, volume_name=r["volume_name"], patient=r["patient"], split="test") for i, r in enumerate(rows)]
    calls = []

    def fetch(name):
        calls.append(name)
        if name == "train_2_a_1" and len(calls) < 3:
            raise OSError("network down")
        return str(tmp_path / "v.npy")

    st = ct_prep.build(subset, tmp_path / "ct", rows, fetch=fetch, workers=1, k=4, size=32)
    assert [s["ok"] for s in st] == [True, False] and "network down" in st[1]["msg"]
    st = ct_prep.build(subset, tmp_path / "ct", rows, fetch=fetch, workers=1, k=4, size=32)  # resumed: only the failed volume again
    assert [s["ok"] for s in st] == [True, True] and calls == ["train_1_a_1", "train_2_a_1", "train_2_a_1"]
    assert np.load(tmp_path / "ct" / "ct.npy").shape == (2, 4, 3, 32, 32) and np.load(tmp_path / "ct" / "labels.npy")[1].tolist() == [0, 0, 1, 0, 0, 0]
    st = ct_prep.build(subset, tmp_path / "late", rows, fetch=fetch, workers=1, k=4, size=32, deadline=time.time() - 1)
    assert [s["msg"] for s in st] == ["deadline", "deadline"]


def test_ctnet_shapes_and_gradients():
    m = xm.CTNet("resnet18", 6, K, size=S, pretrained=False)
    out = m(torch.rand(2, K, 3, S, S))
    assert out.shape == (2, 6) and m.attn.shape == (2, 6, K, 2, 2) and m.emb_out.shape == (2, 256)
    out.sum().backward()
    assert m.zemb.grad.abs().sum() > 0  # the slice-depth embedding reaches the logits


def test_kd_loss_pulls_the_student_to_the_teacher():
    t = torch.tensor([[2.0, -1.0, 0.5]])
    s = t.clone().requires_grad_(True)
    xm.kd_loss(s, t, 2.0).backward()
    assert s.grad.abs().max() < 1e-6
    s0 = torch.zeros_like(t, requires_grad=True)
    xm.kd_loss(s0, t, 2.0).backward()
    assert (torch.sign(-s0.grad) == torch.sign(t)).all()


def test_perturbations_nodules_and_thresholds(tmp_path):
    x = torch.stack([torch.full((8, 8), 0.5), torch.full((8, 8), 0.3), torch.full((8, 8), 0.2)])[None]
    r = xm_halluc.perturb(x, "rib")
    assert torch.allclose(r[0, 0], torch.full((8, 8), 0.6)) and torch.allclose(r[0, 2], torch.full((8, 8), 0.3))
    assert torch.allclose(xm_halluc.perturb(x, "gamma")[0, 0], torch.full((8, 8), 0.5 ** 0.8))
    a, b = (xm_halluc.perturb(x, "noise", torch.Generator().manual_seed(3)) for _ in range(2))
    assert torch.equal(a, b) and not torch.equal(a[0, 0], x[0, 0])  # the same seed gives every model the same noise
    t = xm_halluc.add(x, torch.full((1, 8, 8), 0.1))
    assert torch.allclose(t[0, 0], torch.full((8, 8), 0.6)) and torch.allclose(t[0, 1], torch.full((8, 8), 0.4)) and torch.allclose(t[0, 2], x[0, 2])
    n = xm_halluc.nodule(np.ones((16, 16), np.float32), 128, 128, 256)  # 16 px in the 512 frame: 8 px at 256
    assert n.sum() == 64 and n[124:132, 124:132].all()
    assert xm_halluc.crossed(np.array([[0.1, 0.9], [0.1, 0.1]]), np.array([[0.1, 0.95], [0.6, 0.1]]), np.array([0.5, 0.5])).tolist() == [False, True]
    rng = np.random.default_rng(0)
    y = (rng.random((1000, 2)) < 0.3).astype(np.float32)
    for s in range(2):
        (tmp_path / f"r{s}").mkdir()
        np.save(tmp_path / f"r{s}" / "val_idx.npy", np.arange(1000))
        np.save(tmp_path / f"r{s}" / "val_pred.npy", rng.random((1000, 2)).astype(np.float16))
    pv, vidx = xm_halluc.val_mean([tmp_path / "r0", tmp_path / "r1"])
    thr = xm_halluc.thresholds(pv, y[vidx])
    p = np.mean([np.load(tmp_path / f"r{s}" / "val_pred.npy").astype(np.float32) for s in range(2)], 0)
    assert all(abs((p[y[:, c] == 0, c] <= thr[c]).mean() - 0.9) < 0.01 for c in range(2))


def test_frame_helpers():
    f = xm_xai.to_frame(np.arange(24 * 200, dtype=np.float32).reshape(24, 200))
    assert f.shape == (32, 32) and f[0, 0] < f[0, -1] < f[-1, -1]
    L = np.zeros((64, 64))
    L[10:50, 5:60] = 1
    assert xm_xai.lung_box(L) == (10, 49, 5, 59) and xm_xai.lung_box(np.zeros((4, 4))) == (0, 3, 0, 3)
    a = np.random.default_rng(0).random((8, 8))
    assert abs(xm_xai.pearson(a, 2 * a + 1) - 1) < 1e-9 and abs(xm_xai.pearson(a, -a) + 1) < 1e-9


def fake_ct_and_qp(root, n=16):
    """CT prep and projection pack of the same n scans (8 train, 4 val, 4 test); findings 0 and 3 alternate, so each split has both."""
    rng = np.random.default_rng(0)
    names, split = [f"train_{i}_a_1" for i in range(n)], np.array(["train"] * 8 + ["val"] * 4 + ["test"] * 4)
    y = np.zeros((n, 6), np.float32)
    y[::2, 0] = y[1::2, 3] = 1
    ct, qp = root / "ct", root / "qp"
    ct.mkdir()
    qp.mkdir()
    np.save(ct / "ct.npy", rng.integers(0, 255, (n, K, 3, S, S), dtype=np.uint8))
    np.save(ct / "ap.npy", rng.random((n, K, S)).astype(np.float16))
    np.save(ct / "lung_box.npy", np.tile(np.array([8, 55], np.int16), (n, 1)))
    np.save(ct / "labels.npy", y)
    np.save(ct / "split.npy", split)
    np.save(ct / "rows.npy", np.arange(n))
    (ct / "names.json").write_text(json.dumps(names))
    (ct / "status.json").write_text(json.dumps([dict(ok=True, msg="ok")] * n))
    index, blob = np.zeros((n, 3, 2), np.int64), b""
    for i in range(n):
        for c in range(3):  # T, Bt, L at twice the model size; L: a central lung block
            img = np.pad(np.full((S, S), 200, np.uint8), S // 2) if c == 2 else rng.integers(0, 255, (2 * S, 2 * S), dtype=np.uint8)
            b = cv2.imencode(".png", img)[1].tobytes()
            index[i, c] = (len(blob), len(b))
            blob += b
    (qp / "qp.bin").write_bytes(blob)
    np.save(qp / "qp_index.npy", index)
    with open(qp / "qp_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["volume_name", "patient", "split", *SHARED])
        w.writeheader()
        w.writerows(dict(volume_name=names[i], patient=names[i][:-4], split=split[i], **{s: int(y[i, k]) for k, s in enumerate(SHARED)}) for i in range(n))
    return ct, qp


def fake_nih(root, n=40):
    rng = np.random.default_rng(1)
    d = root / "nih"
    d.mkdir()
    np.save(d / "orig.npy", rng.integers(0, 255, (n, S, S), dtype=np.uint8))
    np.save(d / "soft.npy", rng.integers(0, 200, (n, S, S), dtype=np.uint8))
    y = np.zeros((n, 14), np.float32)
    y[::2, NIH.index("Effusion")] = y[2::4, NIH.index("Nodule")] = 1  # odd rows are finding-free
    np.save(d / "labels.npy", y)
    np.save(d / "split.npy", np.array(["train"] * 16 + ["val"] * 12 + ["test"] * 12))
    np.save(d / "patient.npy", np.array([f"p{i // 2}" for i in range(n)]))
    return d


@needs_ct_lists
def test_pipeline_end_to_end_cpu(tmp_path, monkeypatch, capsys):
    ct, qp = fake_ct_and_qp(tmp_path)
    nih = fake_nih(tmp_path)
    tiny = ["--arch", "resnet18", "--pretrained", "0", "--epochs", "1", "--size", str(S)]
    xm.main(["--stage", "ct", "--data", str(ct), "--out", str(tmp_path / "ct_run"), "--bs", "2", "--accum", "2", *tiny])  # one volume per pass
    t = np.load(tmp_path / "ct_run" / "teacher.npz", allow_pickle=True)
    assert t["logits"].shape == (16, 6) and t["emb"].shape == (16, 256) and np.load(tmp_path / "ct_run" / "test_labels.npy").shape == (4, 6)
    xm.main(["--stage", "xray", "--data", str(qp), "--out", str(tmp_path / "xr_run"), "--bs", "4", "--teacher", str(tmp_path / "ct_run" / "teacher.npz"), *tiny])
    assert json.loads((tmp_path / "xr_run" / "result.json").read_text())["paired"] == 8  # the train scans whose CT is a train teacher volume
    assert not any(k.startswith("align.") for k in torch.load(tmp_path / "xr_run" / "model.pt", map_location="cpu")["state"])
    xm.main(["--stage", "transfer", "--data", str(nih), "--run", str(tmp_path / "xr_run"), "--out", str(tmp_path / "tr"), "--dataset", "nih"])
    assert np.load(tmp_path / "tr" / "test_pred.npy").shape == (12, 6) and (tmp_path / "tr" / "result.json").exists()
    from s4.cls_hybrid import main as fine_tune
    capsys.readouterr()
    fine_tune(["--data", str(nih), "--config", "orig", "--bs", "8", "--max_steps", "2", "--init", str(tmp_path / "xr_run" / "model.pt"),
               "--out", str(tmp_path / "fin"), *tiny])
    got, total = map(int, re.search(r"init from .*: (\d+) of (\d+) tensors", capsys.readouterr().out).groups())
    assert 0 < got == total - 3  # everything but the 14-class queries and heads (query, head_w, head_b)

    xm_xai.main(["--data", str(ct), "--qp", str(qp), "--ct_run", str(tmp_path / "ct_run"), "--xr_run", str(tmp_path / "xr_run"), "--out", str(tmp_path / "xai")])
    z = np.load(tmp_path / "xai" / "maps.npz")
    assert z["ct"].shape == z["xr"].shape == (4, 32, 32) and z["scan"].tolist() == [12, 13, 14, 15] and np.isfinite(z["frame_r"]).all()

    box = np.zeros((S, S), bool)
    box[16:48, 8:56] = True
    monkeypatch.setattr(xm_halluc, "lung_fields", lambda images, dev: (np.array([box] * len(images)), np.array([box] * len(images))))
    for out in ("h", "h2"):
        xm_halluc.main(["--data", str(nih), "--runs", str(tmp_path / "fin"), "--out", str(tmp_path / out), "--n_nodule", "4", "--n_cam", "6"])
    r = json.loads((tmp_path / "h" / "result.json").read_text())
    assert r["n_free"] == 6 and all(0 <= r[f"rate_{k}"] <= 1 for k in xm_halluc.PERTURB)
    h, h2 = np.load(tmp_path / "h" / "halluc.npz"), np.load(tmp_path / "h2" / "halluc.npz")
    assert h["p_nodule"].shape == (4, 14) and np.array_equal(h["p_noise"], h2["p_noise"]) and np.array_equal(h["nod_site"], h2["nod_site"])
    assert int(h["n_runs"]) == 1 and h["val_p"].shape == (12, 14)
    assert ((h["cam_off"] >= 0) & (h["cam_off"] <= 1)).all()


def test_xm_stats_on_synthetic_runs(tmp_path):
    import xm_stats
    rng = np.random.default_rng(0)
    n, G = 300, 32
    y = (rng.random((n, 14)) < 0.2).astype(np.float32)
    pat = np.array([f"p{i // 2}" for i in range(n)])

    def run(d, quality):
        d.mkdir(parents=True)
        np.save(d / "test_pred.npy", np.clip(y * quality + rng.random((n, 14)) * (1 - quality), 0, 1).astype(np.float16))
        np.save(d / "test_idx.npy", np.arange(n))
        np.save(d / "test_labels.npy", y)
        np.save(d / "test_patient.npy", pat)

    root, v3 = tmp_path / "v4", tmp_path / "v3"
    for s in range(2):
        for suffix, q in (("", 0.2), ("+xray", 0.3), ("+xraykd", 0.5)):
            run(root / "down" / "nih" / f"nih_hyb_orig@swinv2_tiny_window8_256{suffix}_s{s}", q)
        run(v3 / f"nih_hyb_orig@densenet121_s{s}", 0.2)
    scan, cls = np.repeat(np.arange(20), 2), np.tile([0, 3], 20)
    ctm = rng.random((40, G, G)).astype(np.float32)
    (root / "xai" / "ct_xmx_xraykd_s0").mkdir(parents=True)
    np.savez(root / "xai" / "ct_xmx_xraykd_s0" / "maps.npz", ct=ctm, xr=ctm[..., ::-1] + 0.05 * rng.random((40, G, G)), scan=scan, cls=cls,
             frame_r=np.full(20, 0.1), frame_r_flip=np.full(20, 0.8))  # the frames say the projection is mirrored
    m = 200
    val_p = np.tile(np.linspace(0, 0.6 / 0.9, 1000)[:, None], (1, 14))  # 90 % specificity threshold 0.6 for every class
    for name, extra, groups in (("fin", 0.0, ((0, 3), (3, 2))), ("fp1", 0.3, ((0, 5),))):  # F_in as two seed groups (two accounts)
        p0 = rng.random((m, 14)) * 0.5
        p_rib = p0 + extra * (rng.random((m, 14)) < 0.1)
        for first, runs in groups:
            d = root / "halluc" / f"nih_xmh_{name}_s{first}"
            d.mkdir(parents=True)
            np.savez(d / "halluc.npz", thr=np.full(14, 0.6), val_p=val_p, val_y=np.zeros((1000, 14)), n_runs=runs, free=np.arange(m),
                     free_patient=np.arange(m) // 2, p_clean=p0, p_rib=p_rib, p_noise=p0, p_gamma=p0, p_nodule=p0[:50] + 0.2, nod=np.arange(50),
                     nod_pos=np.arange(50), nod_site=np.ones((50, 3), int), nod_diam=rng.uniform(4, 30, 50),
                     cam_off=rng.random(10).astype(np.float32), cam_pairs=np.stack([np.arange(10), np.full(10, NIH.index("Nodule"))], 1))
    out = tmp_path / "stats.json"
    xm_stats.main(["--root", str(root), "--v3", str(v3), "--n", "40", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["H11"]["diff"] > 0 and all(0 < res[k]["p_holm"] <= 1 for k in ("H11", "H12_p1_p0", "H12_p0_in")) and "H13" in res
    h14 = res["crossmodal_xraykd"]
    assert h14["mirrored"] and h14["H14"]["lo"] > 0 and h14["matched"] > 0.9
    assert res["H15"]["diff"] > 0 and res["hallucination"]["F_p1"]["rate_rib"] > res["hallucination"]["F_in"]["rate_rib"]
    assert res["hallucination"]["F_in"]["seeds"] == 5 and res["hallucination"]["F_in"]["files"] == 2 and res["hallucination"]["F_in"]["rate_rib"] == 0
    assert set(res["missing"]) == {"halluc:F_p0", "halluc:DN121", "xai:xray"}
