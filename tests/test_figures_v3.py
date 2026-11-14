"""tools/generate_all_thesis_figures.py on synthetic results trees (CPU, Agg): runs of one v3 encoder, v2 runs kept apart, file names, and
the journal standard of tools/figstyle.py for every saved figure (figstyle.problems() empty, nothing drawn up to the edge of the page)."""
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import generate_all_thesis_figures as G  # noqa: E402

GROUPS = ("focal", "diffuse", "boundary")
CURVES = ("diagnostic_curves", *(f"{k}_{g}" for k in ("roc", "pr") for g in GROUPS), "auroc_per_class", "bdcxr_roc")


@pytest.fixture
def saved(monkeypatch):
    """{figure name: figstyle.problems()} of every figure the script saves; the PNG's outermost pixels must be blank (else text or a legend
    runs off the page at the print size)."""
    out, save = {}, G.fs.save

    def spy(fig, stem):
        out[Path(stem).name] = G.fs.problems(fig)
        save(fig, stem)
        with Image.open(f"{stem}.png") as png:
            im = np.asarray(png.convert("L"))
        assert min(im[0].min(), im[-1].min(), im[:, 0].min(), im[:, -1].min()) > 200, f"{stem}: drawn up to the edge"
    monkeypatch.setattr(G.fs, "save", spy)
    return out


def _run(d, p, y, idx, pat):
    d.mkdir(parents=True)
    for k, v in dict(pred=p.astype(np.float16), labels=y, idx=idx, patient=pat).items():  # float16 predictions, as s4/cls_hybrid.py
        np.save(d / f"test_{k}.npy", v)


def _v3_tree(root, n=300, nb=150):
    """v3 runs of DenseNet-121 and of a second encoder: NIH 14 labels x n rows, 2 seeds; BDCXR 3 classes x nb images, 5 folds, 2 seeds."""
    rng = np.random.default_rng(0)
    y = (rng.random((n, 14)) < 0.3).astype(np.float32)
    yb = np.eye(3, dtype=np.float32)[rng.integers(0, 3, nb)]
    d = root / "results/v3/down"
    for arch in ("densenet121", "rad-dino"):
        for cfg in G.V3:
            for s in range(2):
                _run(d / f"nih/nih_hyb_{cfg}@{arch}_s{s}", 0.4 * y + 0.6 * rng.random(y.shape), y, np.arange(n), np.arange(n) // 3)
                for f in range(5):
                    i = np.arange(f, nb, 5)
                    _run(d / f"bdcxr/bdcxr_hyb_{cfg}@{arch}_s{s}_f{f}", 0.5 * yb[i] + rng.random((len(i), 3)), yb[i], i, i)
    return y, yb


def _figs(root):
    return {p.name for p in (root / "results/figures").glob("*")}


def test_curves_and_confusion_on_v3_runs(tmp_path, saved):
    _, yb = _v3_tree(tmp_path)
    a = SimpleNamespace(root=tmp_path, arch="densenet121", with_v2=False)
    runs = G.cls_runs(a, "nih")
    assert list(runs) == list(G.V3)  # the five configurations, labelled by the existing NAME/COL entries; no rad-dino run mixed in
    seeds = [np.load(tmp_path / f"results/v3/down/nih/nih_hyb_hybrid@densenet121_s{s}/test_pred.npy").astype(np.float32) for s in range(2)]
    np.testing.assert_allclose(runs["hybrid"][0], np.mean(seeds, 0), rtol=1e-6)
    bd = G.cls_runs(a, "bdcxr")
    assert list(bd) == list(G.V3)
    np.testing.assert_array_equal(bd["hybrid"][1], yb)  # five folds concatenated, back in image order
    G.main(["--root", str(tmp_path), "--only", "curves,confusion", "--boot", "3"])
    want = {f"fig_{n}_densenet121" for n in (*CURVES, "confusion_matrices", "error_comovement")}  # no detection runs here: no fig_froc
    assert set(saved) == want
    assert not any(saved.values()), saved
    assert {f"{s}.{e}" for s in want for e in ("pdf", "png")} <= _figs(tmp_path)


def test_unfinished_folds_are_left_out(tmp_path, capsys):
    _v3_tree(tmp_path)
    for d in (tmp_path / "results/v3/down/bdcxr").glob("bdcxr_hyb_hybrid@densenet121_s*_f4"):
        shutil.rmtree(d)
    bd = G.cls_runs(SimpleNamespace(root=tmp_path, arch="densenet121", with_v2=False), "bdcxr")
    assert list(bd) == ["orig", "soft", "early", "ctrl"]  # hybrid lacks fold 4, so it is not compared on another image set
    assert "hybrid left out" in capsys.readouterr().out


def test_with_v2_keeps_v2_runs_apart(tmp_path, saved):
    y, _ = _v3_tree(tmp_path)
    rng, r = np.random.default_rng(1), np.arange(len(y))[::-1]  # v2 rows in another order
    for v in ("orig", "soft", "orig-soft", "clahe"):
        _run(tmp_path / f"results/final/down/nih/nih_cls_{v}_s0", rng.random(y.shape), y[r], r, r // 3)
    gts = [(np.array([[10.0, 10, 30, 30]]), np.array([0])) for _ in range(8)]  # v2 detection run (s4/det.py test_preds.pkl): FROC
    preds = [(np.array([[11.0, 9, 31, 29], [60, 60, 80, 80]]), rng.random(2), np.zeros(2, int)) for _ in range(8)]
    (tmp_path / "results/final/down/node21/node21_det_soft_s0").mkdir(parents=True)
    (tmp_path / "results/final/down/node21/node21_det_soft_s0/test_preds.pkl").write_bytes(pickle.dumps(dict(preds=preds, gts=gts)))
    runs = G.cls_runs(SimpleNamespace(root=tmp_path, arch="densenet121", with_v2=True), "nih")
    assert list(runs) == [*G.V3, "v2:orig", "v2:soft", "v2:orig-soft"]  # v2 'orig' does not overwrite v3 'orig'
    np.testing.assert_array_equal(runs["v2:orig"][1], runs["orig"][1])  # one row order, so the bootstrap weights fit both
    assert not np.allclose(runs["v2:orig"][0], runs["orig"][0])
    G.main(["--root", str(tmp_path), "--only", "curves", "--boot", "3", "--with_v2"])
    G.main(["--root", str(tmp_path), "--only", "curves", "--boot", "3", "--with_v2", "--arch", "convnextv2_tiny"])  # no v3 runs: v2 only
    want = {f"fig_{n}_densenet121_with_v2" for n in CURVES[:-1]} | {f"fig_{n}" for n in CURVES[:-1]}
    assert set(saved) == want | {"fig_bdcxr_roc_densenet121", "fig_froc"}  # no v2 BDCXR runs here: the BDCXR figure has v3 runs only
    assert not any(saved.values()), saved
    assert "fig_diagnostic_curves_densenet121.pdf" not in _figs(tmp_path)


def test_uncertainty_and_xai_read_the_encoder_files(tmp_path, saved):
    v3 = tmp_path / "results/v3"
    (v3 / "xai").mkdir(parents=True)
    rc = [[c, 0.2 * c] for c in np.linspace(0.1, 1, 19)]
    nih = dict(risk_coverage_brier=rc, aurc_brier=0.1, lam=0.97, fnr_test=0.09, mean_set_size=5.6)
    bd = dict(risk_coverage=rc, aurc=0.1, coverage=0.9, coverage_per_class=[0.9, 0.88, 0.93], mean_set_size=1.9)
    uq = {f"nih_{c}": nih for c in G.V3} | {f"bdcxr_{c}": bd for c in G.V3}
    uq["nih_hybrid"] = nih | dict(score_b=dict(n=200, risk_coverage=rc[::-1], aurc=0.09))  # drawn dashed
    uq["bdcxr_soft"] = bd | dict(score_b="unavailable: no ids.json")  # a string: no curve
    uq |= dict(arch="densenet121", alpha=0.1, missing=[], score_b_definition="...")  # the cls3 metadata keys are not curves
    (v3 / "uq_cls_densenet121.json").write_text(json.dumps(uq))
    v2 = {f"{d}_{v}": e for d, e in (("nih", nih), ("bdcxr", bd)) for v in ("orig", "soft", "orig-soft", "clahe", "soft_qure")}
    (v3 / "uq_cls.json").write_text(json.dumps(v2))  # the v2 file (s4/uq_conformal.py cls): other variants, no alpha key
    curve = [[0.0, 1.0], [0.5, 0.6], [0.9, 0.2]]
    test = {t: dict(per_image=np.linspace(0.8, 0.97, 51).tolist()) for t in ("pooled", "image")}
    bone = dict(n_cal=24, n_test=51, held_sparsification=curve, held_oracle=curve, held_ause=0.01, reliability=[[0.5, 0.4], [0.9, 0.8]])
    bone |= {"global": dict(test=test), "mondrian": dict(test=test)}
    (v3 / "uq_bone.json").write_text(json.dumps(bone))
    rng = np.random.default_rng(2)
    maps = {}
    for k in ("5_4", "9_1", "12_7"):
        maps |= {f"{k}__x": rng.random((3, 64, 64)), f"{k}__boxes": np.array([[8, 8, 30, 30]], np.float32)}
        maps |= {f"{k}__{m}": rng.random((64, 64)) for m in ("cam_0", "cam_1", "cam_2", "attn", "ig_X", "ig_S", "ig_B")}
    np.savez_compressed(v3 / "xai/nih_hyb_hybrid@densenet121_s1.maps.npz", **maps)  # no seed 0: the lowest seed present is used
    np.savez_compressed(v3 / "xai/nih_hyb_hybrid@convnextv2_tiny_s0.maps.npz", junk=np.zeros(1))  # another encoder; sorts first, unreadable
    hits = ("cam_0", "cam_1", "cam_2", "attn", "ig_X", "ig_S", "ig_B", "ig_all")
    rows = [dict(image=i, cls=c, rho=[0.5, 0.3, 0.2], ig_share=[0.4, 0.35, 0.25], hits={m: bool((i + j) % 3) for j, m in enumerate(hits)})
            for i in range(6) for c in ("Nodule", "Mass", "Effusion", "Cardiomegaly")]
    for s in (1, 2):  # s4/xai_multistream.py output of each seed's hybrid model
        (v3 / f"xai/nih_hyb_hybrid@densenet121_s{s}.json").write_text(json.dumps(dict(tol=15.0, n_targets=len(rows), rows=rows)))
    G.main(["--root", str(tmp_path), "--only", "uncertainty,xai,design"])
    G.main(["--root", str(tmp_path), "--only", "uncertainty", "--arch", "rad-dino"])  # no uq_cls_rad-dino.json: the v2 file, old names
    want = {f"fig_{n}_densenet121" for n in ("selective_classification", "conformal_sets", "xai_comparison", "xai_ig", "pointing_game")}
    assert set(saved) == want | {"fig_selective_classification", "fig_conformal_sets", "fig_bone_uncertainty", "fig_study_design"}
    assert not any(saved.values()), saved


def test_stats_figures_read_the_final_statistics(tmp_path, saved):
    _v3_tree(tmp_path)
    for d in (tmp_path / "results/v3/down").glob("*/*@densenet121_*"):
        (d / "result.json").write_text(json.dumps({"test_macro_auc": 0.8}))
    c = dict(diff=0.001, lo=-0.002, hi=0.004, p=0.4, p_holm=0.0004)
    enc = {ds: dict(n_boot=10, macro_auroc={k: 0.81 for k in G.V3}, **{h: {"auroc": c} for h in ("H6", "H7_ctrl", "H7_early")})
           for ds in ("nih", "bdcxr")}
    enc["nih"]["H8_groups"] = {f"{g}:{k}-orig": c for g in GROUPS for k in ("soft", "hybrid")}  # tools/hybrid_stats.py keys
    (tmp_path / "results/final_hybrid_stats.json").write_text(json.dumps({"encoders": {"densenet121": enc}}))
    G.main(["--root", str(tmp_path), "--only", "stats"])
    G.main(["--root", str(tmp_path), "--only", "stats", "--arch", "rad-dino"])  # not in the statistics: skipped
    assert set(saved) == {"fig_hypothesis_forest_densenet121", "fig_seed_spread_densenet121"}
    assert not any(saved.values()), saved
