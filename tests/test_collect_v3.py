import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import collect_v3 as C  # noqa: E402
import hybrid_stats  # noqa: E402
import v3_phase2 as V  # noqa: E402
from s4.runtag import tag as job_tag  # noqa: E402
from s4.uq_conformal import load_runs  # noqa: E402


def _run(d, val=0.8, test=0.79, drop=(), empty=(), n=6):
    """A finished run folder as s4/cls_hybrid.py leaves it, plus model.pt and a stale ckpt.pt.tmp that must not be copied."""
    d.mkdir(parents=True)
    arrays = dict(val_pred=np.full((n, 3), val, np.float16), val_idx=np.arange(n), test_pred=np.full((n, 3), test, np.float16),
                  test_idx=np.arange(n), test_labels=((np.arange(n)[:, None] + np.arange(3)) % 2).astype(np.float32),
                  test_patient=np.arange(n) // 2, test_routing=np.zeros((n, 3, 3), np.float16))
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)
    (d / "log.jsonl").write_text(json.dumps(dict(epoch=0, val_macro_auc=val)) + "\n")
    (d / "result.json").write_text(json.dumps(dict(val_macro_auc=val, test_macro_auc=test)))
    (d / "model.pt").write_bytes(b"w")
    (d / "ckpt.pt.tmp").write_bytes(b"c")
    for f in drop:
        (d / f).unlink()
    for f in empty:  # what a killed download leaves behind
        (d / f).write_bytes(b"")


def _collect(src, dst, expect="densenet121"):
    C.main(["--src", str(src), "--dst", str(dst), "--expect", expect])
    return json.loads((dst / "collect_report.json").read_text())


def test_layout_feeds_hybrid_stats_and_load_runs(tmp_path, monkeypatch):
    src, dst = tmp_path / "p2", tmp_path / "results/v3/down"
    for cfg in ("orig", "hybrid"):
        for s in (0, 1):
            _run(src / "s4-p2a-nih-dn-01/results" / f"nih_hyb_{cfg}@densenet121_s{s}")
    for k in (0, 4):
        _run(src / "s4-p2a-bdcxr-cnx-01/results" / f"bdcxr_hyb_ctrl@convnextv2_tiny_s2_f{k}")
    _run(src / "s4-p2a-vin-rd-01/results/vin_hyb_orig@rad-dino_s3")
    stale = dst / "nih/.nih_hyb_orig@densenet121_s0.tmp"  # left by an interrupted copy
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    rep = _collect(src, dst)
    run = dst / "nih/nih_hyb_orig@densenet121_s0"
    assert sorted(p.name for p in run.iterdir()) == sorted(C.FILES) and not stale.exists()  # model.pt and ckpt.pt* stay behind
    assert set(load_runs(dst / "bdcxr", "bdcxr_hyb_")) == {("ctrl@convnextv2_tiny", "f0"), ("ctrl@convnextv2_tiny", "f4")}
    assert set(load_runs(dst / "vin", "vin_hyb_")) == {("orig@rad-dino", None)}
    monkeypatch.setattr(hybrid_stats, "ROOT", tmp_path)
    y, P, g = hybrid_stats.gather("nih", "densenet121")
    assert sorted(P) == ["hybrid", "orig"] and y.shape == (6, 3) and len(g) == 6
    c = rep["counts"]
    assert c["densenet121"]["nih"]["orig"] == [2, 5] and c["densenet121"]["bdcxr"]["ctrl"] == [0, 25]
    assert c["convnextv2_tiny"]["bdcxr"]["ctrl"] == [2, 25] and c["rad-dino"]["vin"]["orig"] == [1, 5]  # encoders outside --expect show too
    assert rep["done"] == 7 and rep["expected"] == 3 * 175 and rep["copied"]["vin_hyb_orig@rad-dino_s3"] == "s4-p2a-vin-rd-01"


def test_duplicates_conflicts_and_existing_destination(tmp_path, capsys):
    src, dst = tmp_path / "p2", tmp_path / "down"
    a, b, c, d, e = (f"nih_hyb_{x}@densenet121_s0" for x in ("orig", "soft", "early", "ctrl", "hybrid"))
    _run(src / "k0/results" / a, drop=("val_pred.npy",))  # broken fetch of the same run: agrees, but is not the copy taken
    _run(src / "k1/results" / a)
    _run(src / "k1-r/results" / a)  # done_before copy in the rerun kernel: agrees
    _run(src / "k1/results" / b, val=0.80)
    _run(src / "k2/results" / b, val=0.81)  # the same run trained twice: disagrees
    _run(src / "k1/results" / c, test=0.70)
    _run(dst / "nih" / c, test=0.71)  # collected earlier from another training
    _run(src / "k1/results" / d)
    _run(dst / "nih" / d)  # collected earlier: the same run
    _run(src / "k1/results" / e)
    _run(src / "k3/results" / e, val=0.9, drop=("test_pred.npy",))  # an incomplete copy still takes part in the agreement check
    rep = _collect(src, dst)
    assert rep["copied"] == {a: "k1"} and rep["duplicates"] == {a: ["k0", "k1", "k1-r"]}
    assert rep["conflicts"] == {b: dict(val_macro_auc={"k1": 0.8, "k2": 0.81}, test_differs=False),
                                e: dict(val_macro_auc={"k1": 0.8, "k3": 0.9}, test_differs=False)}
    assert not (dst / "nih" / b).exists() and not (dst / "nih" / e).exists()
    assert rep["dst_differs"] == {c: dict(val_macro_auc={"dst": 0.8, "k1": 0.8}, test_differs=True)}
    shown = [x for x in capsys.readouterr().out.splitlines() if x.startswith(("conflicts:", "dst_differs:"))]
    assert len(shown) == 3 and "0.7" not in "".join(shown) + json.dumps([rep["conflicts"], rep["dst_differs"]])  # no test value
    assert json.loads((dst / "nih" / c / "result.json").read_text())["test_macro_auc"] == 0.71  # never overwritten
    assert rep["kept"] == [d]
    assert rep["counts"]["densenet121"]["nih"] == dict(orig=[1, 5], soft=[0, 5], early=[1, 5], hybrid=[0, 5], ctrl=[1, 5])
    again = _collect(src, dst)
    assert again["copied"] == {} and sorted(again["kept"]) == sorted([a, d]) and again["done"] == 3


def test_unfinished_incomplete_missing_and_exit_code(tmp_path):
    src, dst = tmp_path / "p2", tmp_path / "down"
    u = src / "k1/results/nih_hyb_hybrid@densenet121_s0"
    u.mkdir(parents=True)
    (u / "log.jsonl").write_text("")
    (u / "ckpt.pt").write_bytes(b"c")  # fetched without the phase-2 pattern: the checkpoint came home
    summary = dict(jobs=[], done=[], done_before=[], warm_up={}, warm_up_s=0,
                   unfinished=["nih_hyb_ctrl@densenet121_s1", "nih_hyb_hybrid@densenet121_s2"])
    (src / "k1/run_summary.json").write_text(json.dumps(summary))
    _run(src / "k1-r/results/nih_hyb_hybrid@densenet121_s2")  # resumed and finished in the rerun kernel
    (src / "k1-r/run_summary.json").write_text(json.dumps(summary)[:40])  # killed downloads: truncated, and 0 bytes below
    (src / "k2").mkdir()
    (src / "k2/run_summary.json").write_bytes(b"")
    _run(src / "k1/results/nih_hyb_orig@densenet121_s4", drop=("test_pred.npy",))  # broken fetch
    _run(src / "k1/results/nih_hyb_soft@densenet121_s4", empty=("val_idx.npy",))
    (src / "k1/results/nih_cls_orig_s0").mkdir()  # not a v3 run
    (src / "k1/results/nih_hyb_orig@densenet121_s3.log").write_text("")  # a run's stdout next to the run folders
    r = subprocess.run([sys.executable, str(ROOT / "tools/collect_v3.py"), "--src", str(src), "--dst", str(dst)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr  # a report: runs missing is not an error
    rep = json.loads((dst / "collect_report.json").read_text())
    assert rep["unfinished"] == ["nih_hyb_ctrl@densenet121_s1", "nih_hyb_hybrid@densenet121_s0"]
    assert rep["incomplete"] == {"nih_hyb_orig@densenet121_s4": ["k1"], "nih_hyb_soft@densenet121_s4": ["k1"]}
    assert not (dst / "nih/nih_hyb_orig@densenet121_s4").exists() and not (dst / "nih/nih_hyb_soft@densenet121_s4").exists()
    assert rep["ignored"] == ["k1/results/nih_cls_orig_s0"] and rep["done"] == 1 and rep["expected"] == 175
    assert rep["bad_summaries"] == ["k1-r/run_summary.json", "k2/run_summary.json"]  # reported; k1-r's run is still collected
    assert "bad_summaries (2): k1-r/run_summary.json, k2/run_summary.json" in r.stdout
    assert len(rep["missing"]) == 175 - 1 - 2 - 2 and "bdcxr_hyb_early@densenet121_s4_f2" in rep["missing"]
    assert any(line.split() == ["densenet121", "nih", "0/5", "0/5", "0/5", "1/5", "0/5", "1/25"] for line in r.stdout.splitlines()), r.stdout
    with pytest.raises(SystemExit):
        C.main(["--src", str(tmp_path / "nothing"), "--dst", str(dst)])


def test_expected_matrix_is_the_generated_schedule():
    """Per encoder, the tags expected are those of the jobs tools/v3_phase2.py schedules, as the kernel names their result folders."""
    tim = json.loads((ROOT / "results/v3/timings.json").read_text())
    for arch in V.ENC:
        jobs = {job_tag([0, "hyb", j["ds"], j["cfg"], j["seed"], V.EPOCHS, *V.argv(j, tim)]) for j in V.make_jobs([arch], tim, 1.15)}
        want = {C.tag(ds, cfg, arch, s, k) for ds, folds in C.FOLDS.items() for cfg in C.CONFIGS for s in C.SEEDS for k in folds}
        assert jobs == want and len(want) == 175 and {C.TAG.fullmatch(t).group(2) for t in want} == {arch}
