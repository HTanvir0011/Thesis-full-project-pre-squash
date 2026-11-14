"""R1 review fixes: argument check and log trim on resume, --prev placeholder, second rerun level, chain sessions by whole epochs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import ops.jobqueue as Q
from s4.cls_hybrid import main
from s4.runtag import tag

ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path):
    rng = np.random.default_rng(0)
    n, s = 48, 64
    np.save(tmp_path / "orig.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    np.save(tmp_path / "soft.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    np.save(tmp_path / "labels.npy", (rng.random((n, 4)) < 0.5).astype(np.float32))
    np.save(tmp_path / "split.npy", np.array(["train"] * 24 + ["val"] * 12 + ["test"] * 12))
    return ["--data", str(tmp_path), "--config", "hybrid", "--arch", "resnet18", "--pretrained", "0", "--epochs", "3", "--max_steps", "2",
            "--bs", "8", "--size", "64"]


def test_checkpoint_every_epoch_and_settings_check(tmp_path):
    args, out = _args(tmp_path), tmp_path / "r"
    with pytest.raises(SystemExit) as e:
        main(args + ["--out", str(out), "--stop_after_epochs", "1"])
    assert e.value.code == 75
    with pytest.raises(SystemExit) as e:  # another learning rate: the checkpoint is refused, not silently mixed
        main([a if a != "3" else "3" for a in args] + ["--lr", "1e-3", "--out", str(out), "--resume", str(out / "ckpt.pt")])
    assert "other settings" in str(e.value.code)


def test_resume_trims_log_lines_after_the_checkpoint(tmp_path):
    args, out = _args(tmp_path), tmp_path / "r"
    with pytest.raises(SystemExit):
        main(args + ["--out", str(out), "--stop_after_epochs", "1"])
    with open(out / "log.jsonl", "a") as f:  # a session that died after logging epoch 1 but before its checkpoint
        f.write(json.dumps(dict(epoch=1, val_macro_auc=0.0, loss=0.0, sec=0)) + "\n")
    main(args + ["--out", str(out), "--resume", str(out / "ckpt.pt")])
    epochs = [json.loads(x)["epoch"] for x in (out / "log.jsonl").read_text().splitlines()]
    assert epochs == [0, 1, 2]


def test_render_fills_jobs_and_prev():
    sys.path.insert(0, str(ROOT / "ops"))
    from launch_down import render
    t = (ROOT / "kernels/down/run_down.py").read_text(encoding="utf-8")
    s = render(t, [[0, "hyb", "nih", "orig", 0, 1]], "s4-x-c00-0")
    assert "PREV_SLUG = 's4-x-c00-0'" in s and "__JOBS__" not in s and "__PREV__" not in s
    assert "PREV_SLUG = None" in render(t, [], None)


def test_second_rerun_level_only_for_runs_with_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    (tmp_path / "outputs/jobs").mkdir(parents=True)
    jobs = [[0, "hyb", "nih", c, 0, 20, "--arch", "densenet121.ra_in1k"] for c in ("hybrid", "ctrl", "orig")]
    (tmp_path / "outputs/jobs/k-r.json").write_text(json.dumps(jobs))
    d = tmp_path / "outputs/f-r"
    d.mkdir(parents=True)
    (d / "run_summary.json").write_text(json.dumps(dict(jobs=jobs, done=["nih_hyb_orig@densenet121_s0"], unfinished=["nih_hyb_hybrid@densenet121_s0"])))
    it = dict(id="k-r", ac="ac_6", kernel="k-r", est_h=9.0, rerun_missing=True, fetch="outputs/f-r",
              cmd="python ops/launch_down.py ac_6 k-r outputs/jobs/k-r.json --prep s4-prep-nih --prev k")
    q = tmp_path / "q.json"
    q.write_text(json.dumps([it]))
    state = {"k-r": dict(phase="done", **it)}
    Q.check_missing(it, state, q, dry=False)
    new = state["k-r2"]
    assert [tag(j) for j in json.loads((tmp_path / "outputs/jobs/k-r2.json").read_text())] == ["nih_hyb_hybrid@densenet121_s0"]
    assert new["cmd"].endswith("--prev k-r")
    Q.check_missing(new, state, q, dry=False)  # -r2 never spawns a third level
    assert not any(k.endswith("-r3") or k.endswith("-r2-r") for k in state)


def test_chain_sessions_count_whole_epochs():
    sys.path.insert(0, str(ROOT / "tools"))
    import v3_phase2 as V
    tim = json.loads((ROOT / "results/v3/timings.json").read_text())
    for winner, want in (("swinv2_base_window16_256", 7), ("convnextv2_tiny", 3)):  # three-stream NIH runs; epochs of ~3.1 h / ~1.0 h
        b = V.build(winner, tim, 1.15, True, True, "t")
        nseg = {k["nseg"] for k in b["plan"] if k["chain"] is not None and k["arch"] == winner and k["ds"] == "nih"
                and any(j["cfg"] in ("hybrid", "ctrl") for lane in k["lanes"] for j in lane)}
        assert nseg == {want}, (winner, nseg)
