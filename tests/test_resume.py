"""R1: a run interrupted at epoch ends and resumed from its checkpoint equals an uninterrupted run (CPU, fixed seed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from s4.cls_hybrid import main
from s4.runtag import previous_run


def _data(tmp_path):
    rng = np.random.default_rng(0)
    n, s = 48, 64
    np.save(tmp_path / "orig.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    np.save(tmp_path / "soft.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    np.save(tmp_path / "labels.npy", (rng.random((n, 4)) < 0.5).astype(np.float32))
    np.save(tmp_path / "split.npy", np.array(["train"] * 24 + ["val"] * 12 + ["test"] * 12))
    return ["--data", str(tmp_path), "--config", "hybrid", "--arch", "resnet18", "--pretrained", "0", "--epochs", "3", "--max_steps", "3",
            "--bs", "8", "--size", "64"]


def test_resume_equals_uninterrupted(tmp_path):
    args = _data(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    main(args + ["--out", str(a)])
    for _ in range(2):  # two interruptions: after epoch 1, then after epoch 2
        with pytest.raises(SystemExit) as e:
            main(args + ["--out", str(b), "--stop_after_epochs", "1"] + (["--resume", str(b / "ckpt.pt")] if (b / "ckpt.pt").exists() else []))
        assert e.value.code == 75 and (b / "ckpt.pt").exists() and not (b / "result.json").exists()
    main(args + ["--out", str(b), "--resume", str(b / "ckpt.pt")])
    assert not (b / "ckpt.pt").exists()  # removed once the run is finished
    assert np.array_equal(np.load(a / "test_pred.npy"), np.load(b / "test_pred.npy"))
    assert np.array_equal(np.load(a / "val_pred.npy"), np.load(b / "val_pred.npy"))
    la = [json.loads(x) for x in (a / "log.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (b / "log.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lb] == [0, 1, 2] and [x["val_macro_auc"] for x in la] == [x["val_macro_auc"] for x in lb]


def test_deadline_stops_before_the_next_epoch(tmp_path):
    args = _data(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(args + ["--out", str(tmp_path / "c"), "--deadline_h", "1e-9"])
    assert e.value.code == 75 and (tmp_path / "c" / "ckpt.pt").exists()


def test_previous_run_lookup(tmp_path):
    r1, r2 = tmp_path / "k1" / "results", tmp_path / "k2" / "results"
    (r1 / "x_s0").mkdir(parents=True)
    (r1 / "x_s0" / "ckpt.pt").write_bytes(b"0")
    (r2 / "x_s0").mkdir(parents=True)
    (r2 / "x_s0" / "result.json").write_text("{}")
    (r2 / "y_s0").mkdir(parents=True)
    assert previous_run("x_s0", [str(r1), str(r2)]) == ("done", str(r2 / "x_s0"))  # a finished copy wins over a checkpoint
    assert previous_run("y_s0", [str(r1), str(r2)]) == (None, None)
    (r1 / "y_s0").mkdir()
    (r1 / "y_s0" / "ckpt.pt").write_bytes(b"0")
    assert previous_run("y_s0", [str(r1), str(r2)]) == ("resume", str(r1 / "y_s0"))


def test_launch_down_arguments():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
    from launch_down import parse
    assert parse(["ac_6", "k", "j.json", "--prep", "s4-prep-nih"]) == ("ac_6", "k", "j.json", ["s4-prep-nih"], None, [])
    assert parse(["ac_6", "k", "j.json", "--prep", "a,b", "--prev", "k0"])[3:] == (["a", "b"], "k0", [])
    assert parse(["ac_6", "k", "j.json", "--prep", "a", "--datasets", "s4-qproj"])[3:] == (["a"], None, ["s4-qproj"])
    with pytest.raises(SystemExit):
        parse(["ac_6", "k", "j.json", "--prep", "a", "--prv", "k0"])  # a typo is an error, not silently ignored
