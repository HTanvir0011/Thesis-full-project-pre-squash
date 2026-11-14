import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ops.jobqueue as Q
from s4.cls_hybrid import load_x
from s4.runtag import tag


def test_tags_match_run_down_naming():
    assert tag([0, "hyb", "nih", "hybrid", 0, 20, "--arch", "densenet121.ra_in1k"]) == "nih_hyb_hybrid@densenet121_s0"
    assert tag([1, "hyb", "nih", "orig", 3, 20, "--arch", "hf:microsoft/rad-dino"]) == "nih_hyb_orig@rad-dino_s3"
    assert tag([0, "hyb", "bdcxr", "ctrl", 2, 20, "--arch", "swinv2_base_window16_256.ms_in1k", "--accum", "8", "--split", "split_f4.npy"]) \
        == "bdcxr_hyb_ctrl@swinv2_base_window16_256_s2_f4"
    assert tag([0, "cls", "nih", "orig+soft", 1, 10]) == "nih_cls_orig-soft_s1"
    assert tag([0, "xm", "qp", "xraykd", 2, 15, "--stage", "xray", "--teacher", "@run:ct_xm_ct_s2/teacher.npz"]) == "qp_xm_xraykd_s2"
    assert tag([0, "hyb", "nih", "orig", 2, 10, "--arch", "swinv2_tiny_window8_256.ms_in1k", "--init", "@run:qp_xm_xraykd_s2/model.pt"]) \
        == "nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s2"  # fine-tuned from distilled projections: not the folder of the ImageNet start


def test_missing_runs_and_rerun_item(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    (tmp_path / "outputs" / "jobs").mkdir(parents=True)
    jobs = [[0, "hyb", "nih", c, 0, 20, "--arch", "densenet121.ra_in1k"] for c in ("orig", "soft", "hybrid")]
    (tmp_path / "outputs/jobs/s4-x.json").write_text(json.dumps(jobs))
    it = dict(id="s4-x", ac="ac_6", kernel="s4-x", est_h=9.0, cmd="python ops/launch_down.py ac_6 s4-x outputs/jobs/s4-x.json --prep s4-prep-nih",
              fetch="outputs/v3/p2/s4-x", rerun_missing=True)
    assert len(Q.missing_runs(it)) == 3  # nothing fetched (failed kernel without output): every job is missing
    d = tmp_path / "outputs/v3/p2/s4-x/results/nih_hyb_orig@densenet121_s0"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{}")
    miss = Q.missing_runs(it)
    assert [tag(j) for j in miss] == ["nih_hyb_soft@densenet121_s0", "nih_hyb_hybrid@densenet121_s0"]
    q = tmp_path / "q.json"
    q.write_text(json.dumps([it]))
    state = {"s4-x": dict(phase="done", **it)}
    Q.check_missing(it, state, q, dry=False)
    new = state["s4-x-r"]
    assert new["phase"] == "waiting" and "--prep s4-prep-nih" in new["cmd"] and new["ac"] == "ac_6"
    assert new["cmd"].endswith("--prev s4-x")  # the rerun mounts the kernel it replaces: unfinished runs resume from their checkpoint
    assert [j[0] for j in json.loads((tmp_path / "outputs/jobs/s4-x-r.json").read_text())] == [0, 1]  # spread over both GPUs
    assert [i["id"] for i in json.loads(q.read_text())] == ["s4-x", "s4-x-r"]  # restart-safe
    Q.check_missing(state["s4-x-r"], state, q, dry=False)  # a rerun item never spawns another
    assert "s4-x-r-r" not in state


def test_load_x_on_device_equals_cpu_path():
    rng = np.random.default_rng(0)
    arr = dict(orig=rng.integers(0, 256, (6, 64, 64), dtype=np.uint8), soft=rng.integers(0, 256, (6, 64, 64), dtype=np.uint8))
    idx = np.array([0, 2, 5])
    xo, xs = arr["orig"][idx].astype(np.int16), arr["soft"][idx].astype(np.int16)
    old = torch.from_numpy(np.stack([xo, xs, np.clip(xo - xs, 0, 255)], 1).astype(np.float32) / 255)  # the CPU path before R4
    assert torch.equal(load_x(arr, idx), old)
    assert torch.equal(load_x(arr, idx, 32), F.interpolate(old, size=(32, 32), mode="area"))


def test_load_x_exact_on_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(1)
    arr = dict(orig=rng.integers(0, 256, (10, 64, 64), dtype=np.uint8), soft=rng.integers(0, 256, (10, 64, 64), dtype=np.uint8))
    idx = np.array([1, 4, 7])
    xo, xs = arr["orig"][idx].astype(np.int16), arr["soft"][idx].astype(np.int16)
    old = torch.from_numpy(np.stack([xo, xs, np.clip(xo - xs, 0, 255)], 1).astype(np.float32) / 255)
    assert torch.equal(load_x(arr, idx, None, "cuda").cpu(), old)
    assert torch.equal(load_x(arr, idx, 32, "cuda").cpu(), F.interpolate(old, size=(32, 32), mode="area"))


def test_queue_flags_decide_reruns(tmp_path, monkeypatch):
    """The scheduler follows the item's rerun_missing flag; tools/v3_phase2.py sets it on normal kernels and on the last segment of
    each resume chain only (earlier segments end with unfinished runs by design)."""
    import tools.v3_phase2 as V
    monkeypatch.setattr(V, "ROOT", tmp_path)
    tim = json.loads((Path(__file__).resolve().parents[1] / "results/v3/timings.json").read_text())
    b = V.build("convnextv2_tiny", tim, 1.15, True, True, "t")
    V.write(b, "convnextv2_tiny", tim, "t")
    q = json.loads((tmp_path / "outputs/queue_v3_t_convnextv2_tiny.json").read_text())
    seg = {k["slug"]: k for k in b["plan"]}
    chains = [i for i in q if i["id"] in seg and seg[i["id"]]["chain"] is not None]
    assert chains, "the ConvNeXt-V2-T plan has resume chains"
    for i in q:
        k = seg.get(i["id"])
        if k is None:
            continue  # NIH prep copies
        last = k["chain"] is None or k["seg"] == k["nseg"] - 1
        assert i["rerun_missing"] == last, i["id"]
        assert ("--prev" in i["cmd"]) == (k["chain"] is not None and k["seg"] > 0), i["id"]

    monkeypatch.setattr(Q, "ROOT", tmp_path)
    first = next(i for i in chains if not i["rerun_missing"])
    state = {first["id"]: dict(phase="done", **first)}
    Q.check_missing(first, state, tmp_path / "q.json", dry=False)
    assert list(state) == [first["id"]]  # an earlier chain segment never spawns a rerun


def test_runs_finished_in_earlier_kernels_are_not_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    (tmp_path / "outputs" / "jobs").mkdir(parents=True)
    jobs = [[0, "hyb", "nih", c, 0, 20, "--arch", "densenet121.ra_in1k"] for c in ("orig", "soft")]
    (tmp_path / "outputs/jobs/k.json").write_text(json.dumps(jobs))
    d = tmp_path / "outputs/f"
    d.mkdir(parents=True)
    (d / "run_summary.json").write_text(json.dumps(dict(jobs=jobs, done=[], done_before=["nih_hyb_orig@densenet121_s0"])))
    it = dict(id="k", ac="ac_6", kernel="k", est_h=1.0, cmd="python ops/launch_down.py ac_6 k outputs/jobs/k.json --prep s4-prep-nih", fetch="outputs/f")
    assert [tag(j) for j in Q.missing_runs(it)] == ["nih_hyb_soft@densenet121_s0"]
