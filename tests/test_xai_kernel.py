import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

import s4.xai_runs as X

ROOT = Path(__file__).resolve().parents[1]


def run_dir(root, *parts, finished=True):
    d = Path(root, *parts)
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"0")
    if finished:
        (d / "result.json").write_text("{}")
    return d


def test_runs_found_once_per_tag(tmp_path):
    run_dir(tmp_path, "s4-k1", "results", "nih_hyb_hybrid@densenet121_s0")
    a = run_dir(tmp_path, "notebooks", "u", "s4-k2", "results", "nih_hyb_hybrid@densenet121_s0")  # done_before copy, other depth; sorts first
    c = run_dir(tmp_path, "notebooks", "u", "s4-k2", "results", "nih_hyb_hybrid@rad-dino_s3")
    run_dir(tmp_path, "a-cut", "results", "nih_hyb_hybrid@densenet121_s2", finished=False)  # cut off before result.json
    e = run_dir(tmp_path, "z-rerun", "results", "nih_hyb_hybrid@densenet121_s2")
    run_dir(tmp_path, "s4-k3", "results", "nih_hyb_hybrid@densenet121_s1", finished=False)
    run_dir(tmp_path, "s4-k1", "results", "nih_hyb_orig@densenet121_s0")
    run_dir(tmp_path, "s4-k1", "results", "vin_hyb_hybrid@densenet121_s0")
    run_dir(tmp_path, "s4-k1", "xai", "nih_hyb_hybrid@densenet121_s4")  # not a results folder
    paths = [str(p) for p in tmp_path.rglob("model.pt")] + [str(tmp_path / "gone/results/nih_hyb_hybrid@densenet121_s3/model.pt")]
    assert X.runs(paths) == {"nih_hyb_hybrid@densenet121_s0": str(a), "nih_hyb_hybrid@densenet121_s2": str(e),
                             "nih_hyb_hybrid@rad-dino_s3": str(c)}


def test_v4_pattern_takes_orig_swin_and_densenet_runs(tmp_path):
    v4 = re.compile(r"nih_hyb_orig@(swinv2_tiny_window8_256(\+xray|\+xraykd)?|densenet121)_s\d+")
    a = run_dir(tmp_path, "s4-v4-p1-b", "results", "nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s3")
    b = run_dir(tmp_path, "datasets", "u", "s4-dn121", "results", "nih_hyb_orig@densenet121_s4")
    run_dir(tmp_path, "s4-v4-p1-b", "results", "qp_xm_xraykd_s3")
    h = run_dir(tmp_path, "s4-k1", "results", "nih_hyb_hybrid@densenet121_s0")
    paths = [str(p) for p in tmp_path.rglob("model.pt")]
    assert X.runs(paths, v4) == {"nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s3": str(a), "nih_hyb_orig@densenet121_s4": str(b)}
    assert X.runs(paths) == {"nih_hyb_hybrid@densenet121_s0": str(h)}  # the default (v3) is unchanged
    j = tmp_path / "s4-v4-p1-b.json"
    j.write_text(json.dumps([[0, "xm", "qp", "xraykd", 3, 15, "--stage", "xray", "--arch", "swinv2_tiny_window8_256.ms_in1k"],
                             [0, "hyb", "nih", "orig", 3, 10, "--arch", "swinv2_tiny_window8_256.ms_in1k", "--enc_lr", "0.25",
                              "--init", "@run:qp_xm_xraykd_s3/model.pt"]]))
    assert X.expected([j], v4) == ["nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s3"]


def test_lanes_alternate_over_gpus():
    found = {f"nih_hyb_hybrid@densenet121_s{s}": f"d{s}" for s in (4, 0, 3, 1, 2)}
    lanes = X.lanes(found, 2)
    assert [[d for _, d in lane] for lane in lanes] == [["d0", "d2", "d4"], ["d1", "d3"]]
    assert X.lanes({"t": "d"}, 2) == [[("t", "d")], []]


def test_command_is_accepted_by_the_xai_cli(monkeypatch):
    c = X.command("R", "D", "B.csv", "o/t.json")
    assert c == [sys.executable, "-m", "s4.xai_multistream", "--run", "R", "--data", "D", "--boxes", "B.csv", "--out", "o/t.json",
                 "--maps", "24"]
    assert X.command("R", "D", "B.csv", "o/t.json", limit=20)[-2:] == ["--limit", "20"]
    import s4.xai_multistream as M

    def stop(run, dev):
        raise RuntimeError(f"parsed {run}")
    monkeypatch.setattr(M, "load_model", stop)  # argparse exits on an unknown flag before this is reached
    with pytest.raises(RuntimeError, match="parsed R"):
        M.main(X.command("R", "D", "B.csv", "o/t.json", limit=20)[3:])


def test_find_boxes(tmp_path):
    f = tmp_path / "datasets" / "nih-chest-xrays" / "data" / "BBox_List_2017.csv"
    f.parent.mkdir(parents=True)
    f.write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,\n")
    other = f.parent / "Data_Entry_2017.csv"
    other.write_text("x\n")
    assert X.find_boxes([tmp_path / "nih-chest-xrays/data/BBox_List_2017.csv", other, f]) == str(f)
    with pytest.raises(SystemExit):
        X.find_boxes([other])


def test_run_all_one_process_per_gpu(tmp_path, monkeypatch):
    prog = {"a": "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
            "b": "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); raise SystemExit(3)",
            "c": "import time; time.sleep(20)"}
    monkeypatch.setattr(X, "command", lambda d, *a: [sys.executable, "-c", prog[d]])
    t0 = time.time()
    rc = X.run_all({"t_a": "a", "t_b": "b", "t_c": "c"}, "D", "B", tmp_path / "xai", dict(os.environ), 0, time.time() + 4)
    assert rc == {"t_a": 0, "t_b": 3, "t_c": "timeout"}  # lanes: GPU 0 runs t_a then t_c, GPU 1 runs t_b
    assert [(tmp_path / "xai" / f"t_{k}.log").read_text().strip() for k in "ab"] == ["0", "1"]
    assert time.time() - t0 < 15  # the run past the deadline was killed, not awaited


def test_launch_xai_arguments():
    sys.path.insert(0, str(ROOT / "ops"))
    from launch_xai import parse
    assert parse(["ac_6", "k", "--sources", "s4-a,s4-b"]) == ("ac_6", "k", ["s4-a", "s4-b"], "s4-prep-nih", 0, X.TAG.pattern, [])
    assert parse(["ac_7", "k", "--prep", "p", "--sources", "s4-a", "--limit", "20"])[2:5] == (["s4-a"], "p", 20)
    assert parse(["ac_9", "k", "--sources", "s4-a", "--runs", "r_s\\d", "--datasets", "s4-dn121"])[5:] == ("r_s\\d", ["s4-dn121"])
    for bad in (["ac_6", "k"], ["ac_6", "k", "--prep", "p"], ["ac_6", "k", "--sources"], ["ac_6", "k", "--sources", "a,,b"],
                ["ac_6", "k", "--sources", "a", "--source", "b"]):  # a typo is an error, not silently ignored
        with pytest.raises(SystemExit):
            parse(bad)


def test_expected_runs_from_job_files(tmp_path):
    a, b, c = tmp_path / "s4-k1.json", tmp_path / "s4-k1-r.json", tmp_path / "s4-vin.json"
    a.write_text(json.dumps([[0, "hyb", "nih", "hybrid", 0, 20, "--arch", "densenet121.ra_in1k"], [1, "hyb", "nih", "orig", 0, 20],
                             [1, "hyb", "nih", "hybrid", 3, 20, "--arch", "hf:microsoft/rad-dino", "--enc_lr", "0.25"]]))
    b.write_text(json.dumps([[0, "hyb", "nih", "hybrid", 0, 20, "--arch", "densenet121.ra_in1k"]]))  # a rerun repeats a planned run
    c.write_text(json.dumps([[0, "hyb", "vin", "hybrid", 0, 20], [1, "hyb", "bdcxr", "hybrid", 0, 20, "--split", "split_f2.npy"]]))
    assert X.expected(p for p in (a, b, c)) == ["nih_hyb_hybrid@densenet121_s0", "nih_hyb_hybrid@rad-dino_s3"]
    with pytest.raises(SystemExit):  # sources without an NIH hybrid run: nothing to explain
        X.expected([c])


def test_template_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(ROOT / "ops"), *sys.path])  # restored afterwards: the kernel also puts its code folder first
    from launch_xai import render
    t, k = "nih_hyb_hybrid@densenet121_s", tmp_path / "kaggle"
    for f in ("input/s4-code/s4/__init__.py", "input/s4-prep-nih/nih/split.npy", "input/datasets/nih-chest-xrays/data/BBox_List_2017.csv"):
        (k / f).parent.mkdir(parents=True)
        (k / f).write_text("")
    run_dir(k, "input", "notebooks", "u", "s4-k1", "results", t + "0")
    run_dir(k, "input", "s4-k1-r", "results", t + "1")  # finished by a rerun kernel
    src = render((ROOT / "kernels/xai/run_xai.py").read_text(), 20, [t + "0", t + "1", t + "2"], X.TAG.pattern)  # s2: expected, not mounted
    monkeypatch.setattr(X, "command", lambda d, *a: [sys.executable, "-c", f"raise SystemExit({3 if d.endswith('1') else 0})"])
    exec(compile(src.replace("/kaggle/", k.as_posix() + "/"), "run_xai.py", "exec"), {"__name__": "__main__"})
    s = json.loads((k / "working/run_summary.json").read_text())
    assert s["jobs"] == [t + "0", t + "1", t + "2"]  # ops/fetch.py reads run_summary.json['jobs'] of every fetched kernel
    assert (s["done"], s["failed"], s["missing"], sorted(s["runs"])) == ([t + "0"], [t + "1", t + "2"], [t + "2"], [t + "0", t + "1"])


def test_collect_copies_finished_outputs_once(tmp_path):
    t = "nih_hyb_hybrid@densenet121_s"

    def kernel(slug, summary, files):
        (tmp_path / "dl" / slug / "xai").mkdir(parents=True)
        if summary:
            (tmp_path / "dl" / slug / "run_summary.json").write_text(json.dumps(dict(done=summary[0], failed=summary[1])))
        for name, body in files.items():
            (tmp_path / "dl" / slug / "xai" / (t + name)).write_text(body)

    kernel("a", ([t + "0", t + "1"], [t + "2", t + "4"]), {"0.json": "r0", "0.maps.npz": "m0", "1.json": "r1", "2.json": "partial"})
    kernel("b", ([t + "0", t + "2", t + "3"], []), {"0.json": "r0", "0.maps.npz": "m0", "2.json": "r2", "2.maps.npz": "m2", "3.json": "r3",
                                                    "3.maps.npz": "m3"})
    kernel("c", ([t + "3"], []), {"3.json": "other", "3.maps.npz": "m3"})
    kernel("d", None, {})  # died before its summary
    rep = X.collect(tmp_path / "dl", tmp_path / "xai")
    assert rep == dict(copied=[t + "0", t + "2", t + "3"], kept=[t + "0"], differs=[f"c/{t}3"], incomplete=[f"a/{t}1"], no_summary=["d"],
                       failed=[t + "4"])  # a: s1 maps never came home, s2 failed there but b delivered it
    assert {p.name: p.read_text() for p in (tmp_path / "xai").iterdir()} == {
        t + "0.json": "r0", t + "0.maps.npz": "m0", t + "2.json": "r2", t + "2.maps.npz": "m2", t + "3.json": "r3", t + "3.maps.npz": "m3"}
    assert X.collect(tmp_path / "dl", tmp_path / "xai")["copied"] == []  # nothing copied or overwritten on a second pass
