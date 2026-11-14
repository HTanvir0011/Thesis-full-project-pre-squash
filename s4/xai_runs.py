"""H10 XAI over many runs (SAP_v3.md H10): find the finished NIH hybrid runs in the mounted outputs of earlier phase-2 kernels and run
s4/xai_multistream.py on each, one process per GPU, runs one after another per GPU; then collect the fetched outputs.

usage: from s4.xai_runs import expected, find_boxes, run_all, runs, summary   (ops/launch_xai.py; kernels/xai/run_xai.py only collects paths)
       python -m s4.xai_runs [--src outputs/v3/xai_dl] [--dst results/v3/xai]   (fetched kernel outputs -> where tools/hybrid_stats.py reads)
"""
import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .runtag import tag as job_tag

TAG = re.compile(r"nih_hyb_hybrid@[\w-]+_s\d+")  # s4/runtag.py name of an NIH hybrid run (NIH has no folds)


def runs(paths, tag=TAG):
    """Candidate model.pt paths -> {run tag: run dir}, one dir per tag. Only .../results/<tag>/model.pt next to a result.json counts:
    s4/cls_hybrid.py saves model.pt before the final predictions, so without result.json the run was cut off and is not the evaluated one.
    A finished run also sits in every later kernel that copied it (done_before, identical files): the first path in sorted order is used."""
    out = {}
    for p in map(Path, sorted(map(str, paths))):
        d = p.parent
        if p.name == "model.pt" and d.parent.name == "results" and tag.fullmatch(d.name) and p.is_file() and (d / "result.json").is_file():
            out.setdefault(d.name, str(d))
    return out


def expected(job_files, tag=TAG):
    """Job files of the source kernels (outputs/jobs/<slug>.json: kernels/down/run_down.py jobs) -> sorted NIH hybrid run tags they were to
    produce. None at all is an error: the --sources hold nothing to explain."""
    files = [str(f) for f in job_files]
    tags = sorted({t for f in files for t in map(job_tag, json.loads(Path(f).read_text())) if tag.fullmatch(t)})
    if not tags:
        raise SystemExit(f"no run matching {tag.pattern} in the job files {files}")
    return tags


def lanes(found, n=2):
    """{tag: dir} -> n lists of (tag, dir), dealt round-robin in tag order: list g runs on GPU g."""
    items = sorted(found.items())
    return [items[g::n] for g in range(n)]


def command(run_dir, data, boxes, out, limit=0):
    """argv of one s4/xai_multistream.py run; 24 maps are kept for the figures; limit > 0: only the first `limit` boxed test images."""
    return [sys.executable, "-m", "s4.xai_multistream", "--run", str(run_dir), "--data", str(data), "--boxes", str(boxes), "--out", str(out),
            "--maps", "24", *(["--limit", str(limit)] if limit else [])]


def find_boxes(paths):
    """BBox_List_2017.csv among candidate paths (the Kaggle data set nih-chest-xrays/data mounts at varying depths)."""
    paths = [str(p) for p in paths]
    hit = sorted(p for p in paths if Path(p).name == "BBox_List_2017.csv" and os.path.isfile(p))
    if not hit:
        raise SystemExit(f"BBox_List_2017.csv not among {paths[:10]}")
    return hit[0]


def run_all(found, data, boxes, out, env, limit, deadline, ngpu=2):
    """Every run of `found` -> <out>/<tag>.json (+ .maps.npz, .log), lanes(found, ngpu) in parallel -> {tag: return code}. A run still
    going at `deadline` (a time.time() value) is killed (code 'timeout'), so the kernel ends and saves its output before the session limit."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rc = {}

    def lane(gpu, items):
        for tag, d in items:
            t0, cmd = time.time(), command(d, data, boxes, out / f"{tag}.json", limit)
            try:
                with open(out / f"{tag}.log", "w") as f:
                    rc[tag] = subprocess.run(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT,
                                             timeout=max(1.0, deadline - t0)).returncode
            except subprocess.TimeoutExpired:
                rc[tag] = "timeout"
            print(tag, "rc", rc[tag], f"{time.time() - t0:.0f}s", flush=True)

    ts = [threading.Thread(target=lane, args=(g, items)) for g, items in enumerate(lanes(found, ngpu))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return rc


def summary(found, rc, expect):
    """The kernel's run_summary.json: jobs (the key ops/fetch.py reads) = expected and found tags, done = return code 0, failed = the rest.
    missing (also failed) = expected runs with no finished model.pt mounted, e.g. one only a rerun kernel <slug>-r finished: launch again
    with that kernel among --sources."""
    jobs = sorted(set(expect) | set(found))
    return dict(jobs=jobs, runs=found, done=[t for t in jobs if rc.get(t) == 0], failed=[t for t in jobs if rc.get(t) != 0],
                missing=sorted(set(expect) - set(found)))


def collect(src, dst):
    """Fetched XAI kernels <src>/<slug>/ (run_summary.json, xai/<tag>.json, .maps.npz) -> <dst>/<tag>.json, .maps.npz, read by
    tools/hybrid_stats.py (H10) and tools/generate_all_thesis_figures.py -> report. Only tags in a run_summary 'done' are copied (a run
    killed at the deadline can leave partial files); an existing destination file is never overwritten (one that differs is reported).
    failed: tags a kernel failed or missed that are not in dst; no_summary: kernels without run_summary.json (died or not fetched)."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    rep, failed = dict(copied=[], kept=[], differs=[], incomplete=[], no_summary=[]), set()
    for k in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        if not (k / "run_summary.json").is_file():
            rep["no_summary"].append(k.name)
            continue
        s = json.loads((k / "run_summary.json").read_text())
        failed |= set(s["failed"])
        for t in s["done"]:
            pairs = [(k / "xai" / f"{t}{e}", dst / f"{t}{e}") for e in (".json", ".maps.npz")]
            if not all(f.is_file() and f.stat().st_size for f, _ in pairs):
                rep["incomplete"].append(f"{k.name}/{t}")  # not every file came home: fetch again
            elif any(o.exists() and not filecmp.cmp(f, o, shallow=False) for f, o in pairs):
                rep["differs"].append(f"{k.name}/{t}")
            else:
                new = [(f, o) for f, o in pairs if not o.exists()]
                for f, o in new:  # copy, then rename: an interrupted copy never looks like a collected file
                    shutil.copyfile(f, o.with_name(o.name + ".tmp"))
                    os.replace(o.with_name(o.name + ".tmp"), o)
                rep["copied" if new else "kept"].append(t)
    rep["failed"] = sorted(t for t in failed if not ((dst / f"{t}.json").exists() and (dst / f"{t}.maps.npz").exists()))
    return rep


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs/v3/xai_dl"), help="fetched XAI kernel outputs, one folder per kernel")
    ap.add_argument("--dst", default=str(ROOT / "results/v3/xai"))
    a = ap.parse_args()
    print(json.dumps(collect(a.src, a.dst), indent=1))
