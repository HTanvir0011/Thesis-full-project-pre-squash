"""Kaggle GPU kernel (template): downstream jobs (classification / detection), one process per GPU, jobs sequential per GPU.

[[0, 'hyb', 'nih', 'hybrid', 0, 2, '--arch', 'densenet121.ra_in1k', '--max_steps', '40', '--max_eval', '256'], [1, 'hyb', 'nih', 'orig', 0, 2, '--arch', 'densenet121.ra_in1k', '--max_steps', '40', '--max_eval', '256']] = [[gpu, kind, data_dir_name, variant, seed, epochs, extra_argv...], ...]
kind 'cls': variant is a '+'-joined channel list (orig | soft | orig+soft); kind 'det': orig | soft | both;
kind 'hyb' (s4/cls_hybrid.py): variant is the configuration (orig | soft | early | hybrid | ctrl), the tag carries @<arch> from --arch.
The prepared arrays come from a kernel-output source mounted under /kaggle/input/<prep kernel slug>/<dataset>/.
Before the lanes start, every large array of the data sets in use is read once sequentially (RUNPLAN_v3 R2): the first epoch then finds
it in the page cache instead of reading it at random (v2: +44 min per 7.3 GB array); the time is logged and kept in run_summary.json.
Resume across sessions (RUNPLAN_v3 R1): a run that an earlier kernel (mounted with launch_down --prev) finished is copied over and
skipped; one it left unfinished continues from its checkpoint; every hyb run gets a deadline so it checkpoints and exits (75) before the
session reaches SESSION_H, and a run is not started when less than 0.75 h is left.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

JOBS = [[0, 'hyb', 'nih', 'hybrid', 0, 2, '--arch', 'densenet121.ra_in1k', '--max_steps', '40', '--max_eval', '256'], [1, 'hyb', 'nih', 'orig', 0, 2, '--arch', 'densenet121.ra_in1k', '--max_steps', '40', '--max_eval', '256']]
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
ENV = dict(os.environ, PYTHONPATH=CODE, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
sys.path.insert(0, CODE)
from s4.runtag import previous_run  # noqa: E402
from s4.runtag import tag as job_tag  # noqa: E402  (same folder names as ops/jobqueue.py's missing-run check)
KERNEL_T0 = time.time()
SESSION_H = 11.5  # Kaggle ends a session at 12 h; the rest covers saving the output
PREV = sorted({os.path.dirname(os.path.dirname(p)) for d in range(1, 5)  # kernel outputs mount at varying depths
               for p in glob.glob("/kaggle/input/" + "*/" * d + "results/*/log.jsonl")})  # results folders of earlier kernels (--prev)
DONE_BEFORE = []
RES = Path("/kaggle/working/results")
RES.mkdir(parents=True, exist_ok=True)


def find_data(name):
    hit = sorted(glob.glob(f"/kaggle/input/**/{name}/split.npy", recursive=True))  # kernel outputs mount at varying depths
    assert hit, f"prepared data {name} not mounted; inputs: {sorted(glob.glob('/kaggle/input/*/*'))[:20]}"
    return str(Path(hit[0]).parent)


def warm_up(names, block=64 << 20):
    """Read orig.npy and soft.npy of the given data sets once, sequentially, two files at a time -> {file: [GB, seconds]}.
    Only these two: s4/cls_hybrid.py derives the bone stream from them, and an unused bone.npy (detection preps) would only crowd the cache."""
    dirs = [os.path.dirname(h[0]) for n in names for h in [sorted(glob.glob(f"/kaggle/input/**/{n}/split.npy", recursive=True))] if h]
    files = sorted({os.path.join(d, f"{a}.npy") for d in dirs for a in ("orig", "soft") if os.path.exists(os.path.join(d, f"{a}.npy"))})
    out = {}

    def read(f):
        t0 = time.time()
        with open(f, "rb", buffering=0) as fh:
            while fh.read(block):
                pass
        out[f] = [round(os.path.getsize(f) / 1e9, 2), round(time.time() - t0, 1)]
        print("warm-up", f, out[f], flush=True)

    for i in range(0, len(files), 2):
        ts = [threading.Thread(target=read, args=(f,)) for f in files[i:i + 2]]
        [t.start() for t in ts]
        [t.join() for t in ts]
    return out


def worker(gpu, jobs):
    for job in jobs:
        _, kind, name, variant, seed, epochs, *extra = job
        tag = job_tag(job)
        out = RES / tag
        state, pdir = previous_run(tag, PREV)
        if state == "done":  # finished in an earlier session: keep a copy here so later kernels and the fetch see it
            shutil.copytree(pdir, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt.pt*"))
            DONE_BEFORE.append(tag)
            print(tag, "finished in an earlier kernel; copied", flush=True)
            continue
        left = SESSION_H - (time.time() - KERNEL_T0) / 3600
        if kind == "hyb" and left < 0.75:
            print(tag, f"not started: {left:.2f} h of the session left", flush=True)
            continue
        base = [sys.executable, "-m", {"cls": "s4.cls", "det": "s4.det", "hyb": "s4.cls_hybrid"}[kind], "--data", find_data(name), "--seed", str(seed),
                "--epochs", str(epochs), "--out", str(out)]
        flag = {"cls": ["--variants", *variant.split("+")], "det": ["--variant", variant], "hyb": ["--config", variant]}[kind]
        base += flag + [str(e) for e in extra]
        if kind == "hyb":
            base += ["--deadline_h", f"{left - 0.5:.2f}"]  # >= 0.25; leaves 0.5 h for the final predictions (Swin hybrid: ~23 min)
            if state == "resume":
                out.mkdir(parents=True, exist_ok=True)
                shutil.copy(os.path.join(pdir, "log.jsonl"), out / "log.jsonl")
                base += ["--resume", os.path.join(pdir, "ckpt.pt")]
        if kind == "cls" and "soft_qure" in variant and not os.path.exists(os.path.join(find_data(name), "soft_qure.npy")):
            base += ["--qdata", find_data(name + "q")]  # NIH keeps the Qure array in its own prep output (s4-prep-nihq)
        t0 = time.time()
        with open(RES / f"{tag}.log", "w") as f:
            rc = subprocess.run(base, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
        print(tag, "rc", rc, "(checkpoint saved, continues in the next session)" if rc == 75 else "", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    warm = warm_up({j[2] for j in JOBS})
    warm_s = round(time.time() - t0)
    print(f"warm-up total {warm_s}s", flush=True)
    by_gpu = {}
    for j in JOBS:
        by_gpu.setdefault(j[0], []).append(j)
    ts = [threading.Thread(target=worker, args=(g, js)) for g, js in by_gpu.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    json.dump(dict(jobs=JOBS, done=sorted(p.parent.name for p in RES.glob("*/result.json")), done_before=sorted(DONE_BEFORE),
                   unfinished=sorted(p.parent.name for p in RES.glob("*/ckpt.pt")), warm_up=warm, warm_up_s=warm_s),
              open("/kaggle/working/run_summary.json", "w"))
    print("ALL DONE", flush=True)
