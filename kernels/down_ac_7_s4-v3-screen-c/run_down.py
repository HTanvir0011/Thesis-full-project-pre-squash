"""Kaggle GPU kernel (template): downstream jobs (classification / detection), one process per GPU, jobs sequential per GPU.

[[0, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'convnextv2_tiny.fcmae_ft_in22k_in1k', '--enc_lr', '0.25'], [0, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'hf:microsoft/rad-dino', '--enc_lr', '0.25'], [1, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'swinv2_base_window16_256.ms_in1k', '--enc_lr', '0.25', '--accum', '4']] = [[gpu, kind, data_dir_name, variant, seed, epochs, extra_argv...], ...]
kind 'cls': variant is a '+'-joined channel list (orig | soft | orig+soft); kind 'det': orig | soft | both;
kind 'hyb' (s4/cls_hybrid.py): variant is the configuration (orig | soft | early | hybrid | ctrl), the tag carries @<arch> from --arch.
The prepared arrays come from a kernel-output source mounted under /kaggle/input/<prep kernel slug>/<dataset>/.
Before the lanes start, every large array of the data sets in use is read once sequentially (RUNPLAN_v3 R2): the first epoch then finds
it in the page cache instead of reading it at random (v2: +44 min per 7.3 GB array); the time is logged and kept in run_summary.json.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

JOBS = [[0, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'convnextv2_tiny.fcmae_ft_in22k_in1k', '--enc_lr', '0.25'], [0, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'hf:microsoft/rad-dino', '--enc_lr', '0.25'], [1, 'hyb', 'nih', 'orig', 0, 10, '--arch', 'swinv2_base_window16_256.ms_in1k', '--enc_lr', '0.25', '--accum', '4']]
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
ENV = dict(os.environ, PYTHONPATH=CODE, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
sys.path.insert(0, CODE)
from s4.runtag import tag as job_tag  # noqa: E402  (same folder names as ops/jobqueue.py's missing-run check)
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
        base = [sys.executable, "-m", {"cls": "s4.cls", "det": "s4.det", "hyb": "s4.cls_hybrid"}[kind], "--data", find_data(name), "--seed", str(seed),
                "--epochs", str(epochs), "--out", str(out)]
        flag = {"cls": ["--variants", *variant.split("+")], "det": ["--variant", variant], "hyb": ["--config", variant]}[kind]
        base += flag + [str(e) for e in extra]
        if kind == "cls" and "soft_qure" in variant and not os.path.exists(os.path.join(find_data(name), "soft_qure.npy")):
            base += ["--qdata", find_data(name + "q")]  # NIH keeps the Qure array in its own prep output (s4-prep-nihq)
        t0 = time.time()
        with open(RES / f"{tag}.log", "w") as f:
            rc = subprocess.run(base, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
        print(tag, "rc", rc, f"{time.time() - t0:.0f}s", flush=True)


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
    json.dump(dict(jobs=JOBS, done=sorted(p.parent.name for p in RES.glob("*/result.json")), warm_up=warm, warm_up_s=warm_s),
              open("/kaggle/working/run_summary.json", "w"))
    print("ALL DONE", flush=True)
