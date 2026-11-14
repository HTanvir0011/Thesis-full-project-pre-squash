"""Kaggle GPU kernel (template): downstream jobs (classification / detection), one process per GPU, jobs sequential per GPU.

[[0, 'det', 'tbx', 'soft_qure', 1, 12], [1, 'det', 'tbx', 'clahe', 1, 12], [0, 'det', 'tbx', 'orig', 2, 12], [1, 'det', 'tbx', 'soft', 2, 12]] = [[gpu, kind, data_dir_name, variant, seed, epochs, extra_argv...], ...]
kind 'cls': variant is a '+'-joined channel list (orig | soft | orig+soft); kind 'det': orig | soft | both.
The prepared arrays come from a kernel-output source mounted under /kaggle/input/<prep kernel slug>/<dataset>/.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

JOBS = [[0, 'det', 'tbx', 'soft_qure', 1, 12], [1, 'det', 'tbx', 'clahe', 1, 12], [0, 'det', 'tbx', 'orig', 2, 12], [1, 'det', 'tbx', 'soft', 2, 12]]
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
ENV = dict(os.environ, PYTHONPATH=CODE, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
RES = Path("/kaggle/working/results")
RES.mkdir(parents=True, exist_ok=True)


def find_data(name):
    hit = sorted(glob.glob(f"/kaggle/input/**/{name}/split.npy", recursive=True))  # kernel outputs mount at varying depths
    assert hit, f"prepared data {name} not mounted; inputs: {sorted(glob.glob('/kaggle/input/*/*'))[:20]}"
    return str(Path(hit[0]).parent)


def worker(gpu, jobs):
    for _, kind, name, variant, seed, epochs, *extra in jobs:
        tag = f"{name}_{kind}_{variant.replace('+', '-')}_s{seed}"
        out = RES / tag
        base = [sys.executable, "-m", "s4.cls" if kind == "cls" else "s4.det", "--data", find_data(name), "--seed", str(seed),
                "--epochs", str(epochs), "--out", str(out)]
        base += (["--variants", *variant.split("+")] if kind == "cls" else ["--variant", variant]) + [str(e) for e in extra]
        if kind == "cls" and "soft_qure" in variant and not os.path.exists(os.path.join(find_data(name), "soft_qure.npy")):
            base += ["--qdata", find_data(name + "q")]  # NIH keeps the Qure array in its own prep output (s4-prep-nihq)
        t0 = time.time()
        with open(RES / f"{tag}.log", "w") as f:
            rc = subprocess.run(base, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
        print(tag, "rc", rc, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    by_gpu = {}
    for j in JOBS:
        by_gpu.setdefault(j[0], []).append(j)
    ts = [threading.Thread(target=worker, args=(g, js)) for g, js in by_gpu.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    json.dump(dict(jobs=JOBS, done=sorted(p.parent.name for p in RES.glob("*/result.json"))), open("/kaggle/working/run_summary.json", "w"))
    print("ALL DONE", flush=True)
