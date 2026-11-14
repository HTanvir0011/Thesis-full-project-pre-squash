"""Kaggle GPU kernel (template): train suppressor arms (one process per GPU, jobs sequential per GPU), then score each
finished model on the synthetic test split and on the 75 real SZCH pairs. Placeholders are filled by ops/launch_train.py."""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

JOBS = [[0, 'nofreq', 2, 200000, 8, 0.0002], [0, 'rot0', 0, 200000, 8, 0.0002], [1, 'rot0', 1, 200000, 8, 0.0002], [1, 'rot0', 2, 200000, 8, 0.0002]]  # [[gpu, arm, seed, samples, bs, lr], ...]
EXTRA = []  # extra argv for s4.train, e.g. ["--limit_ct", "20"] for the pilot
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
BANK = next(iter(glob.glob("/kaggle/input/s4-bank") + glob.glob("/kaggle/input/datasets/*/s4-bank")))
ENV = dict(os.environ, PYTHONPATH=CODE, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
RUNS, RES = Path("/kaggle/working/runs"), Path("/kaggle/working/results")
RUNS.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def sh(cmd, gpu, log):
    with open(log, "a") as f:
        return subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode


def worker(gpu, jobs):
    for _, arm, seed, samples, bs, lr, *tag in jobs:
        name = f"{arm}_s{seed}" + (tag[0] if tag else "")
        out = RUNS / name
        rc = sh([sys.executable, "-m", "s4.train", "--arm", arm, "--seed", str(seed), "--samples", str(samples), "--bs", str(bs),
                 "--lr", str(lr), "--out", str(out), "--bank", BANK, "--split", f"{BANK}/splits.csv", "--ref_q", f"{BANK}/ref_q.npy",
                 *EXTRA], gpu, RUNS / f"{name}.log")
        print(name, "train rc", rc, time.strftime("%H:%M:%S"), flush=True)
        if rc == 0 and (out / "final_ema.pt").exists():
            rc2 = sh([sys.executable, "-m", "s4.eval_synth", "--ckpt", str(out / "final_ema.pt"), "--bank", BANK, "--split",
                      f"{BANK}/splits.csv", "--out", str(RES / name / "synth_test")], gpu, RUNS / f"{name}.log")
            rc3 = sh([sys.executable, "-m", "s4.eval_real", "--root", "/kaggle/working/szch", "--method", f"ours:{out / 'final_ema.pt'}",
                      "--out", str(RES / name / "szch.csv")], gpu, RUNS / f"{name}.log")
            print(name, "eval rc", rc2, rc3, flush=True)
            (out / "ckpt_last.pt").unlink(missing_ok=True)  # 465 MB each; the final EMA weights are what we keep


def fetch_szch():
    dst = Path("/kaggle/working/szch")
    if dst.exists():
        return
    url = "https://huggingface.co/datasets/diaoquesang/SZCH-X-Rays/resolve/main/SZCH-X-Rays_testset.zip"
    zp = "/kaggle/working/szch.zip"
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            assert os.path.abspath(os.path.join(dst, n)).startswith(str(dst)), n
        z.extractall(dst)
    os.remove(zp)


if __name__ == "__main__":
    fetch_szch()
    print("jobs:", JOBS, flush=True)
    by_gpu = {}
    for j in JOBS:
        by_gpu.setdefault(j[0], []).append(j)
    ts = [threading.Thread(target=worker, args=(g, js)) for g, js in by_gpu.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    json.dump(dict(jobs=JOBS, done=sorted(p.parent.name for p in RUNS.glob("*/DONE"))), open("/kaggle/working/run_summary.json", "w"))
    shutil.rmtree("/kaggle/working/szch", ignore_errors=True)
    print("ALL DONE", flush=True)
