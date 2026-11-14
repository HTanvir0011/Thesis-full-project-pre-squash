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

JOBS = [[0, 'qp', 1, 300000, 8, 0.0002, '_cnx2_300k', '--backbone convnextv2_tiny.fcmae_ft_in22k_in1k --enc_lr_mult 0.25'], [1, 'qp', 2, 300000, 8, 0.0002, '_cnx2_300k', '--backbone convnextv2_tiny.fcmae_ft_in22k_in1k --enc_lr_mult 0.25']]  # [[gpu, arm, seed, samples, bs, lr], ...]
EXTRA = []  # extra argv for s4.train, e.g. ["--limit_ct", "20"] for the pilot
MAX_HOURS = 10.0  # per session and run; a longer run stops with ckpt_last.pt and is continued by a chained kernel (ops/launch_train.py --resume)
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
BANK = next(iter(glob.glob("/kaggle/input/s4-bank") + glob.glob("/kaggle/input/datasets/*/s4-bank")), None)  # None: bank-free QP arms (accounts that cannot take the 1.2 GB upload)
_qb = glob.glob("/kaggle/input/**/qp.bin", recursive=True)  # Qure ct2xr-projections, packed: dataset s4-qproj or the output of the s4-qpprep kernel
QP = str(Path(_qb[0]).parent) if _qb else None
if QP:  # random access to a 6 GB file: use the local disk, not the input mount
    _dst = Path("/tmp/s4_qp")
    _dst.mkdir(exist_ok=True)
    for _f in ("qp.bin", "qp_index.npy", "qp_meta.csv"):
        shutil.copy2(Path(QP) / _f, _dst / _f)
    QP = str(_dst)
LIB = f"{CODE}/s4/data/lesions_test.npz"  # test-CT lesions for the real-image audit
ENV = dict(os.environ, PYTHONPATH=CODE, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
RUNS, RES = Path("/kaggle/working/runs"), Path("/kaggle/working/results")
RUNS.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def sh(cmd, gpu, log):
    with open(log, "a") as f:
        return subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode


def resume_ckpt(name, out):
    """A run cut by the session time budget leaves ckpt_last.pt in its kernel's output; a chained kernel (kernel_sources) mounts it here."""
    found = glob.glob(f"/kaggle/input/**/runs/{name}/ckpt_last.pt", recursive=True)
    if found and not (out / "ckpt_last.pt").exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], out / "ckpt_last.pt")
        print(name, "resuming from", found[0], flush=True)


def worker(gpu, jobs):
    for _, arm, seed, samples, bs, lr, *tag in jobs:  # tag = [name suffix, "extra s4.train argv"]
        name = f"{arm}_s{seed}" + (tag[0] if tag else "")
        out = RUNS / name
        resume_ckpt(name, out)
        rc = sh([sys.executable, "-m", "s4.train", "--arm", arm, "--seed", str(seed), "--samples", str(samples), "--bs", str(bs),
                 "--lr", str(lr), "--out", str(out), *(["--bank", BANK, "--split", f"{BANK}/splits.csv", "--ref_q", f"{BANK}/ref_q.npy"] if BANK else
                 ["--bank", "none", "--split", "none", "--lib_file", f"{CODE}/s4/data/lib_train.npz"]), "--max_hours", str(MAX_HOURS), *(["--qp_pack", QP] if QP else []), *EXTRA, *(tag[1].split() if len(tag) > 1 else [])],
                gpu, RUNS / f"{name}.log")
        print(name, "train rc", rc, time.strftime("%H:%M:%S"), flush=True)
        if rc == 0 and (out / "final_ema.pt").exists():
            rc2 = None if not BANK else sh([sys.executable, "-m", "s4.eval_synth", "--ckpt", str(out / "final_ema.pt"), "--bank", BANK, "--split",
                      f"{BANK}/splits.csv", "--out", str(RES / name / "synth_test")], gpu, RUNS / f"{name}.log")
            rc3 = sh([sys.executable, "-m", "s4.eval_real", "--root", "/kaggle/working/szch", "--method", f"ours:{out / 'final_ema.pt'}",
                      "--out", str(RES / name / "szch.csv")], gpu, RUNS / f"{name}.log")
            ck = str(out / "final_ema.pt")
            rc4 = sh([sys.executable, "-m", "s4.audit_real", "--root", "/kaggle/working/szch", "--lib", LIB, "--methods", f"ours:{ck}",
                      "--out", str(RES / name / "audit.csv")], gpu, RUNS / f"{name}.log")
            rc5 = sh([sys.executable, "-m", "s4.eval_qp", "--ckpt", ck, "--pack", QP, "--out", str(RES / name / "qp_test")], gpu, RUNS / f"{name}.log") if QP else None
            print(name, "eval rc", rc2, rc3, "audit", rc4, "qp", rc5, flush=True)
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision"], check=False)  # lung masks for the real-image audit
    subprocess.run([sys.executable, "-c", "import torchxrayvision as x; x.baseline_models.chestx_det.PSPNet()"], check=False)  # weights downloaded once, before the workers race
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
