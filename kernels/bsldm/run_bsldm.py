"""Kaggle GPU kernel (template): the released BS-LDM SZCH weights as an external baseline (s4/bsldm.py). Placeholders: MODE ('smoke' | 'full'), FP16.

Steps: install MONAI / diffusers (no dependency changes), copy the BS-LDM sources (MIT) from the code dataset, download the two weight files from the
authors' Google Drive links (size-checked), scan both pickles for non-allowlisted globals (refuse to load otherwise), then
  smoke: time 1000-step sampling in fp32 and fp16 on one SZCH image and compare the two outputs (decides the fp16 flag of the full run);
  full : fidelity metrics on the 75 SZCH test pairs and the 450-site lesion audit, one process per GPU (image shards), plus the saved soft estimates.
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

MODE, FP16, SITES = __MODE__, __FP16__, __SITES__  # SITES: '0:3' = evaluate sites 0..2 of the 6 drawn per image
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
HOME, OUT = Path("/kaggle/working/bsldm"), Path("/kaggle/working/out")
ENV = dict(os.environ, PYTHONPATH=CODE, BSLDM_HOME=str(HOME), HF_HUB_DISABLE_SYMLINKS_WARNING="1")
WEIGHTS = [("1KcVK0F7lG5L9Zc0-pWPS9pucAWIG3yFc", "2024-02-13-VQGAN.pth", 153365647),
           ("1zt5rV-d5wXVXCOgYqqM3C3r4wap6XkBe", "ldm-final-2024-02-16-myModel.pth", 1532925051)]
(HOME / "bsldm_weights").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SMOKE = Path(CODE) / "s4" / "bsldm_smoke.py"


def run(cmd, gpu=None, log=None):
    env = dict(ENV, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}))
    with open(log or OUT / "kernel.log", "a") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def download(fid, name, size):
    dst = HOME / "bsldm_weights" / name
    for attempt in range(8):
        subprocess.run(["curl", "-s", "-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-o", str(dst),
                        f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"])
        if dst.exists() and dst.stat().st_size == size:
            return
        print("download", name, dst.stat().st_size if dst.exists() else 0, "of", size, "attempt", attempt, flush=True)
        time.sleep(20)
    raise SystemExit(f"{name}: download incomplete (Google Drive quota or network); size {dst.stat().st_size if dst.exists() else 0} of {size}")


def fetch_szch():
    dst = Path("/kaggle/working/szch")
    if dst.exists():
        return
    zp = "/kaggle/working/szch.zip"
    urllib.request.urlretrieve("https://huggingface.co/datasets/diaoquesang/SZCH-X-Rays/resolve/main/SZCH-X-Rays_testset.zip", zp)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            assert os.path.abspath(os.path.join(dst, n)).startswith(str(dst)), n
        z.extractall(dst)
    os.remove(zp)


if __name__ == "__main__":
    t0 = time.time()
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "monai==1.3.2", "monai-generative==0.2.3", "diffusers==0.35.2", "torchxrayvision", "lpips"])
    shutil.copytree(f"{CODE}/s4/data/bsldm_src", HOME / "bsldm_src", dirs_exist_ok=True)
    for w in WEIGHTS:
        download(*w)
    print("weights downloaded", f"{time.time() - t0:.0f}s", flush=True)
    for _, name, _ in WEIGHTS:  # never unpickle before the scan passes
        rc = run([sys.executable, "-I", f"{CODE}/s4/bsldm_safe.py", "scan", str(HOME / "bsldm_weights" / name)], log=OUT / f"scan_{name}.txt")  # -I: no PYTHONPATH, no cwd imports
        assert rc == 0, f"pickle scan failed for {name}: see scan_{name}.txt"
    print("pickle scans passed", flush=True)
    fetch_szch()
    if MODE == "smoke":
        run([sys.executable, "-m", "s4.bsldm_smoke", "/kaggle/working/szch", str(OUT)], log=OUT / "smoke.log")
    else:
        spec = "bsldm:fp16" if FP16 else "bsldm"
        lib = f"{CODE}/s4/data/lesions_test.npz"

        def worker(g):
            if MODE == "full":
                run([sys.executable, "-m", "s4.eval_real", "--root", "/kaggle/working/szch", "--method", spec, "--shard", str(g), "2", "--save_est", str(OUT / "soft"),
                     "--out", str(OUT / f"fidelity_{g}.csv")], gpu=g, log=OUT / f"fidelity_{g}.log")
            run([sys.executable, "-m", "s4.audit_real", "--root", "/kaggle/working/szch", "--lib", lib, "--methods", spec, "--shard", str(g), "2",
                 "--eval_sites", *SITES.split(":"), "--out", str(OUT / f"audit_{g}.csv")], gpu=g, log=OUT / f"audit_{g}.log")
            print("worker", g, "done", f"{time.time() - t0:.0f}s", flush=True)
        ts = [threading.Thread(target=worker, args=(g,)) for g in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    shutil.rmtree("/kaggle/working/szch", ignore_errors=True)
    shutil.rmtree(HOME / "bsldm_weights", ignore_errors=True)  # 1.7 GB, public and re-downloadable: keep the kernel output small
    print("BSLDM DONE", MODE, f"{time.time() - t0:.0f}s", flush=True)
