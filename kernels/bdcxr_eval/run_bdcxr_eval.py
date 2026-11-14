"""Kaggle GPU kernel: SAP_bdcxr.md E2 (lesion audit on the fixed 300-image BDCXR subset) and E3 (ensemble spread on BDCXR and SZCH).
BDCXR comes from figshare (the lab's DUA forbids re-uploading it), SZCH and the Qure model from Hugging Face; weights from the account's s4-wts3
(cnx_t16 = ours, cnx_nt = no-twin). GPU 0 runs the four audit methods one after another (same GPU and masks, so the same sites); GPU 1 the spread.
Push: python ops/push_kernel.py ac_8 kernels/bdcxr_eval s4-bdcxr-eval {user}/s4-wts3 --shape=NvidiaTeslaT4
"""
import glob
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
sys.path.insert(0, CODE)
ENV = dict(os.environ, PYTHONPATH=CODE, PYTHONUTF8="1")
RES = Path("/kaggle/working/results")
OURS, NOTWIN = ("+".join(sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))) for p in ("cnx_t16_s[012].pt", "cnx_nt_s[012].pt"))
QURE, SZCH, LIB = "/tmp/bone_suppression.ts", "/tmp/szch", f"{CODE}/s4/data/lesions_test.npz"
FAILED = []


def fetch(url, dst):
    for attempt in range(5):
        try:
            return urllib.request.urlretrieve(url, dst)
        except Exception as e:  # network hiccup: retry
            print("retry", url, attempt, repr(e), flush=True)
            time.sleep(10)
    sys.exit(f"download failed: {url}")


def sh(gpu, args, log):
    t0 = time.time()
    with open(RES / log, "a") as f:
        rc = subprocess.run([sys.executable, "-m", *args], env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
    print(args[0], args[-1], "rc", rc, f"{time.time() - t0:.0f}s", flush=True)
    FAILED.extend([args[-1]] if rc else [])


def audit(bd):
    for name, spec in (("ours", f"ens:{OURS}"), ("notwin", f"ens:{NOTWIN}"), ("qure", f"qure:{QURE}"), ("identity", "identity")):
        sh(0, ["s4.audit_real", "--dataset", "bdcxr", "--root", bd, "--sample", "300", "--lib", LIB, "--overlap_ref", f"qure:{QURE}",
               "--methods", spec, "--out", str(RES / "audit_bdcxr" / f"{name}.csv")], "audit.log")


def spread(bd):
    sh(1, ["s4.spread", "--weights", OURS, "--dataset", "bdcxr", "--root", bd, "--out", str(RES / "spread_bdcxr.csv")], "spread.log")
    sh(1, ["s4.spread", "--weights", OURS, "--dataset", "szch", "--root", SZCH, "--out", str(RES / "spread_szch.csv")], "spread.log")


if __name__ == "__main__":
    assert OURS.count("+") == 2 and NOTWIN.count("+") == 2, f"weights missing: {OURS} | {NOTWIN}"
    RES.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision"], check=False)
    subprocess.run([sys.executable, "-c", "import torchxrayvision as x; x.baseline_models.chestx_det.PSPNet()"], check=False, env=ENV)  # weights once
    fetch("https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts", QURE)
    fetch("https://huggingface.co/datasets/diaoquesang/SZCH-X-Rays/resolve/main/SZCH-X-Rays_testset.zip", "/tmp/szch.zip")
    with zipfile.ZipFile("/tmp/szch.zip") as z:
        assert all(os.path.abspath(os.path.join(SZCH, n)).startswith(SZCH) for n in z.namelist())
        z.extractall(SZCH)
    from s4.prep import bdcxr_fetch
    bd = str(bdcxr_fetch())
    ts = [threading.Thread(target=audit, args=(bd,)), threading.Thread(target=spread, args=(bd,))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("ALL DONE", sorted(str(p.relative_to(RES)) for p in RES.rglob("*.csv")), flush=True)
    sys.exit(f"failed: {FAILED}" if FAILED else 0)  # a failed step marks the kernel as errored instead of a quiet partial result
