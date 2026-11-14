"""Kaggle GPU kernel (template): locked-set evaluation. tools/final_local.sh runs unchanged inside a workspace (/tmp/ws) built from the mounted
datasets: code + evalpack (s4-code), weights (s4-wte), SZCH from Hugging Face, JSRT (raddar), the Qure weights, and with a qp job the packed Qure
projections of the s4-qpprep kernel. Placeholders are filled by ops/launch_eval.py.

[[0, 'unc']] = [[gpu, step, method...], ...]   step: szch | audit | qp | unc (unc takes no method). One process per GPU, jobs sequential per GPU.
Outputs go to /kaggle/working/results/final, the same layout as the local results/final.
"""
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

JOBS = [[0, 'unc']]
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
WS, RES = Path("/tmp/ws"), Path("/kaggle/working/results")
ENV = dict(os.environ, PYTHONPATH=CODE, PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, str(dst) + ".part")
            os.replace(str(dst) + ".part", dst)
            return
        except Exception as e:  # network hiccup: retry
            print("retry", url, attempt, repr(e), flush=True)
            time.sleep(10)
    sys.exit(f"download failed: {url}")


def setup(need_qp):
    shutil.rmtree(WS, ignore_errors=True)
    shutil.copytree(Path(CODE) / "evalpack", WS)
    for f in (WS / "tools").glob("*"):  # files packed on Windows may carry CRLF
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    ENV["S4_AUDIT_MASKS"] = str(WS / "audit_masks")  # pinned local masks: the audit sites match the local final table
    assert (WS / "audit_masks" / "szch.npz").exists() and (WS / "audit_masks" / "jsrt.npz").exists(), "audit masks missing from evalpack"
    (WS / "outputs").mkdir(exist_ok=True)
    shutil.copy(Path(CODE) / "s4" / "data" / "lesions_test.npz", WS / "outputs" / "lesions_test.npz")
    RES.mkdir(parents=True, exist_ok=True)
    os.symlink(RES, WS / "results")
    for f in glob.glob("/kaggle/input/**/*__*.pt", recursive=True):  # s4-wte files: <v2|fa>__<run dir>.pt
        tag, name = Path(f).name[:-3].split("__", 1)
        if tag in ("v2", "fa"):
            d = WS / "outputs" / {"v2": "final_v2", "fa": "fin_all"}[tag] / "weights" / name
            d.mkdir(parents=True, exist_ok=True)
            os.symlink(f, d / "final_ema.pt")
    zp, dst = Path("/tmp/szch.zip"), WS / "data" / "szch"
    fetch("https://huggingface.co/datasets/diaoquesang/SZCH-X-Rays/resolve/main/SZCH-X-Rays_testset.zip", zp)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            assert os.path.abspath(os.path.join(dst, n)).startswith(str(dst)), n
        z.extractall(dst)
    meta = glob.glob("/kaggle/input/**/jsrt_metadata.csv", recursive=True)
    assert meta, "JSRT dataset not mounted"
    shutil.copytree(Path(meta[0]).parent / "images", WS / "data" / "jsrt" / "images")
    fetch("https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts", WS / "data" / "qure" / "weights" / "bone_suppression.ts")
    if need_qp:
        qb = glob.glob("/kaggle/input/**/qp.bin", recursive=True)
        assert qb, "packed Qure projections not mounted (kernel_sources s4-qpprep)"
        Path("/tmp/s4_qp").mkdir(exist_ok=True)  # random access to a 6 GB file: local disk, not the input mount
        for f in ("qp.bin", "qp_index.npy", "qp_meta.csv"):
            shutil.copy2(Path(qb[0]).parent / f, Path("/tmp/s4_qp") / f)
        assert os.path.getsize("/tmp/s4_qp/qp.bin") > 6e9, "qp.bin is not the full 6.3 GB pack"
        os.symlink("/tmp/s4_qp", WS / "outputs" / "qp_pack")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision", "lpips"], check=False)  # lung masks for the audit; LPIPS for eval_real
    subprocess.run([sys.executable, "-c", "import torchxrayvision as x; x.baseline_models.chestx_det.PSPNet()"], check=False, env=ENV)  # weights once, before the workers race


def worker(gpu, jobs, out):
    with open(RES / f"worker{gpu}.log", "a") as f:
        for step, *methods in jobs:
            t0 = time.time()
            f.flush()
            start = f.tell()
            rc = subprocess.run(["bash", "tools/final_local.sh", step, *methods], cwd=WS, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdout=f, stderr=subprocess.STDOUT).returncode
            f.flush()
            err = b"Error" in Path(f.name).read_bytes()[start:]  # final_local.sh pipes through tail: rc is always 0, so look at the log
            out.append(dict(gpu=gpu, step=step, methods=methods, rc=rc, error_in_log=err, seconds=round(time.time() - t0)))
            print(step, methods, "rc", rc, "error_in_log", err, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    setup(any(j[1] in ("qp", "unc") for j in JOBS))
    by_gpu, done = {}, []
    for j in JOBS:
        by_gpu.setdefault(j[0], []).append(j[1:])
    ts = [threading.Thread(target=worker, args=(g, js, done)) for g, js in by_gpu.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    json.dump(dict(jobs=JOBS, done=done, files=sorted(str(p.relative_to(RES)) for p in (RES / "final").rglob("*") if p.is_file())),
              open("/kaggle/working/run_summary.json", "w"))
    print("ALL DONE", flush=True)
