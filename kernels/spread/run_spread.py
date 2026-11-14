"""Kaggle GPU kernel (template): lung-field suppressor spread on the NIH test split (s4/spread.py --dataset nih), the second score of the
SAP_v3 selective classification (entropy + spread). One s4.spread process per GPU over alternate 32-image batches of the test split; the
parts are merged in prep index order. ops/launch_spread.py sets WEIGHTS = __WEIGHTS__ (the frozen suppressor ensemble: file names joined
by '+', found by name).
Mounts (varying depths under /kaggle/input): s4-code; the weights data set (s4-wts3 or s4-wte); nih-chest-xrays/data (raw 1024-px PNGs,
Data_Entry_2017.csv); the account's NIH prep output (folder nih/: ids.json, split.npy).
Writes /kaggle/working/spread_nih.csv (dataset,image,lung_frac,spread_lung,spread_all,index), spread_<gpu>.log and run_summary.json.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WEIGHTS = __WEIGHTS__
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
ENV = dict(os.environ, PYTHONPATH=CODE, PYTHONUTF8="1")
sys.path.insert(0, CODE)
OUT = Path("/kaggle/working")


def find(pattern, what):
    hit = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))  # mounts sit at varying depths
    assert hit, f"{what} not mounted; inputs: {sorted(glob.glob('/kaggle/input/*/*'))[:20]}"
    return str(Path(hit[0]).parent)


def part(g, cmd, rcs):
    t0 = time.time()
    with open(OUT / f"spread_{g}.log", "w") as f:
        rcs[g] = subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g)), stdout=f, stderr=subprocess.STDOUT).returncode
    print("gpu", g, "rc", rcs[g], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision"], check=False)  # PSPNet lung masks (s4/audit_real.py)
    # PSPNet weights downloaded once, before the workers race for them
    subprocess.run([sys.executable, "-c", "import torchxrayvision as x; x.baseline_models.chestx_det.PSPNet()"], check=False, env=ENV)
    import numpy as np
    import torch
    from s4.prep import resolve_weights
    from s4.spread import merge
    weights = resolve_weights(WEIGHTS)
    root, prep = find("Data_Entry_2017.csv", "nih-chest-xrays/data"), find("nih/split.npy", "NIH prep output")
    n = max(1, torch.cuda.device_count())
    print("weights", weights, "| NIH", root, "| prep", prep, "| GPUs", n, flush=True)
    cmd = [sys.executable, "-m", "s4.spread", "--weights", weights, "--dataset", "nih", "--root", root, "--prep", prep, "--nworkers", str(n)]
    parts, rcs = [OUT / f"spread_nih_{g}.csv" for g in range(n)], [None] * n
    ts = [threading.Thread(target=part, args=(g, cmd + ["--worker", str(g), "--out", str(parts[g])], rcs)) for g in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    failed, rows = any(rcs), 0
    if not failed:  # a failed part leaves the others for inspection, never a partial spread_nih.csv
        rows = merge(parts, OUT / "spread_nih.csv")
        [p.unlink() for p in parts]
    test = int((np.load(Path(prep) / "split.npy") == "test").sum())
    json.dump(dict(weights=weights, nih_root=root, prep=prep, gpus=n, rc=rcs, test=test, rows=rows, seconds=round(time.time() - t0)),
              open(OUT / "run_summary.json", "w"))
    print(f"rows {rows} of {test} test images, rc {rcs} (skipped images: spread_<gpu>.log)", flush=True)
    sys.exit(f"failed: rc {rcs}" if failed else 0)  # an errored kernel instead of a quiet partial result
