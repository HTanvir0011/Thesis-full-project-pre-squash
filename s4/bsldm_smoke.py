"""Smoke test of the BS-LDM baseline on one SZCH image: 1000-step sampling time and memory in fp32 and fp16 autocast, agreement of the two outputs
(same noise), and the paired-image metrics of both. Decides whether the full run may use fp16 (rule: mean |fp16 - fp32| < 0.01 on the [0,1] scale and
bone-region PSNR_cal within 0.05 dB).

python -m s4.bsldm_smoke <szch_root> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .bsldm import BSLDM
from .eval_real import load_pairs, metrics

if __name__ == "__main__":
    root, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    n, x, g = load_pairs(root)[0]
    res = {}
    for fp16 in (False, True):
        m = BSLDM("cuda", fp16=fp16)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        s = m.soft(x, seed=0)
        torch.cuda.synchronize()
        res["fp16" if fp16 else "fp32"] = dict(sec=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), est=s)
        del m
        torch.cuda.empty_cache()
    a, b = res["fp32"]["est"], res["fp16"]["est"]
    rep = dict(image=n, sec_fp32=res["fp32"]["sec"], sec_fp16=res["fp16"]["sec"], peak_gb_fp32=res["fp32"]["peak_gb"], peak_gb_fp16=res["fp16"]["peak_gb"],
               mean_abs_diff=float(np.abs(a - b).mean()), max_abs_diff=float(np.abs(a - b).max()), range_fp32=[float(a.min()), float(a.max())],
               metrics_fp32=metrics(x, g, a), metrics_fp16=metrics(x, g, b))
    rep["fp16_ok"] = bool(rep["mean_abs_diff"] < 0.01 and abs(rep["metrics_fp32"]["psnr_cal_bone"] - rep["metrics_fp16"]["psnr_cal_bone"]) < 0.05)
    json.dump(rep, open(out / "smoke.json", "w"), indent=1, default=float)
    np.save(out / "smoke_fp32.npy", a.astype(np.float16))
    np.save(out / "smoke_fp16.npy", b.astype(np.float16))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("metrics")}, indent=1, default=float))
