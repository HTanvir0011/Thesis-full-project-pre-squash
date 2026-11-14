"""Bone-image error of checkpoints on the held-out Qure-projection samples (patient-level test split; exact additive bone image).

python -m s4.eval_qp --ckpt a/final_ema.pt [b/final_ema.pt ...] --pack DIR --out results/qp_test [--split test] [--limit 400]
Several --ckpt values write to --out/<run dir name>/per_sample.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .eval_synth import load_model, predict
from .losses import ssim
from .qp import QPBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    dev = "cuda"
    bank = QPBank(a.pack, a.split, dev, workers=4, limit=a.limit)
    for ck in a.ckpt:
        model, _ = load_model(ck, dev)
        rows = []
        for s in range(0, len(bank), 8):
            idx = list(range(s, min(s + 8, len(bank))))
            X, B, _ = bank.raw(len(idx), idx=idx)
            err = predict(model, X) - B
            pr = predict(model, X)
            for k, i in enumerate(idx):
                rows.append(dict(name=bank.names[i], mae_bone=float(err[k].abs().mean()),
                                 psnr_bone=float(10 * torch.log10(1.0 / (err[k] ** 2).mean().clamp_min(1e-10))),
                                 ssim_bone=float(ssim(pr[k:k + 1].clamp(0, 1), B[k:k + 1].clamp(0, 1)))))
        out = Path(a.out) / Path(ck).parent.name if len(a.ckpt) > 1 else Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(ck, f"n={len(rows)}", {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("mae_bone", "psnr_bone", "ssim_bone")})


if __name__ == "__main__":
    main()
