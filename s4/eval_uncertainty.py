"""Ensemble uncertainty + split-conformal coverage for a set of seeds of one arm.

Synthetic: calibrate the conformal scale on VAL CTs, report coverage / width / AUSE / Spearman on TEST CTs (exact truth).
Real (SZCH): coverage of the pseudo-truth bone (x - DES soft aligned to x by least squares) with the same scale, as measured under shift.

python -m s4.eval_uncertainty --ckpts a/final_ema.pt b/final_ema.pt c/final_ema.pt --bank DIR --split splits.csv --out unc.json
QP models: --qp_pack DIR instead of --bank/--split (calibration on Qure-projection VAL rows, test on its TEST rows; exact additive bone).
--szch_subset held restricts the real-shift part to the 51 held-out SZCH pairs (SAP_v2.md H5').
"""
import argparse
import csv
import json

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from . import compose as C
from . import uncertainty as U
from .bank import load_bank
from .eval_real import SIZE, load_pairs
from .eval_synth import canonical, load_model, predict


def qp_stats(models, bank, dev):
    out = []
    for k in range(len(bank)):
        X, B, _ = bank.raw(1, idx=[k])
        preds = np.stack([predict(m, X)[0, 0].cpu().numpy() for m in models])
        out.append((*U.ensemble(preds), B[0, 0].cpu().numpy()))
    return out


def synth_stats(models, bank, idx, phys, dev):
    out = []
    for j in idx:
        _, o = canonical(bank, j, phys, dev, seed=j)
        preds = np.stack([predict(m, o["X"])[0, 0].cpu().numpy() for m in models])
        out.append((*U.ensemble(preds), o["B"][0, 0].cpu().numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--bank", nargs="+", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--qp_pack", default=None)
    ap.add_argument("--szch_subset", default="all", choices=["all", "dev", "held"])
    ap.add_argument("--limit", type=int, default=0, help="rows per QP split / SZCH pairs (smoke tests)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--szch", default="data/szch")
    a = ap.parse_args()
    dev = "cuda"
    models = [load_model(c, dev)[0] for c in a.ckpts]
    if a.qp_pack:
        from .qp import QPBank
        cal = qp_stats(models, QPBank(a.qp_pack, "val", dev, workers=2, limit=a.limit or 200), dev)
    else:
        rows = [r for r in csv.DictReader(open(a.split)) if r["usable"] == "1"]
        bank = load_bank(a.bank, [r["name"] for r in rows], dev)
        sp = {r["name"]: r["split"] for r in rows}
        va = [i for i, n in enumerate(bank["names"]) if sp[n] == "val"][:40]
        te = [i for i, n in enumerate(bank["names"]) if sp[n] == "test"]
        phys = C.Physics(dev)
        cal = synth_stats(models, bank, va, phys, dev)
    q = U.conformal_scale(np.stack([c[0] for c in cal]), np.stack([c[1] for c in cal]), np.stack([c[2] for c in cal]), a.alpha)
    test = qp_stats(models, QPBank(a.qp_pack, "test", dev, workers=2, limit=a.limit), dev) if a.qp_pack else synth_stats(models, bank, te, phys, dev)
    cov = [U.coverage(m, s, t, q) for m, s, t in test]
    err = np.concatenate([np.abs(m - t).ravel()[::17] for m, s, t in test])
    unc = np.concatenate([s.ravel()[::17] for m, s, t in test])
    res = dict(k=len(models), alpha=a.alpha, q=q, synth_coverage=float(np.mean([c[0] for c in cov])), synth_width=float(np.mean([c[1] for c in cov])),
               synth_ause=U.ause(err, unc), synth_spearman=float(spearmanr(err, unc).statistic),
               synth_items=[[float(c[0]), float(c[1])] for c in cov])  # per test item, for the CIs of tools/final_stats.py
    real = []
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from szch_split import is_dev
    keep = {"all": lambda n: True, "dev": is_dev, "held": lambda n: not is_dev(n)}[a.szch_subset]
    for name, x, gt in [p for p in load_pairs(a.szch) if keep(p[0])][: a.limit or None]:
        x5 = cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
        x5 = (x5 - x5.min()) / (x5.max() - x5.min() + 1e-10)
        xt = torch.from_numpy(x5)[None, None].to(dev)
        preds = np.stack([predict(m, xt)[0, 0].cpu().numpy() for m in models])
        m, s = U.ensemble(preds)
        g = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_AREA)
        A = np.stack([g.ravel(), np.ones(g.size, np.float32)], 1)
        (p0, p1), *_ = np.linalg.lstsq(A, x5.ravel(), rcond=None)
        truth = x5 - (p0 * g + p1)  # pseudo-truth bone from the real DES pair
        real.append((*U.coverage(m, s, truth, q), float(spearmanr(np.abs(m - truth).ravel()[::17], s.ravel()[::17]).statistic)))
    res.update(szch_subset=a.szch_subset, n_real=len(real), real_coverage=float(np.mean([r[0] for r in real])), real_width=float(np.mean([r[1] for r in real])), real_spearman=float(np.mean([r[2] for r in real])),
               real_items=[[float(x) for x in r] for r in real])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
