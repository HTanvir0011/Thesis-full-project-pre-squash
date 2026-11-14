"""Exact-truth lesion audit on the held-out Qure-projection test split (430 CT-RATE volumes; SAP_v2.md H1' secondary).

A test-CT lesion n is added to the soft component: X' = X + a*n with the bone image B unchanged, so the truth is exact.
Per site: absorption A = <B^(X') - B^(X), a*n> / <a*n, a*n> on the lesion support, lesion contrast ratio LCR = C(X' - B^(X')) / C(S + a*n)
(C = mean inside the support - mean of a surrounding ring), bone overlap = share of the support where the TRUE bone image is in its top 20%.
Per image: bone MAE / PSNR on the clean image. Methods use the same specs as audit_real (ours:, ens:, qure:, identity); every method sees
the same 512 px image, upsampled to the 1024 px frame audit_real's bone functions expect.

python -m s4.audit_qp --pack outputs/qp_pack --lib outputs/lesions_test.npz --methods ens:a.pt+b.pt+c.pt qure:w.ts --out results/final/qp/x
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from . import lesion_lib
from .audit_real import bone_fn
from .qp import QPBank, lung_field


def contrast(img, sup):
    ring = (cv2.dilate(sup.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~(cv2.dilate(sup.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    return float(img[sup].mean() - img[ring].mean()) if ring.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lib = lesion_lib.load(a.lib)
    bank = QPBank(a.pack, a.split, a.device, workers=2, limit=a.limit)
    up = lambda z: cv2.resize(z, (1024, 1024), interpolation=cv2.INTER_CUBIC)  # noqa: E731
    down = lambda z: cv2.resize(z, (512, 512), interpolation=cv2.INTER_AREA)  # noqa: E731
    fns = {s: (lambda f: (lambda x512: down(f(up(x512)))))(bone_fn(s, a.device)) for s in a.methods}
    sites, imgs = [], []
    for k in range(len(bank)):
        X, B, L = bank.raw(1, idx=[k])
        x, b = X[0, 0].cpu().numpy(), B[0, 0].cpu().numpy()
        field = lung_field(L)[0, 0].cpu().numpy() > 0
        ys, xs = np.nonzero(field)
        pred0 = {s: f(x) for s, f in fns.items()}
        for s, p in pred0.items():
            e = p - b
            imgs.append(dict(method=s.split(":")[0], name=bank.names[k], mae_bone=float(np.abs(e).mean()),
                             psnr_bone=float(10 * np.log10(1.0 / max(float((e ** 2).mean()), 1e-10)))))
        if len(ys) == 0:
            continue
        hi = b > np.quantile(b, 0.80)
        for site in range(a.sites):
            les = lib[int(rng.integers(len(lib)))]
            patch = les["x_patch"]  # library units are 512-frame composite intensities, the frame of these projections
            cy, cx = int(ys[rng.integers(len(ys))]), int(xs[rng.integers(len(xs))])
            h, w = patch.shape
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
            y1, x1 = min(512, y0 + h), min(512, x0 + w)
            n = np.zeros((512, 512), np.float32)
            n[y0:y1, x0:x1] = patch[: y1 - y0, : x1 - x0]
            sup = n > 0.02 * n.max()
            if sup.sum() < 10:
                continue
            an = a.alpha * n
            xp = np.clip(x + an, 0, 1).astype(np.float32)
            c_true = contrast(x - b + an, sup)
            for s, f in fns.items():
                p1 = f(xp)
                db = p1 - pred0[s]
                sites.append(dict(method=s.split(":")[0], name=bank.names[k], site=site, overlap=float(hi[sup].mean()), diam_mm=les["diam_mm"],
                                  absorption=float((db * an)[sup].sum() / (an[sup] ** 2).sum()), lcr=contrast(xp - p1, sup) / c_true if c_true else float("nan")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("sites.csv", sites), ("images.csv", imgs)):
        with open(out / fn, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    for m in sorted({r["method"] for r in sites}):
        r = [x for x in sites if x["method"] == m]
        q = [x for x in imgs if x["method"] == m]
        print(f"{m}: sites={len(r)} A={np.mean([x['absorption'] for x in r]):.3f} LCR={np.nanmean([x['lcr'] for x in r]):.3f} "
              f"bone MAE={np.mean([x['mae_bone'] for x in q]):.4f}", flush=True)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
