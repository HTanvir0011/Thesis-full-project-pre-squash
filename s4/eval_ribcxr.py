"""Rib benchmark on VinDr-RibCXR (licensed: run locally, write metrics only, never images or masks).

For each of the 245 radiographs and each method (soft-tissue estimate at 1024):
  R_rib  = 1 - G(soft)/G(orig) on the rib-boundary band inside the lung fields     (how much rib edge energy was removed)
  R_non  = 1 - G(soft)/G(orig) on lung pixels away from ribs                        (how much of everything else was removed)
  selectivity = R_rib - R_non ; AUC = ROC-AUC of the gradient-reduction map separating rib-band pixels from non-rib lung pixels.
G = Sobel gradient magnitude after Gaussian smoothing (sigma 1), so overall brightness shifts do not count.

python -m s4.eval_ribcxr --root data/ribcxr_raw --method qure:<w.ts> --out results/rib_qure.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from .audit_real import lung_masks
from .eval_real import SIZE, clahe, homomorphic, make_ens, make_ours, make_qure


def grad(x):
    g = cv2.GaussianBlur(x, (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def rib_masks(ann, idx, w0, h0):
    m = np.zeros((SIZE, SIZE), np.uint8)
    for side in "RL":
        for k in range(1, 11):
            pts = ann[f"{side}{k}"][str(idx)]
            if pts:
                poly = np.array([[p["x"] * SIZE / w0, p["y"] * SIZE / h0] for p in pts], np.int32)
                cv2.fillPoly(m, [poly], 1)
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ribcxr_raw")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    kind, _, arg = a.method.partition(":")
    fn = {"identity": lambda: (lambda x: x), "clahe": lambda: clahe, "homomorphic": lambda: homomorphic,
          "qure": lambda: make_qure(arg, a.device), "ours": lambda: make_ours(arg, a.device), "ens": lambda: make_ens(arg, a.device)}[kind]()
    root = Path(a.root)
    items = []
    for split, f in (("train", "Annotations/train/Vindr_RibCXR_train_mask.json"), ("val", "Annotations/test/Vindr_RibCXR_val_mask.json")):
        ann = json.load(open(root / f))
        items += [(ann, i, root / ann["img"][str(i)]) for i in range(len(ann["img"]))]
    items = items[: a.limit or None]
    ims, masks = [], []
    for ann, i, p in items:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
        h0, w0 = raw.shape[:2]
        x = cv2.resize(raw.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ims.append((x - x.min()) / (x.max() - x.min() + 1e-10))
        masks.append(rib_masks(ann, i, w0, h0))
    cache = os.environ.get("S4_AUDIT_MASKS")  # lung masks of the final-table machine (tools/export_audit_masks.py), as in s4/audit_real.py
    if cache:
        z = np.load(Path(cache) / "ribcxr.npz")
        assert list(z["names"][: len(items)]) == [p.name for _, _, p in items] and (a.limit or len(items) == len(z["names"])), "mask cache does not match the image list"
        lungs = [np.unpackbits(m, count=SIZE * SIZE).reshape(SIZE, SIZE) > 0 for m in z["lung"]]
    else:
        lungs = lung_masks(ims, a.device)
    rows = []
    for (ann, i, p), x, rib, lung in zip(items, ims, masks, lungs):
        lung = cv2.dilate(lung.astype(np.uint8), np.ones((49, 49), np.uint8)) > 0  # undo the audit's erosion: full lung field
        band = (cv2.morphologyEx(rib.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0) & lung
        non = lung & ~(cv2.dilate(rib.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        if band.sum() < 100 or non.sum() < 100:
            continue
        g0, g1 = grad(x), grad(np.clip(fn(x), 0, 1).astype(np.float32))
        r_rib, r_non = 1 - g1[band].mean() / g0[band].mean(), 1 - g1[non].mean() / g0[non].mean()
        d = (g0 - g1)
        y = np.r_[np.ones(int(band.sum())), np.zeros(int(non.sum()))]
        rows.append(dict(image=p.name, r_rib=r_rib, r_non=r_non, selectivity=r_rib - r_non, auc=roc_auc_score(y, np.r_[d[band], d[non]])))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(a.method, f"n={len(rows)}", {k: f"{np.mean([r[k] for r in rows]):.4f} +- {np.std([r[k] for r in rows], ddof=1):.4f}" for k in ("r_rib", "r_non", "selectivity", "auc")})


if __name__ == "__main__":
    main()
