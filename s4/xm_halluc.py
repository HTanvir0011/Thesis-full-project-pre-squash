"""Hallucination tests of NIH classifiers (DESIGN_v4.md section 5, SAP_v4.md H15).

python -m s4.xm_halluc --data NIH_PREP --runs RUN_s0 RUN_s1 ... --out DIR [--lib lesions_test.npz] [--n_nodule 2000] [--n_cam 500]

A model is the mean probability of its seed runs (s4/cls_hybrid.py folders with model.pt, val_pred.npy, val_idx.npy); the operating
threshold of each class gives 90 % specificity on the validation split (positive: probability above it). On NIH test images without any
finding:
- perturbations that add no disease: rib emphasis (X + 0.5 B, where B = clip(X - S) and S is our suppressor's soft image), Gaussian noise
  (sigma 0.02) and gamma 0.8; every class's probability before and after;
- counterfactual nodules (--n_nodule images): one LIDC test nodule of the audit library (s4/lesion_lib.py, 512-frame px, resized to the
  prep) added to X and S at a random pixel of the eroded lung mask of s4/audit_real.py; every class's probability after.
On --n_cam random test images: for every parenchymal finding the model calls positive, the share of Grad-CAM++ mass outside the lung field
(mean over the seeds). Noise, sites, nodules and images depend only on --seed, so all models see the same inputs. Writes DIR/halluc.npz
(with the seed-mean validation probabilities, so tools/xm_stats.py can combine seed groups of one model run on different accounts) and
DIR/result.json; intervals and H15 in tools/xm_stats.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import lesion_lib
from .cls_hybrid import load_x
from .prep import NIH
from .xm_xai import load, upsampled

PARENCHYMA = [NIH.index(c) for c in ("Atelectasis", "Infiltration", "Mass", "Nodule", "Pneumonia", "Consolidation", "Edema", "Emphysema",
                                      "Fibrosis")]  # findings inside the lung field (not cardiac, pleural or diaphragmatic)
LESION = [NIH.index("Mass"), NIH.index("Nodule")]
PERTURB = ("rib", "noise", "gamma")
BS = 64  # fixed: the noise of a batch is seeded by its offset, so every model must see the same batches


def val_mean(runs):
    """Seed-mean validation probabilities [Nval, C] of the runs and their prep indices."""
    vidx = np.load(Path(runs[0]) / "val_idx.npy")
    assert all(np.array_equal(np.load(Path(r) / "val_idx.npy"), vidx) for r in runs), "seed runs with different validation sets"
    return np.mean([np.load(Path(r) / "val_pred.npy").astype(np.float32) for r in runs], 0), vidx


def thresholds(p, y, spec=0.90):
    """Per class, the spec-quantile of validation probabilities p over that class's validation negatives (labels y)."""
    return np.array([np.quantile(p[y[:, c] == 0, c], spec) for c in range(y.shape[1])])


def perturb(x, kind, gen=None):
    """x [n,3,S,S] channels X, S, B (s4/cls_hybrid.py load_x) -> X changed, S kept, B = clip(X - S) again."""
    X, S = x[:, :1], x[:, 1:2]
    if kind == "rib":
        X = X + 0.5 * x[:, 2:]
    elif kind == "noise":
        X = X + 0.02 * torch.randn(X.shape, generator=gen, device=X.device)
    else:
        X = X.clamp_min(0) ** 0.8
    X = X.clamp(0, 1)
    return torch.cat([X, S, (X - S).clamp_min(0)], 1)


def nodule(patch, cy, cx, S):
    """Lesion image [S, S]: a 512-frame library patch resized to the S-px frame and centred at (cy, cx); placed as in s4/audit_real.py
    (moved inside at the top and left border, cut at the bottom and right; lung sites keep it away from both)."""
    p = cv2.resize(patch, (max(1, round(patch.shape[1] * S / 512)), max(1, round(patch.shape[0] * S / 512))), interpolation=cv2.INTER_AREA)
    h, w = p.shape
    y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
    y1, x1 = min(S, y0 + h), min(S, x0 + w)
    n = np.zeros((S, S), np.float32)
    n[y0:y1, x0:x1] = p[:y1 - y0, :x1 - x0]
    return n


def add(x, n):
    """Lesion images n [b,S,S] (tensor) added to X and S (soft tissue; the bone image stays) -> clipped, B recomputed."""
    X, S = (x[:, :2] + n[:, None]).clamp(0, 1).unbind(1)
    return torch.stack([X, S, (X - S).clamp_min(0)], 1)


def lung_fields(images, dev):
    """torchxrayvision lung masks of s4/audit_real.py at the image size: (eroded as in the audit: nodule sites; dilated back with the same
    kernel, an opening of the raw mask: the lung field of the off-target share)."""
    from .audit_real import lung_masks
    S, k = images[0].shape[0], np.ones((49, 49), np.uint8)
    rs = lambda m: cv2.resize(m.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    er = lung_masks(images, dev)
    return np.array([rs(m) for m in er]), np.array([rs(cv2.dilate(m.astype(np.uint8), k)) for m in er])


@torch.no_grad()
def predict(model, arrays, idx, size, dev, change=None):
    """Probabilities [n, C] of the prep images idx; change(x, start) alters each batch (start: its offset in idx)."""
    out = []
    for s in range(0, len(idx), BS):
        x = load_x(arrays, idx[s:s + BS], size, dev)
        x = change(x, s) if change else x
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(x)).float().cpu())
    return torch.cat(out).numpy()


def crossed(p0, p1, t):
    """Per image: some class goes from at most its threshold to above it."""
    return ((p0 <= t) & (p1 > t)).any(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="NIH prep (orig.npy, soft.npy, labels.npy, split.npy, patient.npy)")
    ap.add_argument("--runs", nargs="+", required=True, help="the seed runs of one model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=str(Path(__file__).parent / "data" / "lesions_test.npz"))
    ap.add_argument("--n_nodule", type=int, default=2000)
    ap.add_argument("--n_cam", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("orig", "soft")}
    y, split = np.load(d / "labels.npy"), np.load(d / "split.npy", allow_pickle=True)
    patient = np.load(d / "patient.npy", allow_pickle=True)
    rng = np.random.default_rng(a.seed)
    free = np.nonzero((split == "test") & (y == 0).all(1))[0]
    nod = np.sort(rng.choice(free, min(a.n_nodule, len(free)), replace=False))
    cam = np.sort(rng.choice(np.nonzero(split == "test")[0], min(a.n_cam, int((split == "test").sum())), replace=False))
    S = arrays["orig"].shape[-1]
    eroded, _ = lung_fields([arrays["orig"][i] / 255.0 for i in nod], dev)
    _, field = lung_fields([arrays["orig"][i] / 255.0 for i in cam], dev)
    lib = lesion_lib.load(a.lib)
    site = np.full((len(nod), 3), -1)  # lesion, row, column; -1: empty lung mask, no nodule
    for k, m in enumerate(eroded):
        ys, xs = np.nonzero(m)
        if len(ys):
            j = rng.integers(len(ys))
            site[k] = (rng.integers(len(lib)), ys[j], xs[j])
    pv, vidx = val_mean(a.runs)
    thr = thresholds(pv, y[vidx])

    def noisy(kind):
        return lambda x, s: perturb(x, kind, torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + s))

    def with_nodule(x, s):
        n = [nodule(lib[li]["x_patch"], cy, cx, S) if li >= 0 else np.zeros((S, S), np.float32) for li, cy, cx in site[s:s + len(x)]]
        return add(x, torch.from_numpy(np.stack(n)).to(x.device))

    models = [load(r, dev) for r in a.runs]

    def mean_p(idx, change=None):
        return np.mean([predict(m, arrays, idx, st["size"], dev, change) for m, st in models], 0)

    def outside(m, st, i, c):
        cam_ = upsampled(m, load_x(arrays, [cam[i]], st["size"], dev), c)[0]
        return float(cam_[~field[i]].sum() / (cam_.sum() + 1e-12))

    P = dict(clean=mean_p(free), **{k: mean_p(free, noisy(k)) for k in PERTURB}, nodule=mean_p(nod, with_nodule), cam_test=mean_p(cam))
    pairs = [(i, c) for i in range(len(cam)) for c in PARENCHYMA if P["cam_test"][i, c] > thr[c]]  # positives of the seed-mean model
    off = np.array([np.mean([outside(m, st, i, c) for m, st in models]) for i, c in pairs], np.float32)
    pos = np.searchsorted(free, nod)
    res = dict(runs=[str(r) for r in a.runs], n_free=int(len(free)), thresholds=dict(zip(NIH, map(float, thr))),
               rate_clean=float((P["clean"] > thr).any(1).mean()),
               **{f"rate_{k}": float(crossed(P["clean"], P[k], thr).mean()) for k in PERTURB})
    ok = (site[:, 0] >= 0) & (P["clean"][pos][:, LESION] <= thr[LESION]).all(1)
    res |= dict(n_nodule=int(ok.sum()), nodule_sensitivity=float((P["nodule"][ok][:, LESION] > thr[LESION]).any(1).mean()) if ok.any() else None,
                n_cam_pairs=len(pairs), off_target=float(off.mean()) if len(off) else None)
    np.savez_compressed(out / "halluc.npz", thr=thr, val_p=pv, val_y=y[vidx], n_runs=len(a.runs), free=free, free_patient=patient[free], **{f"p_{k}": v.astype(np.float32) for k, v in P.items()},
                        nod=nod, nod_pos=pos, nod_site=site, nod_diam=np.array([lib[li]["diam_mm"] if li >= 0 else np.nan for li in site[:, 0]]),
                        cam=cam, cam_pairs=np.array(pairs, int).reshape(-1, 2), cam_off=off)
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
