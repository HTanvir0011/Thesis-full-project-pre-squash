"""Counterfactual lesion audit on REAL radiographs (SZCH test set): paste a held-out-CT lesion into a real image, x' = x + a*n, and
measure how much of it the bone prediction absorbs:  A = <B(x') - B(x), a*n> / <a*n, a*n>  over the lesion support.
Ideal A = 0 (the lesion stays in the soft image); A = 1 means it was removed as 'bone'. Identity (no suppression) has A = 0 by
construction, so A is only meaningful next to the suppression gain from eval_real.py. Sites are drawn in the lung fields; bone
overlap of a site is measured on the real dual-energy bone signal (calibrated CXR - DES), not on any model output.

python -m s4.audit_real --lib lesions_test.npz --methods qure:<w.ts> ours:<ckpt> identity --out audit.csv
Other image sets (no paired truth needed): --dataset ribcxr (bone overlap from the radiologist rib masks; licensed, local only, metrics only)
or --dataset jsrt (bone overlap from the bone map of --overlap_ref, e.g. Qure's model, so it is not measured by the model under test).
Sites depend only on the images, the lung masks, the library and --seed, so separate runs with different --methods share them.
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from . import lesion_lib
from .eval_real import calibrate, load_pairs

SIZE = 1024


def lung_masks(images, dev):
    import torchxrayvision as xrv
    m = xrv.baseline_models.chestx_det.PSPNet().to(dev).eval()
    out = []
    for x in images:
        img = cv2.resize(xrv.datasets.normalize((x * 255).astype(np.uint8), 255), (512, 512), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(img)[None, None].float().to(dev)))[0].cpu().numpy()
        lung = ((p[4] > 0.5) | (p[5] > 0.5)).astype(np.uint8)
        lung = cv2.resize(lung, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        out.append(cv2.erode(lung, np.ones((49, 49), np.uint8)) > 0)
    return out


def load_images(dataset, root, sample=0):
    """-> [(name, x 1024x1024 in [0,1], bone mask or None)]. SZCH: strongest 20% of the real DES bone signal; RibCXR: annotated ribs.
    sample > 0: a fixed class-stratified subset of that size (class = parent folder; SAP_bdcxr.md E2)."""
    if dataset == "szch":
        out = []
        for name, x, gt in load_pairs(root):
            d = calibrate(x, gt) - gt
            out.append((name, x, d > np.quantile(d, 0.80)))
        return out
    paths = []
    if dataset == "ribcxr":
        import json
        from .eval_ribcxr import rib_masks
        for f in ("Annotations/train/Vindr_RibCXR_train_mask.json", "Annotations/test/Vindr_RibCXR_val_mask.json"):
            ann = json.load(open(Path(root) / f))
            paths += [(Path(root) / ann["img"][str(i)], (ann, i)) for i in range(len(ann["img"]))]
    else:
        paths = [(p, None) for p in sorted(Path(root).glob("**/*.jpg" if dataset == "bdcxr" else "**/*.png"))]
        if sample:
            rng, by = np.random.default_rng(7), {}
            for p, _ in paths:
                by.setdefault(p.parent.name, []).append(p)
            paths = [(by[c][i], None) for c in sorted(by) for i in sorted(rng.choice(len(by[c]), round(sample * len(by[c]) / len(paths)), replace=False))]
    out = []
    for p, ann in paths:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print("unreadable", p.name, flush=True)
            continue
        raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        h0, w0 = raw.shape[:2]
        x = cv2.resize(raw.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        out.append((p.name, x, rib_masks(ann[0], ann[1], w0, h0) if ann else None))
    return out


def bone_fn(spec, dev):
    kind, _, arg = spec.partition(":")
    if kind == "ens":  # mean bone of several checkpoints: ens:<ckpt>+<ckpt>...
        fs = [bone_fn(f"ours:{c}", dev) for c in arg.split("+")]
        return lambda x: np.mean([f(x) for f in fs], 0)
    if kind == "identity":
        return lambda x: np.zeros_like(x)
    if kind == "qure":
        m = torch.jit.load(arg, map_location=dev).eval()

        def f(x):
            with torch.no_grad():
                o = m(torch.from_numpy(x)[None, None].to(dev))
                return (o[0] if isinstance(o, (tuple, list)) else o)[0, 0].float().cpu().numpy()
        return f
    if kind == "bsldm":  # bsldm[:fp16]: generator-based soft estimate turned into a bone estimate (s4/bsldm.py, BoneFn)
        from .bsldm import BSLDM, BoneFn
        return BoneFn(BSLDM(dev, fp16="fp16" in arg.split(":")))
    from .eval_synth import load_model
    m, _ = load_model(arg, dev)

    def g(x):  # our suppressor works at 512; the lesion signal is added BEFORE this resize, nothing is renormalised
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            b = m(torch.from_numpy(cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA))[None, None].to(dev)).float()[0, 0].cpu().numpy()
        return cv2.resize(b, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/szch")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default="szch", choices=["szch", "ribcxr", "jsrt", "bdcxr"])
    ap.add_argument("--sample", type=int, default=0, help="class-stratified fixed subset of this many images (bdcxr: 300)")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "N"), help="evaluate images with index %% N == I; the site draws are replayed for all images")
    ap.add_argument("--eval_sites", type=int, nargs=2, default=[0, 10**6], metavar=("A", "B"), help="evaluate only sites A <= site < B of each image (all --sites are still drawn, so the sites match other methods')")
    ap.add_argument("--overlap_ref", default=None, help="method spec whose bone map defines bone overlap when the set has no bone truth (jsrt)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lib = lesion_lib.load(a.lib)
    pairs = load_images(a.dataset, a.root, a.sample)[: a.limit or None]
    cache = os.environ.get("S4_AUDIT_MASKS")  # masks exported on the machine of the final table (tools/export_audit_masks.py): another GPU then draws the same sites
    if cache:
        z = np.load(Path(cache) / f"{a.dataset}.npz")
        assert list(z["names"][: len(pairs)]) == [n for n, _, _ in pairs] and (a.limit or len(pairs) == len(z["names"])), "mask cache does not match the image list"
        unpack = lambda m: np.unpackbits(m, count=SIZE * SIZE).reshape(SIZE, SIZE) > 0  # noqa: E731
        lungs = [unpack(m) for m in z["lung"]]
        pairs = [(n, x, unpack(b)) for (n, x, _), b in zip(pairs, z["bone"])]
    else:
        lungs = lung_masks([x for _, x, _ in pairs], a.device)
    fns = {s: bone_fn(s, a.device) for s in a.methods}
    ref = bone_fn(a.overlap_ref, a.device) if a.overlap_ref else None
    rows = []
    for k, ((name, x, bone_proxy), lung) in enumerate(zip(pairs, lungs)):
        if a.dataset == "bdcxr" and lung.mean() < 0.02:  # SAP_bdcxr.md E2: the adult-trained lung segmenter failed on this child (counted, no sites)
            print("lung mask under 2%:", name, flush=True)
            continue
        mine = k % a.shard[1] == a.shard[0]
        if mine and bone_proxy is None and ref is not None:
            r = ref(x)
            bone_proxy = r > 0.25 * np.percentile(r, 99)  # same rule as the downstream box overlap (s4/prep.py)
        b0 = {s: f(x) for s, f in fns.items()} if mine else None
        ys, xs = np.nonzero(lung)
        for site in range(a.sites):
            les = lib[int(rng.integers(len(lib)))]
            patch = cv2.resize(les["x_patch"], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)  # 512-frame px -> 1024 px
            cy, cx = int(ys[rng.integers(len(ys))]), int(xs[rng.integers(len(xs))])
            h, w = patch.shape
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
            y1, x1 = min(SIZE, y0 + h), min(SIZE, x0 + w)
            n = np.zeros((SIZE, SIZE), np.float32)
            n[y0:y1, x0:x1] = patch[: y1 - y0, : x1 - x0]
            sup = n > 0.02 * n.max()
            if sup.sum() < 30 or not mine or not a.eval_sites[0] <= site < a.eval_sites[1]:
                continue
            ov = float(bone_proxy[sup].mean()) if bone_proxy is not None else float("nan")
            wide = cv2.dilate(sup.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            for alpha in a.alphas:
                an = alpha * n
                xp = np.clip(x + an, 0, 1).astype(np.float32)
                for s, f in fns.items():
                    db = f(xp) - b0[s]
                    A = float((db * an)[sup].sum() / (an[sup] ** 2).sum())
                    rows.append(dict(method=s.split(":")[0], image=name, site=site, alpha=alpha, overlap=ov, diam_mm=les["diam_mm"],
                                     absorption=A, ltr=1 - A, leak=float(np.abs(db[~wide]).mean() / n.max())))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in sorted({r["method"] for r in rows}):
        r = [x for x in rows if x["method"] == m]
        lo, hi = np.quantile([x["overlap"] for x in r], [1 / 3, 2 / 3])
        t = lambda sel: float(np.mean([x["absorption"] for x in r if sel(x["overlap"])]))  # noqa: E731
        print(f"{m}: n={len(r)} A_mean={np.mean([x['absorption'] for x in r]):.3f} low-overlap={t(lambda o: o <= lo):.3f} "
              f"high-overlap={t(lambda o: o > hi):.3f} leak={np.mean([x['leak'] for x in r]):.4f}")


if __name__ == "__main__":
    main()
