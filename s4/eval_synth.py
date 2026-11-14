"""Synthetic held-out evaluation with exact ground truth: bone/soft image error, lesion contrast ratio, and the lesion-absorption
audit (insert a lesion from ANOTHER CT, measure how much of it the bone prediction absorbs). One row per CT / per insertion site.

python -m s4.eval_synth --ckpt runs/x/final_ema.pt --bank DIR --split splits.csv --out results/x
Several --ckpt values share one bank load; each writes to --out/<its run directory name>.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from . import compose as C
from .bank import load_bank
from .losses import ssim
from .model import SwinUNetPP
from .train import lesion_contrast

KVP_IDX = 3  # 100 kVp


def load_model(ckpt, dev):
    st = torch.load(ckpt, map_location="cpu")
    m = SwinUNetPP(st["backbone"], 512, pretrained=False)
    m.load_state_dict({k: v.float() for k, v in st["ema"].items()})
    return m.to(dev).eval(), st


@torch.no_grad()
def predict(model, x, bs=8):
    out = []
    for s in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x[s:s + bs]).float())
    return torch.cat(out)


def canonical(bank, j, phys, dev, seed):
    a4 = torch.cat([bank["a"][j:j + 1].float(), bank["lesion"][j:j + 1][:, None].float()], 1)
    return a4, C.compose(a4.clone(), torch.tensor([KVP_IDX], device=dev), phys, gen=torch.Generator(device=dev).manual_seed(seed),
                         tone="gamma", detector=False)


def bone_overlap(a_bone_frame, lesion_mask):
    """Fraction of lesion pixels where the bone areal density exceeds 25% of its 99th percentile (project convention)."""
    thr = 0.25 * torch.quantile(a_bone_frame.flatten().float(), 0.99)
    m = lesion_mask > 0.5
    return float((a_bone_frame[m] > thr).float().mean()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--which", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    rows = [r for r in csv.DictReader(open(args.split)) if r["usable"] == "1"]
    bank = load_bank(args.bank, [r["name"] for r in rows], dev)
    split = {r["name"]: r["split"] for r in rows}
    idx = [i for i, n in enumerate(bank["names"]) if split[n] == args.which][: args.limit or None]
    # lesion library from the evaluated split itself: these lesions were never seen by training
    lib = [p["patch"] for p in bank["patches"] if p["ct"] in set(idx)]
    phys = C.Physics(dev)
    for ckpt in args.ckpt:
        model, _ = load_model(ckpt, dev)
        per_ct, per_site = [], []
        for j in idx:
            a4, o = canonical(bank, j, phys, dev, seed=j)
            pred = predict(model, o["X"])
            err = pred - o["B"]
            lc_hat, lc_true = lesion_contrast(o["X"] - pred, o["nod"]), lesion_contrast(o["S"], o["nod"])
            per_ct.append(dict(name=bank["names"][j], mae_bone=float(err.abs().mean()),
                               psnr_bone=float(10 * torch.log10(1.0 / (err**2).mean().clamp_min(1e-10))),
                               ssim_bone=float(ssim(pred.clamp(0, 1), o["B"].clamp(0, 1))), lcr=float(lc_hat / lc_true),
                               n_lesion_px=int(o["nod"].sum()),
                               slope_bone=float((pred * o["B"]).sum() / (o["B"] ** 2).sum().clamp_min(1e-12))))  # share of the true bone signal removed
            for s in range(args.sites):  # lesion-absorption audit on a clean copy of the same CT
                g = torch.Generator(device=dev).manual_seed(1000 * j + s)
                a_ins = a4.clone()
                C.insert_nodules(a_ins, lib, 1, g)
                if a_ins[:, 3].sum() == a4[:, 3].sum():
                    continue
                new = (a_ins[:, 3] > a4[:, 3])[:, None].float()
                oi = C.compose(a_ins.clone(), torch.tensor([KVP_IDX], device=dev), phys, gen=torch.Generator(device=dev).manual_seed(j),
                               tone="gamma", detector=False)
                delta = (oi["X"] - o["X"]) * new
                if float((delta**2).sum()) < 1e-8:
                    continue
                absorbed = (predict(model, oi["X"]) - pred) * new
                A = float((absorbed * delta).sum() / (delta * delta).sum())
                per_site.append(dict(name=bank["names"][j], site=s, absorption=A, ltr=1 - A,
                                     bone_overlap=bone_overlap(a_ins[0, 0], new[0, 0]), n_px=int(new.sum())))
        out = Path(args.out) / Path(ckpt).parent.name if len(args.ckpt) > 1 else Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, data in (("per_ct.csv", per_ct), ("per_site.csv", per_site)):
            if data:
                with open(out / name, "w", newline="") as f:
                    w = csv.DictWriter(f, list(data[0]))
                    w.writeheader()
                    w.writerows(data)
        m = lambda k: float(np.nanmean([r[k] for r in per_ct]))  # noqa: E731
        print(dict(n_ct=len(per_ct), n_sites=len(per_site), mae=m("mae_bone"), psnr=m("psnr_bone"), ssim=m("ssim_bone"), lcr=m("lcr"),
                   absorption=float(np.mean([r["absorption"] for r in per_site])) if per_site else None))
    

if __name__ == "__main__":
    main()
