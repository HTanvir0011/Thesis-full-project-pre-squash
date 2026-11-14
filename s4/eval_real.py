"""Real paired evaluation on the SZCH-X-Rays test set (75 radiographs with dual-energy-subtraction soft-tissue references).

Protocol (fixed before any model is scored on it):
  input  = CXR resized to 1024 (area), min-max normalised to [0,1]; method returns an estimate of the soft-tissue image
  raw    = metrics against the DES image as is (tone differs between CXR and DES, so raw penalises predict-and-subtract models)
  cal    = same metrics after a per-image least-squares gain/offset of the estimate onto the DES image  (PRIMARY)
  bone   = `cal` metrics restricted to the pixels where DES removed the most signal (top 10% of calibrated CXR - DES), and
           `nonbone` = the complement (preservation)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

SIZE = 1024


def load_pairs(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "CXR", "*.png"))):
        n = os.path.basename(p)
        x = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        g = cv2.imread(os.path.join(root, "BS", n), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        x = cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        out.append((n, x, cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)))
    return out


def calibrate(est, ref):
    """Least-squares gain/offset mapping est onto ref (per image)."""
    A = np.stack([est.ravel(), np.ones(est.size, np.float32)], 1)
    (a, b), *_ = np.linalg.lstsq(A, ref.ravel(), rcond=None)
    return np.clip(a * est + b, 0, 1)


def _psnr(a, b, m=None):
    d = (a - b) ** 2
    mse = float(d[m].mean() if m is not None else d.mean())
    return 10 * np.log10(1.0 / max(mse, 1e-10)), mse


def ms_ssim(a, b):
    """MS-SSIM (Wang 2003: 11x11 Gaussian window sigma 1.5, 5 scales, standard exponents) of two images in [0,1]."""
    import torch
    import torch.nn.functional as F
    w = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)
    g = torch.exp(-(torch.arange(11, dtype=torch.float64) - 5) ** 2 / (2 * 1.5 ** 2))
    k = ((g / g.sum())[:, None] * (g / g.sum())[None, :])[None, None]
    x, y = (torch.from_numpy(np.ascontiguousarray(z, np.float64))[None, None] for z in (a, b))
    vals = []
    for i in range(5):
        mx, my = F.conv2d(x, k), F.conv2d(y, k)
        sxx, syy, sxy = F.conv2d(x * x, k) - mx ** 2, F.conv2d(y * y, k) - my ** 2, F.conv2d(x * y, k) - mx * my
        cs = ((2 * sxy + 0.03 ** 2) / (sxx + syy + 0.03 ** 2)).mean().clamp_min(1e-6)
        if i == 4:
            cs = cs * ((2 * mx * my + 0.01 ** 2) / (mx ** 2 + my ** 2 + 0.01 ** 2)).mean()  # luminance term only at the coarsest scale
        vals.append(cs)
        x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return float(torch.prod(torch.stack(vals) ** w))


_LPIPS = {}


def lpips_dist(a, b, device):
    """LPIPS (AlexNet) between two grey images in [0,1], replicated to 3 channels."""
    import lpips
    import torch
    if device not in _LPIPS:
        _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    t = lambda z: (torch.from_numpy(np.ascontiguousarray(z, np.float32))[None, None].expand(1, 3, -1, -1) * 2 - 1).to(device)  # noqa: E731
    with torch.no_grad():
        return float(_LPIPS[device](t(a), t(b)))


def metrics(x, gt, est, lpips_device=None):
    est = np.clip(est, 0, 1).astype(np.float32)
    r = {}
    r["psnr_raw"], r["mse_raw"] = _psnr(est, gt)
    r["ssim_raw"] = structural_similarity(est, gt, data_range=1.0)
    cal = calibrate(est, gt)
    r["psnr_cal"], r["mse_cal"] = _psnr(cal, gt)
    _, smap = structural_similarity(cal, gt, data_range=1.0, full=True)
    r["ssim_cal"] = float(smap.mean())
    d = calibrate(x, gt) - gt
    bone = d > np.quantile(d, 0.90)
    r["psnr_cal_bone"], _ = _psnr(cal, gt, bone)
    r["psnr_cal_nonbone"], _ = _psnr(cal, gt, ~bone)
    r["ssim_cal_bone"], r["ssim_cal_nonbone"] = float(smap[bone].mean()), float(smap[~bone].mean())
    # secondary (added after the first ablation, see DEVIATIONS): the DES soft image has ~9% less high-frequency energy than the CXR, which
    # penalises full-resolution SSIM of methods that keep the CXR texture; also fidelity of the removed signal to the DES bone signal d.
    bl = lambda z, s: cv2.GaussianBlur(z, (0, 0), s)  # noqa: E731
    r["ssim_cal_s2"] = float(structural_similarity(bl(cal, 2), bl(gt, 2), data_range=1.0))
    r["ssim_cal_s4"] = float(structural_similarity(bl(cal, 4), bl(gt, 4), data_range=1.0))
    rem = bl(calibrate(x, gt) - cal, 2).ravel()
    ds = bl(d, 2).ravel()
    ds, rem = ds - ds.mean(), rem - rem.mean()
    r["bone_corr_s2"] = float((ds * rem).sum() / max(np.sqrt((ds * ds).sum() * (rem * rem).sum()), 1e-12))
    r["bone_slope_s2"] = float((ds * rem).sum() / max((ds * ds).sum(), 1e-12))
    hp = lambda z: z - bl(z, 3)  # noqa: E731
    r["hp_ratio_nonbone"] = float(hp(cal)[~bone].std() / max(hp(gt)[~bone].std(), 1e-12))
    r["msssim_cal"] = ms_ssim(cal, gt)
    if lpips_device:  # needs the lpips package; used by the local final evaluation only
        r["lpips_raw"], r["lpips_cal"] = lpips_dist(est, gt, lpips_device), lpips_dist(cal, gt, lpips_device)
    return r


def clahe(x):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply((x * 255).astype(np.uint8)).astype(np.float32) / 255.0


def homomorphic(x, sigma=40.0, g_low=0.6, g_high=1.5):
    """Appearance-only control: attenuate low spatial frequencies and boost high ones in the log domain."""
    lg = np.log(x + 0.02)
    low = cv2.GaussianBlur(lg, (0, 0), sigma)
    y = np.exp(g_low * low + g_high * (lg - low))
    return (y - y.min()) / (y.max() - y.min() + 1e-10)


def make_qure(weights, device):
    import torch
    m = torch.jit.load(weights, map_location=device).eval()

    def f(x):
        with torch.no_grad():
            out = m(torch.from_numpy(x)[None, None].to(device))
            out = out[0] if isinstance(out, (tuple, list)) else out
            return x - out[0, 0].float().cpu().numpy()
    return f


def make_ours(ckpt, device):
    import torch
    from .model import SwinUNetPP
    st = torch.load(ckpt, map_location="cpu")
    m = SwinUNetPP(st["backbone"], 512, pretrained=False)
    m.load_state_dict({k: v.float() for k, v in st["ema"].items()})
    m = m.to(device).eval()

    def f(x):
        x5 = cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
        x5 = (x5 - x5.min()) / (x5.max() - x5.min() + 1e-10)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            b = m(torch.from_numpy(x5)[None, None].to(device)).float()[0, 0].cpu().numpy()
        return x - cv2.resize(b, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return f


def make_ens(spec, device):
    fs = [make_ours(c, device) for c in spec.split("+")]
    return lambda x: np.mean([f(x) for f in fs], 0)  # mean soft image = x - mean bone


def make_bsldm(spec, device):
    """bsldm[:fp16][:steps=N] -> f(x) = the soft-tissue estimate of the released BS-LDM SZCH weights (s4/bsldm.py); metrics() calibrates its scale."""
    from .bsldm import BSLDM
    opts = spec.split(":") if spec else []
    g = BSLDM(device, fp16="fp16" in opts, steps=int(next((o.split("=")[1] for o in opts if o.startswith("steps=")), 1000)))
    return lambda x: g.soft(x, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/szch")
    ap.add_argument("--method", required=True, help="identity | clahe | homomorphic | qure:<weights.ts> | ours:<final_ema.pt> | ens:<ckpt>+<ckpt>...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lpips", action="store_true", help="also LPIPS (AlexNet) of the raw and calibrated estimate")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "N"), help="process images with index %% N == I (one process per GPU)")
    ap.add_argument("--save_est", default=None, help="directory: also store each soft-tissue estimate as float16 .npy")
    a = ap.parse_args()
    kind, _, arg = a.method.partition(":")
    f = {"identity": lambda: (lambda x: x), "clahe": lambda: clahe, "homomorphic": lambda: homomorphic, "qure": lambda: make_qure(arg, a.device),
         "ours": lambda: make_ours(arg, a.device), "ens": lambda: make_ens(arg, a.device), "bsldm": lambda: make_bsldm(arg, a.device)}[kind]()
    rows = []
    for k, (n, x, g) in enumerate(load_pairs(a.root)[: a.limit or None]):
        if k % a.shard[1] != a.shard[0]:
            continue
        est = f(x)
        if a.save_est:
            Path(a.save_est).mkdir(parents=True, exist_ok=True)
            np.save(Path(a.save_est) / f"{n}.npy", np.asarray(est, np.float16))
        rows.append(dict(image=n, **metrics(x, g, est, a.device if a.lpips else None)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    Path(a.out).write_text("\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]))
    summ = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))) for k in keys[1:]}
    print(json.dumps({"method": a.method, "n": len(rows), **{k: f"{m:.4f} +- {s:.4f}" for k, (m, s) in summ.items()}}, indent=1))


if __name__ == "__main__":
    main()
