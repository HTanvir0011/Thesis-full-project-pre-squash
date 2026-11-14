"""GPU sample composer: areal-density maps -> input radiograph X' and its EXACT components (bone B', soft S'), X' = S' + B'.

Order: lesion insertion -> geometry (one affine for every map) -> polychromatic physics -> component weighting ->
monotone tone curve g (gamma or histogram matching) -> X' = g(S+B), S' = g(S), B' = X' - S' -> detector effects.
Detector effects: Gaussian blur is linear and applied to X' and B' alike; noise and scatter touch X' only."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import xray

KVPS = (70, 80, 90, 100, 110, 120, 130, 140)
BMAX = 0.5  # largest bone amplitude; X is divided by 1 + BMAX so it stays within [0, ~1]


class Physics:
    def __init__(self, device, a_max=80.0, n=8192):
        rows = [[xray.projection_lut(k, m, a_max, n)[1] for m in xray.MATERIALS] for k in KVPS]  # bone, soft, lung
        self.lut = torch.tensor(np.array(rows), dtype=torch.float32, device=device)  # [K, 3, n]
        self.step, self.n = a_max / (n - 1), n

    def __call__(self, a, kvp_idx):  # a: [B,3,H,W] (bone, non-lung soft, lung) in g/cm^2
        B, C, H, W = a.shape
        pos = (a.float() / self.step).clamp(0, self.n - 1.001)
        i0 = pos.floor().long()
        flat = i0.flatten(2)
        lut = self.lut[kvp_idx]
        v0, v1 = torch.gather(lut, 2, flat), torch.gather(lut, 2, flat + 1)
        return (v0 + (v1 - v0) * (pos - i0).flatten(2)).view(B, C, H, W)


def random_affine(x, rot_deg=15.0, zoom=(0.9, 1.15), shift=0.06, flip=True, gen=None):
    """One random affine (flip = AP mirror) shared by every channel of x [B,C,H,W]; border padding."""
    B = x.shape[0]
    r = lambda: torch.rand(B, device=x.device, generator=gen)  # noqa: E731
    ang = (r() * 2 - 1) * math.radians(rot_deg)
    s = zoom[0] + r() * (zoom[1] - zoom[0])
    fl = torch.where(r() < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=x.device)
    tx, ty = (r() * 2 - 1) * shift * 2, (r() * 2 - 1) * shift * 2
    c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([c * fl, -sn, tx], 1), torch.stack([sn * fl, c, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def insert_nodules(a, library, n_max, gen=None, thr=0.02):
    """Add random lesion signals (areal-density patches from TRAIN CTs) at random lung pixels, over bone or not.
    a: [B,4,H,W] (bone, nl, lung, lesion mask) is modified in place; the bone channel is never touched."""
    if n_max <= 0 or not library:
        return a
    B, _, H, W = a.shape
    dev = a.device
    for b in range(B):
        lung_px = (a[b, 2] > 0).nonzero()
        if len(lung_px) == 0:
            continue
        for _ in range(int(torch.randint(0, n_max + 1, (1,), device=dev, generator=gen))):
            p = library[int(torch.randint(len(library), (1,), device=dev, generator=gen))]
            s = 0.7 + 0.8 * float(torch.rand(1, device=dev, generator=gen))
            amp = (0.6 + float(torch.rand(1, device=dev, generator=gen))) * s
            p = F.interpolate(p[None, None].to(dev), scale_factor=s, mode="bilinear", align_corners=False)[0, 0] * amp
            cy, cx = lung_px[int(torch.randint(len(lung_px), (1,), device=dev, generator=gen))].tolist()
            h, w = p.shape
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
            y1, x1 = min(H, y0 + h), min(W, x0 + w)
            sub = p[: y1 - y0, : x1 - x0]
            a[b, 2, y0:y1, x0:x1] += sub
            a[b, 3, y0:y1, x0:x1] = torch.maximum(a[b, 3, y0:y1, x0:x1], (sub > thr).float())
    return a


def _quantiles(x, probs):
    return torch.quantile(x.flatten(1).float(), probs, dim=1).T  # [B, K]


def apply_tone(x, kx, ky):
    """Per-sample monotone piecewise-linear map; kx strictly increasing [B,K], ky [B,K]; x [B,1,H,W]."""
    B = x.shape[0]
    v = x.flatten(1).float().contiguous()
    i = torch.searchsorted(kx.contiguous(), v).clamp(1, kx.shape[1] - 1)
    x0, x1 = torch.gather(kx, 1, i - 1), torch.gather(kx, 1, i)
    y0, y1 = torch.gather(ky, 1, i - 1), torch.gather(ky, 1, i)
    t = ((v - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0, 1)
    return (y0 + (y1 - y0) * t).view_as(x)


def gaussian_blur(x, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2).float() / (2 * sigma**2))
    g = (g / g.sum()).to(x.dtype)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))


def compose(a4, kvp_idx, phys, gen=None, tone="gamma", ref_q=None, detector=True, lung_gain=(1.0, 3.0), bone_amp=(0.2, 0.5)):
    """a4 [B,4,H,W] augmented maps -> dict(X, B, S, roi, nod). tone: 'gamma' | 'hist' (needs ref_q [R,257] real-image quantiles)."""
    dev, Bn = a4.device, a4.shape[0]
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(Bn, 1, 1, device=dev, generator=gen)  # noqa: E731
    P = phys(a4[:, :3], kvp_idx)
    s_raw = P[:, 1] + r(*lung_gain) * P[:, 2]
    lo, hi = (_quantiles(s_raw, torch.tensor([q], device=dev))[:, 0].view(Bn, 1, 1) for q in (0.01, 0.995))
    S = (s_raw - lo) / (hi - lo).clamp_min(1e-6)
    p99 = _quantiles(P[:, 0], torch.tensor([0.99], device=dev))[:, 0].view(Bn, 1, 1).clamp_min(1e-6)
    Bone = r(*bone_amp) * P[:, 0] / p99
    S, Bone = (S / (1 + BMAX))[:, None], (Bone / (1 + BMAX))[:, None]
    X = S + Bone
    if tone == "hist" and ref_q is not None:
        probs = torch.linspace(0, 1, ref_q.shape[1], device=dev)
        ramp = torch.linspace(0, 1e-5, ref_q.shape[1], device=dev)
        kx = _quantiles(X, probs) + ramp
        ky = ref_q[torch.randint(len(ref_q), (Bn,), device=dev, generator=gen)].to(dev) + ramp
    else:  # gamma curve on a fixed grid
        kx = torch.linspace(0, 1.5, 65, device=dev).expand(Bn, -1)
        gam = (0.7 + 0.7 * torch.rand(Bn, 1, device=dev, generator=gen))
        ky = kx.clamp_min(0) ** gam
    Xp, Sp = apply_tone(X, kx, ky), apply_tone(S, kx, ky)
    Bp = Xp - Sp  # >= 0 because the tone curve is monotone
    if detector:
        sig = 1e-3 + 1.2 * torch.rand(Bn, 1, 1, 1, device=dev, generator=gen) * (torch.rand(Bn, 1, 1, 1, device=dev, generator=gen) < 0.5)
        for i in range(Bn):  # linear blur: identical on X' and B' so the decomposition stays exact
            if sig[i] > 0.05:
                Xp[i:i + 1], Bp[i:i + 1] = gaussian_blur(Xp[i:i + 1], float(sig[i])), gaussian_blur(Bp[i:i + 1], float(sig[i]))
        noise = (0.004 + 0.02 * Xp.clamp_min(0).sqrt()) * torch.randn(Xp.shape, device=dev, generator=gen) * torch.rand(Bn, 1, 1, 1, device=dev, generator=gen)
        scatter = gaussian_blur(Xp, 20.0) * 0.08 * torch.rand(Bn, 1, 1, 1, device=dev, generator=gen)
        Xp = Xp + noise + scatter
    pl = P[:, 2:3]
    q = _quantiles(pl, torch.tensor([0.999], device=dev))[:, 0].view(Bn, 1, 1, 1).clamp_min(1e-6)
    roi = ((pl / q) > 0.3).float()
    nod = F.max_pool2d(a4[:, 3:4], 7, 1, 3)
    return dict(X=Xp, B=Bp, S=Sp, roi=roi, nod=nod)
