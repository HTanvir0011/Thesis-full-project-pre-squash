"""Qure ct2xr-projections (CC BY-NC-SA 4.0) as an extra training source with EXACT additive components.

A row holds translated T, bone_translated Bt, soft_translated St and lung L (uint8 512x512). tools/qp_prep.py recovered T = a*St + b*Bt + c*L + d
(R^2 = 1.000), so the additive bone image of X = T/255 is B = b*Bt/255 and the rest is S = X - B. One sample per CT, kVp 100, real-domain translated.

Augmentation mirrors compose.py: bone re-mix (exact: X' = S + w*B) -> one affine -> lesion insertion into S (bone untouched) -> monotone gamma tone
(X'=g(S+B), S'=g(S), B'=X'-S') -> blur (linear, X' and B') -> noise/scatter (X' only).
Packed layout (tools/qp_pack.py): qp.bin = concatenated PNG bytes; qp_index.npy [N,3,2] (offset, length) for T, Bt, L; qp_meta.csv rows in the same order.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import compose as C


class QPBank:
    def __init__(self, pack_dir, split="train", device="cuda", workers=4, seed=0, limit=0):
        d = Path(pack_dir)
        self.dev = device
        self.buf = np.memmap(d / "qp.bin", dtype=np.uint8, mode="r")
        index = np.load(d / "qp_index.npy")
        meta = list(csv.DictReader(open(d / "qp_meta.csv")))
        keep = [i for i, r in enumerate(meta) if r["split"] == split][: limit or None]
        self.index, self.w_bone = index[keep], np.array([float(meta[i]["w_bone"]) for i in keep], np.float32)
        self.names = [meta[i]["volume_name"] for i in keep]
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(workers)  # PNG decoding
        self.pre = ThreadPoolExecutor(1)  # prefetching the next batch (separate so a prefetch never waits on its own pool)
        self._next = None

    def __len__(self):
        return len(self.names)

    def _png(self, off, n):
        return cv2.imdecode(np.asarray(self.buf[off:off + n]), cv2.IMREAD_GRAYSCALE)

    def _load(self, k):
        (o0, n0), (o1, n1), (o2, n2) = self.index[k]
        return self._png(o0, n0), self._png(o1, n1), self._png(o2, n2)

    def _fetch(self, idx):
        rows = list(self.pool.map(self._load, idx))
        t = torch.from_numpy(np.stack([[r[0], r[1], r[2]] for r in rows]))  # [B,3,H,W] uint8: T, Bt, L
        return idx, t.pin_memory() if self.dev == "cuda" else t

    def raw(self, bs, idx=None):
        """-> (X, B, L) float on device in [0,1] units; random rows (prefetched) unless idx is given."""
        if idx is None:
            if self._next is None:
                self._next = self.pre.submit(self._fetch, self.rng.integers(len(self), size=bs))
            idx, t = self._next.result()
            self._next = self.pre.submit(self._fetch, self.rng.integers(len(self), size=bs))
        else:
            idx, t = self._fetch(np.asarray(idx))
        t = t.to(self.dev, non_blocking=True).float() / 255.0
        b = torch.from_numpy(self.w_bone[idx]).to(self.dev).view(-1, 1, 1, 1)
        return t[:, 0:1], b * t[:, 1:2], t[:, 2:3]


def lung_field(L, thr=0.04, k=41):
    """Candidate lesion positions: the lung component (vessel tree) closed into a field. L [B,1,H,W] in [0,1]."""
    m = (L > thr).float()
    return (F.max_pool2d(m, k, 1, k // 2) > 0).float() * (F.avg_pool2d(m, 61, 1, 30) > 0.02).float()


def insert_lesions(S, field, library, n_max, gen=None, thr=0.01, n_min=0):
    """Image-space CLIA: add random lesion patches (composite units, from TRAIN CTs) into S at random lung-field pixels. -> lesion mask [B,1,H,W]."""
    B, _, H, W = S.shape
    dev = S.device
    mask = torch.zeros_like(S)
    if n_max <= 0 or not library:
        return mask
    for b in range(B):
        px = (field[b, 0] > 0).nonzero()
        if len(px) == 0:
            continue
        for _ in range(int(torch.randint(n_min, n_max + 1, (1,), device=dev, generator=gen))):
            p = library[int(torch.randint(len(library), (1,), device=dev, generator=gen))]
            s = 0.7 + 0.8 * float(torch.rand(1, device=dev, generator=gen))
            amp = (0.6 + float(torch.rand(1, device=dev, generator=gen))) * s
            p = F.interpolate(p[None, None].to(dev), scale_factor=s, mode="bilinear", align_corners=False)[0, 0] * amp
            cy, cx = px[int(torch.randint(len(px), (1,), device=dev, generator=gen))].tolist()
            h, w = p.shape
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
            y1, x1 = min(H, y0 + h), min(W, x0 + w)
            sub = p[: y1 - y0, : x1 - x0]
            S[b, 0, y0:y1, x0:x1] += sub
            mask[b, 0, y0:y1, x0:x1] = torch.maximum(mask[b, 0, y0:y1, x0:x1], (sub > thr).float())
    return mask


def qp_compose(X, B, L, library=None, n_insert=0, rot_deg=15.0, remix=(0.6, 1.7), gamma=(0.85, 1.2), detector=True, gen=None):
    """X [B,1,H,W] translated composite, B its exact additive bone image, L lung component -> dict(X, B, S, roi, nod) like compose.compose."""
    Bn, dev = X.shape[0], X.device
    u = lambda: torch.rand(Bn, 1, 1, 1, device=dev, generator=gen)  # noqa: E731
    w = torch.exp(torch.log(torch.tensor(remix[0], device=dev)) + u() * (np.log(remix[1]) - np.log(remix[0]))) if remix else torch.ones(Bn, 1, 1, 1, device=dev)
    S = X - B
    Bw = w * B
    q = torch.quantile(L.flatten(1), 0.999, dim=1).view(Bn, 1, 1, 1).clamp_min(1e-4)
    m = (S + Bw).flatten(1).amax(1).view(Bn, 1, 1, 1).clamp_min(1.0)  # keep X in [0,1]: scale all components together
    maps = C.random_affine(torch.cat([S / m, Bw / m, L / q], 1), rot_deg=rot_deg, gen=gen)
    S, Bw, Lq = maps[:, 0:1].contiguous(), maps[:, 1:2].contiguous(), maps[:, 2:3]
    nod = torch.zeros_like(S)
    if n_insert > 0 and library:
        nod = insert_lesions(S, lung_field(Lq * q), library, n_insert, gen)
        nod = F.max_pool2d(nod, 7, 1, 3)
    kx = torch.linspace(0, 1.5, 65, device=dev).expand(Bn, -1)
    gam = (gamma[0] + (gamma[1] - gamma[0]) * torch.rand(Bn, 1, device=dev, generator=gen)) if gamma else torch.ones(Bn, 1, device=dev)
    ky = kx.clamp_min(0) ** gam
    Xp, Sp = C.apply_tone(S + Bw, kx, ky), C.apply_tone(S, kx, ky)
    Bp = Xp - Sp
    if detector:
        sig = 1e-3 + 1.2 * u() * (u() < 0.5)
        for i in range(Bn):
            if sig[i] > 0.05:
                Xp[i:i + 1], Bp[i:i + 1] = C.gaussian_blur(Xp[i:i + 1], float(sig[i])), C.gaussian_blur(Bp[i:i + 1], float(sig[i]))
        noise = (0.004 + 0.02 * Xp.clamp_min(0).sqrt()) * torch.randn(Xp.shape, device=dev, generator=gen) * u()
        scatter = C.gaussian_blur(Xp, 20.0) * 0.08 * u()
        Xp = Xp + noise + scatter
    return dict(X=Xp, B=Bp, S=Sp, roi=(Lq > 0.3).float(), nod=nod)
