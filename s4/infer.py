"""Run a suppressor over a list of real images and keep the input variants used downstream, all at one fixed resolution.

Variants written per image (uint8, SIZE x SIZE): orig, soft (= x - bone), and optionally bone. Conventions (same as training
and as the reference model): area-resize to 512, per-image min-max to [0,1]; soft = x - bone is computed at 512 and clipped.
"""
import cv2
import numpy as np
import torch

from .model import SwinUNetPP


class Ensemble(torch.nn.Module):
    """Deep ensemble: mean bone image of several suppressors, called like one model."""

    def __init__(self, models):
        super().__init__()
        self.members = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)


def load_suppressor(ckpt, device="cuda"):
    """ckpt: one final_ema.pt or several joined by '+' (deep ensemble of the same architecture family)."""
    ms = []
    for c in str(ckpt).split("+"):
        st = torch.load(c, map_location="cpu")
        m = SwinUNetPP(st["backbone"], 512, pretrained=False)
        m.load_state_dict({k: v.float() for k, v in st["ema"].items()})
        ms.append(m.to(device).eval())
    return ms[0] if len(ms) == 1 else Ensemble(ms).eval()


class Clahe:
    """Appearance control: on-the-fly CLAHE (clip 2, 8x8 tiles) of a uint8 array [N,S,S]; indexed like the array it wraps."""

    def __init__(self, X):
        self.X, self.c = X, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __getitem__(self, idx):
        a = np.asarray(self.X[idx])
        return self.c.apply(a) if a.ndim == 2 else np.stack([self.c.apply(m) for m in a])


def prep512(gray):
    x = cv2.resize(gray.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


@torch.no_grad()
def suppress_batch(model, x512, device="cuda"):
    """x512: [B,512,512] float32 in [0,1] -> bone [B,512,512] float32."""
    t = torch.from_numpy(x512)[:, None].to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        return model(t).float()[:, 0].cpu().numpy()


def to_u8(x, size):
    y = np.clip(x, 0, 1)
    if size != y.shape[-1]:
        y = np.stack([cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA) for a in y])
    return (y * 255.0 + 0.5).astype(np.uint8)


def variants(model, grays, size, device="cuda"):
    """list of 2-D float images -> dict(orig, soft, bone) uint8 arrays [B,size,size]."""
    x = np.stack([prep512(g) for g in grays])
    b = suppress_batch(model, x, device)
    return dict(orig=to_u8(x, size), soft=to_u8(x - b, size), bone=to_u8(np.clip(b, 0, 1), size))
