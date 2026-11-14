"""Lesion library from rendered CTs: patches converted to composite-image intensity (units of the training input X), so a lesion
can be pasted into a REAL radiograph for the counterfactual audit."""
import numpy as np
import torch

from . import compose as C
from . import xray

LUNG_GAIN_MEAN = 2.0  # mid-point of the training range (1, 3)


def conversion(bank, idx, phys, dev, n=64):
    """-> (x_per_dP, a_ref): composite intensity per unit lung pixel value, and the typical lung areal density (g/cm^2)."""
    a = bank["a"][torch.tensor(idx[:n], device=dev)].float()
    P = phys(a, torch.full((len(a),), 3, device=dev))
    s_raw = P[:, 1] + LUNG_GAIN_MEAN * P[:, 2]
    lo, hi = (torch.quantile(s_raw.flatten(1), q, dim=1) for q in (0.01, 0.995))
    x_per_dP = LUNG_GAIN_MEAN * float(torch.median(1.0 / ((hi - lo) * (1 + C.BMAX))))
    return x_per_dP, float(torch.median(a[:, 2][a[:, 2] > 0]))


def build(bank, idx, phys, dev, min_diam_mm=4.0):
    """Library from the CTs in `idx` (use the TEST split so audited lesions were never seen in training)."""
    x_per_dP, a_ref = conversion(bank, idx, phys, dev)
    keep = set(idx)
    base = xray.to_pixel(np.float32(a_ref), 100, "lung")
    out = []
    for p in bank["patches"]:
        if p["ct"] in keep and p["diam_mm"] >= min_diam_mm:
            dP = xray.to_pixel(a_ref + p["patch"].cpu().numpy(), 100, "lung") - base
            out.append(dict(x_patch=(dP * x_per_dP).astype(np.float32), diam_mm=p["diam_mm"]))
    return out, dict(x_per_dP=x_per_dP, a_ref=a_ref)


def save(lib, consts, path):
    np.savez_compressed(path, diam=np.array([p["diam_mm"] for p in lib]), consts=np.array([consts["x_per_dP"], consts["a_ref"]]),
                        **{f"p{i}": p["x_patch"] for i, p in enumerate(lib)})


def load(path):
    z = np.load(path)
    return [dict(x_patch=z[f"p{i}"], diam_mm=float(d)) for i, d in enumerate(z["diam"])]
