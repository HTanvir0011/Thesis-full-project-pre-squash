"""Polychromatic projection physics: areal density (g/cm^2) -> pixel value P = -log(sum phi*E*exp(-mu*a) / sum phi*E)."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MATERIALS = ("bone_cortical", "soft_tissue", "lung")


@lru_cache(maxsize=1)
def tables():
    return json.loads((Path(__file__).parent / "data" / "xray_tables.json").read_text())


def spectrum(kvp):
    """Normalised photon spectrum on the 1 keV grid; linear blend between tabulated kVp values."""
    spec = tables()["spectra"]
    ks = sorted(int(k) for k in spec)
    if not ks[0] <= kvp <= ks[-1]:
        raise ValueError(f"kvp {kvp} outside tabulated {ks[0]}-{ks[-1]}")
    lo = max(k for k in ks if k <= kvp)
    hi = min(k for k in ks if k >= kvp)
    if lo == hi:
        return np.array(spec[str(lo)])
    w = (kvp - lo) / (hi - lo)
    return (1 - w) * np.array(spec[str(lo)]) + w * np.array(spec[str(hi)])


def projection_lut(kvp, material, a_max=80.0, n=8192, mono_keV=None):
    """Return (a_grid, P_grid). mono_keV set -> single-energy beam (exactly additive, used by tests)."""
    energy = np.array(tables()["energy_keV"])
    mu = np.array(tables()["mu_over_rho_cm2_per_g"][material])
    if mono_keV is None:
        w = spectrum(kvp) * energy
    else:
        w = (np.abs(energy - mono_keV) < 0.5).astype(float)
    a = np.linspace(0.0, a_max, n)
    inten = (w[None, :] * np.exp(-mu[None, :] * a[:, None])).sum(1)
    return a, -np.log(inten / w.sum())


def to_pixel(a_map, kvp, material, **kw):
    a, p = projection_lut(kvp, material, **kw)
    return np.interp(a_map, a, p).astype(np.float32)
