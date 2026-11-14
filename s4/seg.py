"""CT -> body / lung / bone masks and densities. HU thresholds follow the published recipe (Angaitkar et al. 2026, Alg. 1)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball, disk


def rho_from_hu(hu):
    """Water-equivalent density in g/cm^3 (air 0, water 1)."""
    return np.clip(1.0 + hu.astype(np.float32) / 1000.0, 0.0, 3.0)


def _largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(np.argmax(sizes)))


def body_mask(hu):
    m = hu > -300
    for z in range(m.shape[0]):  # fill first: opening a thin chest wall would otherwise leave the lungs unfilled
        m[z] = ndi.binary_fill_holes(m[z])
    return _largest(ndi.binary_opening(m, structure=np.ones((1, 9, 9), bool)))  # cuts the couch off the body


def box_fraction(mask, spacing, size_mm=10.0, coarse_mm=3.0):
    """Fraction of `mask` inside a size_mm cube around each voxel, computed on a coarse grid and expanded back."""
    f = [max(1, int(round(coarse_mm / s))) for s in spacing]
    cs = [mask.shape[i] // f[i] for i in range(3)]
    core = mask[: cs[0] * f[0], : cs[1] * f[1], : cs[2] * f[2]].astype(np.float32)
    coarse = core.reshape(cs[0], f[0], cs[1], f[1], cs[2], f[2]).mean(axis=(1, 3, 5))
    k = [max(1, int(round(size_mm / (spacing[i] * f[i])))) for i in range(3)]
    coarse = ndi.uniform_filter(coarse, size=k, mode="constant")
    out = np.zeros(mask.shape, np.float32)
    up = np.repeat(np.repeat(np.repeat(coarse, f[0], 0), f[1], 1), f[2], 2)
    out[: up.shape[0], : up.shape[1], : up.shape[2]] = up
    return out


def bone_mask(hu, spacing, taus=(200, 150, 100), theta=0.05, kappa=0.01, ref_hu=300):
    ref = hu > ref_hu
    m = ref.copy()
    for tau in taus:
        m = ((hu > tau) & (box_fraction(m, spacing) >= theta)) | ref
    zs, ys, xs = np.nonzero(m)
    if len(zs) == 0:
        return m
    pad = 4
    sl = tuple(slice(max(0, a.min() - pad), a.max() + pad + 1) for a in (zs, ys, xs))
    # fills marrow inside the cortical shell; edge-padded because scipy's closing erodes anything touching the border
    m[sl] = ndi.binary_closing(np.pad(m[sl], 3, mode="edge"), structure=ball(2))[3:-3, 3:-3, 3:-3]
    lab, n = ndi.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(sizes >= kappa * sizes.max())[0] + 1  # prune speckle islands to soft tissue
    return np.isin(lab, keep)


def lung_mask(hu, body, bone, min_rel=0.2):
    air = (hu < -320) & body
    lab, n = ndi.label(air)
    if n == 0:
        return air
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(sizes >= min_rel * sizes.max())[0] + 1
    lung = np.isin(lab, keep)
    se = disk(5)
    for z in np.nonzero(lung.any(axis=(1, 2)))[0]:
        lung[z] = ndi.binary_fill_holes(ndi.binary_closing(lung[z], structure=se))
    return lung & body & ~bone


def soft_density(rho, body, lung, bone, spacing, sigma_mm=4.0):
    """Non-lung soft-tissue density with bone voxels filled from their soft neighbours (normalised convolution)."""
    soft = body & ~lung & ~bone
    out = np.where(soft, rho, 0.0).astype(np.float32)
    if not bone.any():
        return out
    zs, ys, xs = np.nonzero(bone)
    pad = 20
    sl = tuple(slice(max(0, a.min() - pad), a.max() + pad + 1) for a in (zs, ys, xs))
    w = soft[sl].astype(np.float32)
    v = (rho[sl] * w).astype(np.float32)
    fill = np.full(w.shape, np.nan, np.float32)
    for mult in (1.0, 3.0, 9.0):
        sig = [sigma_mm * mult / s for s in spacing]
        den = ndi.gaussian_filter(w, sig)
        num = ndi.gaussian_filter(v, sig)
        ok = np.isnan(fill) & (den > 1e-2)
        fill[ok] = num[ok] / den[ok]
    fill[np.isnan(fill)] = 1.0
    sub = out[sl]
    b = bone[sl] & body[sl]
    sub[b] = fill[b]
    return out
