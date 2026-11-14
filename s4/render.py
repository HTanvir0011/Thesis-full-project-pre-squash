"""CT series -> per-component areal-density maps (g/cm^2) in a square 512x512 body-crop PA frame, plus nodule signals.
The detector physics (spectrum, kVp, scatter, noise) is applied later, so one render serves every kVp."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import seg
from .ct import SkipCT
from .nodule import consensus_nodules

FRAME = 512


def project(rho_masked, dy):
    """Parallel-beam AP projection of a [z, y, x] density volume -> [z, x] areal density (g/cm^2)."""
    return rho_masked.sum(axis=1, dtype=np.float32) * (dy / 10.0)


def frame_box(body, lung, spacing, margin=1.08, extra_mm=60.0, min_lung_cm2=8.0, aspect=(0.75, 1.2)):
    dz, dy, dx = spacing
    # parenchyma extent: slices holding only the trachea (a few cm^2) must not count as lung, or every scan that starts in
    # the neck looks apex-truncated and the frame is centred too high
    lz = np.nonzero(lung.sum(axis=(1, 2)) * dy * dx / 100.0 > min_lung_cm2)[0]
    bx = np.nonzero(body.any(axis=(0, 1)))[0]
    if len(lz) == 0 or len(bx) == 0:
        raise SkipCT("no lung or body found")
    top, bot = int(lz[0]), int(lz[-1])
    width = margin * (bx[-1] - bx[0] + 1) * dx
    # height follows the lungs but stays within the aspect range of real radiographs (H/W 0.75-1.2; short LIDC scans cannot reach 1.2); the frame is later
    # squashed to a square, exactly as real radiographs are
    height = float(np.clip((bot - top + 1) * dz + extra_mm, aspect[0] * width, aspect[1] * width))
    zc, xc = (top + bot) / 2.0, (bx[0] + bx[-1]) / 2.0
    hz, hx = height / dz / 2.0, width / dx / 2.0
    nz = body.shape[0]
    if 2 * hz <= nz:  # slide the frame inside the scan instead of padding
        zc = min(max(zc, hz), nz - 1 - hz)
    return dict(z0=int(round(zc - hz)), z1=int(round(zc + hz)), x0=int(round(xc - hx)), x1=int(round(xc + hx)),
                width_mm=float(width), height_mm=height, lung_top=top, lung_bot=bot)


def to_frame(img, box, edge=True):
    """Crop (z0:z1, x0:x1) of a [z, x] map with edge (or zero) padding, then area-resize to FRAME x FRAME."""
    z0, z1, x0, x1 = box["z0"], box["z1"], box["x0"], box["x1"]
    pad = ((max(0, -z0), max(0, z1 - img.shape[0])), (max(0, -x0), max(0, x1 - img.shape[1])))
    if any(p for pp in pad for p in pp):
        img = np.pad(img, pad, mode="edge" if edge else "constant")
        z0, z1, x0, x1 = z0 + pad[0][0], z1 + pad[0][0], x0 + pad[1][0], x1 + pad[1][0]
    return cv2.resize(np.ascontiguousarray(img[z0:z1, x0:x1], np.float32), (FRAME, FRAME), interpolation=cv2.INTER_AREA)


def nodule_delta_a(nod, rho, lung, bone, spacing):
    """Excess density of the nodule over the surrounding lung, projected -> [z, x] native map (g/cm^2)."""
    zs = sorted(nod["mask"])
    sl = (slice(zs[0], zs[-1] + 1),)
    m3 = np.zeros((zs[-1] - zs[0] + 1,) + next(iter(nod["mask"].values())).shape, bool)
    for z, m in nod["mask"].items():
        m3[z - zs[0]] = m
    ring = ndi.binary_dilation(m3, iterations=3) & ~m3
    bg_ok = ring & lung[sl] & ~bone[sl]
    bg = float(np.median(rho[sl][bg_ok])) if bg_ok.sum() >= 20 else 0.2
    drho = np.clip(rho[sl] - bg, 0.0, None) * m3
    out = np.zeros((rho.shape[0], rho.shape[2]), np.float32)
    out[sl] = drho.sum(axis=1, dtype=np.float32) * (spacing[1] / 10.0)
    return out, bg


def render_ct(ct, xml_paths, cfg=None):
    """-> (arrays dict, meta dict). Raises SkipCT for unusable CTs."""
    t0 = time.time()
    hu, spacing = ct["hu"], ct["spacing"]
    dz, dy, dx = spacing
    meta = dict(ct["meta"])
    body = seg.body_mask(hu)
    bone = seg.bone_mask(hu, spacing) & body
    lung = seg.lung_mask(hu, body, bone)
    t1 = time.time()
    rho = seg.rho_from_hu(hu)
    box = frame_box(body, lung, spacing)
    a = {
        "a_bone": project(np.where(bone, rho, 0.0).astype(np.float32), dy),
        "a_nl": project(seg.soft_density(rho, body, lung, bone, spacing), dy),
        "a_lung": project(np.where(lung, rho, 0.0).astype(np.float32), dy),
    }
    out = {k: to_frame(v, box) for k, v in a.items()}
    out["lung2d"] = (to_frame(lung.any(axis=1).astype(np.float32), box, edge=False) > 0.5).astype(np.uint8)
    out["body2d"] = (to_frame(body.any(axis=1).astype(np.float32), box, edge=False) > 0.5).astype(np.uint8)
    z0, z1 = box["z0"], box["z1"]
    nz = hu.shape[0]
    meta.update(box=box, pad_top_rows=round(FRAME * max(0, -z0) / (z1 - z0)),
                pad_bottom_rows=round(FRAME * max(0, z1 - nz) / (z1 - z0)),
                lung_truncated=bool(box["lung_top"] <= 1 or box["lung_bot"] >= nz - 2),
                bone_vol_cm3=float(bone.sum() * dz * dy * dx / 1000), lung_vol_cm3=float(lung.sum() * dz * dy * dx / 1000),
                seg_seconds=round(t1 - t0, 1))
    nods, meta["nodules"] = [], []
    if xml_paths:
        for k, nod in enumerate(consensus_nodules(xml_paths, ct["locate"], hu.shape[1:], hu.shape[0], spacing)):
            da, bg = nodule_delta_a(nod, rho, lung, bone, spacing)
            fr = to_frame(da, box, edge=False)
            ys, xs = np.nonzero(fr > 0)
            if len(ys) == 0:
                continue
            y0, y1, x0_, x1_ = max(0, ys.min() - 2), min(FRAME, ys.max() + 3), max(0, xs.min() - 2), min(FRAME, xs.max() + 3)
            out[f"nod{len(nods)}"] = fr[y0:y1, x0_:x1_].copy()
            nods.append(1)
            meta["nodules"].append(dict(k=len(nods) - 1, y0=int(y0), x0=int(x0_), n_readers=nod["n_readers"],
                                        malignancy=nod["malignancy"], diam_mm=nod["diam_mm"], vol_mm3=nod["vol_mm3"],
                                        bg_density=bg, peak_da=float(fr.max()),
                                        overlap_bone=float(np.mean(out["a_bone"][ys, xs] > 0.25 * np.percentile(out["a_bone"], 99)))))
    meta["n_nodules"] = len(nods)
    meta["seconds"] = round(time.time() - t0, 1)
    return out, meta


def save(out_dir, uid, arrays, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{uid}.npz.partial"  # a half-written file must never look finished
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    (out_dir / f"{uid}.json").write_text(json.dumps(meta))
    tmp.replace(out_dir / f"{uid}.npz")

