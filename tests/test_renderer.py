"""Synthetic-phantom tests: every check has a closed-form answer, so none depends on real data."""
import numpy as np
import pytest

from s4 import seg, xray
from s4.ct import SkipCT
from s4.render import nodule_delta_a, project, render_ct

SP = (2.5, 1.5, 1.5)  # dz, dy, dx (mm)


def phantom(nz=120, ny=160, nx=200):
    z, y, x = np.mgrid[:nz, :ny, :nx].astype(np.float32)
    hu = np.full((nz, ny, nx), -1000, np.int16)
    body = ((y - 80) / 60) ** 2 + ((x - 100) / 85) ** 2 < 1
    hu[body] = 40
    for cx in (55, 145):
        lung = (((y - 75) / 40) ** 2 + ((x - cx) / 28) ** 2 < 1) & (z > 15) & (z < 105)
        hu[lung] = -850
    spine = ((y - 125) ** 2 + (x - 100) ** 2 < 10 ** 2)
    hu[spine & body] = 600
    ring = (np.abs(((y - 80) / 56) ** 2 + ((x - 100) / 81) ** 2 - 1) < 0.06) & body
    for zr in (30, 50, 70, 90):
        hu[ring & (np.abs(z - zr) < 2)] = 700
    return hu


@pytest.fixture(scope="module")
def vol():
    return phantom()


def test_bone_mask_contains_reference_voxels(vol):
    body = seg.body_mask(vol)
    bone = seg.bone_mask(vol, SP)
    ref = vol > 300
    assert (bone & ref).sum() / ref.sum() > 0.99
    assert not (bone & (vol < -500)).any()


def test_lung_volume_close_to_analytic(vol):
    body = seg.body_mask(vol)
    lung = seg.lung_mask(vol, body, seg.bone_mask(vol, SP))
    expected = 2 * np.pi * 40 * 28 * (105 - 15 - 1)  # two ellipsoid-free cylinders (voxels)
    assert abs(lung.sum() - expected) / expected < 0.15


def test_projection_matches_chord_length(vol):
    spine = np.where(vol == 600, 1.6, 0.0).astype(np.float32)
    a = project(spine, SP[1])
    expected = 2 * 10 * SP[1] / 10 * 1.6  # chord 2r through the cylinder axis, 1.6 g/cm^3
    assert abs(a[60, 100] - expected) / expected < 0.08


def test_mono_additivity_is_exact_and_poly_gap_is_measured():
    a_s, a_b = 20.0, 3.0
    for kvp in (80, 100, 120):
        ps = xray.to_pixel(np.float32(a_s), kvp, "soft_tissue", mono_keV=60)
        pb = xray.to_pixel(np.float32(a_b), kvp, "bone_cortical", mono_keV=60)
        e = np.array(xray.tables()["energy_keV"])
        mu = {m: np.array(xray.tables()["mu_over_rho_cm2_per_g"][m])[np.argmin(abs(e - 60))] for m in xray.MATERIALS}
        assert abs((ps + pb) - (mu["soft_tissue"] * a_s + mu["bone_cortical"] * a_b)) < 1e-3
    en = np.array(xray.tables()["energy_keV"])
    t = xray.tables()["mu_over_rho_cm2_per_g"]
    w = xray.spectrum(100) * en
    full = -np.log((w * np.exp(-np.array(t["soft_tissue"]) * a_s - np.array(t["bone_cortical"]) * a_b)).sum() / w.sum())
    parts = float(xray.to_pixel(np.float32(a_s), 100, "soft_tissue") + xray.to_pixel(np.float32(a_b), 100, "bone_cortical"))
    gap = abs(full - parts) / full
    print(f"polychromatic additivity gap at 100 kVp: {gap:.3f}")
    assert gap < 0.15  # pre-set gate; the measured value is reported in the thesis


def test_pixel_monotone_and_zero_at_zero():
    for m in xray.MATERIALS:
        a, p = xray.projection_lut(100, m)
        assert p[0] == pytest.approx(0, abs=1e-6) and (np.diff(p) > 0).all()


def test_nodule_peak_signal(vol):
    z, y, x = np.mgrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    sph = (z - 60) ** 2 * (SP[0] / SP[1]) ** 2 + (y - 70) ** 2 + (x - 55) ** 2 < 6 ** 2
    v = vol.copy()
    v[sph] = 30
    body = seg.body_mask(v)
    bone = seg.bone_mask(v, SP)
    lung = seg.lung_mask(v, body, bone) | sph
    nod = dict(mask={int(zz): sph[zz] for zz in np.nonzero(sph.any(axis=(1, 2)))[0]})
    da, bg = nodule_delta_a(nod, seg.rho_from_hu(v), lung, bone, SP)
    expected = (1.03 - 0.15) * 2 * 6 * SP[1] / 10
    assert bg == pytest.approx(0.15, abs=0.03)
    assert abs(da.max() - expected) / expected < 0.12


def test_render_outputs_and_skip(vol):
    ct = dict(hu=vol, spacing=SP, sop_to_idx={}, meta={})
    out, meta = render_ct(ct, [])
    for k in ("a_bone", "a_nl", "a_lung"):
        assert out[k].shape == (512, 512) and np.isfinite(out[k]).all() and out[k].min() >= 0
    assert out["a_bone"].max() > 1.0 and not meta["lung_truncated"]
    with pytest.raises(SkipCT):
        render_ct(dict(hu=np.full((90, 64, 64), -1000, np.int16), spacing=SP, sop_to_idx={}, meta={}), [])
