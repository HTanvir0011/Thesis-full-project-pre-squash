import numpy as np

from s4.uq_conformal import apply_bone, calibrate_bone, cq, crc_fnr, fnr, load_runs, mondrian_sets, risk_coverage


def test_conformal_quantile_rank():
    assert cq(np.arange(1, 10), 0.1) == 9  # n = 9: ceil(10 * 0.9) = 9th smallest
    assert cq(np.arange(1, 9), 0.1) == float("inf")  # n = 8 cannot reach 0.9


def test_mondrian_covers_each_class():
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.integers(0, 3, n)
    p = rng.dirichlet([1, 1, 1], n) + 0.8 * np.eye(3)[y] * rng.random((n, 1))
    s = mondrian_sets(p[:3000], y[:3000], p[3000:], 0.1)
    yt = y[3000:]
    for k in range(3):
        assert s[yt == k, k].mean() >= 0.87


def test_crc_controls_fnr():
    rng = np.random.default_rng(1)
    y = (rng.random((8000, 6)) < 0.2).astype(float)
    p = np.clip(0.3 * y + 0.7 * rng.random(y.shape), 0, 1)
    lam = crc_fnr(p[:4000], y[:4000], 0.1)
    assert fnr(p[4000:], y[4000:], lam) <= 0.12


def test_bone_recalibration_reaches_nominal():
    rng = np.random.default_rng(2)
    items = []
    for _ in range(60):
        m = rng.random((32, 32))
        s = 0.05 + 0.05 * rng.random((32, 32))
        items.append((m, s, m + 2.0 * s * rng.standard_normal((32, 32))))  # errors 2x the spread: uncalibrated
    for mondrian in (False, True):
        c = calibrate_bone(items[:30], mondrian=mondrian)
        r = apply_bone(c, items[30:])
        assert 0.87 <= r["pooled"]["pixel_coverage"] <= 0.93
        assert r["image"]["images_reaching_nominal"] >= 0.8


def test_risk_coverage_falls_with_informative_confidence():
    rng = np.random.default_rng(3)
    loss = rng.random(1000)
    rc, aurc = risk_coverage(-loss, loss)
    assert rc[0][1] < rc[-1][1] and aurc < loss.mean()


def test_load_runs_groups_seeds_and_folds(tmp_path):
    for v in ("orig-soft", "soft_qure"):
        for s in (0, 1):
            d = tmp_path / f"bdcxr_cls_{v}_s{s}_f2"
            d.mkdir()
            np.save(d / "test_pred.npy", np.full((4, 3), s, np.float16))
            np.save(d / "test_idx.npy", np.arange(4))
            np.save(d / "test_labels.npy", np.eye(3)[[0, 1, 2, 0]])
    runs = load_runs(tmp_path, "bdcxr_cls_")
    assert set(runs) == {("orig-soft", "f2"), ("soft_qure", "f2")}
    assert np.allclose(runs[("soft_qure", "f2")][0], 0.5)
