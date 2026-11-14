import numpy as np

from s4 import uncertainty as U


def test_conformal_reaches_nominal_coverage_on_fresh_data():
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.01, 0.1, (64, 64))  # heteroscedastic noise the ensemble std tracks
    def draw():
        truth = rng.normal(0, 1, (64, 64))
        preds = truth + rng.normal(0, 1, (5, 64, 64)) * scale
        return U.ensemble(preds), truth
    (m, s), t = draw()
    q = U.conformal_scale(m, s, t, alpha=0.1)
    covs = []
    for _ in range(20):
        (m2, s2), t2 = draw()
        covs.append(U.coverage(m2, s2, t2, q)[0])
    assert abs(np.mean(covs) - 0.9) < 0.04


def test_informative_uncertainty_beats_random_in_ause():
    rng = np.random.default_rng(1)
    err = rng.exponential(1.0, 20000)
    good = err * rng.uniform(0.8, 1.2, 20000)
    bad = rng.exponential(1.0, 20000)
    assert U.ause(err, good) < U.ause(err, bad)
    assert U.ause(err, err) == 0.0
