import numpy as np

from s4 import stats


def test_holm_matches_hand_computation():
    assert stats.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_bootstrap_ci_covers_true_mean_and_respects_clusters():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 400)
    m, lo, hi = stats.cluster_bootstrap_ci(d, n=2000)
    assert lo < 0.5 < hi and abs(m - d.mean()) < 1e-12
    cl = np.repeat(np.arange(40), 10)  # identical values inside a cluster -> clustered CI must be wider
    d2 = np.repeat(rng.normal(0, 1, 40), 10)
    w_plain = np.subtract(*stats.cluster_bootstrap_ci(d2, n=2000)[2:0:-1])
    w_clus = np.subtract(*stats.cluster_bootstrap_ci(d2, cl, n=2000)[2:0:-1])
    assert w_clus > w_plain


def test_paired_tests_and_non_inferiority():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.3, 80)
    r = stats.paired_tests(a + 0.2, a)
    assert r["wilcoxon_p"] < 1e-6 and r["cliffs_delta"] == 1.0
    assert stats.non_inferior(np.full(80, 0.01) + rng.normal(0, 0.01, 80), margin=0.05)["non_inferior"]
    assert not stats.non_inferior(np.full(80, -0.2) + rng.normal(0, 0.01, 80), margin=0.05)["non_inferior"]
