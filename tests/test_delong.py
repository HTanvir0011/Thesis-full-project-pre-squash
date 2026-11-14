import numpy as np
from sklearn.metrics import roc_auc_score

from s4.delong import delong_paired


def test_auc_matches_sklearn_and_detects_a_real_difference():
    rng = np.random.default_rng(0)
    y = rng.random(600) < 0.3
    s = rng.normal(size=600)
    p1 = s + 1.5 * y + rng.normal(0, 0.3, 600)  # strong
    p2 = s + 0.5 * y + rng.normal(0, 0.3, 600)  # weaker, correlated through s
    r = delong_paired(y, p1, p2)
    assert abs(r["auc1"] - roc_auc_score(y, p1)) < 1e-9 and abs(r["auc2"] - roc_auc_score(y, p2)) < 1e-9
    assert r["diff"] > 0 and r["p"] < 0.01


def test_identical_scores_give_p_one():
    rng = np.random.default_rng(1)
    y = rng.random(200) < 0.4
    p = rng.normal(size=200) + y
    assert delong_paired(y, p, p)["p"] == 1.0


def test_null_p_values_are_roughly_uniform():
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(200):
        y = rng.random(150) < 0.4
        s = rng.normal(size=150)
        ps.append(delong_paired(y, s + y + rng.normal(0, 1, 150), s + y + rng.normal(0, 1, 150))["p"])
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10  # nominal 5%
