"""Paired statistics used for every comparison: cluster bootstrap CI, Wilcoxon, effect size, Holm correction, non-inferiority."""
import numpy as np
from scipy import stats


def cluster_bootstrap_ci(diff, clusters=None, n=10000, alpha=0.05, seed=0):
    """Percentile CI of mean(diff), resampling whole clusters (e.g. patients) when given. -> (mean, lo, hi)."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(diff), (n, len(diff)))
        means = diff[idx].mean(1)
    else:
        labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
        sums = np.bincount(inv, weights=diff, minlength=len(labels))
        cnt = np.bincount(inv, minlength=len(labels)).astype(float)
        pick = rng.integers(0, len(labels), (n, len(labels)))
        means = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(diff.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_tests(a, b):
    """a, b: per-case scores of two methods on the same cases. -> dict(mean_diff, wilcoxon_p, cliffs_delta, dz)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return dict(mean_diff=float(d.mean()), wilcoxon_p=p, cliffs_delta=float((np.sum(d > 0) - np.sum(d < 0)) / len(d)),
                dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def non_inferior(diff, margin, clusters=None, n=10000, seed=0):
    """H1: mean(diff) > -margin, where diff = new - reference (higher is better). True when the lower 95% CI bound clears it."""
    m, lo, hi = cluster_bootstrap_ci(diff, clusters, n, alpha=0.10, seed=seed)  # one-sided 5%
    return dict(mean=m, lower=lo, margin=margin, non_inferior=bool(lo > -margin))
