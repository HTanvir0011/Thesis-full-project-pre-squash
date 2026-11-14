"""Paired DeLong test for two correlated AUCs (Sun & Xu 2014 fast algorithm)."""
import numpy as np
from scipy import stats


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds, m):
    """preds [K, N] with the m positives first. -> aucs [K], covariance [K,K]."""
    n = preds.shape[1] - m
    k = preds.shape[0]
    tx, ty, tz = (np.empty((k, c)) for c in (m, n, m + n))
    for r in range(k):
        tx[r], ty[r], tz[r] = _midrank(preds[r, :m]), _midrank(preds[r, m:]), _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n


def delong_paired(y, p1, p2):
    """-> dict(auc1, auc2, diff, z, p). y in {0,1}; NaN labels must be removed by the caller."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(~y, kind="stable")
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, int(y.sum()))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), diff=float(aucs[0] - aucs[1]), z=float(z), p=float(2 * stats.norm.sf(abs(z))))
