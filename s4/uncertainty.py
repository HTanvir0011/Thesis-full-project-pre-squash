"""Deep-ensemble disagreement and split-conformal intervals for the bone image (soft = x - bone shares the same interval)."""
import numpy as np


def ensemble(preds):
    """preds [K,H,W] -> (mean, std)."""
    p = np.asarray(preds, np.float32)
    return p.mean(0), p.std(0, ddof=1) if len(p) > 1 else np.zeros_like(p[0])


def conformal_scale(mean, std, truth, alpha=0.1, eps=1e-3, sample=200_000, seed=0):
    """Split-conformal quantile q of |truth - mean| / (std + eps) pooled over calibration pixels, so that
    [mean - q (std+eps), mean + q (std+eps)] covers the truth with probability >= 1 - alpha (exchangeable pixels)."""
    s = (np.abs(truth - mean) / (std + eps)).ravel()
    if len(s) > sample:
        s = np.random.default_rng(seed).choice(s, sample, replace=False)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))


def coverage(mean, std, truth, q, eps=1e-3):
    """-> (empirical coverage, mean interval width)."""
    half = q * (std + eps)
    return float((np.abs(truth - mean) <= half).mean()), float(2 * half.mean())


def sparsification_curve(err, unc, fractions=np.linspace(0, 0.95, 20)):
    """Mean error of the pixels kept after removing the most uncertain fraction f (lower and falling = uncertainty is informative)."""
    e, u = np.asarray(err).ravel(), np.asarray(unc).ravel()
    order = np.argsort(-u)
    e = e[order]
    n = len(e)
    return [(float(f), float(e[int(f * n):].mean())) for f in fractions]


def ause(err, unc):
    """Area under the sparsification error curve minus that of the oracle (error-sorted) ordering; 0 is perfect."""
    f = np.linspace(0, 0.95, 20)
    a = np.array([v for _, v in sparsification_curve(err, unc, f)])
    o = np.array([v for _, v in sparsification_curve(err, err, f)])
    return float(np.trapezoid(a - o, f))
