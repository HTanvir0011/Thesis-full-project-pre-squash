"""Detection metrics: FROC sensitivity at fixed false-positive rates (CPM = their mean) and AP@IoU, with per-image matching."""
import numpy as np

FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def iou_matrix(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match(preds, gts, iou_thr, per_class=False):
    """preds: list per image of (boxes[N,4], scores[N], labels[N]); gts: list per image of (boxes[M,4], labels[M]).
    -> (scores, is_tp, hit [n_gt bool per image list]) over all images; greedy by score, each GT matched once."""
    scores, tps, hits = [], [], []
    for (pb, ps, pl), (gb, gl) in zip(preds, gts):
        order = np.argsort(-np.asarray(ps))
        hit = np.zeros(len(gb), bool)
        iou = iou_matrix(pb, gb) if len(pb) and len(gb) else np.zeros((len(pb), len(gb)))
        for i in order:
            ok = (iou[i] >= iou_thr) & ~hit
            if per_class:
                ok &= np.asarray(gl) == pl[i]
            j = int(np.argmax(np.where(ok, iou[i], -1))) if ok.any() else -1
            scores.append(ps[i])
            tps.append(j >= 0)
            if j >= 0:
                hit[j] = True
        hits.append(hit)
    return np.array(scores), np.array(tps, bool), hits


def curve(s, tp):
    """-> (distinct score thresholds high to low, cumulative TP, cumulative FP at each). Tied scores enter together, so the result does not depend
    on how a sort orders ties (np.argsort's default sort does that differently across machines; DEVIATIONS.md 2026-11-03)."""
    o = np.argsort(-s, kind="stable")
    so = s[o]
    last = np.r_[so[1:] != so[:-1], True] if len(s) else np.zeros(0, bool)
    return so[last], np.cumsum(tp[o])[last], np.cumsum(~tp[o])[last]


def froc(preds, gts, iou_thr=0.3, rates=FP_RATES):
    """-> dict(sens={rate: sensitivity}, cpm). Class-agnostic."""
    s, tp, hits = match(preds, gts, iou_thr)
    n_gt, n_img = sum(len(g[0]) for g in gts), len(gts)
    _, tpc, fpc = curve(s, tp)
    fp_per_img = fpc / n_img
    sens = {r: float(tpc[fp_per_img <= r][-1] / n_gt) if (fp_per_img <= r).any() else 0.0 for r in rates}
    return dict(sens=sens, cpm=float(np.mean(list(sens.values()))))


def average_precision(preds, gts, iou_thr=0.5, n_classes=1):
    """Mean over classes present in the ground truth of the area under the precision-recall curve (all-point interpolation)."""
    aps = []
    for c in range(n_classes):
        p_c = [(pb[pl == c], ps[pl == c], pl[pl == c]) for pb, ps, pl in preds]
        g_c = [(gb[np.asarray(gl) == c], np.asarray(gl)[np.asarray(gl) == c]) for gb, gl in gts]
        n_gt = sum(len(g[0]) for g in g_c)
        if n_gt == 0:
            continue
        s, tp, _ = match(p_c, g_c, iou_thr)
        _, tpc, fpc = curve(s, tp)
        rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1)
        rec, prec = np.r_[0, rec, 1], np.r_[0, prec, 0]
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        aps.append(float(np.sum((rec[1:] - rec[:-1]) * prec[1:])))
    return float(np.mean(aps)) if aps else float("nan")


def recall_at_fp(preds, gts, fp_rate=2.0, iou_thr=0.3):
    """Per-GT hit flags at the score threshold giving `fp_rate` false positives per image (for subgroup analysis)."""
    s, tp, hits = match(preds, gts, iou_thr)
    th, _, fpc = curve(s, tp)
    keep = fpc / len(gts) <= fp_rate
    thr = th[keep][-1] if keep.any() else np.inf  # lowest threshold within the FP budget; every prediction tied at it is kept
    cut = [(pb[ps >= thr], ps[ps >= thr], pl[ps >= thr]) for pb, ps, pl in preds]
    _, _, hits = match(cut, gts, iou_thr)
    return np.concatenate(hits) if hits else np.array([], bool)
