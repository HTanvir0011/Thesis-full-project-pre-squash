"""Shift-aware uncertainty (SAP_v3.md H9 and the classification-uncertainty section).

Bone image, real-pair recalibration (post hoc; exploratory):
  python -m s4.uq_conformal bone --ckpts a.pt b.pt c.pt --out results/v3/uq_bone.json [--szch data/szch] [--limit N]
  Calibrate on the 24 SZCH development pairs, test on the 51 held-out pairs. Score r = |b - b_hat| / (spread + eps) per pixel.
  'pooled': quantile of the calibration pixels (marginal pixel coverage, images exchangeable). 'image': per-image (1-alpha) pixel quantiles,
  then the conformal quantile over the calibration images at level 1-delta (with probability >= 1-delta a new image has >= 1-alpha of its
  pixels covered). Mondrian strata: predicted bone overlap (ensemble bone > 25 % of its p99, the prep.py overlap rule) x spread tercile
  (tercile edges from the calibration images); the stratum is known at test time without the truth.
Classification, from saved predictions (no GPU):
  python -m s4.uq_conformal cls --bdcxr results/bdcxr/down --nih results/final/down/nih --out results/v3/uq_cls.json
  BDCXR-3257: Mondrian (class-conditional) conformal sets, cross-conformal over the five folds (calibrate on the out-of-fold predictions of
  the other folds). NIH: conformal risk control of the per-image false-negative rate (Angelopoulos et al. 2024) on a calibration half of
  the test split (patient-disjoint), applied to the other half (shortcut: s4/cls.py saved test predictions only; the v3 runs save
  validation predictions, see cls3). Selective classification: risk-coverage and AURC with 1 - ensemble entropy.
Classification v3 (s4/cls_hybrid.py runs <root>/<ds>/<ds>_hyb_<config>@<arch>_s<seed>[_f<k>], seed means; no GPU):
  python -m s4.uq_conformal cls3 --arch densenet121 [--root results/v3/down] [--spread_nih results/v3/spread_nih.csv]
      [--spread_bdcxr results/bdcxr/spread_bdcxr.csv] [--nih_labels NIH_PREP/labels.npy] [--bdcxr_ids BDCXR_PREP/ids.json]
      --out results/v3/uq_cls_densenet121.json
  Per configuration: NIH conformal risk control of the per-image false-negative rate, calibrated on the validation predictions (labels
  --nih_labels at val_idx), tested on the test split; BDCXR-3257 cross-conformal Mondrian sets as in 'cls' (a configuration with fewer
  than two folds is listed under 'missing'). Selective classification (risk: NIH per-image Brier score, BDCXR top-1 error): score A =
  1 - entropy; score B (with a spread CSV) = -(z(entropy) + z(lung-field suppressor spread)), each z-scored over the images of the data
  set that have a spread, so both terms weigh equally; its AURC is reported with score A's on the same images. Spread join: NIH on the
  CSV 'index' (prep index = test_idx), BDCXR on the image name (--bdcxr_ids, the prep's ids.json, read only with --spread_bdcxr;
  unavailable without it).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import uncertainty as U


def cq(scores, alpha):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (inf when n is too small for that level)."""
    s = np.sort(np.asarray(scores, np.float64).ravel())
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def strata(mean, std, edges):
    """Stratum per pixel: 3 * bone_overlap + spread tercile (0..5)."""
    bone = mean > 0.25 * np.percentile(mean, 99)
    return 3 * bone.astype(np.int64) + np.digitize(std, edges)


def calibrate_bone(cal, alpha=0.1, delta=0.1, eps=1e-3, mondrian=True, sample=20_000, seed=0):
    """cal: list of (mean, std, truth) -> dict with spread edges and, per stratum, the pooled and per-image thresholds."""
    rng = np.random.default_rng(seed)
    std_all = np.concatenate([s.ravel()[::17] for _, s, _ in cal])
    edges = np.quantile(std_all, [1 / 3, 2 / 3]) if mondrian else np.array([np.inf, np.inf])
    pooled, per_img = {}, {}
    for k in range(6):
        pix, imgq = [], []
        for m, s, t in cal:
            st = strata(m, s, edges) if mondrian else np.zeros(m.shape, np.int64)
            r = (np.abs(t - m) / (s + eps))[st == k]
            if len(r):
                pix.append(r if len(r) <= sample else rng.choice(r, sample, replace=False))
                imgq.append(np.quantile(r, 1 - alpha))
        if pix:
            pooled[k] = cq(np.concatenate(pix), alpha)
            per_img[k] = cq(imgq, delta)
    return dict(edges=edges.tolist(), pooled=pooled, image=per_img, alpha=alpha, delta=delta, eps=eps, mondrian=mondrian)


def apply_bone(cal, items):
    """-> per-image coverage and mean width for both thresholds, pooled pixel coverage."""
    out = {}
    for kind in ("pooled", "image"):
        cov, wid, hit, tot = [], [], 0, 0
        for m, s, t in items:
            st = strata(m, s, np.array(cal["edges"])) if cal["mondrian"] else np.zeros(m.shape, np.int64)
            q = np.vectorize(lambda k: cal[kind].get(k, np.inf))(st)
            half = q * (s + cal["eps"])
            ok = np.abs(t - m) <= half
            cov.append(float(ok.mean()))
            wid.append(float(2 * half[np.isfinite(half)].mean()))
            hit, tot = hit + int(ok.sum()), tot + ok.size
        out[kind] = dict(pixel_coverage=hit / tot, image_coverage_median=float(np.median(cov)), image_coverage_p10=float(np.percentile(cov, 10)),
                         images_reaching_nominal=float(np.mean(np.array(cov) >= 1 - cal["alpha"])), width_mean=float(np.mean(wid)), per_image=cov)
    return out


def szch_items(ckpts, szch, limit=0):
    """Ensemble mean/spread and the least-squares pseudo-truth bone per SZCH pair (as s4/eval_uncertainty.py) -> {name: (mean, std, truth)}."""
    import cv2
    import torch

    from .eval_real import load_pairs
    from .eval_synth import load_model, predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(c, dev)[0] for c in ckpts]
    out = {}
    for name, x, gt in load_pairs(szch)[: limit or None]:
        x5 = cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
        x5 = (x5 - x5.min()) / (x5.max() - x5.min() + 1e-10)
        xt = torch.from_numpy(x5)[None, None].to(dev)
        m, s = U.ensemble(np.stack([predict(mo, xt)[0, 0].cpu().numpy() for mo in models]))
        g = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_AREA)
        (p0, p1), *_ = np.linalg.lstsq(np.stack([g.ravel(), np.ones(g.size, np.float32)], 1), x5.ravel(), rcond=None)
        out[name] = (m, s, x5 - (p0 * g + p1))
    return out


def mondrian_sets(p_cal, y_cal, p_test, alpha=0.1):
    """Class-conditional conformal sets for single-label data. p rows are normalised to sum 1; score = 1 - p[true class]."""
    pc, pt = p_cal / p_cal.sum(1, keepdims=True), p_test / p_test.sum(1, keepdims=True)
    q = np.array([cq(1 - pc[y_cal == k, k], alpha) for k in range(pc.shape[1])])
    return (1 - pt) <= q[None]


def crc_fnr(p_cal, y_cal, alpha=0.1, grid=np.linspace(0, 1, 1001)):
    """Conformal risk control of the per-image false-negative rate (images with >= 1 positive): set = {c : p_c >= 1 - lam};
    the smallest lam with (n R_n(lam) + 1) / (n + 1) <= alpha (loss bounded by 1, non-increasing in lam)."""
    pos = np.nansum(y_cal, 1) > 0
    p, y = p_cal[pos], np.nan_to_num(y_cal[pos])
    n = len(p)
    for lam in grid:
        fnr = 1 - ((p >= 1 - lam) * y).sum(1) / y.sum(1)
        if (n * fnr.mean() + 1) / (n + 1) <= alpha:
            return float(lam)
    return 1.0


def fnr(p, y, lam):
    pos = np.nansum(y, 1) > 0
    yy = np.nan_to_num(y[pos])
    return float((1 - ((p[pos] >= 1 - lam) * yy).sum(1) / yy.sum(1)).mean())


def risk_coverage(conf, loss, grid=np.linspace(0.1, 1, 19)):
    """Keep the most confident fraction c; -> [(c, mean loss of kept)], AURC (mean loss over all coverages, by rank)."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    l = np.asarray(loss, np.float64)[order]
    cum = np.cumsum(l) / np.arange(1, len(l) + 1)
    return [(float(c), float(cum[max(1, int(round(c * len(l)))) - 1])) for c in grid], float(cum.mean())


def entropy(p, multilabel):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if multilabel:  # mean binary entropy over classes
        return -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(1)
    q = p / p.sum(1, keepdims=True)
    return -(q * np.log(q)).sum(1)


def load_runs(root, prefix, val=False):
    """{(variant, fold or None): (pred mean over seeds [N,C], labels, idx, patient)} from s4/cls.py result dirs; val=True (s4/cls_hybrid.py
    runs) appends the number of seeds, the validation prediction mean and its prep indices."""
    groups = {}
    for d in sorted(Path(root).glob(prefix + "*")):
        if not (d / "test_pred.npy").exists():
            continue
        parts = d.name[len(prefix):].split("_")
        fold = parts[-1] if parts[-1].startswith("f") else None
        variant = "_".join(p for p in parts if not (p.startswith("s") and p[1:].isdigit()) and p != fold)
        groups.setdefault((variant, fold), []).append(d)
    out = {}
    for key, ds in groups.items():
        idx = np.load(ds[0] / "test_idx.npy")
        assert all(np.array_equal(np.load(d / "test_idx.npy"), idx) for d in ds), key
        pat = np.load(ds[0] / "test_patient.npy", allow_pickle=True) if (ds[0] / "test_patient.npy").exists() else idx
        out[key] = (np.mean([np.load(d / "test_pred.npy").astype(np.float32) for d in ds], 0), np.load(ds[0] / "test_labels.npy"), idx, pat)
        if val:
            vidx = np.load(ds[0] / "val_idx.npy")
            assert all(np.array_equal(np.load(d / "val_idx.npy"), vidx) for d in ds), key
            out[key] += (len(ds), np.mean([np.load(d / "val_pred.npy").astype(np.float32) for d in ds], 0), vidx)
    return out


def cross_mondrian(runs, alpha):
    """runs: {fold: (pred [N,K], one-hot labels, idx, ...)} -> (metrics of the pooled out-of-fold Mondrian sets, each fold calibrated on
    the other folds' predictions; pooled prep indices, ensemble entropy, top-1 errors)."""
    folds = sorted(runs)
    sets, ys, P, I = [], [], [], []
    for f in folds:
        p, y, idx = runs[f][:3]
        pc = np.concatenate([runs[g][0] for g in folds if g != f])
        yc = np.concatenate([runs[g][1] for g in folds if g != f]).argmax(1)
        sets.append(mondrian_sets(pc, yc, p, alpha))
        ys.append(y.argmax(1))
        P.append(p)
        I.append(idx)
    S, Yc, P, I = np.concatenate(sets), np.concatenate(ys), np.concatenate(P), np.concatenate(I)
    err = (P.argmax(1) != Yc).astype(float)
    ent = entropy(P, False)
    rc, aurc = risk_coverage(-ent, err)
    return dict(n=len(Yc), coverage=float(S[np.arange(len(Yc)), Yc].mean()),
                coverage_per_class=[float(S[Yc == k, k].mean()) for k in range(S.shape[1])],
                mean_set_size=float(S.sum(1).mean()), singletons=float((S.sum(1) == 1).mean()),
                top1_error=float(err.mean()), risk_coverage=rc, aurc=aurc), I, ent, err


def cls_main(a):
    res = {}
    if a.bdcxr:
        runs = load_runs(a.bdcxr, "bdcxr_cls_")
        for v in sorted({k[0] for k in runs}):
            res[f"bdcxr_{v}"] = cross_mondrian({f: r for (w, f), r in runs.items() if w == v}, a.alpha)[0]
    if a.nih:
        for (v, _), (p, y, idx, pat) in sorted(load_runs(a.nih, "nih_cls_").items()):
            up = np.unique(pat)
            cal_p = set(np.random.default_rng(0).permutation(up)[: len(up) // 2])  # patient-disjoint halves, fixed seed
            c = np.array([q in cal_p for q in pat])
            lam = crc_fnr(p[c], y[c], a.alpha)
            brier = np.nanmean((p - y) ** 2, 1)
            rc, aurc = risk_coverage(-entropy(p, True), brier)
            res[f"nih_{v}"] = dict(n_cal=int(c.sum()), n_test=int((~c).sum()), lam=lam, fnr_test=fnr(p[~c], y[~c], lam),
                                   mean_set_size=float((p[~c] >= 1 - lam).sum(1).mean()), risk_coverage_brier=rc, aurc_brier=aurc)
    return res


CONFIGS = ("orig", "soft", "early", "hybrid", "ctrl")  # s4/cls_hybrid.py configurations (SAP_v3)


def read_spread(path, key):
    """{row[key]: lung-field suppressor spread} from an s4/spread.py CSV (key 'index' = NIH prep index, 'image' = BDCXR image name)."""
    with open(path, newline="") as f:
        return {r[key]: float(r["spread_lung"]) for r in csv.DictReader(f)}


def score_b(ent, loss, spread):
    """Risk-coverage and AURC with confidence -(z(entropy) + z(spread)) on the images with a spread (NaN: none); score A's AURC there."""
    ok = np.isfinite(spread)
    if not ok.any():
        return "unavailable: no evaluated image has a spread in the CSV"
    z = [(v - v.mean()) / (v.std() or 1.0) for v in (ent[ok], spread[ok])]
    rc, aurc = risk_coverage(-(z[0] + z[1]), loss[ok])
    return dict(n=int(ok.sum()), risk_coverage=rc, aurc=aurc, aurc_score_a_same_images=risk_coverage(-ent[ok], loss[ok])[1])


def cls3_main(a):
    res = dict(arch=a.arch, alpha=a.alpha, missing=[], score_b_definition="confidence = -(z(ensemble entropy) + z(lung-field suppressor "
               "spread)), each z-scored over the evaluated images of the data set that have a spread")
    ids = json.loads(Path(a.bdcxr_ids).read_text()) if a.spread_bdcxr and Path(a.bdcxr_ids).exists() else None  # only for the spread join
    spread = dict(nih=a.spread_nih and read_spread(a.spread_nih, "index"), bdcxr=a.spread_bdcxr and read_spread(a.spread_bdcxr, "image"))
    for ds in ("nih", "bdcxr"):
        for cfg in CONFIGS:
            runs = {f: r for (v, f), r in load_runs(Path(a.root, ds), f"{ds}_hyb_{cfg}@{a.arch}_", val=True).items() if v == ""}
            if not runs or ds == "bdcxr" and len(runs) < 2:  # cross-conformal: each fold is calibrated on the other folds
                res["missing"].append(f"{ds}_{cfg}" + (f" (only fold {next(iter(runs))})" if runs else ""))
                continue
            if ds == "nih":
                p, y, idx, _, n, vp, vidx = runs[None]
                Y = np.load(a.nih_labels)
                assert np.array_equal(Y[idx], y, equal_nan=True), f"{a.nih_labels} does not match the runs' test_labels.npy (another prep?)"
                lam = crc_fnr(vp, Y[vidx], a.alpha)
                ent, loss = entropy(p, True), np.nanmean((p - y) ** 2, 1)
                rc, aurc = risk_coverage(-ent, loss)
                r = dict(n_runs=n, n_cal=len(vidx), n_test=len(idx), lam=lam, fnr_test=fnr(p, y, lam),
                         mean_set_size=float((p >= 1 - lam).sum(1).mean()), risk_coverage_brier=rc, aurc_brier=aurc)
                keys = [str(i) for i in idx]
            else:
                r, idx, ent, loss = cross_mondrian(runs, a.alpha)
                r.update(n_runs=sum(v[4] for v in runs.values()), folds=sorted(runs))
                keys = None
                if ids is not None:
                    # the image folder is the class, so a wrong ids.json (another prep) cannot join silently
                    pairs = {(ids[i].split("/")[0], c) for v in runs.values() for i, c in zip(v[2], v[1].argmax(1))}
                    assert len(pairs) == len({d for d, _ in pairs}), f"{a.bdcxr_ids} does not match the runs' labels (another prep?)"
                    keys = [ids[i] for i in idx]
            if spread[ds] is not None:
                r["score_b"] = (score_b(ent, loss, np.array([spread[ds].get(k, np.nan) for k in keys])) if keys is not None
                                else f"unavailable: no {a.bdcxr_ids} (BDCXR image names) to join the spread CSV")
            res[f"{ds}_{cfg}"] = r
    return res


def bone_main(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from szch_split import is_dev
    items = szch_items(a.ckpts, a.szch, a.limit)
    cal = [v for n, v in items.items() if is_dev(n)]
    test = [v for n, v in items.items() if not is_dev(n)]
    res = dict(n_cal=len(cal), n_test=len(test), note="post hoc: the 51 held-out pairs were already used in v2 (SAP_v3.md H9)")
    for name, mondrian in (("global", False), ("mondrian", True)):
        c = calibrate_bone(cal, a.alpha, a.delta, mondrian=mondrian)
        res[name] = dict(calibration={k: v for k, v in c.items() if k in ("edges", "pooled", "image")}, test=apply_bone(c, test))
    res["reliability"] = []  # nominal vs held-out pixel coverage, Mondrian pooled thresholds recalibrated at each level
    for nominal in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        c = calibrate_bone(cal, 1 - nominal, a.delta, mondrian=True)
        res["reliability"].append((nominal, apply_bone(c, test)["pooled"]["pixel_coverage"]))
    err = np.concatenate([np.abs(m - t).ravel()[::17] for m, s, t in test])
    unc = np.concatenate([s.ravel()[::17] for m, s, t in test])
    res["held_sparsification"] = U.sparsification_curve(err, unc)
    res["held_oracle"] = U.sparsification_curve(err, err)
    res["held_ause"] = U.ause(err, unc)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bone")
    b.add_argument("--ckpts", nargs="+", required=True)
    b.add_argument("--szch", default="data/szch")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--delta", type=float, default=0.1)
    c = sub.add_parser("cls")
    c.add_argument("--bdcxr", default=None)
    c.add_argument("--nih", default=None)
    c3 = sub.add_parser("cls3")
    c3.add_argument("--arch", required=True, help="encoder as in the run tag, e.g. densenet121, convnextv2_tiny, rad-dino")
    c3.add_argument("--root", default="results/v3/down")
    c3.add_argument("--nih_labels", default="outputs/prep_dl/ac_6-s4-prep-nih/nih/labels.npy")
    c3.add_argument("--bdcxr_ids", default="outputs/prep_dl/ac_8-s4-prep-bdcxr/bdcxr/ids.json")
    c3.add_argument("--spread_nih", default=None)
    c3.add_argument("--spread_bdcxr", default=None)
    for p in (b, c, c3):
        p.add_argument("--alpha", type=float, default=0.1)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = dict(bone=bone_main, cls=cls_main, cls3=cls3_main)[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}) for k, v in res.items()}, indent=1,
                     default=float)[:3000])


if __name__ == "__main__":
    main()
