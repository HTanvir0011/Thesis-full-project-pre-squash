"""v4 statistics (SAP_v4.md H11-H15 and the descriptive results) -> results/final_xm_stats.json.

python tools/xm_stats.py [--n 10000] [--root results/v4] [--out results/final_xm_stats.json]
Layout under --root (kernel outputs without model.pt; folder names from s4/runtag.py):
  down/nih/nih_hyb_orig@swinv2_tiny_window8_256[+xray|+xraykd]_s<k>  NIH fine-tuning from ImageNet (F_in), from P0 (F_p0), from P1 (F_p1)
  down/ct/ct_xm_ct_s<k>                                              CT teacher (s4/xm.py stage ct)
  down/qp/qp_xm_<xray|xraykd>_s<k>                                   projection models P0, P1 (stage xray)
  down/<nih|vin>/<ds>_xm_transfer-<xray|xraykd>_s<k>                 zero-shot transfer to real X-rays (stage transfer)
  xai/ct_xmx_<xray|xraykd>_s<k>/maps.npz                             cross-modal maps (s4/xm_xai.py), CT seed k with X-ray seed k
  halluc/nih_xmh_<fin|fp0|fp1|dn121>_s<k>/halluc.npz                 hallucination tests (s4/xm_halluc.py), one file per seed group
The v3 DenseNet-121 orig runs (results/v3/down/nih) are the CNN reference of H13 and of the hallucination tests. A model's prediction is the
mean of its seeds' probabilities. Bootstrap: patients (NIH), scans (H14); the contrasts of one data set share their resamples (paired).
p-values: two-sided percentile bootstrap with the +1 rule (tools/hybrid_stats.py). Holm: H11 with the two H12 contrasts. Parts whose runs
are missing are listed under 'missing' and skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from hybrid_stats import class_metric  # noqa: E402
from s4.prep import NIH  # noqa: E402
from s4.stats import holm  # noqa: E402
from s4.uq_conformal import load_runs  # noqa: E402
from s4.xm_halluc import LESION, PERTURB, crossed, thresholds  # noqa: E402
from s4.xm_labels import EXACT_NIH, SHARED, to_shared  # noqa: E402
from s4.xm_xai import pearson  # noqa: E402

SWIN = "orig@swinv2_tiny_window8_256"
NIH_MODELS = {"F_in": SWIN, "F_p0": SWIN + "+xray", "F_p1": SWIN + "+xraykd"}
HALLUC = {"F_in": "fin", "F_p0": "fp0", "F_p1": "fp1", "DN121": "dn121"}


def resamples(g, n, seed=0):
    """n cluster-bootstrap weight vectors over the rows (cluster of each row: g)."""
    labels, inv = np.unique(np.asarray(g), return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))[inv].astype(float)


def summary(d, bd):
    n = len(bd)
    return dict(diff=float(d), lo=float(np.quantile(bd, 0.025)), hi=float(np.quantile(bd, 0.975)),
                p=min(1.0, 2 * min((1 + (bd <= 0).sum()) / (n + 1), (1 + (bd >= 0).sum()) / (n + 1))))


def auc_contrasts(y, P, g, pairs, n, seed=0):
    """y [N,C], P {model: [N,C]}, pairs [(name, a, b)] -> ({name: macro-AUROC of a minus b with interval and p}, {model: per-class AUROC})."""
    names = list(P)
    base = np.stack([class_metric(y, P[m], None, roc_auc_score) for m in names])
    boot = np.stack([np.stack([class_metric(y, P[m], w, roc_auc_score) for m in names]) for w in resamples(g, n, seed)])
    return ({h: summary(np.nanmean(base[names.index(a)]) - np.nanmean(base[names.index(b)]),
                        np.nanmean(boot[:, names.index(a)], 1) - np.nanmean(boot[:, names.index(b)], 1)) for h, a, b in pairs},
            {m: base[k].tolist() for k, m in enumerate(names)})


def mean_contrast(diff, g, n, seed=0):
    """Mean of per-row differences, cluster bootstrap (each row weighted by its cluster's draw count)."""
    diff = np.asarray(diff, float)
    return summary(diff.mean(), np.array([(w * diff).sum() / w.sum() for w in resamples(g, n, seed)]))


def ensemble(root, prefix):
    """{variant: (seed-mean prediction, labels, idx, clusters, seeds)} of the runs root/prefix* (s4/uq_conformal.py load_runs)."""
    out = {}
    for (v, _), r in (load_runs(root, prefix) if Path(root).exists() else {}).items():
        out[v] = (*r[:4], len(list(Path(root).glob(f"{prefix}{v}_s*/test_pred.npy"))))
    return out


def macro(y, p):
    a = class_metric(y, p, None, roc_auc_score)
    return dict(macro=float(np.nanmean(a)), per_class=a.tolist())


def nih_hypotheses(root, v3, n, res):
    runs = ensemble(root / "down" / "nih", "nih_hyb_")
    dn = ensemble(v3, "nih_hyb_").get("orig@densenet121")
    have = {m: runs[v] for m, v in NIH_MODELS.items() if v in runs} | ({"DN121": dn} if dn else {})
    res["missing"] += [m for m in [*NIH_MODELS, "DN121"] if m not in have]
    if not have:
        return
    ref = next(iter(have.values()))
    for m, r in have.items():
        assert np.array_equal(r[2], ref[2]), f"{m}: other NIH test rows"
    y, g = ref[1], ref[3]
    res["nih_seeds"] = {m: r[4] for m, r in have.items()}
    res["nih_macro_auroc"] = {m: macro(y, r[0]) for m, r in have.items()}
    if {"F_in", "F_p0", "F_p1"} <= set(have):
        cols = [SHARED.index(s) for s in EXACT_NIH]
        P = {m: to_shared(have[m][0], "nih")[:, cols] for m in ("F_in", "F_p0", "F_p1")}
        h, per = auc_contrasts(to_shared(y, "nih")[:, cols], P, g, [("H11", "F_p1", "F_in"), ("H12_p1_p0", "F_p1", "F_p0"), ("H12_p0_in", "F_p0", "F_in")], n)
        for k, p in zip(h, holm([v["p"] for v in h.values()])):
            h[k]["p_holm"] = p
        res |= h
        res["shared_auroc"] = dict(findings=list(EXACT_NIH), **per)
    if {"F_in", "DN121"} <= set(have):
        h, _ = auc_contrasts(y, {m: have[m][0] for m in ("F_in", "DN121")}, g, [("H13", "F_in", "DN121")], n)
        res |= h


def h14(files, n):
    """Matched against mismatched CT/X-ray map correlation; maps averaged over the seed pairs (each map normalised to sum 1 first)."""
    zs = [np.load(f) for f in files]
    scan, cls = zs[0]["scan"], zs[0]["cls"]
    assert all(np.array_equal(z["scan"], scan) and np.array_equal(z["cls"], cls) for z in zs), "map files list other scans"
    norm = lambda m: m / (m.sum((1, 2), keepdims=True) + 1e-12)  # noqa: E731
    ct, xr = np.mean([norm(z["ct"]) for z in zs], 0), np.mean([norm(z["xr"]) for z in zs], 0)
    fr, frf = float(zs[0]["frame_r"].mean()), float(zs[0]["frame_r_flip"].mean())
    if frf > fr:  # pre-specified: orientation from the attenuation frames only (no heat map involved)
        xr = xr[..., ::-1]
    match = np.array([pearson(ct[i], xr[i]) for i in range(len(cls))])
    other = [np.nonzero((cls == cls[i]) & (scan != scan[i]))[0] for i in range(len(cls))]
    mis = np.array([np.mean([pearson(ct[i], xr[j]) for j in o]) if len(o) else np.nan for i, o in enumerate(other)])
    k = ~np.isnan(mis)
    return dict(H14=mean_contrast(match[k] - mis[k], scan[k], n), seeds=len(zs), n_maps=int(k.sum()), n_scans=int(len(np.unique(scan[k]))),
                frame_r=fr, frame_r_flip=frf, mirrored=frf > fr, matched=float(match[k].mean()), mismatched=float(mis[k].mean()),
                per_finding={SHARED[c]: dict(n=int((k & (cls == c)).sum()), matched=float(match[k & (cls == c)].mean()),
                                             mismatched=float(mis[k & (cls == c)].mean())) for c in np.unique(cls[k])})


def combine(files):
    """halluc.npz files of one model (seed groups run on different accounts) -> one record as if all seeds had run together: probabilities
    averaged with the number of seeds as weights, thresholds from the averaged validation probabilities; off-target pairs pooled (each
    group's pairs are the positives of its own seed mean)."""
    zs = [dict(np.load(f)) for f in files]
    w = np.array([int(z["n_runs"]) for z in zs], float)
    out = dict(zs[0], n_runs=int(w.sum()), n_files=len(zs))
    for z in zs[1:]:
        assert all(np.array_equal(z[k], out[k]) for k in ("free", "nod", "val_y")), "seed groups of one model tested on other images"
    for k in ("p_clean", *(f"p_{p}" for p in PERTURB), "p_nodule", "val_p"):
        out[k] = np.average([z[k] for z in zs], 0, weights=w)
    out["thr"] = thresholds(out["val_p"], out["val_y"])
    out["cam_off"] = np.concatenate([z["cam_off"] for z in zs])
    out["cam_pairs"] = np.concatenate([z["cam_pairs"] for z in zs])
    return out


def halluc(root, n, res):
    Z = {m: combine(fs) for m, v in HALLUC.items() for fs in [sorted((root / "halluc").glob(f"nih_xmh_{v}_s*/halluc.npz"))] if fs}
    res["missing"] += [f"halluc:{m}" for m in HALLUC if m not in Z]
    out = {}
    for m, z in Z.items():
        ok = z["nod_site"][:, 0] >= 0
        p0, p1 = z["p_clean"][z["nod_pos"]][:, LESION], z["p_nodule"][:, LESION]
        below = ok & (p0 <= z["thr"][LESION]).all(1)
        dp = p1.max(1) - p0.max(1)
        edges = np.quantile(z["nod_diam"][ok], [1 / 3, 2 / 3]) if ok.any() else [np.nan, np.nan]
        size = np.digitize(z["nod_diam"], edges)
        out[m] = dict(seeds=z["n_runs"], files=z["n_files"], n_free=int(len(z["free"])), rate_clean=float((z["p_clean"] > z["thr"]).any(1).mean()),
                      **{f"rate_{k}": float(crossed(z["p_clean"], z[f"p_{k}"], z["thr"]).mean()) for k in PERTURB},
                      nodule_sensitivity=float((p1[below] > z["thr"][LESION]).any(1).mean()) if below.any() else None, n_nodule=int(below.sum()),
                      nodule_dp_by_size={f"{lo:.1f}-{hi:.1f} mm": float(dp[ok & (size == t)].mean()) for t, (lo, hi) in
                                         enumerate(zip([z["nod_diam"][ok].min(), *edges], [*edges, z["nod_diam"][ok].max()]))} if ok.any() else {},
                      off_target=float(z["cam_off"].mean()) if len(z["cam_off"]) else None, n_cam_pairs=int(len(z["cam_off"])),
                      off_target_by_class={NIH[c]: float(z["cam_off"][z["cam_pairs"][:, 1] == c].mean()) for c in np.unique(z["cam_pairs"][:, 1])})
    res["hallucination"] = out
    if {"F_in", "F_p1"} <= set(Z):
        a, b = Z["F_p1"], Z["F_in"]
        assert np.array_equal(a["free"], b["free"]), "hallucination tests on other images"
        d = crossed(a["p_clean"], a["p_rib"], a["thr"]).astype(float) - crossed(b["p_clean"], b["p_rib"], b["thr"]).astype(float)
        res["H15"] = mean_contrast(d, a["free_patient"], n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "v4"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "results" / "final_xm_stats.json"))
    ap.add_argument("--v3", default=str(ROOT / "results" / "v3" / "down" / "nih"), help="v3 NIH runs (DenseNet-121 orig reference)")
    a = ap.parse_args(argv)
    root, res = Path(a.root), dict(n_boot=a.n, missing=[])
    nih_hypotheses(root, Path(a.v3), a.n, res)
    for name, sub, prefix in (("ct_teacher", "ct", "ct_xm_"), ("projection", "qp", "qp_xm_")):
        res[name] = {v: dict(seeds=r[4], **macro(r[1], r[0])) for v, r in ensemble(root / "down" / sub, prefix).items()}
    res["transfer"] = {ds: {v: dict(seeds=r[4], **macro(r[1], r[0])) for v, r in ensemble(root / "down" / ds, f"{ds}_xm_transfer-").items()}
                       for ds in ("nih", "vin")}
    for v in ("xraykd", "xray"):  # P1 is the H14 pair; P0 descriptive
        files = sorted((root / "xai").glob(f"ct_xmx_{v}_s*/maps.npz"))
        if files:
            res[f"crossmodal_{v}"] = h14(files, a.n)
        else:
            res["missing"].append(f"xai:{v}")
    halluc(root, a.n, res)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("H") or k == "missing"}, indent=1))


if __name__ == "__main__":
    main()
