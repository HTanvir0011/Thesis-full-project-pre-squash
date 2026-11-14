"""v3 statistics (SAP_v3.md H6 primary: hybrid vs orig; H7: vs ctrl and vs early; H8; H10; uncertainty summary) -> results/final_hybrid_stats.json, results/v3/tables/*.tex.

python tools/hybrid_stats.py [--n 10000] [--archs densenet121,<screen winner>]
python tools/hybrid_stats.py --tables_from results/final_hybrid_stats.json --tables paper/tables   (tables only, nothing recomputed)
Holm families (DEVIATIONS.md 2026-11-04, the author's R7 decision): H6 over encoders x data sets; H7 over encoders x data sets x both
contrasts; H8 and per-class DeLong within encoder and data set. H10 (pointing game): s4/xai_multistream.py outputs of the DenseNet-121
hybrid models, all seeds (results/v3/xai/nih_hyb_hybrid@densenet121_s*.json); the winner's the same way, as secondary.
Runs: results/v3/down/<dataset>/<dataset>_hyb_<config>@<arch>_s<seed>[_f<k>] (s4/cls_hybrid.py through kernels/down/run_down.py).
Seed means per configuration; BDCXR-3257: the five folds' test predictions concatenated (out-of-fold). Clusters: patients (NIH), images
(VinBigData), near-duplicate groups (BDCXR-3257), from test_patient.npy. Every contrast of a data set uses the same resamples (paired).
--dry_map 'early=orig-soft,...' runs the code on v2 predictions (results/final/down, results/bdcxr/down) to test it; output goes to --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from final_stats import tex as write_tex  # noqa: E402
from s4.delong import delong_paired  # noqa: E402
from s4.prep import BDCXR, NIH  # noqa: E402
from s4.stats import holm  # noqa: E402
from s4.uq_conformal import load_runs  # noqa: E402

GROUPS = {  # DESIGN_v3.md section 3 (class indices of s4/prep.py NIH and of the VinBigData class list)
    "nih": {"focal": [4, 5], "diffuse": [2, 3, 8, 9, 6], "boundary": [1, 0, 7, 12, 10, 11, 13]},
    "vin": {"focal": [8], "diffuse": [10, 6, 4, 7, 5], "boundary": [3, 1, 12, 11, 0, 13, 2]},
}
CONFIGS = ("orig", "soft", "early", "hybrid", "ctrl")
DS = ("nih", "vin", "bdcxr")
FOCAL_BOX = {"Mass", "Nodule"}  # NIH box classes; the other six (Atelectasis, Cardiomegaly, Effusion, Infiltration, Pneumonia, Pneumothorax) are diffuse/boundary
VIN = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung Opacity",
       "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"]  # VinBigData class_id order (GROUPS indices)
NAMES = {"nih": NIH, "vin": VIN, "bdcxr": BDCXR}
MAPS = {"cam_0": "Grad-CAM++, X stream", "cam_1": "Grad-CAM++, S stream", "cam_2": "Grad-CAM++, B stream", "attn": "decoder attention",
        "ig_X": "Integrated Gradients, X", "ig_S": "Integrated Gradients, S", "ig_B": "Integrated Gradients, B", "ig_all": "Integrated Gradients, sum"}


def gather(ds, arch, dry_map=None):
    """-> (y [N,C], {config: pred [N,C]}, clusters [N]) or None."""
    if dry_map:
        root, prefix = (ROOT / "results/bdcxr/down", "bdcxr_cls_") if ds == "bdcxr" else (ROOT / "results/final/down" / ds, f"{ds}_cls_")
    else:
        root, prefix = ROOT / "results/v3/down" / ds, f"{ds}_hyb_"
    if not root.exists():
        return None
    runs = load_runs(root, prefix)
    P, Y, G = {}, None, None
    for cfg in CONFIGS:
        want = dry_map.get(cfg) if dry_map else f"{cfg}@{arch}"
        keys = sorted(k for k in runs if k[0] == want)
        if not keys:
            continue
        idx = np.concatenate([runs[k][2] for k in keys])
        o = np.argsort(idx, kind="stable")  # one row order for every configuration
        P[cfg] = np.concatenate([runs[k][0] for k in keys])[o]
        y, g = np.concatenate([runs[k][1] for k in keys])[o], np.concatenate([runs[k][3] for k in keys])[o]
        assert Y is None or np.array_equal(np.nan_to_num(Y, nan=-1), np.nan_to_num(y, nan=-1)), f"{ds}: label order differs between configurations"
        Y, G = y, g
    return (Y, P, G) if P else None


def class_metric(y, p, w, fn):
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        k = ~np.isnan(y[:, c]) if w is None else ~np.isnan(y[:, c]) & (w > 0)
        if 0 < y[k, c].sum() < k.sum():
            out[c] = fn(y[k, c], p[k, c], sample_weight=None if w is None else w[k])
    return out


def analyse(ds, y, P, g, n, seed=0):
    cfgs = list(P)
    labels, inv = np.unique(np.asarray(g), return_inverse=True)
    rng = np.random.default_rng(seed)
    base = {m: np.stack([class_metric(y, P[c], None, f) for c in cfgs]) for m, f in (("auroc", roc_auc_score), ("auprc", average_precision_score))}
    boot = {m: [] for m in base}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))[inv].astype(float)
        for m, f in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            boot[m].append(np.stack([class_metric(y, P[c], w, f) for c in cfgs]))
    boot = {m: np.stack(v) for m, v in boot.items()}  # [n, n_cfg, C]

    def contrast(a, b, cols, m="auroc"):
        i, j = cfgs.index(a), cfgs.index(b)
        d = float(np.nanmean(base[m][i, cols]) - np.nanmean(base[m][j, cols]))
        bd = np.nanmean(boot[m][:, i, cols], 1) - np.nanmean(boot[m][:, j, cols], 1)
        p = min(1.0, 2 * min((1 + (bd <= 0).sum()) / (n + 1), (1 + (bd >= 0).sum()) / (n + 1)))  # two-sided percentile bootstrap
        return dict(diff=d, lo=float(np.quantile(bd, 0.025)), hi=float(np.quantile(bd, 0.975)), p=p)

    allc = list(range(y.shape[1]))
    res = dict(n=int(len(y)), clusters=int(len(labels)), configs=cfgs, n_boot=n,
               macro_auroc={c: float(np.nanmean(base["auroc"][k])) for k, c in enumerate(cfgs)},
               macro_auprc={c: float(np.nanmean(base["auprc"][k])) for k, c in enumerate(cfgs)})
    for h, a, b in (("H6", "hybrid", "orig"), ("H7_ctrl", "hybrid", "ctrl"), ("H7_early", "hybrid", "early")):
        if a in P and b in P:
            res[h] = dict(auroc=contrast(a, b, allc), auprc=contrast(a, b, allc, "auprc"))
    if ds in GROUPS:
        h8 = {f"{grp}:{a}-orig": contrast(a, "orig", cols) for grp, cols in GROUPS[ds].items() for a in ("hybrid", "soft") if a in P and "orig" in P}
        for k, v in zip(h8, holm([v["p"] for v in h8.values()])):
            h8[k]["p_holm"] = v
        res["H8_groups"] = h8
    pc = {}
    for a, b in (("hybrid", "orig"), ("hybrid", "early")):
        if a in P and b in P:
            for c in allc:
                k = ~np.isnan(y[:, c])
                if 0 < y[k, c].sum() < k.sum():
                    pc[f"{a}-{b}:{c}"] = delong_paired(y[k, c], P[a][k, c], P[b][k, c])
    for k, v in zip(pc, holm([v["p"] for v in pc.values()])):
        pc[k]["p_holm"] = v
    res["per_class_delong"] = pc
    return res


def holm_families(enc):
    """enc[arch][ds][hypothesis]['auroc']['p'] -> adds 'p_holm': H6 one family over encoders x data sets, H7 one family over encoders x
    data sets x {ctrl, early} (DEVIATIONS.md 2026-11-04)."""
    for fam in (("H6",), ("H7_ctrl", "H7_early")):
        keys = [(a, ds, h) for a in enc for ds in DS for h in fam if h in enc[a].get(ds, {})]
        for (a, ds, h), p in zip(keys, holm([enc[a][ds][h]["auroc"]["p"] for a, ds, h in keys])):
            enc[a][ds][h]["auroc"]["p_holm"] = p


def pointing(files, n=10000, seed=0):
    """H10 from s4/xai_multistream.py outputs (one per seed's hybrid model). Per target (image, box class): S-stream minus X-stream
    Grad-CAM++ hit (cam_1 - cam_0), averaged over the seeds; interaction = focal mean - diffuse/boundary mean; image-cluster bootstrap."""
    per, maps = {}, {}
    for f in files:
        for r in json.loads(Path(f).read_text())["rows"]:
            k = (r["image"], r["cls"])
            per.setdefault(k, []).append(float(r["hits"]["cam_1"]) - float(r["hits"]["cam_0"]))
            for m, h in r["hits"].items():
                maps.setdefault(m, {}).setdefault(k, []).append(float(h))
    keys = sorted(per)
    d = np.array([np.mean(per[k]) for k in keys])
    foc = np.array([k[1] in FOCAL_BOX for k in keys])
    if not foc.any() or foc.all():
        return dict(n_targets=len(keys), note="needs focal and non-focal targets")
    labels, inv = np.unique(np.array([k[0] for k in keys]), return_inverse=True)

    def stat(w):
        a, b = w[foc].sum(), w[~foc].sum()
        return np.dot(w[foc], d[foc]) / a - np.dot(w[~foc], d[~foc]) / b if a > 0 and b > 0 else np.nan

    rng = np.random.default_rng(seed)
    est = stat(np.ones(len(keys)))
    bd = np.array([stat(np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))[inv].astype(float)) for _ in range(n)])
    bd = bd[np.isfinite(bd)]
    p = min(1.0, 2 * min((1 + (bd <= 0).sum()) / (len(bd) + 1), (1 + (bd >= 0).sum()) / (len(bd) + 1)))
    rate = {m: dict(focal=float(np.mean([np.mean(v[k]) for k in keys if k[1] in FOCAL_BOX])),
                    other=float(np.mean([np.mean(v[k]) for k in keys if k[1] not in FOCAL_BOX]))) for m, v in maps.items()}
    return dict(files=[str(f) for f in files], n_seeds=len(files), n_targets=len(keys), n_focal=int(foc.sum()), n_images=int(len(labels)),
                interaction=float(est), lo=float(np.quantile(bd, 0.025)), hi=float(np.quantile(bd, 0.975)), p=p, n_boot=int(len(bd)), hit_rates=rate)


def fmt(c):
    return f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def write_tables(out, tdir):
    """Bare tabulars (tex()) for the papers from a statistics dict as main() writes it to --out (--tables_from: without recomputing)."""
    enc, archs = out["encoders"], out["archs"]
    tdir.mkdir(parents=True, exist_ok=True)
    dsl = {"nih": "NIH", "vin": "VinBigData", "bdcxr": "BDCXR-3257"}

    def tex(path, cols, rows, caption):  # one encoder: its name goes to the caption comment instead of every row
        if len(archs) == 1 and cols[0] == "encoder":
            cols, rows, caption = cols[1:], [r[1:] for r in rows], f"{caption}; encoder {archs[0]}"
        write_tex(path, cols, [[dsl.get(c, c) for c in r] for r in rows], caption)
    hyps = ("H6", "H7_ctrl", "H7_early")
    label = {"H6": "H6: hybrid vs orig", "H7_ctrl": "H7: hybrid vs ctrl", "H7_early": "H7: hybrid vs early"}
    rows = [[arch, ds, label[h], fmt(enc[arch][ds][h]["auroc"]), f"{enc[arch][ds][h]['auroc']['p_holm']:.3g}", fmt(enc[arch][ds][h]["auprc"])]
            for h in hyps for arch in archs for ds in DS if h in enc[arch].get(ds, {})]
    if rows:
        tex(tdir / "v3_primary.tex", ["encoder", "data set", "hypothesis", r"$\Delta$ macro-AUROC [95\,\% CI]", "Holm $p$", r"$\Delta$ macro-AUPRC [95\,\% CI]"], rows,
            "H6 (primary): hybrid - original, Holm over encoders x data sets; H7 (secondary): hybrid - control X x3 and hybrid - early-fused, "
            "Holm over encoders x data sets x contrasts (SAP_v3.md; DEVIATIONS.md 2026-11-04); cluster bootstrap")
    g = [[arch, ds, k.replace(":", ": "), fmt(v), f"{v['p_holm']:.3g}"] for arch in archs for ds in ("nih", "vin")
         for k, v in enc[arch].get(ds, {}).get("H8_groups", {}).items()]
    if g:
        tex(tdir / "v3_groups.tex", ["encoder", "data set", "group: contrast", r"$\Delta$ mean AUROC [95\,\% CI]", "Holm $p$"], g,
            "H8 per taxonomy group (DESIGN_v3.md section 3), Holm within encoder and data set")
    m = [[arch, f"{dsl[ds]} ({e['n']:,})", name] + [f"{e[key][c]:.4f}" for c in CONFIGS]
         for arch in archs for ds, e in enc[arch].items() for name, key in (("macro-AUROC", "macro_auroc"), ("macro-AUPRC", "macro_auprc"))]
    if m:
        tex(tdir / "v3_macro.tex", ["encoder", "data set (test images)", "metric", *CONFIGS], m,
            "Seed-ensemble test metrics per configuration, descriptive (BDCXR-3257: out-of-fold)")
    pc = []
    for arch in archs:
        for ds, e in enc[arch].items():
            d = e.get("per_class_delong", {})
            for i in sorted({int(k.split(":")[1]) for k in d}):
                ho, he = d[f"hybrid-orig:{i}"], d[f"hybrid-early:{i}"]  # delong_paired(y, first, second): auc1 = hybrid
                pc.append([arch, ds, NAMES[ds][i].replace("_", " "), f"{ho['auc2']:.4f}", f"{ho['auc1']:.4f}", f"{ho['diff']:+.4f}", f"{ho['p_holm']:.3g}",
                           f"{he['diff']:+.4f}", f"{he['p_holm']:.3g}"])
    if pc:
        tex(tdir / "v3_per_class.tex", ["encoder", "data set", "class", "AUROC orig", "AUROC hybrid", r"$\Delta$ hybrid $-$ orig", "Holm $p$",
                                        r"$\Delta$ hybrid $-$ early", "Holm $p$"], pc,
            "H8 per class: paired DeLong on the seed ensembles, Holm within encoder and data set over classes x both contrasts")
    pt = []
    for arch, h in out.get("H10", {}).items():
        r = h.get("hit_rates", {})
        pt += [[arch, MAPS.get(k, k), f"{v['focal']:.3f}", f"{v['other']:.3f}", f"{v['focal'] - v['other']:+.3f}"] for k, v in r.items()]
        if "cam_0" in r and "cam_1" in r:
            pt.append([arch, "S minus X (H10)", f"{r['cam_1']['focal'] - r['cam_0']['focal']:+.3f}", f"{r['cam_1']['other'] - r['cam_0']['other']:+.3f}",
                       f"{h['interaction']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}], p = {h['p']:.2g}"])
    if pt:
        n = next(iter(out["H10"].values()))
        tex(tdir / "v3_pointing.tex", ["encoder", "map", f"focal targets ({n['n_focal']})", f"other targets ({n['n_targets'] - n['n_focal']})",
                                       "focal $-$ other"], pt,
            "H10 pointing game on the NIH expert boxes: hit rates averaged over the seeds' hybrid models; interaction: image-cluster bootstrap")
    if "uncertainty" in out:
        c, b = out["uncertainty"]["cls"], out["uncertainty"]["bone"]
        tex(tdir / "v3_bone_intervals.tex", ["calibration (nominal 0.90)", "pooled pixel coverage", "median per-image coverage", "mean width"],
            [[m, f"{b[k]['test']['pooled']['pixel_coverage']:.3f}", f"{b[k]['test']['pooled']['image_coverage_median']:.3f}",
              f"{b[k]['test']['pooled']['width_mean']:.3f}"] for m, k in (("global", "global"), ("Mondrian", "mondrian"))],
            f"H9 (exploratory, post hoc): bone-image intervals calibrated on {b['n_cal']} SZCH development pairs, tested on {b['n_test']} held-out pairs")
        u = [[f"BDCXR-3257 {k[6:]}, Mondrian sets", f"coverage {v['coverage']:.3f}", f"{v['mean_set_size']:.2f}"] for k, v in c.items() if k.startswith("bdcxr_")]
        u += [[f"NIH {k[4:]}, FNR control", f"FNR {v['fnr_test']:.3f}", f"{v['mean_set_size']:.2f}"] for k, v in c.items() if k.startswith("nih_")]
        tex(tdir / "v3_conformal.tex", ["setting (nominal 0.90)", "test result", "mean set size"], u, "Conformal classification (s4/uq_conformal.py cls3)")
        s = [[{"nih": "NIH (Brier score)", "bdcxr": "BDCXR-3257 (top-1 error)"}[k.partition("_")[0]], k.partition("_")[2],
              f"{v['aurc_brier'] if k.startswith('nih_') else v['aurc']:.4f}", f"{v['score_b']['aurc_score_a_same_images']:.4f}", f"{v['score_b']['aurc']:.4f}"]
             for k, v in c.items() if k.partition("_")[0] in ("nih", "bdcxr") and isinstance(v.get("score_b"), dict)]
        if s:
            tex(tdir / "v3_selective.tex", ["data set (risk)", "configuration", "AURC, 1 $-$ entropy", "same, images with spread",
                                            "AURC, + suppressor spread"], s,
                "Selective classification: area under the risk-coverage curve, lower is better; score B adds the lung-field suppressor spread")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--archs", default="densenet121", help="comma-separated: DenseNet-121 and the screen winner")
    ap.add_argument("--xai", default=str(ROOT / "results/v3/xai"), help="folder with s4/xai_multistream.py outputs <run tag>.json")
    ap.add_argument("--dry_map", default=None, help="code test on v2 runs, e.g. 'orig=orig,soft=soft,early=orig-soft'")
    ap.add_argument("--out", default=str(ROOT / "results/final_hybrid_stats.json"))
    ap.add_argument("--tables", default=str(ROOT / "results/v3/tables"))
    ap.add_argument("--tables_from", default=None, help="only write the tables from this statistics file (as written to --out)")
    a = ap.parse_args(argv)
    if a.tables_from:
        return write_tables(json.loads(Path(a.tables_from).read_text()), Path(a.tables))
    dry =dict(kv.split("=") for kv in a.dry_map.split(",")) if a.dry_map else None
    archs = ["dry"] if dry else a.archs.split(",")
    out = dict(archs=archs, dry_run=bool(dry), sap="SAP_v3.md", decisions="DEVIATIONS.md 2026-11-04 (R7)", encoders={})
    for arch in archs:
        out["encoders"][arch] = {}
        for ds in DS:
            got = gather(ds, arch, dry)
            if got is None:
                print(f"{arch} {ds}: no runs")
                continue
            out["encoders"][arch][ds] = analyse(ds, *got, n=a.n)
            print(arch, ds, json.dumps(out["encoders"][arch][ds]["macro_auroc"]))
    enc = out["encoders"]
    holm_families(enc)
    out["H10"] = {}
    for arch in archs:
        files = sorted(Path(a.xai).glob(f"nih_hyb_hybrid@{arch}_s*.json"))
        if files:
            out["H10"][arch] = dict(pointing(files, a.n), role="primary" if arch == "densenet121" else "secondary")
    uc, ub = ROOT / "results/v3/uq_cls_densenet121.json", ROOT / "results/v3/uq_bone.json"  # cls3 on the primary encoder; uq_cls.json is the v2 'cls' file
    if uc.exists() and ub.exists():
        c, b = json.load(open(uc)), json.load(open(ub))
        out["uncertainty"] = dict(cls=c, bone={k: b[k] for k in ("n_cal", "n_test", "global", "mondrian", "reliability", "held_ause", "note")})
    write_tables(out, Path(a.tables))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
