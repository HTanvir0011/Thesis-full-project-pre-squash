"""Confirmatory statistics of SAP_v2.md (H1'-H5') from saved result files; writes results/final/stats.json and paper/tables/*.tex.

python tools/final_stats.py [--root results/final] [--tables paper/tables] [--subset held|dev|all]
Every section is skipped (and says so) when its inputs are missing. --subset dev exists only to test this script before the freeze.
Method names are the file names written by tools/final_local.sh (ours, notwin, qure, ours_s0, w4_s0, ...).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from s4 import froc  # noqa: E402
from s4.delong import delong_paired  # noqa: E402
from s4.stats import cluster_bootstrap_ci, holm, non_inferior, paired_tests  # noqa: E402
from szch_split import is_dev  # noqa: E402

DOSE = {0: ["swin_nt_s0", "swin_nt_s1", "swin_nt_s2"], 4: ["w4_s0", "w4_s1"], 8: ["w8_s0", "w8_s1"], 16: ["swin_s0", "swin_s1", "swin_s2", "swin_s3"],
        32: ["w32_s0", "w32_s1"], 64: ["w64_s0", "w64_s1"]}  # SwinV2-T family (the headline ConvNeXt pair has only w = 0 and 16)
SZCH_COLS = ["psnr_cal_bone", "psnr_cal", "ssim_cal", "ssim_cal_s2", "ssim_cal_s4", "msssim_cal", "lpips_cal", "psnr_raw", "ssim_raw", "hp_ratio_nonbone"]


def ci_str(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def esc(cell):
    """Method names and labels in text mode (main.tex loads neither `underscore` nor T1 fontenc); math between $...$ is kept as written."""
    parts = str(cell).split("$")  # even parts are text, odd parts math ("swin $-$ swin_nt" needs swin\_nt)
    return "$".join(p if i % 2 else p.replace("_", r"\_").replace(">=", r"$\geq$").replace("<", r"$<$") for i, p in enumerate(parts))


def tex(path, cols, rows, caption):
    """Bare booktabs tabular: paper/sections/results.tex supplies the float, caption and label; the caption text goes in a comment."""
    body = "\n".join(" & ".join(map(esc, r)) + r" \\" for r in rows)
    path.write_text("% " + caption + "\n\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n"
                    + " & ".join(cols) + r" \\" + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def macro_auc(y, p, w=None):
    """Mean one-vs-rest AUROC over the classes with both labels present (NaN label = not annotated); w = bootstrap weights per row."""
    a = []
    for c in range(y.shape[1]):
        k = ~np.isnan(y[:, c]) if w is None else ~np.isnan(y[:, c]) & (w > 0)
        if 0 < y[k, c].sum() < k.sum():
            a.append(roc_auc_score(y[k, c], p[k, c], sample_weight=None if w is None else w[k]))
    return float(np.mean(a))


def boot_vs_ref(y, P, ref, g, n=1000, seed=0):
    """{input: (diff, lo, hi)}: macro-AUROC(input) - macro-AUROC(ref) with a 95% CI from resampling clusters g (patients, duplicate
    groups) with replacement; a resample is applied as row weights, and every input uses the same resamples (paired)."""
    labels, inv = np.unique(np.asarray(g), return_inverse=True)
    rng, base = np.random.default_rng(seed), {v: macro_auc(y, p) for v, p in P.items()}
    boot = {v: [] for v in P if v != ref}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))[inv].astype(float)
        r0 = macro_auc(y, P[ref], w)
        for v in boot:
            boot[v].append(macro_auc(y, P[v], w) - r0)
    return {v: (base[v] - base[ref], float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) for v, b in boot.items()}


def szch(root, out, tables, keep):
    d = root / "szch"
    files = {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))} if d.exists() else {}
    if "ours" not in files:
        return print("H2': skipped (results/final/szch/ours.csv missing)")
    sub = {k: v[v.image.map(keep)].set_index("image").sort_index() for k, v in files.items()}
    out["szch_n"] = len(sub["ours"])
    out["szch_desc"] = {k: {c: float(v[c].mean()) for c in SZCH_COLS if c in v} for k, v in sub.items()}
    tests = []
    for ref, col, margin in (("qure", "psnr_cal_bone", 0.5), ("qure", "ssim_cal", 0.01), ("notwin", "psnr_cal_bone", 0.25)):
        if ref not in sub:
            continue
        diff = (sub["ours"][col] - sub[ref][col]).values
        tests.append(dict(ref=ref, metric=col, **non_inferior(diff, margin), ci95=cluster_bootstrap_ci(diff)[1:],
                          **paired_tests(sub["ours"][col], sub[ref][col])))
    out["H2"] = tests
    cols = ("psnr_cal_bone", "psnr_cal", "ssim_cal", "ssim_cal_s4", "msssim_cal", "lpips_cal", "psnr_raw", "ssim_raw", "hp_ratio_nonbone")
    rows = [[k] + [f"{v[c].mean():.3f}" if c in v else "-" for c in cols] for k, v in sub.items()]
    tex(tables / "szch_heldout.tex", ["method", "bone PSNR$_{cal}$", "PSNR$_{cal}$", "SSIM$_{cal}$", r"SSIM$_{cal,\sigma4}$", "MS-SSIM", "LPIPS",
                                      "PSNR raw", "SSIM raw", "HF ratio"], rows,
        f"SZCH paired dual-energy test, {out['szch_n']} pairs ({out['subset']} subset; mean). Generated by tools/final_stats.py.")


def load_audit(d):
    return {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))} if d.exists() else {}


def audit_contrasts(a, ref, thr=0.15, tw="ours"):
    """tw - ref per site (sites shared across files by image and site index) -> CI, Holm-ready p, overlap interaction."""
    k = ["image", "site", "alpha"]
    m = a[tw].merge(a[ref], on=k, suffixes=("", "_r"))
    diff, cl = (m.absorption - m.absorption_r).values, m.image.values
    mean, lo, hi = cluster_bootstrap_ci(diff, cl)
    img = m.assign(d=diff).groupby("image").d.mean()
    p = paired_tests(img.values, np.zeros(len(img)))["wilcoxon_p"]
    hi_ov = m.overlap >= thr
    did = np.nan
    if hi_ov.any() and (~hi_ov).any():  # images resampled WITH replacement (counts as weights), sums per image and overlap group
        labels, inv = np.unique(cl, return_inverse=True)
        hv = hi_ov.values
        s_hi, n_hi = np.bincount(inv, diff * hv, len(labels)), np.bincount(inv, hv.astype(float), len(labels))
        s_lo, n_lo = np.bincount(inv, diff * ~hv, len(labels)), np.bincount(inv, (~hv).astype(float), len(labels))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels)) for _ in range(2000)]).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = W @ s_hi / (W @ n_hi) - W @ s_lo / (W @ n_lo)
        did = dict(est=float((diff[hv]).mean() - diff[~hv].mean()), ci95=[float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))])
    return dict(ref=ref, n_sites=len(m), mean_diff=mean, ci95=[lo, hi], p_image_wilcoxon=p, overlap_interaction=did)


def audits(root, out, tables, keep):
    for name, sub in (("ribcxr", "audit_ribcxr"), ("jsrt", "audit_jsrt"), ("szch", "audit_szch")):
        a = load_audit(root / sub)
        if name == "szch":
            a = {k: v[v.image.map(keep)] for k, v in a.items()}
        if "ours" not in a:
            print(f"H3' {name}: skipped")
            continue
        res = dict(mean_A={k: list(cluster_bootstrap_ci(v.absorption.values, v.image.values)) for k, v in a.items()})
        res["contrasts"] = [audit_contrasts(a, r) for r in ("qure", "notwin") if r in a]
        for c, ph in zip(res["contrasts"], holm([c["p_image_wilcoxon"] for c in res["contrasts"]])):
            c["p_holm"] = ph
        res["dose_response"] = dose(a)
        res["encoder_pairs"] = [dict(twin=tw, **audit_contrasts(a, nt, tw=tw)) for tw, nt in (("swin", "swin_nt"), ("rd", "rd_nt")) if tw in a and nt in a]  # secondary: twin vs no twin on the other encoders
        out[f"H3_{name}"] = res
        if name == "ribcxr":
            tex(tables / "audit_ribcxr.tex", ["method", "absorption $A$ [95\\% CI]"],
                [[k, ci_str(*v)] for k, v in res["mean_A"].items()], "Counterfactual lesion audit on VinDr-RibCXR (lower is better).")


def dose(a):
    """H1': slope of per-site absorption (mean over seeds) on log2(1 + w), image-cluster bootstrap."""
    have = {w: [n for n in names if n in a] for w, names in DOSE.items()}
    have = {w: n for w, n in have.items() if n}
    if len(have) < 3:
        return "skipped (fewer than 3 twin weights present)"
    k = ["image", "site"]
    per = []
    for w, names in have.items():
        g = pd.concat([a[n][k + ["absorption"]] for n in names]).groupby(k).absorption.mean().reset_index()
        per.append(g.assign(x=np.log2(1 + w)))
    df = pd.concat(per)
    slope = lambda s: float(np.polyfit(s.x, s.absorption, 1)[0])  # noqa: E731
    rng, imgs = np.random.default_rng(0), df.image.unique()
    boot = [slope(pd.concat([df[df.image == i] for i in rng.choice(imgs, len(imgs))])) for _ in range(500)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    means = {int(w): float(df[df.x == np.log2(1 + w)].absorption.mean()) for w in have}
    return dict(weights=sorted(have), mean_A=means, slope=slope(df), ci95=[float(lo), float(hi)], passes=bool(hi < 0))


def qp(root, out):
    d = root / "qp"
    if not (d / "ours" / "sites.csv").exists():
        return print("QP exact-truth audit: skipped")
    s = {p.name: pd.read_csv(p / "sites.csv") for p in d.iterdir() if (p / "sites.csv").exists()}
    im = {p.name: pd.read_csv(p / "images.csv") for p in d.iterdir() if (p / "images.csv").exists()}
    out["qp"] = {k: dict(A=list(cluster_bootstrap_ci(v.absorption.values, v.name.values)), lcr=float(np.nanmedian(v.lcr)),
                         bone_mae=float(im[k].mae_bone.mean())) for k, v in s.items()}
    out["qp_dose_response"] = dose({k: v.rename(columns={"name": "image"}).assign(alpha=1.0) for k, v in s.items()})


def cnr(root, out, tables):
    res = {}
    for ds in ("jsrt", "node21"):
        def boxes(name):
            p = root / ds / name / "ann.json"
            if not p.exists():
                return None
            rows = []
            for i, r in enumerate(json.loads(p.read_text())):
                for j, _ in enumerate(r.get("boxes", [])):
                    if j < len(r.get("cnr_orig", [])):
                        rows.append(dict(img=i, box=j, orig=r["cnr_orig"][j], soft=r["cnr_soft"][j], qure=r.get("cnr_soft_qure", [np.nan] * 99)[j],
                                         ov=r.get("overlap_cons", r.get("overlap", [np.nan] * 99))[j]))
            return pd.DataFrame(rows)
        o, n = boxes("ours"), boxes("notwin")
        if o is None or o.empty:
            print(f"H4'a {ds}: skipped")
            continue
        ok = o[["orig", "soft"]].notna().all(axis=1)  # degenerate boxes (Node21: 2 of 1,476) have NaN contrast and would turn every bootstrap CI into NaN
        if n is not None and len(n) == len(o):
            ok &= n[["orig", "soft"]].notna().all(axis=1)
            n = n[ok]
        print(f"H4'a {ds}: {int((~ok).sum())} of {len(o)} boxes dropped for NaN contrast")
        o = o[ok]
        r, ps = {}, []
        img_p = lambda dd, im: paired_tests(pd.Series(dd).groupby(np.asarray(im)).mean().values, 0)["wilcoxon_p"]  # noqa: E731  image-level
        for lab, sel in (("all", slice(None)), ("bone_overlap>=0.15", o.ov >= 0.15)):
            oo = o[sel]
            r[lab] = dict(n=len(oo), d_cnr_ours=float((oo.soft - oo.orig).mean()), d_cnr_qure=float((oo.qure - oo.orig).mean()),
                          ours_minus_qure=list(cluster_bootstrap_ci((oo.soft - oo.qure).values, oo.img.values)))
            ps.append((lab, "ours_minus_qure", img_p((oo.soft - oo.qure).values, oo.img.values)))
            if n is not None and len(n) == len(o):
                nn = n[sel]
                r[lab]["ours_minus_notwin"] = list(cluster_bootstrap_ci((oo.soft.values - nn.soft.values), oo.img.values))
                ps.append((lab, "ours_minus_notwin", img_p(oo.soft.values - nn.soft.values, oo.img.values)))
        for (lab, k, _), ph in zip(ps, holm([p for *_, p in ps])):  # Holm within dataset (SAP_v2 H4'a)
            r[lab][f"{k}_p_holm"] = ph
        res[ds] = r
    out["H4a_cnr"] = res
    if res:
        tex(tables / "nodule_cnr.tex", ["set", "lesions", r"$\Delta$CNR ours", r"$\Delta$CNR Qure", "ours $-$ Qure [95\\% CI]", "ours $-$ no-twin [95\\% CI]"],
            [[f"{ds} ({lab})", v["n"], f"{v['d_cnr_ours']:.3f}", f"{v['d_cnr_qure']:.3f}", ci_str(*v["ours_minus_qure"]),
              ci_str(*v["ours_minus_notwin"]) if "ours_minus_notwin" in v else "-"]
             for ds, r in res.items() for lab, v in r.items()], "Real nodule contrast after suppression (box CNR change vs original; Holm within dataset in stats_held.json).")


def det_parts(pk):
    """Matches of one detection run, computed per image once and sorted by score as in s4/froc.py, so that an image bootstrap only
    re-weights: {'cpm': (tp, image, n_gt per image)} at IoU 0.3 and {class: (...)} at IoU 0.5 for AP."""
    def one(preds, gts, thr):
        m = [froc.match([p], [g], thr) for p, g in zip(preds, gts)]
        s = np.array([v for x in m for v in x[0]])  # built like froc.match (dtype included): the unstable sort orders tied scores the same way
        tp = np.array([v for x in m for v in x[1]], bool)
        img = np.concatenate([np.full(len(x[0]), i) for i, x in enumerate(m)]).astype(int)
        o = np.argsort(-s, kind="stable")
        last = np.r_[s[o][1:] != s[o][:-1], True] if len(s) else np.zeros(0, bool)  # tied scores enter together, as in s4/froc.py curve()
        return tp[o], img[o], np.array([len(g[0]) for g in gts], float), last
    parts = {"cpm": one(pk["preds"], pk["gts"], 0.3)}
    for c in np.unique(np.concatenate([np.asarray(g[1]) for g in pk["gts"]])).astype(int):
        pc = [(np.asarray(pb)[np.asarray(pl) == c], np.asarray(ps)[np.asarray(pl) == c], np.asarray(pl)[np.asarray(pl) == c]) for pb, ps, pl in pk["preds"]]
        gc = [(np.asarray(gb).reshape(-1, 4)[np.asarray(gl) == c], np.asarray(gl)[np.asarray(gl) == c]) for gb, gl in pk["gts"]]
        parts[int(c)] = one(pc, gc, 0.5)
    return parts


def det_metrics(parts, w):
    """(CPM, mAP@50) with image weights w (all ones = s4/froc.py on the full set; bootstrap counts otherwise)."""
    def curve(tp, img, ngt, last):
        ww = w[img]
        return np.cumsum(tp * ww)[last], np.cumsum(~tp * ww)[last], float(w @ ngt)
    tpc, fpc, n_gt = curve(*parts["cpm"])
    f = fpc / w.sum()
    cpm = float(np.mean([tpc[f <= r][-1] / n_gt if (f <= r).any() else 0.0 for r in froc.FP_RATES]))
    aps = []
    for c, p in parts.items():
        if c == "cpm":
            continue
        tpc, fpc, n_gt = curve(*p)
        if n_gt == 0:
            continue
        rec, prec = np.r_[0, tpc / n_gt, 1], np.r_[0, tpc / np.maximum(tpc + fpc, 1), 0]
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        aps.append(float(np.sum((rec[1:] - rec[:-1]) * prec[1:])))
    return cpm, float(np.mean(aps)) if aps else float("nan")


def downstream(root, out, tables, n_boot=1000):
    """H4'b as SAP.md H4: NIH macro-AUROC with a patient-cluster bootstrap (per-class DeLong, Holm); detection CPM and mAP@50 as mean +- SD over
    seeds with an image bootstrap of the difference to orig; recall at 2 FP/image by bone overlap as a difference in differences with an image
    bootstrap (box hits at the full-set threshold, re-weighted)."""
    d = root / "down"
    if not d.exists():
        return print("H4'b downstream: skipped")
    res = {}
    for ds_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        runs = [json.loads(p.read_text()) | {"dir": p.parent} for p in ds_dir.glob("*/result.json")]
        if not runs:
            continue
        df = pd.DataFrame(runs)
        key = "variants" if "variants" in df else "variant"
        df["input"] = df[key].map(lambda v: "+".join(v) if isinstance(v, list) else v)
        dirs = {i: sorted(g.dir, key=str) for i, g in df.groupby("input")}
        if "test_macro_auc" in df:
            idx0 = np.load(df.dir.iloc[0] / "test_idx.npy")
            assert all((np.load(p / "test_idx.npy") == idx0).all() for p in df.dir), f"{ds_dir.name}: test sets differ between runs"
            y = np.load(df.dir.iloc[0] / "test_labels.npy")
            pm = {i: np.mean([np.load(p / "test_pred.npy").astype(np.float32) for p in ds], 0) for i, ds in dirs.items()}
            r = {i: dict(macro_auc_ensemble=macro_auc(y, pm[i]), macro_auc_seed_mean=float(g.test_macro_auc.mean()),
                         sd=float(g.test_macro_auc.std(ddof=1)) if len(g) > 1 else 0.0, seeds=len(g)) for i, g in df.groupby("input")}
            if "orig" in pm:
                bt = boot_vs_ref(y, pm, "orig", np.load(df.dir.iloc[0] / "test_patient.npy", allow_pickle=True), n=n_boot)
                for i, (dm, lo, hi) in bt.items():
                    dl = [delong_paired(y[k, c], pm[i][k, c], pm["orig"][k, c]) for c in range(y.shape[1]) for k in [~np.isnan(y[:, c])]]
                    r[i]["vs_orig"] = dict(diff=dm, ci95=[lo, hi], per_class_diff=[x["diff"] for x in dl], per_class_p_holm=holm([x["p"] for x in dl]))
        else:
            parts, hits, ref, r, drift = {}, {}, None, {}, []
            for i, ds in dirs.items():
                parts[i], hs, per = [], [], []
                for p in ds:
                    pk = pickle.load(open(p / "test_preds.pkl", "rb"))
                    parts[i].append(det_parts(pk))
                    per.append(det_metrics(parts[i][-1], np.ones(len(pk["gts"]))))  # recomputed: tied scores made the saved values machine-dependent
                    t = json.loads((p / "result.json").read_text())["test"]
                    drift.append(max(abs(per[-1][0] - t["cpm"]), abs(per[-1][1] - t["map50"])))
                    hs.append(froc.recall_at_fp(pk["preds"], pk["gts"]).astype(float))
                    if ref is None:
                        ov = pk.get("overlap_cons") or pk["overlap"]
                        ref = dict(n=len(pk["gts"]), ov=np.concatenate([np.asarray(o if o else [np.nan] * len(gt[0]), float) for o, gt in zip(ov, pk["gts"])]),
                                   img=np.concatenate([np.full(len(gt[0]), k) for k, gt in enumerate(pk["gts"])]).astype(int), test_idx=np.asarray(pk["test_idx"]))
                    assert len(pk["gts"]) == ref["n"] and (np.asarray(pk["test_idx"]) == ref["test_idx"]).all(), f"{p}: test set differs"
                hits[i] = np.mean(hs, 0)
                per = np.array(per)
                r[i] = dict(cpm=float(per[:, 0].mean()), cpm_sd=float(per[:, 0].std(ddof=1)), map50=float(per[:, 1].mean()), map50_sd=float(per[:, 1].std(ddof=1)), seeds=len(per))
            print(f"H4'b {ds_dir.name}: largest change vs the values saved at training time {max(drift):.2g} (tied scores now enter together)", flush=True)
            rng = np.random.default_rng(0)
            W = np.stack([np.bincount(rng.integers(0, ref["n"], ref["n"]), minlength=ref["n"]) for _ in range(n_boot)]).astype(float)
            met = {i: np.array([[det_metrics(pp, w) for pp in parts[i]] for w in W]).mean(1) for i in parts}  # [boot, (cpm, map)], seed mean
            hi, lo = ref["ov"] >= 0.15, ref["ov"] < 0.15  # a box without an overlap value (NaN) is in neither group
            Wb = W[:, ref["img"]]
            for i in r:
                if i == "orig" or "orig" not in r:
                    continue
                for j, m in enumerate(("cpm", "map50")):
                    b = met[i][:, j] - met["orig"][:, j]
                    r[i][f"d_{m}"] = [r[i][m] - r["orig"][m], float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
                dh = hits[i] - hits["orig"]
                if hi.any() and lo.any():
                    boot = (Wb[:, hi] @ dh[hi]) / Wb[:, hi].sum(1) - (Wb[:, lo] @ dh[lo]) / Wb[:, lo].sum(1)
                    r[i]["did_recall2fp"] = dict(est=float(dh[hi].mean() - dh[lo].mean()), ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                                 n_hi=int(hi.sum()), n_lo=int(lo.sum()))
                else:
                    r[i]["did_recall2fp"] = dict(est=None, n_hi=int(hi.sum()), n_lo=int(lo.sum()),
                                                 note="not estimable: every test box is on one side of the 15% bone-overlap cut")
        res[ds_dir.name] = r
    out["H4b_downstream"] = res
    cls = [[ds, i, f"{v['macro_auc_ensemble']:.3f}", f"{v['macro_auc_seed_mean']:.3f} $\\pm$ {v['sd']:.3f}",
            ci_str(v["vs_orig"]["diff"], *v["vs_orig"]["ci95"]) if "vs_orig" in v else "-"] for ds, r in res.items() for i, v in r.items() if "sd" in v]
    if cls:
        tex(tables / "downstream_cls.tex", ["dataset", "input", "macro-AUROC (3-seed ensemble)", "per seed", "vs orig [95\\% CI]"], cls,
            "Downstream classification (SAP_v2 H4'b; patient-cluster bootstrap). Generated by tools/final_stats.py.")
    did = lambda x: ci_str(x["est"], *x["ci95"]) if x.get("est") is not None else "n/a"  # noqa: E731
    det = [[ds, i, f"{v['cpm']:.3f} $\\pm$ {v['cpm_sd']:.3f}", ci_str(*v["d_cpm"]) if "d_cpm" in v else "-", f"{v['map50']:.3f} $\\pm$ {v['map50_sd']:.3f}",
            ci_str(*v["d_map50"]) if "d_map50" in v else "-", did(v["did_recall2fp"]) if "did_recall2fp" in v else "-"]
           for ds, r in res.items() for i, v in r.items() if "cpm" in v]
    if det:
        tex(tables / "downstream_det.tex", ["dataset", "input", "CPM", "$\\Delta$CPM vs orig", "mAP@50", "$\\Delta$mAP vs orig", "DiD recall@2FP"], det,
            "Downstream detection, mean $\\pm$ SD over 3 seeds; differences with image bootstrap 95% CI (SAP_v2 H4'b). Generated by tools/final_stats.py.")


def rib(root, out, tables):
    d = root / "rib"
    files = {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))} if d.exists() else {}
    if not files:
        return print("rib benchmark: skipped")
    out["rib"] = {k: {c: float(v[c].mean()) for c in ("r_rib", "r_non", "selectivity", "auc")} for k, v in files.items()}
    tex(tables / "rib_benchmark.tex", ["method", "rib edge removed", "non-rib removed", "selectivity", "AUC"],
        [[k, *(f"{v[c]:.3f}" for c in ("r_rib", "r_non", "selectivity", "auc"))] for k, v in out["rib"].items()],
        "Rib benchmark on VinDr-RibCXR (metrics only).")


def synth_v1(root, out):
    """Replication of SAP.md H1 (v1 family) with the 300k-sample finals: synthetic LIDC test CTs, exact truth, clia vs recipe. The planned H1
    test (200k ablation runs, all lesion-aware arms) is results/ablation_test (tools/aggregate_ablation.py, 2026-10-27). Per-CT A (mean over
    sites and the 3 seeds), paired by CT, patient-cluster bootstrap and Wilcoxon; non-inferiority of per-CT bone MAE, margin 5% of the recipe mean."""
    d = root / "synth"
    arms = {a: sorted(d.glob(f"{a}_v1/*/per_site.csv")) for a in ("clia", "recipe")}
    if not all(arms.values()):
        return print("H1 (v1, synthetic): skipped")
    A = {a: pd.concat([pd.read_csv(f) for f in fs]).groupby("name").absorption.mean() for a, fs in arms.items()}
    M = {a: pd.concat([pd.read_csv(f.parent / "per_ct.csv") for f in fs]).groupby("name").mae_bone.mean() for a, fs in arms.items()}
    ka, km = A["clia"].index.intersection(A["recipe"].index), M["clia"].index.intersection(M["recipe"].index)
    pat = lambda idx: [n.split("_")[0] for n in idx]  # noqa: E731  LIDC-IDRI-xxxx
    da = (A["clia"][ka] - A["recipe"][ka]).values
    margin = 0.05 * float(M["recipe"][km].mean())
    res = dict(seeds={a: len(fs) for a, fs in arms.items()}, n_ct_A=len(ka), n_ct_mae=len(km), A_clia=float(A["clia"][ka].mean()), A_recipe=float(A["recipe"][ka].mean()),
               A_diff=list(cluster_bootstrap_ci(da, pat(ka))), A_p=paired_tests(A["clia"][ka].values, A["recipe"][ka].values)["wilcoxon_p"],
               mae_clia=float(M["clia"][km].mean()), mae_recipe=float(M["recipe"][km].mean()),
               mae_noninferior=non_inferior((M["recipe"][km] - M["clia"][km]).values, margin, pat(km)))
    res["passes"] = bool(res["A_diff"][0] < 0 and res["A_p"] < 0.05 and res["mae_noninferior"]["non_inferior"])
    out["H1_v1_synth"] = res


def h5_ci(h5):
    """95% CIs for H5' from the per-item values s4/eval_uncertainty.py stores (synth_items: [coverage, width] per test CT; real_items:
    [coverage, width, Spearman] per SZCH image). Older result files carry no items: no CI then. The synthetic Spearman and AUSE pool pixels
    over CTs and get no CI."""
    ci = {}
    for key, cols in (("synth_items", ("synth_coverage", "synth_width")), ("real_items", ("real_coverage", "real_width", "real_spearman"))):
        if h5.get(key):
            v = np.asarray(h5[key], float)
            for j, name in enumerate(cols):
                m, lo, hi = cluster_bootstrap_ci(v[:, j])
                assert abs(m - h5[name]) < 1e-9, f"{name}: items do not reproduce the stored mean"
                ci[name + "_ci95"] = [lo, hi]
    return ci


def extra_tables(out, tables):
    """Tables for results that so far existed only in stats_*.json (labels as in results.tex)."""
    if out.get("H2"):
        tex(tables / "h2_noninferiority.tex", ["ours vs", "metric", "mean difference", "one-sided 95\\% lower bound", "margin", "non-inferior"],
            [[t["ref"], t["metric"], f"{t['mean']:.3f}", f"{t['lower']:.3f}", f"$-${t['margin']}", "yes" if t["non_inferior"] else "no"] for t in out["H2"]],
            "H2' non-inferiority on SZCH held-out (image bootstrap).")
    rows, enc = [], []
    for s in ("ribcxr", "jsrt", "szch"):
        r = out.get(f"H3_{s}")
        if not isinstance(r, dict):
            continue
        for c in r["contrasts"]:
            it = c["overlap_interaction"]
            rows.append([s, f"ours $-$ {c['ref']}", ci_str(c["mean_diff"], *c["ci95"]), f"{c['p_holm']:.2g}", ci_str(it["est"], *it["ci95"]) if isinstance(it, dict) else "-"])
        for c in [c | {"twin": "ours"} for c in r["contrasts"] if c["ref"] == "notwin"] + r.get("encoder_pairs", []):  # ConvNeXt-V2-T, SwinV2-T, RAD-DINO
            it = c["overlap_interaction"]
            enc.append([s, f"{c['twin']} $-$ {c['ref']}", c["n_sites"], ci_str(c["mean_diff"], *c["ci95"]), ci_str(it["est"], *it["ci95"]) if isinstance(it, dict) else "-"])
    if rows:
        tex(tables / "audit_contrasts.tex", ["set", "contrast", "$\\Delta A$ [95\\% CI]", "Holm $p$", "interaction (>=15\\% vs <15\\% overlap)"], rows,
            "H3' paired site-level contrasts, image-cluster bootstrap; SZCH is development evidence.")
        tex(tables / "encoders.tex", ["set", "twin $-$ no twin", "sites", "$\\Delta A$ [95\\% CI]", "interaction"], enc,
            "Twin regulariser per encoder (exploratory): absorption difference to the matched no-twin control.")
    dr, dq = out.get("H3_ribcxr", {}).get("dose_response"), out.get("qp_dose_response")
    if isinstance(dr, dict):
        ws = sorted(set(map(int, dr["mean_A"])) | (set(map(int, dq["mean_A"])) if isinstance(dq, dict) else set()))
        rows = [[w, len(DOSE[w]), f"{dr['mean_A'].get(w, float('nan')):.3f}", f"{dq['mean_A'].get(w, float('nan')):.3f}" if isinstance(dq, dict) else "-"] for w in ws]
        rows.append(["slope on $\\log_2(1+w)$", "", ci_str(dr["slope"], *dr["ci95"], d=4), ci_str(dq["slope"], *dq["ci95"], d=4) if isinstance(dq, dict) else "-"])
        tex(tables / "dose_response.tex", ["twin weight $w$", "seeds", "$A$ VinDr-RibCXR", "$A$ Qure-projection test"], rows,
            "H1' dose-response (SwinV2-T family; per-site mean over seeds; image-cluster bootstrap).")
    h5 = out.get("H5")
    if isinstance(h5, dict):
        ci = lambda k, d=3: ci_str(h5[k], *h5[k + "_ci95"], d=d) if k + "_ci95" in h5 else f"{h5[k]:.{d}f}"  # noqa: E731
        tex(tables / "reliability.tex", ["set", "coverage (nominal 0.90)", "interval width", "Spearman(spread, error)"],
            [["Qure-projection test (exact truth)", ci("synth_coverage"), ci("synth_width", 4), ci("synth_spearman")],
             [f"SZCH held-out ({h5['n_real']}, under shift)", ci("real_coverage"), ci("real_width", 4), ci("real_spearman")]],
            "H5' ensemble spread and split-conformal intervals (K = 3).")
    # the v1 H1 table (planned test + this replication) is written by tools/paper_extras.py (v1_h1.tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "final"))
    ap.add_argument("--tables", default=str(ROOT / "paper" / "tables"))
    ap.add_argument("--subset", default="held", choices=["held", "dev", "all"])
    a = ap.parse_args()
    root, tables = Path(a.root), Path(a.tables)
    tables.mkdir(parents=True, exist_ok=True)
    keep = {"held": lambda n: not is_dev(n), "dev": is_dev, "all": lambda n: True}[a.subset]
    out = dict(subset=a.subset)
    szch(root, out, tables, keep)
    audits(root, out, tables, keep)
    qp(root, out)
    cnr(root, out, tables)
    downstream(root, out, tables)
    rib(root, out, tables)
    unc = root / "unc_heldout.json"
    out["H5"] = json.loads(unc.read_text()) if unc.exists() else "skipped"
    if isinstance(out["H5"], dict):
        out["H5"] = {k: v for k, v in out["H5"].items() if not k.endswith("_items")} | h5_ci(out["H5"])
    synth_v1(root, out)
    extra_tables(out, tables)
    (root / f"stats_{a.subset}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: (v if not isinstance(v, (dict, list)) else "...") for k, v in out.items()}, indent=1))
