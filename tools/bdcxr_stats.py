"""Statistics of SAP_bdcxr.md (E1-E3) from saved result files; writes results/bdcxr/stats.json and paper/tables/bdcxr_*.tex.

python tools/bdcxr_stats.py [--root results/bdcxr] [--meta PATH/metadata.csv]
root/down/<bdcxr_cls_<input>_s<seed>_f<k>>/ (s4.cls outputs), root/audit_bdcxr/<method>.csv, root/spread_{bdcxr,szch}.csv.
Every section is skipped (and says so) when its inputs are missing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from final_stats import audit_contrasts, boot_vs_ref, ci_str, macro_auc, tex  # noqa: E402
from s4.delong import delong_paired  # noqa: E402
from s4.stats import cluster_bootstrap_ci, holm  # noqa: E402
from szch_split import is_dev  # noqa: E402

INPUTS = ["orig", "soft", "soft_qure", "clahe", "orig-soft"]
MARGIN = 0.01  # non-inferiority margin, macro-AUROC (SAP_bdcxr.md E1)


def oof(d):
    """-> {input: P [N,3]} pooled out-of-fold, seed-averaged; Y [N,3]; group [N]; per-fold macro-AUROC {input: [5]}."""
    P, Y, G, folds = {}, None, None, {}
    for v in INPUTS:
        runs = sorted(d.glob(f"bdcxr_cls_{v}_s*_f*/result.json"))
        if not runs:
            continue
        by = {}
        for r in runs:
            by.setdefault(r.parent.name.rsplit("_", 1)[1], []).append(r.parent)
        assert len(by) == 5 and all(len(x) == 3 for x in by.values()), f"{v}: expected 5 folds x 3 seeds, got { {k: len(x) for k, x in by.items()} }"
        for f, dirs in sorted(by.items()):
            idx = np.load(dirs[0] / "test_idx.npy")
            assert all((np.load(x / "test_idx.npy") == idx).all() for x in dirs)
            pm = np.mean([np.load(x / "test_pred.npy").astype(np.float32) for x in dirs], 0)
            if Y is None:  # sized by the largest index: an image left out at prep time stays NaN and fails the check below
                n = max(int(np.load(x[0] / "test_idx.npy").max()) for x in by.values()) + 1
                Y, G = np.full((n, 3), np.nan, np.float32), np.empty(n, object)
            P.setdefault(v, np.full((len(Y), 3), np.nan, np.float32))[idx] = pm
            Y[idx], G[idx] = np.load(dirs[0] / "test_labels.npy"), np.load(dirs[0] / "test_patient.npy", allow_pickle=True)
            folds.setdefault(v, []).append(macro_auc(np.load(dirs[0] / "test_labels.npy"), pm))
    if P:
        assert not np.isnan(Y).any() and all(not np.isnan(p).any() for p in P.values()), "an image is missing from the out-of-fold predictions"
    return P, Y, G, folds


def ages(meta, ids):
    """Age in years per image (None if unknown); metadata names differ from file names in case and separators."""
    key = lambda s: re.sub(r"[-_ ]+", " ", s.strip().lower())  # noqa: E731
    m = pd.read_csv(meta, encoding="utf-8-sig")
    age = {key(f): a for f, a in zip(m.Filename, m.Age.astype(str).str.strip().str.replace("Y", "", regex=False))}
    out = [age.get(key(Path(i).stem)) for i in ids]
    print(f"age known for {sum(a is not None for a in out)} of {len(out)} images", flush=True)
    return np.array([0.5 if a == "<1" else float(a) if a else np.nan for a in out])


def e1(root, out, tables, meta, ids):
    P, Y, G, folds = oof(root / "down")
    if "orig" not in P:
        return print("E1: skipped (no orig runs)")
    yc = Y.argmax(1)
    res, bt = {}, boot_vs_ref(Y, P, "orig", G, n=2000)  # duplicate groups resampled, the same resamples for every input
    for v, p in P.items():
        pn = p[yc > 0]
        r = dict(macro_auc=macro_auc(Y, p), per_class=[float(roc_auc_score(Y[:, c], p[:, c])) for c in range(3)],
                 pneumonia_vs_normal=float(roc_auc_score(yc > 0, 1 - p[:, 0])),
                 bacterial_vs_viral=float(roc_auc_score(yc[yc > 0] == 1, pn[:, 1] / (pn[:, 1] + pn[:, 2] + 1e-9))),
                 balanced_acc=float(balanced_accuracy_score(yc, p.argmax(1))),
                 fold_mean=float(np.mean(folds[v])), fold_sd=float(np.std(folds[v], ddof=1)))
        if v != "orig":
            d, lo, hi = bt[v]
            dl = [delong_paired(Y[:, c], p[:, c], P["orig"][:, c]) for c in range(3)]
            r["vs_orig"] = dict(diff=d, ci95=[lo, hi], non_inferior=bool(lo > -MARGIN), superior=bool(lo > 0),
                                per_class_diff=[x["diff"] for x in dl], per_class_p_holm=holm([x["p"] for x in dl]))
        res[v] = r
    if meta:
        a = ages(meta, ids)
        for v, p in P.items():
            res[v]["age"] = {lab: dict(n=int(sel.sum()), macro_auc=macro_auc(Y[sel], p[sel])) for lab, sel in (("<1y", a < 1), (">=1y", a >= 1))}
    out["E1"] = res
    rows = [[v, f"{r['macro_auc']:.3f}", f"{r['fold_mean']:.3f} $\\pm$ {r['fold_sd']:.3f}",
             ci_str(r["vs_orig"]["diff"], *r["vs_orig"]["ci95"]) if "vs_orig" in r else "-", f"{r['pneumonia_vs_normal']:.3f}",
             f"{r['bacterial_vs_viral']:.3f}", f"{r['balanced_acc']:.3f}"] for v, r in res.items()]
    tex(tables / "bdcxr_cls.tex", ["input", "macro-AUROC (OOF)", "per fold", "vs orig [95\\% CI]", "pneu. vs normal", "bact. vs viral", "bal. acc."],
        rows, f"BDCXR-3257, 5-fold grouped CV x 3 seeds, n = {len(Y)} (SAP_bdcxr.md E1). Generated by tools/bdcxr_stats.py.")


def e2(root, out, tables):
    a = {p.stem: pd.read_csv(p) for p in sorted((root / "audit_bdcxr").glob("*.csv"))}
    if not a:
        return print("E2: skipped")
    assert set(a) == {"ours", "notwin", "qure", "identity"}, f"E2 needs all four methods, got {sorted(a)}"
    k = ["image", "site", "alpha", "overlap", "diam_mm"]
    for m, df in a.items():
        assert df[k].reset_index(drop=True).equals(a["ours"][k].reset_index(drop=True)), f"{m}: sites differ from ours"
    res = dict(n_images=int(a["ours"].image.nunique()), n_without_sites=300 - int(a["ours"].image.nunique()), n_sites=len(a["ours"]),
               mean_A={m: list(cluster_bootstrap_ci(df.absorption.values, df.image.values)) for m, df in a.items()},
               contrasts=[audit_contrasts(a, r) for r in ("qure", "notwin") if r in a])
    for c, ph in zip(res["contrasts"], holm([c["p_image_wilcoxon"] for c in res["contrasts"]])):
        c["p_holm"] = ph
    out["E2"] = res
    tex(tables / "bdcxr_audit.tex", ["method", "absorption $A$ [95\\% CI]"], [[m, ci_str(*v)] for m, v in res["mean_A"].items()],
        f"Lesion audit on {res['n_images']} BDCXR images, {res['n_sites']} sites (SAP_bdcxr.md E2; lower is better).")


def e3(root, out, ids, meta):
    f, s = root / "spread_bdcxr.csv", root / "spread_szch.csv"
    if not (f.exists() and s.exists()):
        return print("E3: skipped")
    b, z = pd.read_csv(f), pd.read_csv(s)
    z = z[~z.image.map(is_dev)]
    res = {n: dict(n=int(df.spread_lung.notna().sum()), median=float(df.spread_lung.median()), iqr=[float(x) for x in df.spread_lung.quantile([0.25, 0.75])],
                   lung_frac_median=float(df.lung_frac.median())) for n, df in (("bdcxr", b), ("szch_held", z))}
    if meta:
        a = ages(meta, b.image.tolist())
        ok = ~np.isnan(a) & b.spread_lung.notna().values
        res["bdcxr"]["spearman_age"] = float(spearmanr(a[ok], b.spread_lung.values[ok]).statistic)
    out["E3"] = res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "bdcxr"))
    ap.add_argument("--tables", default=str(ROOT / "paper" / "tables"))
    ap.add_argument("--meta", default=None, help="BDCXR metadata.csv (age subgroups)")
    a = ap.parse_args()
    root, tables, out = Path(a.root), Path(a.tables), {}
    tables.mkdir(parents=True, exist_ok=True)
    ids = json.loads((root / "ids.json").read_text()) if (root / "ids.json").exists() else None
    e1(root, out, tables, a.meta if ids else None, ids)
    e2(root, out, tables)
    e3(root, out, ids, a.meta)
    (root / "stats.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: "..." for k in out}, indent=1))
