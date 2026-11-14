"""All v3 figure suites (DESIGN_v3.md; directive section 4) -> results/figures/<figure>.pdf + .png, each built with tools/figstyle.py
at its final print size (88 / 140 / 180 mm wide, at most 225 mm high, all text >= 8 pt; PDF + 600-dpi PNG; fs.save refuses anything else).

python tools/generate_all_thesis_figures.py [--only pipeline,curves,confusion,uncertainty,xai,stats] [--boot 200] [--arch densenet121] [--with_v2]
Each suite draws what its inputs allow and prints what it skipped. Images in panels come only from LIDC-IDRI (CC BY 3.0), SZCH-X-Rays
(CC BY-NC 4.0) and NIH ChestX-ray14 (unrestricted with attribution); VinDr, VinBigData, Node21, JSRT and BDCXR-3257 appear only as curves
and numbers. Sources under --root (default: this repository) -> figures:
  pipeline     data/tcia_test + data/lidc_xml (+ data/szch) -> fig_preprocessing_pipeline, fig_lesion_insertion
  curves       the five configurations of --arch, results/v3/down/<data set>/<data set>_hyb_<config>@<arch>_s<seed>[_f<k>] (seed means;
               BDCXR folds out-of-fold), with --with_v2 also the v2 runs (results/final/down, results/bdcxr/down), drawn lighter and
               labelled (v2) -> fig_diagnostic_curves (NIH macro ROC, PR), fig_roc_<group> and fig_pr_<group> (a panel per class, groups of
               DESIGN_v3.md §3), fig_auroc_per_class, fig_bdcxr_roc; v2 detection runs (results/final/down/{node21,vin}) -> fig_froc
  confusion    the same runs -> fig_confusion_matrices (BDCXR-3257), fig_error_comovement (NIH)
  uncertainty  results/v3/uq_cls_<arch>.json (s4/uq_conformal.py cls3), else the v2 results/v3/uq_cls.json -> fig_selective_classification,
               fig_conformal_sets; results/v3/uq_bone.json -> fig_bone_uncertainty
  xai          results/v3/xai/nih_hyb_hybrid@<arch>_s<k>.maps.npz, lowest seed present -> fig_xai_comparison (Grad-CAM++ per stream,
               decoder attention), fig_xai_ig; all seeds' .json -> fig_pointing_game
  stats        results/final_hybrid_stats.json (tools/hybrid_stats.py) -> fig_hypothesis_forest (H6, H7, H8 with Holm p); with each run's
               result.json -> fig_seed_spread (single runs next to the seed ensembles that the tests compare)
  design       one NIH test image from the XAI maps above -> fig_study_design (inputs X, S, B and the five configurations)
A figure drawn from v3 runs is named <figure>_<arch> (<figure>_<arch>_with_v2 with v2 runs next to them); v2-only figures keep their names.
Bootstrap bands and intervals use --boot patient-cluster resamples (the statistics in results/final_hybrid_stats.json use 10,000).
"""
import argparse
import glob
import json
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import figstyle as fs  # noqa: E402
from hybrid_stats import FOCAL_BOX, GROUPS  # noqa: E402
from s4.froc import curve, match  # noqa: E402
from s4.prep import NIH  # noqa: E402
from s4.uq_conformal import load_runs  # noqa: E402

COL = {"orig": "#4a4a4a", "soft": "#2f6fd0", "orig-soft": "#e07b28", "early": "#e07b28", "hybrid": "#2a9d55", "ctrl": "#9b59b6",
       "clahe": "#999999", "soft_qure": "#c0392b", "both": "#e07b28"}
NAME = {"orig": "original", "soft": "soft tissue (ours)", "orig-soft": "early-fused (orig+soft)", "early": "early-fused [X,S,B]",
        "hybrid": "multi-stream hybrid", "ctrl": "control X×3", "both": "orig+soft", "clahe": "CLAHE", "soft_qure": "soft (reference)"}
V3 = ("orig", "soft", "early", "hybrid", "ctrl")  # s4/cls_hybrid.py configurations (SAP_v3.md)
V2 = ("orig", "soft", "orig-soft")  # s4/cls.py variants drawn with --with_v2, keyed 'v2:<variant>' so they never merge with a v3 name
COL |= {f"v2:{v}": to_hex(0.45 * np.array(to_rgb(COL[v])) + 0.55) for v in V2}  # the v3 hue mixed with 55 % white
NAME |= {f"v2:{v}": f"{NAME[v]} (v2)" for v in V2}
GROUP_NAME = {"focal": "focal / nodular", "diffuse": "diffuse / fluid", "boundary": "boundary / structural"}  # DESIGN_v3.md §3
BD = ["normal", "bacterial", "viral"]  # class order of s4/prep.py BDCXR
AB = ["Atel", "Card", "Effu", "Infi", "Mass", "Nodu", "Pna", "Ptx", "Cons", "Edem", "Emph", "Fibr", "PlTh", "Hern"]  # NIH order
STREAM = {"X": "#4a4a4a", "S": "#2f6fd0", "B": "#b8860b"}  # input streams (bone in the mask colour of the pipeline figure)
GRID = np.linspace(0, 1, 201)


def stem(a, name):
    """results/figures/<name> under --root (fs.save adds .pdf and .png)."""
    d = a.root / "results" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / name)


def cls_runs(a, ds):
    """{key: (pred, labels, patient)}, seed means, folds concatenated (BDCXR out-of-fold), rows in prep-index order: the five v3
    configurations of --arch (s4/cls_hybrid.py; key = configuration) and, with --with_v2, the v2 runs (s4/cls.py; key 'v2:<variant>').
    A run with other test rows than the largest run (unfinished folds) is left out, so all curves and contrasts share one image set."""
    v2 = a.root / ("results/bdcxr/down" if ds == "bdcxr" else f"results/final/down/{ds}")
    srcs = [("", f"@{a.arch}", V3, load_runs(a.root / "results/v3/down" / ds, f"{ds}_hyb_"))]
    if a.with_v2:
        srcs.append(("v2:", "", V2, load_runs(v2, f"{ds}_cls_")))
    out, rows = {}, {}
    for pre, suf, names, runs in srcs:
        for v in names:
            ks = sorted(k for k in runs if k[0] == v + suf)
            if ks:  # rows in prep-index order, so one set of bootstrap weights fits v2 and v3 runs alike
                idx = np.concatenate([runs[k][2] for k in ks])
                o = np.argsort(idx, kind="stable")
                out[pre + v] = tuple(np.concatenate([runs[k][i] for k in ks])[o] for i in (0, 1, 3))
                rows[pre + v] = idx[o]
    ref = max(rows.values(), key=len, default=None)
    for k in [k for k in rows if not np.array_equal(rows[k], ref)]:
        print(f"{ds}: {k} left out (its test rows differ from the other runs': unfinished folds?)")
        del out[k]
    return out


def suffix(a, keys):
    """'_<arch>' for a figure with v3 runs ('_<arch>_with_v2' when v2 runs are drawn next to them); v2-only figures keep their names."""
    v2 = [k.startswith("v2:") for k in keys]
    return "" if all(v2) else f"_{a.arch}" + ("_with_v2" if any(v2) else "")


def boot_weights(pat, n, seed=0):
    """Patient-cluster bootstrap as row weights (multinomial counts of the patients)."""
    codes, inv = np.unique(pat, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)))[inv]


def boot_curves(p, y, W):
    """Per class: ROC (sensitivity on the FPR GRID), precision on the recall GRID and AUROC; row 0 from the data, rows 1.. under the
    bootstrap weights W (the same W for every run, so differences between runs are paired) -> roc, pr [1+B, C, G], auroc [1+B, C]."""
    n, C = len(W) + 1, y.shape[1]
    roc, pr, auc = np.empty((n, C, len(GRID))), np.empty((n, C, len(GRID))), np.empty((n, C))
    for b, w in enumerate([None, *W]):
        for c in range(C):
            f, t, _ = roc_curve(y[:, c], p[:, c], sample_weight=w)
            q, r, _ = precision_recall_curve(y[:, c], p[:, c], sample_weight=w)
            roc[b, c], pr[b, c], auc[b, c] = np.interp(GRID, f, t), np.interp(GRID, r[::-1], q[::-1]), np.trapezoid(t, f)
    return roc, pr, auc


def band(x, cur, k, ls="-", label=None):
    """Curve cur[0] over GRID with the 95 % band of its bootstrap rows cur[1:]."""
    if len(cur) > 1:
        x.fill_between(GRID, *np.percentile(cur[1:], [2.5, 97.5], 0), color=COL[k], alpha=0.15, lw=0)
    x.plot(GRID, cur[0], ls, color=COL[k], label=label)


def panel_grid(n, row_mm=56):
    """n panels of about 45 mm (2-4 columns) and one free slot for the legend -> (fig, panel axes, legend axis)."""
    rows = -(-(n + 1) // 4)
    cols = -(-(n + 1) // rows)
    fig, ax = fs.figure({2: "single", 3: "onehalf", 4: "double"}[cols], height_mm=rows * row_mm, nrows=rows, ncols=cols, squeeze=False)
    ax = list(ax.ravel())
    for x in ax[n:]:
        x.axis("off")
    return fig, ax[:n], ax[n]


# ---- 1. pipeline -----------------------------------------------------------------------------------------------------
def pipeline(a):
    import torch
    from scipy.ndimage import binary_dilation
    from szch_split import is_dev

    from s4 import compose as C
    from s4 import seg
    from s4.ct import load_series
    from s4.nodule import xml_series_uid
    from s4.render import render_ct
    files = sorted(glob.glob(str(a.root / "data/tcia_test/series/*.dcm")))
    if not files:
        return print("pipeline: skipped (no CT in data/tcia_test/series)")
    ct = load_series(files)
    xmls = [p for p in glob.glob(str(a.root / "data/lidc_xml/**/*.xml"), recursive=True) if xml_series_uid(p) == ct["meta"]["series_uid"]]
    arr, meta = render_ct(ct, xmls)
    hu = ct["hu"]
    body = seg.body_mask(hu)
    bone = seg.bone_mask(hu, ct["spacing"]) & body
    lung = seg.lung_mask(hu, body, bone)
    z = int(np.argmax(lung.sum((1, 2))))  # axial slice with the most lung
    nod = np.zeros_like(arr["a_bone"])
    if meta["nodules"]:
        n0 = meta["nodules"][0]
        p = arr["nod0"]
        nod[n0["y0"]:n0["y0"] + p.shape[0], n0["x0"]:n0["x0"] + p.shape[1]] = p
    a4 = torch.from_numpy(np.stack([arr["a_bone"], arr["a_nl"], arr["a_lung"], (nod > 0.02).astype(np.float32)]))[None].float()
    phys, kv = C.Physics("cpu"), torch.tensor([C.KVPS.index(100)])
    o = C.compose(a4, kv, phys, gen=torch.Generator().manual_seed(0), detector=False)
    a4l = a4.clone()
    a4l[0, 2] += torch.from_numpy(nod)  # counterfactual: the nodule's areal density added to the lung channel (as compose.insert_nodules)
    ol = C.compose(a4l, kv, phys, gen=torch.Generator().manual_seed(0), detector=False)  # same random draws: only the lesion differs
    X, B, S = (o[k][0, 0].numpy() for k in ("X", "B", "S"))
    Xl = ol["X"][0, 0].numpy()
    n_img = Xl - X  # lesion signal in composite-image units; the bone component is unchanged by construction
    sl = np.clip((hu[z] + 1000) / 1400, 0, 1)
    rgb = np.dstack([sl] * 3) * 0.6
    rgb[bone[z]] = [0.95, 0.85, 0.2]
    rgb[lung[z]] = [0.3, 0.6, 1.0]
    rgb[body[z] & ~bone[z] & ~lung[z]] *= [1.6, 0.8, 0.8]
    panels = [(sl, "gray", "(a) CT slice\n(LIDC-IDRI test CT)"),
              (np.clip(rgb, 0, 1), None, "(b) masks: bone (yellow),\nlung (blue), soft (red)"),
              (arr["a_bone"], "magma", "(c) areal density, bone\n(g cm$^{-2}$)"),
              (arr["a_nl"], "magma", "(d) areal density,\nnon-lung soft tissue"),
              (arr["a_lung"], "magma", "(e) areal density, lung"), (X, "gray", "(f) composite X\n(100 kVp spectrum)"),
              (B, "gray", "(g) exact bone B"), (S, "gray", "(h) exact soft tissue\nS = X − B")]
    real = sorted(p for p in glob.glob(str(a.root / "data/szch/CXR/*.png")) if is_dev(Path(p).stem))
    if real:
        panels.append((plt.imread(real[0]), "gray", "(i) real radiograph\n(SZCH-X-Rays)"))
    fig, ax = fs.figure("onehalf", height_mm=175, nrows=3, ncols=3)
    for x, (img, cmap, t) in zip(ax.ravel(), panels):
        x.imshow(img, cmap=cmap)
        x.set_title(t)
    for x in ax.ravel():
        x.axis("off")
    fig.suptitle("CT decomposition and polychromatic projection")
    fs.save(fig, stem(a, "fig_preprocessing_pipeline"))
    if not meta["nodules"]:
        return print("pipeline: lesion insertion skipped (no LIDC nodule in this CT)")
    n0, p = meta["nodules"][0], arr["nod0"]
    cy, cx = n0["y0"] + p.shape[0] // 2, n0["x0"] + p.shape[1] // 2
    w = 40
    crop = lambda im: im[max(0, cy - w):cy + w, max(0, cx - w):cx + w]  # noqa: E731
    sup = binary_dilation(n_img > 0.02 * n_img.max(), iterations=3)
    fig, ax = fs.figure("onehalf", height_mm=60, ncols=3)
    for x, (img, cmap, t) in zip(ax, [(crop(n_img), "magma", f"(a) lesion signal n\n({n0['diam_mm']:.0f} mm LIDC nodule)"),
                                      (crop(Xl), "gray", "(b) X + n with support M\n(dilated 3 px)"),
                                      (crop(B), "gray", "(c) bone image under\nthe site (unchanged)")]):
        x.imshow(img, cmap=cmap)
        x.set_title(t)
        x.axis("off")
    for x, col in ((ax[1], "#2a9d55"), (ax[2], "#e07b28")):
        x.contour(crop(sup), levels=[0.5], colors=col, linewidths=0.8)
    fig.suptitle("Counterfactual lesion insertion")
    fs.save(fig, stem(a, "fig_lesion_insertion"))


# ---- 2. diagnostic curves --------------------------------------------------------------------------------------------
def curves(a):
    nih = cls_runs(a, "nih")
    if nih:
        sfx, (_, y, pat) = suffix(a, nih), next(iter(nih.values()))
        W = list(boot_weights(pat, a.boot))
        bc = {k: boot_curves(p, yk, W) for k, (p, yk, _) in nih.items()}
        diagnostic(a, nih, bc, sfx)
        for g, cols in GROUPS["nih"].items():
            for kind in ("roc", "pr"):
                what = "ROC" if kind == "roc" else "precision–recall"
                class_figure(a, bc, y, cols, [NIH[c].replace("_", " ") for c in cols], kind, f"fig_{kind}_{g}{sfx}",
                             f"NIH, {GROUP_NAME[g]} findings: {what}, 95 % patient-bootstrap bands")
        auroc_forest(a, bc, sfx)
    else:
        print(f"curves: NIH skipped (no NIH runs of {a.arch}; v2 runs need --with_v2)")
    bd = cls_runs(a, "bdcxr")
    if bd:
        W = list(boot_weights(next(iter(bd.values()))[2], a.boot))
        bc = {k: boot_curves(p, yk, W) for k, (p, yk, _) in bd.items()}
        class_figure(a, bc, None, range(3), BD, "roc", "fig_bdcxr_roc" + suffix(a, bd),
                     "BDCXR-3257 out-of-fold: one-vs-rest ROC, 95 % bootstrap bands over duplicate groups")
    else:
        print(f"curves: BDCXR skipped (no BDCXR runs of {a.arch}; v2 runs need --with_v2)")
    froc_fig(a)


def diagnostic(a, runs, bc, sfx):
    """NIH macro-average ROC and micro / macro precision-recall (macro: mean over the 14 classes, with its bootstrap band)."""
    fig, ax = fs.figure("onehalf", height_mm=95, ncols=2)
    for k, (p, y, _) in runs.items():
        roc, pr, _ = bc[k]
        band(ax[0], roc.mean(1), k, label=NAME[k])
        band(ax[1], pr.mean(1), k, "--")
        q, r, _ = precision_recall_curve(y.ravel(), p.ravel())
        ax[1].plot(r, q, color=COL[k])
    ax[0].plot([0, 1], [0, 1], ":", color="0.6", lw=0.6)
    ax[1].axhline(y.mean(), ls=":", color="0.5", lw=0.8)
    ax[1].legend([Line2D([], [], color="0.3", ls=s) for s in ("-", "--", ":")], ["micro", "macro", "prevalence"], loc="upper right")
    ax[0].set(title="macro-average ROC", xlabel="1 − specificity", ylabel="sensitivity")
    ax[1].set(title="precision–recall", xlabel="recall", ylabel="precision")
    for x, letter in zip(ax, "ab"):
        x.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        fs.panel_label(x, letter)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle(f"NIH ChestX-ray14 test split: 95 % patient-bootstrap bands ({a.boot} resamples)")
    fs.save(fig, stem(a, "fig_diagnostic_curves" + sfx))


def class_figure(a, bc, y, cols, names, kind, name, title):
    """One panel per class with every run's ROC (kind 'roc') or precision-recall curve ('pr') and its 95 % bootstrap band."""
    fig, ax, lx = panel_grid(len(cols))
    for x, c, cname in zip(ax, cols, names):
        for k in bc:
            band(x, bc[k][0 if kind == "roc" else 1][:, c], k, label=NAME[k])
        if kind == "roc":
            x.plot([0, 1], [0, 1], ":", color="0.6", lw=0.6, label="chance")
            x.set(xlabel="1 − specificity", ylabel="sensitivity")
        else:
            x.axhline(y[:, c].mean(), ls=":", color="0.6", lw=0.6, label="prevalence")
            x.set(xlabel="recall", ylabel="precision")
        x.set(title=cname, xlim=(0, 1), ylim=(0, 1), aspect="equal")
    lx.legend(*ax[0].get_legend_handles_labels(), loc="center")
    fig.suptitle(title)
    fs.save(fig, stem(a, name))


def auroc_forest(a, bc, sfx):
    """NIH AUROC per class and per taxonomy group (mean of its classes' AUROC), and each configuration minus the original one (v2 runs:
    minus the v2 original) with 95 % intervals from the paired patient-bootstrap resamples."""
    rows = [r for g, cols in GROUPS["nih"].items() for r in [(f"{g} (mean)", cols)] + [(NIH[c].replace("_", " "), [c]) for c in cols]]
    yy = np.arange(len(rows))
    fig, ax = fs.figure("onehalf", height_mm=4.5 * len(rows) + 45, ncols=2, sharey=True)
    for j, k in enumerate(bc):
        off = (j - (len(bc) - 1) / 2) * 0.7 / max(1, len(bc) - 1)
        m = np.stack([bc[k][2][:, cols].mean(1) for _, cols in rows], 1)  # [1 + resamples, rows]
        ax[0].plot(m[0], yy + off, "o", color=COL[k], label=NAME[k])
        ref = ("v2:" if k.startswith("v2:") else "") + "orig"
        if k != ref and ref in bc:
            d = m - np.stack([bc[ref][2][:, cols].mean(1) for _, cols in rows], 1)
            if len(d) > 1:  # segments, not error bars: a percentile interval need not contain the point estimate
                ax[1].hlines(yy + off, *np.percentile(d[1:], [2.5, 97.5], 0), color=COL[k], lw=0.8)
            ax[1].plot(d[0], yy + off, "o", color=COL[k])
    ax[0].set_yticks(yy, [r[0] for r in rows])
    for t, (label, _) in zip(ax[0].get_yticklabels(), rows):
        t.set_fontweight("bold" if label.endswith("(mean)") else "normal")
    ax[0].invert_yaxis()
    ax[0].locator_params(axis="x", nbins=4)  # AUROC labels such as 0.9875 would overlap at 8 pt
    ax[1].axvline(0, color="0.6", lw=0.6, ls=":")
    ax[0].set(title="per class and group", xlabel="AUROC")
    ax[1].set(title=f"minus original ({a.boot} resamples)", xlabel="AUROC difference")
    for x, letter in zip(ax, "ab"):
        fs.panel_label(x, letter)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("NIH ChestX-ray14: AUROC per class and finding group")
    fs.save(fig, stem(a, "fig_auroc_per_class" + sfx))


def froc_fig(a):
    """FROC of the v2 detection inputs (Node21, VinBigData: curves only)."""
    fps = np.logspace(-2, np.log10(8), 60)
    found = {(ds, v): sorted((a.root / "results/final/down" / ds).glob(f"{ds}_det_{v}_s*/test_preds.pkl"))
             for ds in ("node21", "vin") for v in ("orig", "soft", "both")}
    if not any(found.values()):
        return print("curves: FROC skipped (no detection runs in results/final/down/node21, vin)")
    fig, ax = fs.figure("onehalf", height_mm=80, ncols=2)
    for x, ds, letter in zip(ax, ("node21", "vin"), "ab"):
        for v in ("orig", "soft", "both"):
            sens = []
            for f in found[ds, v]:
                d = pickle.loads(Path(f).read_bytes())
                s, tp, _ = match(d["preds"], d["gts"], 0.3)
                _, tpc, fpc = curve(s, tp)
                n_gt = sum(len(g[0]) for g in d["gts"])
                sens.append(np.interp(fps, fpc / len(d["gts"]), tpc / n_gt))
            if sens:
                x.plot(fps, np.mean(sens, 0), color=COL[v], label=f"{NAME[v]} ({len(sens)} seeds)")
        x.set(xscale="log", xlabel="false positives per image", ylabel="sensitivity",
              title={"node21": "Node21 nodules", "vin": "VinBigData findings"}[ds] + " (FROC, IoU 0.3)")
        fs.panel_label(x, letter)
    h = {lab: hd for x in ax for hd, lab in zip(*x.get_legend_handles_labels())}  # one entry per label across both panels
    fig.legend(h.values(), h.keys(), loc="outside lower center", ncols=2)
    fs.save(fig, stem(a, "fig_froc"))


# ---- 3. confusion ------------------------------------------------------------------------------------------------------
def confusion(a):
    runs = cls_runs(a, "bdcxr")
    if runs:
        n = len(runs)
        cols = n if n <= 5 else 4
        nrow = -(-n // cols)
        width = {1: "single", 2: "single", 3: "onehalf", 4: "double", 5: "double"}[cols]
        fig, ax = fs.figure(width, height_mm=58 * nrow, nrows=nrow, ncols=cols, squeeze=False)
        for x, v in zip(ax.ravel(), runs):
            P, Y = runs[v][0], runs[v][1].argmax(1)
            cm = np.array([[np.mean(P[Y == i].argmax(1) == j) for j in range(3)] for i in range(3)])
            x.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    x.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="w" if cm[i, j] > 0.5 else "k")
            first = x.get_subplotspec().is_first_col()
            x.set_xticks(range(3), BD, rotation=45, ha="right", rotation_mode="anchor")
            x.set_yticks(range(3), BD if first else [])
            x.set(title=textwrap.fill(NAME[v], 20), xlabel="predicted", ylabel="true" if first else "")
        for x in ax.ravel()[n:]:
            x.axis("off")
        fig.suptitle("BDCXR-3257: normalised confusion matrices (out-of-fold, seed means)")
        fs.save(fig, stem(a, "fig_confusion_matrices" + suffix(a, runs)))
    else:
        print(f"confusion: BDCXR skipped (no BDCXR runs of {a.arch}; v2 runs need --with_v2)")
    error_comovement(a)


def error_comovement(a):
    """NIH: correlation between classes of the per-image change of the absolute error (soft or hybrid minus original)."""
    nih = cls_runs(a, "nih")
    pre = next((p for p in ("", "v2:") if p + "orig" in nih and p + "soft" in nih), None)  # the v3 runs of --arch first
    pairs = [] if pre is None else [pre + v for v in ("soft", "hybrid") if pre + v in nih]
    if not pairs:
        return print("confusion: NIH error co-movement skipped (needs the orig and soft runs)")
    one = len(pairs) == 1
    fig, ax = fs.figure("single" if one else "onehalf", height_mm=86 if one else 80, ncols=len(pairs), squeeze=False)
    po, y, _ = nih[pre + "orig"]
    for x, v in zip(ax[0], pairs):
        r = np.corrcoef((np.abs(nih[v][0] - y) - np.abs(po - y)).T)
        np.fill_diagonal(r, np.nan)  # the diagonal is 1 by definition
        im = x.imshow(r, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        x.set_xticks(range(14), AB, rotation=90)
        x.set_yticks(range(14), AB if x is ax[0, 0] else [])
        x.set_title(f"{NAME[v]}\nminus {NAME[pre + 'orig']}")
    fig.colorbar(im, ax=list(ax[0]), label="correlation", shrink=0.8)
    fig.suptitle("NIH: co-movement of error changes")
    fs.save(fig, stem(a, "fig_error_comovement" + suffix(a, [pre + "orig", *pairs])))


# ---- 4. uncertainty ----------------------------------------------------------------------------------------------------
def uncertainty(a):
    uc = a.root / f"results/v3/uq_cls_{a.arch}.json"
    v3 = uc.exists()
    if not v3:
        uc = a.root / "results/v3/uq_cls.json"  # v2 predictions (s4/uq_conformal.py cls)
    if uc.exists():
        cls_uncertainty(a, json.loads(uc.read_text()), f"_{a.arch}" if v3 else "")
    else:
        print("uncertainty: classification skipped (run s4/uq_conformal.py cls3 first)")
    ub = a.root / "results/v3/uq_bone.json"
    if ub.exists():
        bone_uncertainty(a, json.loads(ub.read_text()))
    else:
        print("uncertainty: bone intervals skipped (run s4/uq_conformal.py bone first)")


def cls_uncertainty(a, c, sfx):
    """Selective classification (risk-coverage with 1 - ensemble entropy; dashed: score B with the suppressor spread) and conformal sets."""
    cur = {"bdcxr": {}, "nih": {}}
    for k, v in c.items():
        ds, _, var = k.partition("_")
        if ds in cur and var in COL and var != "clahe":  # cls3's metadata keys (arch, alpha, missing, ...) are no configurations
            cur[ds][var] = v
    alpha = c.get("alpha", 0.1)  # the v2 file predates the key; s4/uq_conformal.py's default
    fig, ax = fs.figure("onehalf", height_mm=110, nrows=2, ncols=2, gridspec_kw=dict(height_ratios=[3, 1.5]))
    for x, lx, ds, letter in zip(ax[0], ax[1], cur, "ab"):
        lx.axis("off")
        if not cur[ds]:
            x.axis("off")
            continue
        rk, ak = ("risk_coverage", "aurc") if ds == "bdcxr" else ("risk_coverage_brier", "aurc_brier")
        for var, v in cur[ds].items():
            rc = np.array(v[rk])
            x.plot(rc[:, 0], rc[:, 1], color=COL[var], label=f"{NAME[var]} ({v[ak]:.3f})")
            if isinstance(v.get("score_b"), dict):  # a string says why score B is unavailable
                rb = np.array(v["score_b"]["risk_coverage"])
                x.plot(rb[:, 0], rb[:, 1], "--", color=COL[var])
        if any(isinstance(v.get("score_b"), dict) for v in cur[ds].values()):
            x.plot([], [], "--", color="0.4", label="dashed: + suppressor spread")
        x.set(title={"bdcxr": "BDCXR-3257", "nih": "NIH ChestX-ray14"}[ds], xlabel="coverage (most confident kept)",
              ylabel={"bdcxr": "top-1 error", "nih": "Brier score"}[ds])
        lx.legend(*x.get_legend_handles_labels(), loc="upper center", title="configuration (AURC)")
        fs.panel_label(x, letter)
    fig.suptitle("Selective classification: risk–coverage with 1 − ensemble entropy")
    fs.save(fig, stem(a, "fig_selective_classification" + sfx))

    b, n = cur["bdcxr"], cur["nih"]
    fig, ax = fs.figure("double", height_mm=80, ncols=4)
    if b:
        wd = 0.8 / len(b)
        for j, (var, v) in enumerate(b.items()):
            ax[0].bar(np.arange(3) + (j - (len(b) - 1) / 2) * wd, v["coverage_per_class"], wd, color=COL[var])
        ax[0].set_xticks(range(3), BD, rotation=30, ha="right", rotation_mode="anchor")
        ax[0].set(title="BDCXR-3257\ncoverage by class", ylabel="coverage", ylim=(0, 1))
    for x, d, key, ttl, yl in ((ax[1], b, "mean_set_size", "BDCXR-3257\nmean set size", "classes per image"),
                               (ax[2], n, "fnr_test", "NIH\nfalse-negative rate", "FNR (test split)"),
                               (ax[3], n, "mean_set_size", "NIH\nmean set size", "classes per image")):
        if d:
            x.bar(range(len(d)), [v[key] for v in d.values()], color=[COL[var] for var in d])
            x.set(xticks=[], title=ttl, ylabel=yl)
    for x, d, level in ((ax[0], b, 1 - alpha), (ax[1], b, None), (ax[2], n, alpha), (ax[3], n, None)):
        if not d:
            x.axis("off")
        elif level is not None:
            x.axhline(level, ls="--", color="0.3", lw=0.8)
    for x, letter in zip(ax, "abcd"):
        fs.panel_label(x, letter)
    keys = list(dict.fromkeys([*b, *n]))
    fig.legend(handles=[Patch(color=COL[v], label=NAME[v]) for v in keys], loc="outside lower center", ncols=3)
    fig.suptitle(f"Conformal prediction at nominal {1 - alpha:.2f}: Mondrian sets (BDCXR-3257), false-negative risk control (NIH)")
    fs.save(fig, stem(a, "fig_conformal_sets" + sfx))


def bone_uncertainty(a, b):
    """Bone-image intervals on the held-out SZCH pairs (SAP_v3.md H9, nominal 0.90): per-image coverage of each calibration, reliability of
    the recalibrated intervals, sparsification of the ensemble spread."""
    fig, ax = fs.figure("double", height_mm=85, ncols=3)
    meth = [(f"{m.replace('mondrian', 'Mondrian')}, {t.replace('image', 'per image')}", b[m]["test"][t]["per_image"])
            for m in ("global", "mondrian") for t in ("pooled", "image")]
    ax[0].boxplot([v for _, v in meth], orientation="horizontal", tick_labels=[m for m, _ in meth], widths=0.5,
                  medianprops=dict(color="k"), flierprops=dict(markersize=2))
    ax[0].invert_yaxis()
    ax[0].axvline(0.9, ls="--", color="0.4", lw=0.8)
    ax[0].set(title="per-image coverage", xlabel="pixel coverage of an image")
    rel = np.array(b["reliability"])
    ax[1].plot([0.45, 1], [0.45, 1], ":", color="0.6")
    ax[1].plot(rel[:, 0], rel[:, 1], "o-", color=COL["hybrid"], label=f"Mondrian, {b['n_cal']} real pairs")
    unc = a.root / "results/final/unc_heldout.json"
    if unc.exists():
        u = json.loads(unc.read_text())
        ax[1].plot([1 - u["alpha"]], [u["real_coverage"]], "x", color=COL["soft_qure"], ms=6, label="synthetic calibration (v2)")
    ax[1].set(title="reliability", xlabel="nominal coverage", ylabel="held-out pixel coverage", xlim=(0.45, 1), ylim=(0, 1))
    sp, orc = np.array(b["held_sparsification"]), np.array(b["held_oracle"])
    ax[2].plot(sp[:, 0], sp[:, 1], color=COL["soft"], label="ensemble spread")
    ax[2].plot(orc[:, 0], orc[:, 1], "--", color="0.4", label="oracle (error-sorted)")
    ax[2].set(title=f"sparsification\n(AUSE {b['held_ause']:.4f})", xlabel="fraction of pixels removed", ylabel="mean |bone error|")
    for x, letter in zip(ax, "abc"):
        fs.panel_label(x, letter)
    h = [x.get_legend_handles_labels() for x in ax[1:]]  # below the panels: inside, (b) and (c) have no empty corner
    fig.legend(h[0][0] + h[1][0], h[0][1] + h[1][1], loc="outside lower center", ncols=2)
    fig.suptitle(f"Bone-image intervals on the {b['n_test']} held-out SZCH pairs (calibrated on {b['n_cal']} development pairs)")
    fs.save(fig, stem(a, "fig_bone_uncertainty"))


# ---- 5. XAI ------------------------------------------------------------------------------------------------------------
def xai(a):
    d = a.root / "results/v3/xai"
    files = sorted(d.glob(f"nih_hyb_hybrid@{a.arch}_s*.maps.npz"))  # NIH maps only (licences: module docstring); lowest seed present
    z = np.load(files[0]) if files else None
    keys = sorted({k.split("__")[0] for k in z.files})[: a.xai_cases] if files else []
    if keys:
        img = [(None, 0, "radiograph X"), (None, 1, "soft tissue S"), (None, 2, "bone B")]
        cam = [(f"cam_{i}", i, f"Grad-CAM++\n{s} stream") for i, s in enumerate("XSB")] + [("attn", 0, "decoder\nattention")]
        maps_figure(a, z, keys, img + cam, f"fig_xai_comparison_{a.arch}",
                    "Multi-stream attributions on NIH ChestX-ray14 test images (boxes: expert annotations)")
        maps_figure(a, z, keys, img + [(f"ig_{s}", i, f"|IG| on {s}") for i, s in enumerate("XSB")], f"fig_xai_ig_{a.arch}",
                    "Integrated Gradients per input stream on NIH ChestX-ray14 test images")
    else:
        print(f"xai: maps skipped (run s4/xai_multistream.py on a trained {a.arch} hybrid model first)")
    js = sorted(d.glob(f"nih_hyb_hybrid@{a.arch}_s*.json"))
    if js:
        pointing_figure(a, js)
    else:
        print(f"xai: pointing game skipped (no s4/xai_multistream.py output of a {a.arch} hybrid model)")


def maps_figure(a, z, keys, panels, name, title):
    """Rows: NIH test targets (image, box class); columns: (heat map or None, image channel X/S/B, title); expert boxes on the heat maps.
    Square panels across the 180-mm width; more rows than fit the page height go to further figures <name>_2, ..."""
    panels = [p for p in panels if p[0] is None or f"{keys[0]}__{p[0]}" in z.files]
    side = fs.WIDTH_MM["double"] / len(panels)
    parts = -(-len(keys) // int((fs.MAX_HEIGHT_MM - 20) // side))
    per = -(-len(keys) // parts)
    for part in range(parts):
        ks = keys[part * per:(part + 1) * per]
        fig, ax = fs.figure("double", height_mm=len(ks) * side + 20, nrows=len(ks), ncols=len(panels), squeeze=False)
        for r, k in enumerate(ks):
            x, bx = z[f"{k}__x"], z[f"{k}__boxes"]
            for j, (m, ch, t) in enumerate(panels):
                v = ax[r, j]
                v.imshow(x[ch], cmap="gray")
                if m is not None:
                    h = z[f"{k}__{m}"]
                    v.imshow(h / (h.max() + 1e-12), cmap="jet", alpha=0.4)
                    for x0, y0, x1, y1 in bx:
                        v.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="w", lw=0.8))
                v.set(xticks=[], yticks=[])
                v.spines[:].set_visible(False)
                if r == 0:
                    v.set_title(t)
            ax[r, 0].set_ylabel(NIH[int(k.split("_")[1])].replace("_", " "))
        fig.suptitle(title)
        fs.save(fig, stem(a, name + (f"_{part + 1}" if part else "")))


def pointing_figure(a, files):
    """Per NIH box class and per group (focal: Mass, Nodule; the rest diffuse/boundary): pointing-game hit rate of each map, decoder routing
    rho and Integrated-Gradients share per stream, means over all targets of all seeds' hybrid models (s4/xai_multistream.py rows)."""
    res = [json.loads(f.read_text()) for f in files]
    rows = [r for d in res for r in d["rows"]]
    if not rows:
        return print("xai: pointing game skipped (no targets)")
    cls = [c for c in NIH if any(r["cls"] == c for r in rows)]
    lines = []
    for g, m in (("focal", [c for c in cls if c in FOCAL_BOX]), ("diffuse/boundary", [c for c in cls if c not in FOCAL_BOX])):
        if m:
            lines += [(f"{g} (mean)", m)] + [(c.replace("_", " "), [c]) for c in m]
    sel = [[r for r in rows if r["cls"] in m] for _, m in lines]
    maps = {"cam_0": ("X stream", STREAM["X"]), "cam_1": ("S stream", STREAM["S"]), "cam_2": ("B stream", STREAM["B"]),
            "attn": ("decoder attention", COL["hybrid"]), "ig_all": ("Integrated Gradients (sum)", COL["ctrl"])}
    maps = {m: v for m, v in maps.items() if m in rows[0]["hits"]}
    yy = np.arange(len(lines))
    fig, ax = fs.figure("double", height_mm=7 * len(lines) + 45, ncols=3, sharey=True)
    h = 0.8 / len(maps)
    for j, (m, (label, col)) in enumerate(maps.items()):
        ax[0].barh(yy + (j - (len(maps) - 1) / 2) * h, [np.mean([r["hits"][m] for r in s]) for s in sel], h, color=col, label=label)
    for x, key in ((ax[1], "rho"), (ax[2], "ig_share")):
        v = np.array([np.mean([r[key] for r in s], 0) for s in sel])  # [lines, streams], each row sums to 1
        left = np.zeros(len(lines))
        for s, col in zip(range(v.shape[1]), STREAM.values()):
            x.barh(yy, v[:, s], 0.6, left=left, color=col)
            left += v[:, s]
    ax[0].set_yticks(yy, [f"{g} [{len({(r['image'], r['cls']) for r in s})}]" for (g, _), s in zip(lines, sel)])
    for t, (g, _) in zip(ax[0].get_yticklabels(), lines):
        t.set_fontweight("bold" if g.endswith("(mean)") else "normal")
    ax[0].invert_yaxis()
    ax[0].set(title="pointing-game hit rate", xlabel="fraction of targets", xlim=(0, 1))
    ax[1].set(title="decoder routing ρ", xlabel="share per stream", xlim=(0, 1))
    ax[2].set(title="Integrated Gradients", xlabel="share per input", xlim=(0, 1))
    for x, letter in zip(ax, "abc"):
        fs.panel_label(x, letter)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    fig.suptitle(f"Pointing game on the NIH expert boxes (tolerance {res[0]['tol']:g} px; [targets]), {len(files)} hybrid "
                 f"model{'s' if len(files) > 1 else ''}")
    fs.save(fig, stem(a, f"fig_pointing_game_{a.arch}"))


# ---- 6. study design ---------------------------------------------------------------------------------------------------
def design(a):
    """One NIH test image as the three inputs (taken from the XAI maps, a focal target if there is one) and the five configurations of
    s4/cls_hybrid.py with the contrasts of SAP_v3.md; drawn in millimetres on one axes."""
    files = sorted((a.root / "results/v3/xai").glob(f"nih_hyb_hybrid@{a.arch}_s*.maps.npz"))
    if not files:
        return print(f"design: skipped (no {a.arch} XAI maps for the example image)")
    z = np.load(files[0])
    keys = sorted({k.split("__")[0] for k in z.files})
    k = next((k for k in keys if NIH[int(k.split("_")[1])] in FOCAL_BOX), keys[0])
    x, bx = z[f"{k}__x"], z[f"{k}__boxes"]
    fig, ax = fs.figure("double", height_mm=105)
    th, s = 26, 26 / x.shape[-1]  # thumbnail side (mm), mm per pixel
    for i, (title, col) in enumerate((("radiograph X", "X"), ("soft tissue S", "S"), (r"bone $\hat{B}$ = X − S", "B"))):
        y0 = 70 - 33 * i
        ax.imshow(x[i], cmap="gray", extent=(4, 4 + th, y0, y0 + th))
        for x0, y1, x1, y2 in bx:  # pixel rows run downwards
            ax.add_patch(plt.Rectangle((4 + x0 * s, y0 + th - y2 * s), (x1 - x0) * s, (y2 - y1) * s, fill=False, ec="w", lw=0.6))
        ax.text(4 + th / 2, y0 + th + 1, title, ha="center", va="bottom", color=STREAM[col])
    arrow = dict(arrowstyle="-|>", lw=0.7, color="0.3", shrinkA=0, shrinkB=0)
    rows = (("orig", "original", ["X"]), ("soft", "soft tissue", ["S"]), ("early", "early-fused", ["XSB"]),
            ("hybrid", "multi-stream hybrid", ["X", "S", "B"]), ("ctrl", "control X×3", ["X", "X", "X"]))
    for r, (cfg, label, chips) in enumerate(rows):
        yc = 90 - 17 * r
        ax.text(40, yc, label, ha="left", va="center", color=COL[cfg], fontweight="bold")
        ys = [yc] if len(chips) == 1 else [yc + 5.5, yc, yc - 5.5]
        for c, y in zip(chips, ys):
            if c == "XSB":  # channel stacking: one RGB input
                for j, cc in enumerate("XSB"):
                    ax.add_patch(plt.Rectangle((72 + 2 * j, y - 3), 2, 6, color=STREAM[cc]))
            else:
                w = 6 if len(chips) == 1 else 4.5
                ax.add_patch(plt.Rectangle((75 - w / 2, y - w / 2), w, w, color=STREAM[c]))
                ax.text(75, y, r"$\hat{B}$" if c == "B" else c, ha="center", va="center", color="w")
            ax.annotate("", xy=(88, yc + (y - yc) * 0.4), xytext=(79, y), arrowprops=arrow)
        multi = len(chips) > 1
        for x0, w, text in ((88, 26, "DenseNet-121\nshared, × 3" if multi else "DenseNet-121"),
                            (122, 30, "decoder over\n" + ("3 × 64 tokens" if multi else "64 tokens"))):
            ax.add_patch(plt.Rectangle((x0, yc - 5), w, 10, fc="0.95", ec="0.4", lw=0.6))
            ax.text(x0 + w / 2, yc, text, ha="center", va="center")
        ax.annotate("", xy=(122, yc), xytext=(114, yc), arrowprops=arrow)
        ax.annotate("", xy=(160, yc), xytext=(152, yc), arrowprops=arrow)
        ax.text(161, yc, "class\nlogits", ha="left", va="center")
    ax.text(40, 3, "H6 (primary): hybrid − original.   H7: hybrid − control, hybrid − early-fused.\n"
                   "H8: soft and hybrid − original per finding group.   H10: S- vs X-stream maps of the hybrid.", ha="left", va="bottom")
    for x0, letter in ((0, "a"), (36, "b")):
        ax.text(x0, 101, letter, fontsize=10, fontweight="bold", ha="left", va="bottom")
    ax.set(xlim=(0, 180), ylim=(0, 106), aspect="equal")
    ax.axis("off")
    fs.save(fig, stem(a, "fig_study_design"))


DS_NAME ={"nih": "NIH ChestX-ray14", "vin": "VinBigData", "bdcxr": "BDCXR-3257"}


def stats(a):
    f = a.root / "results/final_hybrid_stats.json"
    e = json.loads(f.read_text())["encoders"].get(a.arch) if f.exists() else None
    if not e:
        return print(f"stats: skipped (no {a.arch} in {f})")
    hypothesis_forest(a, e)
    seed_spread(a, e)


def hypothesis_forest(a, e):
    """H6/H7 macro-AUROC and H8 group-mean AUROC differences with their 95 % cluster-bootstrap intervals; filled marker: Holm p < 0.05."""
    h67 = {"H6": "orig", "H7_ctrl": "ctrl", "H7_early": "early"}
    top = [r for ds in DS_NAME if ds in e for r in [(DS_NAME[ds], None, None)]
           + [(f"hybrid − {NAME[c]} ({h[:2]})", e[ds][h]["auroc"], COL[c]) for h, c in h67.items() if h in e[ds]]]
    h8 = [(ds, g, [k for k in e[ds].get("H8_groups", {}) if k.startswith(g + ":")]) for ds in DS_NAME if ds in e for g in GROUP_NAME]
    low = [r for ds, g, keys in h8 if keys for r in [(f"{DS_NAME[ds]}: {GROUP_NAME[g]}", None, None)]
           + [(f"{NAME[k[len(g) + 1:].split('-')[0]]} − original", e[ds]["H8_groups"][k], COL[k[len(g) + 1:].split("-")[0]]) for k in keys]]
    panels = [p for p in ((top, "H6 (primary) and H7", "macro-AUROC difference, hybrid − comparator"),
                          (low, "H8: finding groups", "difference in the mean AUROC of the group's classes")) if p[0]]
    fig, ax = fs.figure("onehalf", height_mm=4.2 * sum(len(p[0]) + 1 for p in panels) + 45, nrows=len(panels), squeeze=False,
                        gridspec_kw={"height_ratios": [len(p[0]) + 1 for p in panels]})
    for x, (rows, title, xlabel), letter in zip(ax[:, 0], panels, "ab"):
        tr = x.get_yaxis_transform()  # x in axes fractions, y in rows: the Holm p column right of the axes
        for i, (_, c, col) in enumerate(rows):
            if c is not None:
                x.hlines(i, c["lo"], c["hi"], color=col, lw=0.8)
                x.plot(c["diff"], i, "o", color=col, mfc=col if c["p_holm"] < 0.05 else "white")
                x.text(1.02, i, f"{c['p_holm']:.3f}" if c["p_holm"] >= 0.001 else "<0.001", transform=tr, va="center")
        x.text(1.02, -1, "Holm p", transform=tr, va="center", fontweight="bold")
        x.set_yticks(range(len(rows)), [r[0] for r in rows])
        for t, r in zip(x.get_yticklabels(), rows):
            t.set_fontweight("bold" if r[1] is None else "normal")
        x.set_ylim(len(rows) - 0.5, -1.5)  # top to bottom, with room for the column header
        x.tick_params(axis="y", length=0)
        x.locator_params(axis="x", nbins=4)  # differences such as -0.0075 would overlap at 8 pt
        x.axvline(0, color="0.6", lw=0.6, ls=":")
        x.set(title=title, xlabel=xlabel)
        fs.panel_label(x, letter)
    fig.legend([Line2D([], [], marker="o", ls="", color="0.3", mfc=m) for m in ("0.3", "white")],
               ["Holm p < 0.05", "not significant after Holm"], loc="outside lower center", ncols=2)
    fig.suptitle(f"Hypothesis tests: differences with 95 % cluster-bootstrap intervals ({next(iter(e.values()))['n_boot']:,} resamples)")
    fs.save(fig, stem(a, f"fig_hypothesis_forest_{a.arch}"))


def seed_spread(a, e):
    """Each run's test macro-AUROC (its result.json; BDCXR-3257: on its own test fold) and the seed ensemble that the tests compare."""
    dss = [ds for ds in DS_NAME if ds in e]
    fig, ax = fs.figure("double", height_mm=75, ncols=len(dss), squeeze=False)
    for x, ds, letter in zip(ax[0], dss, "abc"):
        for j, c in enumerate(V3):
            v = [json.loads(f.read_text())["test_macro_auc"] for f in sorted((a.root / "results/v3/down" / ds).glob(f"{ds}_hyb_{c}@{a.arch}_s*/result.json"))]
            x.plot(j + np.linspace(-0.25, 0.25, len(v)), v, "o", ls="", ms=2.5, color=COL[c], mfc="white")
            if c in e[ds]["macro_auroc"]:
                x.plot(j, e[ds]["macro_auroc"][c], "D", ms=5, color=COL[c], label=NAME[c])
        x.set_xticks([])
        x.set(title=DS_NAME[ds], ylabel="test macro-AUROC" if letter == "a" else None)
        fs.panel_label(x, letter)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h + [Line2D([], [], marker=m, ls="", color="0.3", mfc=f, ms=s) for m, f, s in (("o", "white", 2.5), ("D", "0.3", 5))],
               lab + ["single run (BDCXR-3257: its test fold)", "seed ensemble (tested)"], loc="outside lower center", ncols=4)
    fig.suptitle("Test macro-AUROC of every run and of the seed ensembles")
    fs.save(fig, stem(a, f"fig_seed_spread_{a.arch}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="pipeline,curves,confusion,uncertainty,xai,stats,design")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--xai_cases", type=int, default=6)
    ap.add_argument("--arch", default="densenet121", help="v3 encoder as in the run tags (s4/runtag.py), e.g. densenet121, rad-dino")
    ap.add_argument("--with_v2", action="store_true", help="also draw the v2 runs (s4/cls.py) for comparison")
    ap.add_argument("--root", type=Path, default=ROOT, help="folder that holds results/ and data/ (tests use a temporary one)")
    a = ap.parse_args(argv)
    for s in a.only.split(","):
        globals()[s](a)


if __name__ == "__main__":
    main()
