"""v4 figures for the thesis (SAP_v4.md H11-H15 and the descriptive results) from saved outputs only -> results/figures/fig_v4_*.pdf
and .png (tools/figstyle.py sizes), copied to paper/figures.

python tools/fig_v4.py [--only design,forest,per_class,transfer,ct_training,crossmodal,maps,halluc,pointing,conformal,xai]
Inputs: results/final_xm_stats.json (tools/xm_stats.py) and results/v4 (tools/collect_v4.py: per-seed result.json, delong.json,
pointing_summary.json, uq_cls_<model>.json, the cross-modal and pointing-game maps); the CT teacher curves read the training logs of every
CT teacher run in outputs/v4/ct-*/results (collapsed and unused seeds included). Images: NIH ChestX-ray14 test images only; CT-derived
content is limited to the 32x32 heat-map frames, shown as means over scans (DESIGN_v4.md section 2).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402
from s4.prep import NIH  # noqa: E402
from s4.xm_labels import SHARED  # noqa: E402

COL = {"F_in": "#0072B2", "F_p0": "#E69F00", "F_p1": "#009E73", "DN121": "#6E6E6E", "P0": "#E69F00", "P1": "#009E73"}
NAME = {"F_in": r"F$_\mathrm{in}$", "F_p0": r"F$_\mathrm{p0}$", "F_p1": r"F$_\mathrm{p1}$", "DN121": "DenseNet-121", "P0": "P0", "P1": "P1"}
FIND = {s: s.replace("_", " ") for s in SHARED}
SWIN = "nih_hyb_orig@swinv2_tiny_window8_256"
RUN = {"F_in": SWIN, "F_p0": SWIN + "+xray", "F_p1": SWIN + "+xraykd", "DN121": "nih_hyb_orig@densenet121"}


def save(a, fig, name):
    fs.save(fig, str(a.out / name))
    for ext in (".pdf", ".png"):
        shutil.copyfile(a.out / f"{name}{ext}", a.paper / f"{name}{ext}")


def per_seed(d, pattern, key):
    return [json.loads(f.read_text())[key] for f in sorted(d.glob(f"{pattern}/result.json"))]


def design(a, s):
    fig, ax = fs.figure("double", height_mm=92)
    ax.set(xlim=(0, 180), ylim=(0, 92))
    ax.axis("off")

    def box(x, y, w, h, t, c="#F2F2F2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=c, ec="0.35", lw=0.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, linespacing=1.25)

    def arrow(x0, y0, x1, y1, t=None, dy=1.2):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.25", mutation_scale=7, shrinkA=0, shrinkB=0))
        if t:
            ax.text((x0 + x1) / 2 + 1, (y0 + y1) / 2 + dy, t, ha="left", va="center", fontsize=8, style="italic", color="0.25")

    ax.text(24, 89, "data", ha="center", fontsize=9, fontweight="bold")
    ax.text(85, 89, "models (SwinV2-T, five seeds each)", ha="center", fontsize=9, fontweight="bold")
    ax.text(151, 89, "evaluation", ha="center", fontsize=9, fontweight="bold")
    box(1, 68, 46, 16, "CT-RATE subset\n2,156 volumes\n(1,600 / 259 / 297)", "#DCEBF7")
    box(1, 38, 46, 20, "Qure.ai projections of\nCT-RATE volumes, six findings\n(20,533 / 396 / 415;\n1,600 paired with the subset)", "#DCEBF7")
    box(1, 4, 46, 22, "NIH ChestX-ray14\n14 diseases, 984 expert boxes\nVinBigData (zero-shot only)", "#DCEBF7")
    box(54, 68, 62, 16, "CT teacher, 2.5D\n24 slices x 3 HU windows")
    box(54, 46, 29, 14, "P0\nprojections", "#FCE9C8")
    box(87, 46, 29, 14, "P1\nprojections +\ndistillation", "#D5EFE5")
    box(54, 4, 62, 26, "NIH fine-tuning, 10 epochs, 14 diseases\n" + r"F$_\mathrm{in}$ from ImageNet" + "\n" + r"F$_\mathrm{p0}$ from P0,  F$_\mathrm{p1}$ from P1"
        + "\nreference: DenseNet-121 (v3)")
    box(123, 68, 56, 16, "CT test AUROC\nH14: CT vs X-ray heat maps\n(P1; P0 descriptive)", "#EFEFEF")
    box(123, 40, 56, 18, "projection test AUROC\nzero-shot NIH and VinBigData\n(six shared findings)", "#EFEFEF")
    box(123, 2, 56, 30, "NIH test split\nH11, H12: shared findings\nH13: vs DenseNet-121\nH15 and hallucination tests\npointing game, conformal FNR", "#EFEFEF")
    arrow(47, 76, 54, 76)
    arrow(47, 50, 54, 53)
    ax.plot([47, 50, 50, 94], [55, 55, 63.5, 63.5], color="0.25", lw=0.7)  # projections also feed P1, routed above P0
    arrow(94, 63.5, 94, 60)
    arrow(47, 15, 54, 15)
    arrow(101.5, 68, 101.5, 60, "logits (T = 2),\npooled features", dy=0)
    arrow(68.5, 46, 68.5, 30)
    arrow(101.5, 46, 101.5, 30)
    arrow(116, 76, 123, 76)
    arrow(116, 53, 123, 49)
    arrow(116, 17, 123, 17)
    save(a, fig, "fig_v4_design")


def forest(a, s):
    rows = [("a", r"$\Delta$ AUROC", [("H11: F$_\\mathrm{p1}$ $-$ F$_\\mathrm{in}$ (5 shared)", s["H11"], s["H11"]["p_holm"], "Holm"),
                                       ("H12: F$_\\mathrm{p1}$ $-$ F$_\\mathrm{p0}$ (5 shared)", s["H12_p1_p0"], s["H12_p1_p0"]["p_holm"], "Holm"),
                                       ("H12: F$_\\mathrm{p0}$ $-$ F$_\\mathrm{in}$ (5 shared)", s["H12_p0_in"], s["H12_p0_in"]["p_holm"], "Holm"),
                                       ("H13: F$_\\mathrm{in}$ $-$ DenseNet-121 (14)", s["H13"], s["H13"]["p"], "")]),
            ("b", r"$\Delta$ Pearson $r$, matched $-$ mismatched", [("H14: CT teacher vs P1", s["crossmodal_xraykd"]["H14"], s["crossmodal_xraykd"]["H14"]["p"], ""),
                                                                    ("CT teacher vs P0 (descriptive)", s["crossmodal_xray"]["H14"], s["crossmodal_xray"]["H14"]["p"], "")]),
            ("c", r"$\Delta$ share of images flipped", [("H15: F$_\\mathrm{p1}$ $-$ F$_\\mathrm{in}$, rib emphasis", s["H15"], s["H15"]["p"], "")])]
    fig, ax = fs.figure("onehalf", height_mm=92, nrows=3, gridspec_kw=dict(height_ratios=[4, 2, 1.2]))
    for x, (letter, xl, rr) in zip(ax, rows):
        for i, (lab, c, p, kind) in enumerate(rr):
            x.errorbar(c["diff"], -i, xerr=[[c["diff"] - c["lo"]], [c["hi"] - c["diff"]]], fmt="o", color="k", mfc="k" if p < 0.05 else "w",
                       capsize=2, lw=0.8, ms=4)
            x.text(1.01, -i, f"{kind} p = {p:.2g}".strip(), transform=x.get_yaxis_transform(), va="center", fontsize=8)
        x.axvline(0, color="0.55", lw=0.6, ls="--")
        x.xaxis.set_major_locator(MaxNLocator(5))
        x.set_yticks([-i for i in range(len(rr))], [r[0] for r in rr])
        x.set_ylim(-len(rr) + 0.5, 0.5)
        x.set_xlabel(xl)
        x.spines["left"].set_visible(False)
        x.tick_params(axis="y", length=0)
        fs.panel_label(x, letter)
    save(a, fig, "fig_v4_forest")


def per_class(a, s, d):
    m = s["nih_macro_auroc"]
    fig, ax = fs.figure("double", height_mm=105, nrows=2, sharex=True)
    xs = np.arange(len(NIH))
    for j, k in enumerate(m):
        ax[0].plot(xs + (j - 1.5) * 0.17, m[k]["per_class"], "o", color=COL[k], ms=3.5, label=f"{NAME[k]}, macro-AUROC {m[k]['macro']:.3f}")
    ax[0].set(ylabel="test AUROC (seed ensemble)")
    fig.legend(*ax[0].get_legend_handles_labels(), ncols=2, loc="outside upper center")
    dd = [d["nih14"][f"F_in-DN121:{c}"] for c in NIH]
    for i, r in enumerate(dd):
        se = abs(r["diff"] / r["z"]) if r["z"] else 0.0
        ax[1].errorbar(i, r["diff"], yerr=1.96 * se, fmt="o", color="k", mfc="k" if r["p_holm"] < 0.05 else "w", capsize=2, lw=0.8, ms=4)
    ax[1].axhline(0, color="0.55", lw=0.6, ls="--")
    ax[1].set(ylabel=r"F$_\mathrm{in}$ $-$ DenseNet-121")
    ax[1].set_xticks(xs, [c.replace("_", " ") for c in NIH], rotation=35, ha="right", rotation_mode="anchor")
    ax[1].legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="Holm p < 0.05"),
                          Line2D([], [], marker="o", ls="", color="k", mfc="w", label="Holm p ≥ 0.05")], loc="lower left", ncols=2)
    fs.panel_label(ax[0], "a")
    fs.panel_label(ax[1], "b")
    save(a, fig, "fig_v4_per_class")


def transfer(a, s):
    v4 = a.v4 / "down"
    panels = [("projection test split", [("P0", per_seed(v4 / "qp", "qp_xm_xray_s*", "test_macro_auc"), s["projection"]["xray"]["macro"]),
                                         ("P1", per_seed(v4 / "qp", "qp_xm_xraykd_s*", "test_macro_auc"), s["projection"]["xraykd"]["macro"])]),
              ("NIH, zero-shot", [("P0", per_seed(v4 / "nih", "nih_xm_transfer-xray_s*", "mean"), s["transfer"]["nih"]["xray"]["macro"]),
                                  ("P1", per_seed(v4 / "nih", "nih_xm_transfer-xraykd_s*", "mean"), s["transfer"]["nih"]["xraykd"]["macro"])]),
              ("VinBigData, zero-shot", [("P0", per_seed(v4 / "vin", "vin_xm_transfer-xray_s*", "mean"), s["transfer"]["vin"]["xray"]["macro"]),
                                         ("P1", per_seed(v4 / "vin", "vin_xm_transfer-xraykd_s*", "mean"), s["transfer"]["vin"]["xraykd"]["macro"])]),
              ("NIH, fine-tuned", [(k, per_seed(v4 / "nih", f"{RUN[k]}_s*", "test_macro_auc"), s["nih_macro_auroc"][k]["macro"])
                                   for k in ("F_in", "F_p0", "F_p1")])]
    fig, ax = fs.figure("double", height_mm=62, ncols=4)
    for x, (title, items) in zip(ax, panels):
        for i, (k, v, ens) in enumerate(items):
            x.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=COL[k], mfc="w", ms=3.5)
            x.plot(i + 0.3, ens, "D", color=COL[k], ms=4.5)
        x.set_xticks(range(len(items)), [NAME[k] for k, _, _ in items])
        x.set_xlim(-0.5, len(items) - 0.3)
        x.set_title(title)
    ax[0].set_ylabel("macro-AUROC")
    for x, letter in zip(ax, "abcd"):
        fs.panel_label(x, letter)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="0.3", mfc="w", label="single seeds"),
                        Line2D([], [], marker="D", ls="", color="0.3", label="seed ensemble")], loc="outside lower center", ncols=2)
    save(a, fig, "fig_v4_transfer")


def ct_training(a, s):
    logs = sorted(Path(a.ct_runs).glob("ct-*/results/ct_xm_ct_s*/log.jsonl"))
    if not logs:
        return print("ct_training: no CT teacher logs under", a.ct_runs)
    fig, ax = fs.figure("single", aspect=0.95)
    cols = plt.get_cmap("tab10")
    for f in logs:
        k = int(f.parent.name.rsplit("_s", 1)[1])
        r = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        r = [x for x in r if "val_macro_auc" in x]
        style = dict(color="k", ls="--") if k == 2 else dict(color="0.55", ls=":") if k == 6 else dict(color=cols(k))
        lab = f"seed {k}" + (" (collapsed)" if k == 2 else " (unused)" if k == 6 else "")
        ax.plot([x["epoch"] + 1 for x in r], [x["val_macro_auc"] for x in r], lw=0.9, label=lab, **style)
    ax.axhline(0.5, color="0.7", lw=0.5)
    ax.set(xlabel="epoch", ylabel="validation macro-AUROC", xticks=[1, 5, 10, 15, 20])
    fig.legend(*ax.get_legend_handles_labels(), ncols=3, loc="outside lower center", handlelength=1.6, columnspacing=0.8)
    save(a, fig, "fig_v4_ct_training")


def crossmodal(a, s):
    fig, ax = fs.figure("single", aspect=0.85)
    f = list(s["crossmodal_xraykd"]["per_finding"])
    for j, (v, k) in enumerate((("crossmodal_xraykd", "P1"), ("crossmodal_xray", "P0"))):
        for i, name in enumerate(f):
            r = s[v]["per_finding"][name]
            y = -i + (0.15 if j == 0 else -0.15)
            ax.plot([r["mismatched"], r["matched"]], [y, y], color=COL[k], lw=0.8)
            ax.plot(r["matched"], y, "o", color=COL[k], ms=4)
            ax.plot(r["mismatched"], y, "o", color=COL[k], mfc="w", ms=4)
    ax.axvline(0, color="0.55", lw=0.6, ls="--")
    ax.set_yticks([-i for i in range(len(f))], [f"{FIND[n]} ({s['crossmodal_xraykd']['per_finding'][n]['n']})" for n in f])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mean Pearson r, CT map vs X-ray map")
    ax.legend(handles=[Line2D([], [], color=COL["P1"], marker="o", label="P1"), Line2D([], [], color=COL["P0"], marker="o", label="P0"),
                       Line2D([], [], color="0.3", marker="o", ls="", label="same scan"),
                       Line2D([], [], color="0.3", marker="o", ls="", mfc="w", label="other scans")], ncols=2, loc="lower left",
              bbox_to_anchor=(0, 1.0))
    save(a, fig, "fig_v4_crossmodal")


def maps(a, s):
    norm = lambda m: m / (m.sum((1, 2), keepdims=True) + 1e-12)  # noqa: E731  (as tools/xm_stats.py h14)
    rows = []
    for v, label in (("xraykd", "X-ray, P1"), ("xray", "X-ray, P0")):
        zs = [np.load(f) for f in sorted((a.v4 / "xai").glob(f"ct_xmx_{v}_s*/maps.npz"))]
        if not zs:
            return print("maps: no cross-modal maps")
        cls = zs[0]["cls"]
        if not rows:
            rows.append(("CT teacher, projected", np.mean([norm(z["ct"]) for z in zs], 0), cls))
        rows.append((label, np.mean([norm(z["xr"]) for z in zs], 0), cls))
    present = [c for c in range(len(SHARED)) if (rows[0][2] == c).any()]
    side = fs.WIDTH_MM["double"] / len(present)
    fig, ax = fs.figure("double", height_mm=len(rows) * side * 0.95 + 14, nrows=len(rows), ncols=len(present), squeeze=False)
    for r, (label, m, cls) in enumerate(rows):
        for j, c in enumerate(present):
            v = ax[r, j]
            v.imshow(m[cls == c].mean(0), cmap="magma")
            v.set(xticks=[], yticks=[])
            v.spines[:].set_visible(False)
            if r == 0:
                v.set_title(f"{FIND[SHARED[c]]}\n(n = {(cls == c).sum()})")
        ax[r, 0].set_ylabel(label)
    fig.suptitle("Mean heat maps of the CT subset test scans positive for each finding (32 x 32 lung-box frames, each map scaled to sum 1)")
    save(a, fig, "fig_v4_maps")


def halluc(a, s):
    h = s["hallucination"]
    ms = list(h)
    fig, ax = fs.figure("double", height_mm=66, ncols=3, gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
    wd = 0.8 / len(ms)
    for j, k in enumerate(ms):
        off = (j - (len(ms) - 1) / 2) * wd
        ax[0].bar(np.arange(3) + off, [h[k][f"rate_{p}"] for p in ("rib", "noise", "gamma")], wd, color=COL[k], label=NAME[k])
        dp = list(h[k]["nodule_dp_by_size"].values())
        ax[1].bar(np.arange(len(dp)) + off, dp, wd, color=COL[k])
        ax[2].bar(j, h[k]["off_target"], 0.7, color=COL[k])
    ax[0].set_xticks(range(3), ["rib emphasis", "noise", "gamma"])
    ax[0].set(ylabel="share of finding-free images flipped", title="finding-free perturbations")
    ax[1].set_xticks(range(len(dp)), list(next(iter(h.values()))["nodule_dp_by_size"]))
    ax[1].set(ylabel="mean score change", title="inserted nodules, by size")
    ax[2].set_xticks(range(len(ms)), [NAME[k] for k in ms], rotation=35, ha="right", rotation_mode="anchor")
    ax[2].set(ylabel="Grad-CAM++ mass outside lungs", title="off-target", ylim=(0, 1))
    for x, letter in zip(ax, "abc"):
        fs.panel_label(x, letter)
    fig.legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in ms], loc="outside lower center", ncols=len(ms))
    save(a, fig, "fig_v4_halluc")


def pointing(a, s):
    p = json.loads((a.v4 / "pointing_summary.json").read_text())
    ms = [k for k, v in p.items() if isinstance(v, dict)]
    mp = {"cam_0": "Grad-CAM++", "attn": "decoder attention", "ig_X": "Integrated Gradients"}
    fig, ax = fs.figure("onehalf", height_mm=66, ncols=2)
    wd = 0.8 / len(ms)
    for j, k in enumerate(ms):
        off = (j - (len(ms) - 1) / 2) * wd
        r = [p[k]["maps"][m] for m in mp]
        ax[0].bar(np.arange(len(mp)) + off, [x["all"] for x in r], wd, color=COL[k], label=NAME[k],
                  yerr=[[x["all"] - x["lo"] for x in r], [x["hi"] - x["all"] for x in r]], error_kw=dict(lw=0.6, capsize=1.5))
        ax[1].plot(np.arange(len(mp)) + off, [x["focal"] for x in r], "o", color=COL[k], ms=3.5)
        ax[1].plot(np.arange(len(mp)) + off, [x["other"] for x in r], "o", color=COL[k], mfc="w", ms=3.5)
    n = p[ms[0]]
    ax[0].set(ylabel="hit rate", title=f"all {n['n_targets']} boxes", ylim=(0, 1))
    ax[1].set(ylabel="hit rate", title=f"focal ({n['n_focal']}) and other ({n['n_targets'] - n['n_focal']}) boxes", ylim=(0, 1))
    for x, letter in zip(ax, "ab"):
        x.set_xticks(range(len(mp)), list(mp.values()), rotation=20, ha="right", rotation_mode="anchor")
        fs.panel_label(x, letter)
    fig.legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in ms] + [Line2D([], [], marker="o", ls="", color="0.3", label="focal"),
                                                                         Line2D([], [], marker="o", ls="", color="0.3", mfc="w", label="other")],
               loc="outside lower center", ncols=6)
    save(a, fig, "fig_v4_pointing")


def conformal(a, s):
    uq = {k: json.loads((a.v4 / f"uq_cls_{k}.json").read_text())["nih_orig"] for k in ("F_in", "F_p0", "F_p1")}
    fig, ax = fs.figure("onehalf", height_mm=62, ncols=2, gridspec_kw=dict(width_ratios=[1.5, 1]))
    for k, u in uq.items():
        rc = np.array(u["risk_coverage_brier"])
        ax[0].plot(rc[:, 0], rc[:, 1], "o-", color=COL[k], ms=2.5, label=f"{NAME[k]} (AURC {u['aurc_brier']:.4f})")
        ax[1].bar(list(uq).index(k), u["fnr_test"], 0.7, color=COL[k])
        ax[1].text(list(uq).index(k), u["fnr_test"] + 0.003, f"{u['mean_set_size']:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set(xlabel="coverage (most confident images kept)", ylabel="Brier score", title="selective classification")
    ax[0].legend(loc="upper left")
    ax[1].axhline(0.10, color="0.4", ls="--", lw=0.8)
    ax[1].set_xticks(range(len(uq)), [NAME[k] for k in uq])
    ax[1].set(ylabel="test false-negative rate", title="risk control at 0.10\n(above bars: mean set size)", ylim=(0, 0.13))
    for x, letter in zip(ax, "ab"):
        fs.panel_label(x, letter)
    save(a, fig, "fig_v4_conformal")


def xai(a, s):
    zs = {k: np.load(a.v4 / "pointing" / f"{RUN[k]}_s0.maps.npz") for k in RUN if (a.v4 / "pointing" / f"{RUN[k]}_s0.maps.npz").exists()}
    if len(zs) < len(RUN):
        return print("xai: seed-0 pointing-game maps missing")
    keys = sorted({k.split("__")[0] for k in zs["F_in"].files}, key=lambda k: (int(k.split("_")[1]), int(k.split("_")[0])))
    rows = [next(k for k in keys if int(k.split("_")[1]) == c) for c in sorted({int(k.split("_")[1]) for k in keys})]  # first image per class
    cols = [(None, None, "radiograph")] + [(k, "cam_0", f"Grad-CAM++\n{NAME[k]}") for k in RUN] + [("F_p1", "ig_X", "Integrated\nGradients, " + NAME["F_p1"]),
                                                                                                   ("F_p1", "attn", "decoder\nattention, " + NAME["F_p1"])]
    side = fs.WIDTH_MM["double"] / len(cols)
    fig, ax = fs.figure("double", height_mm=min(fs.MAX_HEIGHT_MM, len(rows) * side + 16), nrows=len(rows), ncols=len(cols), squeeze=False)
    for r, key in enumerate(rows):
        z0 = zs["F_in"]
        x, bx = z0[f"{key}__x"][0], z0[f"{key}__boxes"]
        for j, (k, m, t) in enumerate(cols):
            v = ax[r, j]
            v.imshow(x, cmap="gray")
            if m is not None:
                hm = zs[k][f"{key}__{m}"]
                v.imshow(hm / (hm.max() + 1e-12), cmap="jet", alpha=0.4)
            for x0, y0, x1, y1 in bx:
                v.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="w", lw=0.7))
            v.set(xticks=[], yticks=[])
            v.spines[:].set_visible(False)
            if r == 0:
                v.set_title(t)
        ax[r, 0].set_ylabel(NIH[int(key.split("_")[1])].replace("_", " "))
    save(a, fig, "fig_v4_xai_examples")


FIGS = dict(design=design, forest=forest, per_class=per_class, transfer=transfer, ct_training=ct_training, crossmodal=crossmodal, maps=maps,
            halluc=halluc, pointing=pointing, conformal=conformal, xai=xai)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGS))
    ap.add_argument("--stats", default=str(ROOT / "results/final_xm_stats.json"))
    ap.add_argument("--v4", default=str(ROOT / "results/v4"))
    ap.add_argument("--ct_runs", default=str(ROOT / "outputs/v4"))
    ap.add_argument("--out", default=str(ROOT / "results/figures"))
    ap.add_argument("--paper", default=str(ROOT / "paper/figures"))
    a = ap.parse_args(argv)
    a.v4, a.out, a.paper = Path(a.v4), Path(a.out), Path(a.paper)
    a.out.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path(a.stats).read_text())
    d = json.loads((a.v4 / "delong.json").read_text())
    for name in a.only.split(","):
        FIGS[name](a, s, d) if name == "per_class" else FIGS[name](a, s)


if __name__ == "__main__":
    main()
