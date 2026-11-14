"""Workflow and architecture diagrams of the whole project (Studies I-III) at journal size -> results/figures/fig_workflow.pdf/.png and
fig_architecture.pdf/.png (tools/figstyle.py), copied to paper/figures.

python tools/fig_framework.py [--only workflow,architecture]
Drawn from the methods as written in submission/sections (no data are read); the key results quoted in the workflow come from the
generated tables (Study I: tables/audit_contrasts.tex; Study II: v3_primary.tex; Study III: v4_hypotheses.tex and v4_halluc.tex).
"""
import argparse
import shutil
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402

TINT = {"I": "#DCEBF7", "II": "#FCE9C8", "III": "#D5EFE5", "plan": "#EDE7F6", "grey": "#F2F2F2", "white": "white"}
EDGE = {"I": "#0072B2", "II": "#E69F00", "III": "#009E73", "plan": "#6A51A3", "grey": "0.4", "white": "0.4"}


def canvas(height_mm):
    fig, ax = fs.figure("double", height_mm=height_mm)
    ax.set(xlim=(0, 180), ylim=(0, height_mm))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="grey", weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=TINT[kind], ec=EDGE[kind], lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.22, fontweight=weight)


def arrow(ax, p0, p1, text=None, at=0.5, dx=0.0, dy=1.3, color="0.25", ls="-"):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0))
    if text:
        ax.text(p0[0] + (p1[0] - p0[0]) * at + dx, p0[1] + (p1[1] - p0[1]) * at + dy, text, ha="center", va="bottom", fontsize=8,
                style="italic", color="0.25")


def save(a, fig, name):
    fs.save(fig, str(a.out / name))
    for ext in (".pdf", ".png"):
        shutil.copyfile(a.out / f"{name}{ext}", a.paper / f"{name}{ext}")


def workflow(a):
    fig, ax = canvas(132)
    cols = [(8, 40, "Data"), (53, 44, "Models and training"), (102, 46, "Evaluation"), (153, 26, "Key result")]
    for x, w, t in cols:
        ax.text(x + w / 2, 128, t, ha="center", va="center", fontsize=9, fontweight="bold")
    box(ax, 8, 112, 171, 10, "Pre-specified analysis plans (SAP v1-v4), hash-frozen before any test evaluation;\n"
        "every deviation logged and every hypothesis reported, whatever its direction", "plan")
    rows = [
        ("I", 82, ["LIDC-IDRI renders (v1)\nQure.ai CT projections\n(exact bone and soft-tissue\ncomponents)",
                   "Bone suppressor: encoder +\nUNet++ decoder, twin lesion-\ninvariance loss, ensemble of 3,\nconformal intervals",
                   "Counterfactual lesion audit (A)\nfidelity on dual-energy pairs\nrib benchmark, real nodules,\ndownstream tasks, BDCXR-3257",
                   "A = 0.051 vs\n0.149 (released)\nand 0.178 (control)\nPSNR non-inferior"]),
        ("II", 49, ["suppressor outputs $S$, $\\hat B$\nNIH ChestX-ray14,\nVinBigData, BDCXR-3257",
                    "Multi-stream classifier,\n5 input configurations,\n5 seeds, 3 data sets\n(175 runs; 5 folds on\nBDCXR-3257)",
                    "Macro-AUROC contrasts\n(capacity-matched control),\nfinding groups, pointing\ngame, conformal sets",
                    "No overall gain\n(NIH +0.0001);\nsmall focal gain\non NIH only"]),
        ("III", 16, ["CT-RATE subset (2,156)\npaired projections\n(same volumes)\nNIH, VinBigData",
                     "2.5D Swin CT teacher\n-> P0 / P1 (distillation)\n-> F$_\\mathrm{in}$, F$_\\mathrm{p0}$, F$_\\mathrm{p1}$ on NIH\n(Swin vs DenseNet-121)",
                     "AUROC (H11-H13)\nCT-X-ray maps (H14)\nhallucination tests (H15,\nrib emphasis from Study I)\npointing game, conformal",
                     "No gain from CT\n(-0.0001);\nmaps disagree;\nnoise flips\n27-29 vs 78 %"])]
    for kind, y, texts in rows:
        h = 27
        ax.text(3.2, y + h / 2, f"Study {kind}", rotation=90, ha="center", va="center", fontsize=9, fontweight="bold", color=EDGE[kind])
        for (x, w, _), t in zip(cols, texts):
            box(ax, x, y, w, h, t, kind)
        for (x0, w0, _), (x1, _, _) in zip(cols, cols[1:]):
            arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    ax.plot([75, 75, 28], [82, 79, 79], color="0.25", lw=0.7)  # elbow through the gap between Study I and Study II
    arrow(ax, (28, 79), (28, 76))
    ax.text(51.5, 79.4, "soft-tissue and bone images", ha="center", va="bottom", fontsize=8, style="italic", color="0.25")
    box(ax, 8, 2, 171, 10, "Kaggle T4 kernels (code, job lists, logs)  ->  saved outputs  ->  scripted statistics (cluster bootstrap, Holm, "
        "DeLong)\n->  generated tables and figures: no number in the manuscript is typed by hand", "white")
    save(a, fig, "fig_workflow")


def architecture(a):
    fig, ax = canvas(170)
    # (a) Study I suppressor with the twin regulariser
    ax.text(1, 166, "a", fontsize=10, fontweight="bold", va="center")
    ax.text(5, 166, "Study I: bone suppressor with the twin lesion-invariance regulariser", fontsize=9, fontweight="bold", va="center")
    box(ax, 1, 145, 30, 14, "training image $x$\n(radiograph-like\ncomposite)", "I")
    box(ax, 1, 120, 30, 14, "twin $x^{+}$: lesion\nsignal $n$ pasted\nin the lung fields", "I")
    box(ax, 37, 120, 30, 39, "encoder\nConvNeXt-V2-T\nor SwinV2-T\n(512 px)\n\nshared weights", "white")
    box(ax, 73, 120, 30, 39, "UNet++ decoder\nwith CBAM,\nU-Net tail,\n1x1 head\n\nshared weights", "white")
    box(ax, 109, 145, 26, 14, "bone $\\hat b=f(x)$;\nsoft tissue\n$x-\\hat b$", "I")
    box(ax, 109, 120, 26, 14, "$f(x^{+})$; target\nstays $b$", "I")
    box(ax, 141, 120, 38, 39, "loss: $L_{sup}$ on $x$ and $x^{+}$\n$+\\;w\\,L_{inv}$\n\n$L_{inv}$: mean $|f(x^{+})-f(x)|$\non the lesion support $M$\n\n"
        "3-model ensemble:\nmean, spread and\nconformal intervals", "white")
    for y in (152, 127):
        arrow(ax, (31, y), (37, y))
        arrow(ax, (67, y), (73, y))
        arrow(ax, (103, y), (109, y))
        arrow(ax, (135, y), (141, y))
    # (b) Study II multi-stream classifier
    ax.text(1, 112, "b", fontsize=10, fontweight="bold", va="center")
    ax.text(5, 112, "Study II: multi-stream classifier with a class-query decoder", fontsize=9, fontweight="bold", va="center")
    for k, (t, y) in enumerate((("radiograph $X$", 97), ("soft tissue $S$", 86), ("bone $\\hat B=X-S$", 75))):
        box(ax, 1, y, 30, 8.5, t, "II")
        arrow(ax, (31, y + 4.25), (37, y + 4.25))
    box(ax, 37, 75, 30, 30.5, "shared encoder\nDenseNet-121\n(one weight set,\n256 px)", "white")
    box(ax, 73, 75, 30, 30.5, "$8\\times8$ tokens\nper stream\n+ stream embedding\n+ 2-D sine code", "white")
    box(ax, 109, 75, 30, 30.5, "class-query decoder\n2 layers, 8 heads,\none query per class", "white")
    box(ax, 145, 75, 34, 30.5, "linear head per class\n-> probabilities\n(NIH 14, VinBigData\n14, BDCXR-3257 3)", "white")
    for x in (67, 103, 139):
        arrow(ax, (x, 90.25), (x + 6, 90.25))
    ax.text(90, 69.5, "configurations: orig ($X$), soft ($S$), early ($X,S,\\hat B$ as channels), hybrid (three streams), "
            "ctrl ($X$ three times, same capacity)", ha="center", va="center", fontsize=8)
    # (c) Study III CT-to-X-ray distillation
    ax.text(1, 62, "c", fontsize=10, fontweight="bold", va="center")
    ax.text(5, 62, "Study III: CT-to-X-ray knowledge transfer with Swin transformers", fontsize=9, fontweight="bold", va="center")
    box(ax, 1, 38, 30, 18, "CT volume (CT-RATE)\n24 slices x 3 HU\nwindows (lung,\nmediastinum, bone)", "III")
    box(ax, 37, 38, 30, 18, "SwinV2-T\non every slice\n(shared weights)", "white")
    box(ax, 73, 38, 30, 18, "$24\\times8\\times8$ tokens\n+ sine code\n+ slice-depth\nembedding", "white")
    box(ax, 109, 38, 30, 18, "class-query decoder\n-> teacher logits $t$,\npooled output $e^{T}$", "white")
    box(ax, 145, 38, 34, 18, "CT Grad-CAM++ per\nslice, summed along\nAP -> CT heat map\n(H14 agreement)", "white")
    box(ax, 1, 10, 30, 18, "paired projection\n(Qure.ai, same\nvolume, 256 px)", "III")
    box(ax, 37, 10, 30, 18, "SwinV2-T\n(ImageNet\nweights)", "white")
    box(ax, 73, 10, 30, 18, "class-query decoder\n-> student logits $z$,\npooled output $e$", "white")
    box(ax, 109, 10, 30, 18, "loss: $L_{BCE}(z,y)$\n$+\\,\\lambda_{kd}$ distillation ($t$)\n$+\\,\\lambda_{al}$ alignment ($e^{T}$)", "white")
    box(ax, 145, 10, 34, 18, "P1 initialises F$_\\mathrm{p1}$\nfor NIH (14 diseases);\nmaps: Grad-CAM++,\nIG, attention", "white")
    for x in (31, 67, 103, 139):
        arrow(ax, (x, 47), (x + 6, 47))
        arrow(ax, (x, 19), (x + 6, 19))
    arrow(ax, (124, 38), (124, 28), "$t$, $e^{T}$", at=0.5, dx=4, dy=-1.5)
    ax.text(90, 3.5, "$T=2$, $\\lambda_{kd}=1$, $\\lambda_{al}=0.5$; half of every P1 batch from the 1,600 paired projections; "
            "P0: the same model without the teacher terms", ha="center", va="center", fontsize=8)
    save(a, fig, "fig_architecture")


FIGS = dict(workflow=workflow, architecture=architecture)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGS))
    ap.add_argument("--out", default=str(ROOT / "results/figures"))
    ap.add_argument("--paper", default=str(ROOT / "paper/figures"))
    a = ap.parse_args(argv)
    a.out, a.paper = Path(a.out), Path(a.paper)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        FIGS[name](a)


if __name__ == "__main__":
    main()
