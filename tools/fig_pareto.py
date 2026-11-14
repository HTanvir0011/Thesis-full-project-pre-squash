"""Figure: lesion absorption A (x, lower = lesion kept) vs bone-region PSNR_cal (y, higher = better suppression), twin-weight dose-response.

python tools/fig_pareto.py final   # A on VinDr-RibCXR audit, PSNR on SZCH held-out (results/final; after the freeze)
python tools/fig_pareto.py dev     # development data: SZCH audit + SZCH dev-24 PSNR from outputs/screen_*.csv (labelled as such)
Writes paper/figures/pareto.pdf (final) or pareto_dev.pdf (development), each with a .png.
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from szch_split import is_dev  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"  # dataviz reference palette, 3 slots


def dev_points():
    df = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "outputs").glob("screen*.csv"))]).drop_duplicates("run")
    pts = []
    for r in df.itertuples():
        m = re.fullmatch(r"qp(?:_twin(\d+))?_s(\d+)_300k", r.run)
        if m and not np.isnan(r.aud_A):
            pts.append(dict(group="twin", w=int(m.group(1) or 0), seed=int(m.group(2)), A=r.aud_A, psnr=r.szch_dev_psnr_cal_bone))
    refs = {}
    for name, f in (("Qure", "outputs/real_eval/szch2_qure.csv"), ("identity", "outputs/real_eval/szch2_identity.csv")):
        s = pd.read_csv(ROOT / f)
        refs[name] = (0.0 if name == "identity" else float(pd.read_csv(ROOT / "outputs/audit_final/qure.csv").absorption.mean()),
                      float(s[s.image.map(is_dev)].psnr_cal_bone.mean()))
    return pd.DataFrame(pts), refs, "SZCH audit (development) and SZCH dev-24 pairs"


def final_points():
    R = ROOT / "results" / "final"
    A = lambda n: float(pd.read_csv(R / "audit_ribcxr" / f"{n}.csv").absorption.mean())  # noqa: E731
    P = lambda n: float((lambda s: s[~s.image.map(is_dev)].psnr_cal_bone.mean())(pd.read_csv(R / "szch" / f"{n}.csv")))  # noqa: E731
    pts = []
    for p in sorted((R / "audit_ribcxr").glob("*.csv")):  # the dose-response family is SwinV2-T (final_stats.DOSE): swin_nt = w 0, swin = w 16
        m = re.fullmatch(r"(?:w(\d+)|swin|swin_nt)_s(\d+)", p.stem)
        if m and (R / "szch" / f"{p.stem}.csv").exists():
            w = int(m.group(1)) if m.group(1) else (0 if p.stem.startswith("swin_nt") else 16)
            pts.append(dict(group="twin", w=w, seed=int(m.group(2)), A=A(p.stem), psnr=P(p.stem)))
    refs = {lab: (A(n), P(n)) for lab, n in (("Qure", "qure"), ("identity", "identity"), ("v1 clia (LIDC)", "clia_v1"), ("ours (K=3)", "ours"),
                                              ("no twin (K=3)", "notwin"))
            if (R / "audit_ribcxr" / f"{n}.csv").exists() and (R / "szch" / f"{n}.csv").exists()}
    return pd.DataFrame(pts), refs, "VinDr-RibCXR audit and SZCH held-out 51 pairs"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    pts, refs, source = dev_points() if mode == "dev" else final_points()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    if len(pts):
        mean = pts.groupby("w")[["A", "psnr"]].mean().sort_index()
        fam = "SwinV2-T, " if mode == "final" else ""
        ax.scatter(pts.A, pts.psnr, s=14, color=BLUE, alpha=0.35, linewidths=0, label=f"{fam}twin weight w, single seeds")
        ax.plot(mean.A, mean.psnr, "-o", color=BLUE, lw=2, ms=6, mec="white", mew=1.5, label=f"{fam}twin weight w, seed mean")
        off = {0: (6, 4), 4: (8, -12), 8: (-14, 7), 16: (5, -11), 32: (6, -10), 64: (-34, -4)}  # hand-placed so labels do not collide
        for w, r in mean.iterrows():
            ax.annotate(f"w={w}", (r.A, r.psnr), textcoords="offset points", xytext=off.get(w, (5, 4)), fontsize=8, color=INK)
    style = {"Qure": (ORANGE, "s"), "v1 clia (LIDC)": (AQUA, "D"), "identity": (GRAY, "x"), "ours (K=3)": (BLUE, "*"), "no twin (K=3)": (BLUE, "P")}
    roff = {"Qure": (6, 4), "identity": (6, -10), "v1 clia (LIDC)": (-74, -12), "ours (K=3)": (6, 4), "no twin (K=3)": (8, 1)}
    for lab, (a, p) in refs.items():
        c, mk = style.get(lab, (GRAY, "o"))
        ax.scatter([a], [p], s=60 if mk in "*P" else 36, color=c, marker=mk, linewidths=1.5 if mk == "x" else 0.8, edgecolors="white" if mk != "x" else None, zorder=3,
                   label=lab + (" ConvNeXt-V2-T" if "K=3" in lab and mode == "final" else ""))
        ax.annotate(lab, (a, p), textcoords="offset points", xytext=roff.get(lab, (6, -10)), fontsize=8, color=INK)
    ax.set_xlabel("lesion absorption A (lower = lesion kept)")
    ax.set_ylabel("bone-region PSNR$_{cal}$ (dB)")
    ax.set_title(source, fontsize=8, color=MUTED, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pareto{'' if mode == 'final' else '_' + mode}.{ext}", dpi=200)  # the paper includes figures/pareto.pdf
    print("wrote", out, len(pts), "runs")
