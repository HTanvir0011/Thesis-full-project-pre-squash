"""Journal figure standard for every figure of the project (the author's rule, 4 Nov 2026): each figure is drawn at its final print
size, so its text prints at the size set here and is never rescaled in LaTeX or Word (insert at natural size).

    import figstyle as fs                                   # tools/ on sys.path
    fig, ax = fs.figure("single", aspect=0.75)              # 88 mm wide, height = aspect x width
    fig, axes = fs.figure("double", height_mm=110, nrows=2, ncols=3)
    fs.panel_label(axes[0, 0], "a")
    fs.save(fig, "results/figures/fig_name")                # fig_name.pdf (vector, TrueType) + fig_name.png (600 dpi); checks the standard

Widths fit Elsevier (90 / 140 / 190 mm), IEEE (88.9 / 181.9 mm) and Nature (89 / 183 mm) columns: single 88 mm, one-and-a-half 140 mm,
double 180 mm; height at most 225 mm. Text: 8 pt minimum (IEEE asks 8-10 pt, Elsevier about 7 pt or more), titles 9 pt, panel letters
10 pt bold; lines 0.6-1.0 pt. save() refuses a figure with smaller text or a non-standard width.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

MM = 1 / 25.4
WIDTH_MM = {"single": 88.0, "onehalf": 140.0, "double": 180.0}
MAX_HEIGHT_MM = 225.0
MIN_PT = 8.0
RC = {  # Liberation Sans has Arial's metrics, so figures keep their layout where Arial is missing (Linux)
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 9, "legend.title_fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "image.interpolation": "nearest",
}


def apply():
    matplotlib.rcParams.update(RC)


def figure(width="single", aspect=0.75, height_mm=None, **subplots):
    """-> (fig, axes) at print size: width 'single' | 'onehalf' | 'double'; height = height_mm or aspect x width (<= 225 mm)."""
    apply()
    w = WIDTH_MM[width]
    h = height_mm if height_mm is not None else w * aspect
    if h > MAX_HEIGHT_MM:
        raise ValueError(f"height {h:.0f} mm exceeds the {MAX_HEIGHT_MM:.0f} mm page limit")
    return plt.subplots(figsize=(w * MM, h * MM), layout="constrained", **subplots)


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def problems(fig):
    """What breaks the standard: text below 8 pt (visible text only), a width that is not a standard column, a height over the limit."""
    out = []
    w_mm, h_mm = (x / MM for x in fig.get_size_inches())
    if not any(abs(w_mm - v) < 0.5 for v in WIDTH_MM.values()):
        out.append(f"width {w_mm:.1f} mm is not a standard column ({', '.join(f'{v:.0f}' for v in WIDTH_MM.values())} mm)")
    if h_mm > MAX_HEIGHT_MM + 0.5:
        out.append(f"height {h_mm:.0f} mm > {MAX_HEIGHT_MM:.0f} mm")
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
            out.append(f"text {t.get_text()[:30]!r} at {t.get_fontsize():.1f} pt < {MIN_PT:.0f} pt")
    return out


def save(fig, stem):
    """Write stem.pdf and stem.png (600 dpi) at the exact print size; refuse a figure that breaks the standard."""
    bad = problems(fig)
    if bad:
        plt.close(fig)
        raise ValueError(f"{stem}: " + "; ".join(bad[:5]))
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    print("wrote", f"{stem}.pdf")
