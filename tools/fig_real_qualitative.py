"""Qualitative figure on SZCH test images (CC BY-NC 4.0, attribution required): CXR | DES soft | each method's soft estimate | ours bone.

python tools/fig_real_qualitative.py OUT.png ours:<ckpt> qure:<ts> --images 107.png 127.png 142.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4.eval_real import clahe, load_pairs, make_ens, make_ours, make_qure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--images", nargs="+", default=["107.png", "127.png", "142.png"])
    ap.add_argument("--root", default="data/szch")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    fns = {}
    for m in a.methods:
        kind, _, arg = m.partition(":")
        fns[kind] = {"ours": lambda: (make_ens if "+" in arg else make_ours)(arg, a.device), "qure": lambda: make_qure(arg, a.device), "clahe": lambda: clahe}[kind]()
    pairs = {n: (x, g) for n, x, g in load_pairs(a.root)}
    name = {"ours": "ours (K=3)", "qure": "Qure", "clahe": "CLAHE"}
    cols = ["radiograph", "DES soft-tissue reference"] + [f"{name[k]} soft" for k in fns] + (["ours: predicted bone"] if "ours" in fns else [])
    fig, ax = plt.subplots(len(a.images), len(cols), figsize=(2.6 * len(cols), 2.6 * len(a.images)))
    ax = np.atleast_2d(ax)
    for r, n in enumerate(a.images):
        x, g = pairs[n]
        panels = [x, g] + [np.clip(f(x), 0, 1) for f in fns.values()]
        if "ours" in fns:
            panels.append(np.clip(x - fns["ours"](x), 0, None))
        for c, im in enumerate(panels):
            ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=1 if c < len(panels) - (1 if "ours" in fns else 0) else max(float(im.max()), 1e-3))
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
