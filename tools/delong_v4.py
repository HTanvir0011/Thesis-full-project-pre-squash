"""v4 per-class DeLong tests (SAP_v4.md, descriptive) on the NIH seed ensembles -> results/v4/delong.json.

python tools/delong_v4.py [--root results/v4] [--v3 results/v3/down/nih] [--out results/v4/delong.json]
Paired DeLong (s4/delong.py) per class on the seed-mean predictions tools/xm_stats.py uses: the H11 and H12 contrasts on the five exactly
mapped shared findings (one Holm family of 15 tests) and H13's on the 14 NIH classes (a family of 14), as v3's per-class tests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from s4.delong import delong_paired  # noqa: E402
from s4.prep import NIH  # noqa: E402
from s4.stats import holm  # noqa: E402
from s4.xm_labels import EXACT_NIH, SHARED, to_shared  # noqa: E402
from xm_stats import NIH_MODELS, ensemble  # noqa: E402


def family(y, P, contrasts, names):
    """y [N,C], P {model: [N,C]} -> {'a-b:class': DeLong dict with p_holm over the whole family}."""
    out = {}
    for a, b in contrasts:
        for c, name in enumerate(names):
            k = ~np.isnan(y[:, c])
            out[f"{a}-{b}:{name}"] = delong_paired(y[k, c], P[a][k, c], P[b][k, c])
    for k, p in zip(out, holm([v["p"] for v in out.values()])):
        out[k]["p_holm"] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/v4"))
    ap.add_argument("--v3", default=str(ROOT / "results/v3/down/nih"))
    ap.add_argument("--out", default=str(ROOT / "results/v4/delong.json"))
    a = ap.parse_args(argv)
    runs = ensemble(Path(a.root) / "down" / "nih", "nih_hyb_")
    have = {m: runs[v] for m, v in NIH_MODELS.items()} | {"DN121": ensemble(Path(a.v3), "nih_hyb_")["orig@densenet121"]}
    assert all(np.array_equal(r[2], have["F_in"][2]) for r in have.values()), "other NIH test rows"
    y, cols = have["F_in"][1], [SHARED.index(s) for s in EXACT_NIH]
    res = dict(shared=family(to_shared(y, "nih")[:, cols], {m: to_shared(have[m][0], "nih")[:, cols] for m in NIH_MODELS},
                             [("F_p1", "F_in"), ("F_p1", "F_p0"), ("F_p0", "F_in")], list(EXACT_NIH)),
               nih14=family(y, {m: have[m][0] for m in ("F_in", "DN121")}, [("F_in", "DN121")], NIH))
    for fam, d in res.items():
        for k, v in d.items():
            print(f"{fam:7}{k:42}{v['auc1']:.4f} {v['auc2']:.4f} {v['diff']:+.4f}  p {v['p']:.3g}  holm {v['p_holm']:.3g}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
