"""v4 pointing game (SAP_v4.md, descriptive) from the s4/xai_multistream.py outputs of the NIH orig runs -> results/v4/pointing_summary.json.

python tools/pointing_v4.py [--src results/v4/pointing] [--out results/v4/pointing_summary.json] [--n 10000]
Per model (F_in, F_p0, F_p1, DN121) and heat map: the hit rate over the NIH box targets, per seed and for the seed mean of each target's
hits; all targets, focal (Mass, Nodule) and the other six box classes; 95 % image-cluster bootstrap interval of the all-target rate. Maps
with no hit in any run (ig_S, ig_B: the orig models have one stream) are left out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hybrid_stats import FOCAL_BOX  # noqa: E402
from xm_stats import NIH_MODELS, resamples  # noqa: E402

MODELS = {**NIH_MODELS, "DN121": "orig@densenet121"}


def summarize(files, n=10000):
    hits, seeds = {}, {}  # hits: map -> (image, box class) -> [hit per seed]
    for f in files:
        r = json.loads(Path(f).read_text())
        seeds[Path(f).stem] = {k[9:]: v for k, v in r.items() if k.startswith("hit_rate_")}
        for row in r["rows"]:
            for m, h in row["hits"].items():
                hits.setdefault(m, {}).setdefault((row["image"], row["cls"]), []).append(float(h))
    keys = sorted(next(iter(hits.values())))
    foc, g = np.array([k[1] in FOCAL_BOX for k in keys]), [k[0] for k in keys]
    out = dict(n_seeds=len(files), n_targets=len(keys), n_focal=int(foc.sum()), n_images=len(set(g)), per_seed=seeds, maps={})
    for m, v in hits.items():
        h = np.array([np.mean(v[k]) for k in keys])
        if h.any():
            boot = np.array([np.dot(w, h) / w.sum() for w in resamples(g, n)])
            out["maps"][m] = dict(all=float(h.mean()), lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975)),
                                  focal=float(h[foc].mean()), other=float(h[~foc].mean()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results/v4/pointing"))
    ap.add_argument("--out", default=str(ROOT / "results/v4/pointing_summary.json"))
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args(argv)
    rep = {}
    for name, cfg in MODELS.items():
        files = sorted(Path(a.src).glob(f"nih_hyb_{cfg}_s*.json"))
        rep[name] = summarize(files, a.n) if files else "missing"
        if files:
            print(f"{name:6} seeds {rep[name]['n_seeds']} targets {rep[name]['n_targets']}  " +
                  "  ".join(f"{m} {v['all']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] focal {v['focal']:.3f}" for m, v in rep[name]["maps"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
