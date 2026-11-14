"""One row per run: synthetic LIDC test (exact truth), real SZCH pairs (dev-24), real-image audit, Qure-projection test.

python tools/aggregate_screen.py OUT.csv RESULTS_ROOT [RESULTS_ROOT ...] [name=szch.csv ...] [--held]
RESULTS_ROOT holds <run>/{synth_test/per_ct.csv, synth_test/per_site.csv, szch.csv, audit.csv, qp_test/per_sample.csv} (a kernel's results/ dir).
Dev/held-out SZCH split: tools/szch_split.py. Everything is computed from saved files; missing files give NaN.
SZCH paired metrics are computed on dev-24 only; --held adds the all-75 and held-out-51 columns and is for AFTER tools/freeze_v2.py (SAP_v2.md, R4).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from szch_split import is_dev  # noqa: E402

SZCH = ["psnr_cal", "ssim_cal", "psnr_cal_bone", "psnr_cal_nonbone", "ssim_cal_s4", "bone_corr_s2", "bone_slope_s2"]


HELD = "--held" in sys.argv
TAGS = (("all", None), ("dev", True), ("held", False)) if HELD else (("dev", True),)


def rd(p):
    return pd.read_csv(p) if p.exists() else None


def szch_means(sz):
    dev = sz.image.map(is_dev)
    return {f"szch_{t}_{c}": (sz if m is None else sz[dev == m])[c].mean() for t, m in TAGS for c in SZCH if c in sz}


def row(d: Path):
    r = {"run": d.name}
    ct, site = rd(d / "synth_test" / "per_ct.csv"), rd(d / "synth_test" / "per_site.csv")
    if ct is not None:
        r["syn_mae"] = ct.mae_bone.mean()
        r["syn_slope"] = ct.slope_bone.mean() if "slope_bone" in ct else np.nan
    if site is not None:
        r["syn_A"] = site.absorption.mean()
        r["syn_A_overlap0"] = site[site.bone_overlap < 1e-4].absorption.mean()
        r["syn_A_overlap60"] = site[site.bone_overlap > 0.6].absorption.mean()
    sz = rd(d / "szch.csv")
    if sz is not None:
        r.update(szch_means(sz))
    au = rd(d / "audit.csv")
    if au is not None:
        dev = au.image.map(is_dev)
        r["aud_A"], r["aud_A_dev"], r["aud_A_held"] = au.absorption.mean(), au[dev].absorption.mean(), au[~dev].absorption.mean()
        lo, hi = au.overlap.quantile([1 / 3, 2 / 3])
        r["aud_A_lowov"], r["aud_A_highov"] = au[au.overlap <= lo].absorption.mean(), au[au.overlap > hi].absorption.mean()
        r["aud_leak"] = au.leak.mean()
    q = rd(d / "qp_test" / "per_sample.csv")
    if q is not None:
        r["qp_mae"] = q.mae_bone.mean()
    return r


if __name__ == "__main__":
    args = sys.argv[1:]
    refs = [a.split("=", 1) for a in args if "=" in a]
    pos = [a for a in args if "=" not in a and a != "--held"]
    out, roots = pos[0], pos[1:]
    rows = [row(d) for root in roots for d in sorted(Path(root).iterdir()) if d.is_dir() and (d / "szch.csv").exists()]
    for name, path in refs:  # reference methods: name=szch_csv
        rows.append({"run": name, **szch_means(pd.read_csv(path))})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(4).T.to_string())
