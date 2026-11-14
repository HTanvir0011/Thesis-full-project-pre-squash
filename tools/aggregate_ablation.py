"""Aggregate suppressor-arm results into tables (all numbers come from saved result files).

python tools/aggregate_ablation.py RESULTS_ROOT OUT_DIR [--ref recipe]
RESULTS_ROOT contains <arm>_s<seed>/{szch.csv, synth_test/per_ct.csv, synth_test/per_site.csv} (several roots may be merged by copying).
Outputs: arm_summary.csv (mean +- SD over seeds), contrasts_synth.csv (H1: absorption, MAE non-inferiority), contrasts_real.csv (SZCH).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4 import stats  # noqa: E402

SYN_COLS = ["mae_bone", "psnr_bone", "ssim_bone", "lcr"]
REAL_COLS = ["psnr_raw", "ssim_raw", "psnr_cal", "ssim_cal", "psnr_cal_bone", "psnr_cal_nonbone", "ssim_cal_bone", "ssim_cal_nonbone"]


def load(root):
    runs = {}
    for d in sorted(Path(root).glob("*_s[0-9]*")):
        m = re.match(r"(.+)_s(\d+)$", d.name)
        if m and (d / "synth_test" / "per_ct.csv").exists() and (d / "szch.csv").exists():
            runs[(m.group(1), int(m.group(2)))] = dict(
                ct=pd.read_csv(d / "synth_test" / "per_ct.csv"), site=pd.read_csv(d / "synth_test" / "per_site.csv"), real=pd.read_csv(d / "szch.csv"))
    return runs


def seed_mean(runs, arm, part, key, cols):
    frames = [r[part].set_index(key)[cols] for (a, _), r in runs.items() if a == arm]
    return sum(frames) / len(frames)


def patient(name):
    return name.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--ref", default="recipe")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    runs = load(a.root)
    arms = sorted({k[0] for k in runs})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in arms:
        seeds = sorted(s for (x, s) in runs if x == arm)
        per_seed = []
        for s in seeds:
            r = runs[(arm, s)]
            per_seed.append({**{c: (r["ct"][c].median() if c == "lcr" else r["ct"][c].mean()) for c in SYN_COLS},  # LCR is a ratio: heavy tails, so median
                             "absorption": r["site"]["absorption"].mean(),
                             **{c: r["real"][c].mean() for c in REAL_COLS}})
        df = pd.DataFrame(per_seed)
        rows.append({"arm": arm, "n_seeds": len(seeds), **{f"{c}_mean": df[c].mean() for c in df}, **{f"{c}_sd": df[c].std(ddof=1) if len(df) > 1 else np.nan for c in df}})
    pd.DataFrame(rows).to_csv(out / "arm_summary.csv", index=False)

    ref = a.ref
    synth, real = [], []
    for arm in arms:
        if arm == ref:
            continue
        # H1: per-site absorption (seed-averaged), clustered by CT; lower is better, so diff = ref - arm
        sa, sr = seed_mean(runs, arm, "site", ["name", "site"], ["absorption"]), seed_mean(runs, ref, "site", ["name", "site"], ["absorption"])
        k = sa.index.intersection(sr.index)
        d = (sr.loc[k, "absorption"] - sa.loc[k, "absorption"]).to_numpy()
        cl = [patient(n) for n, _ in k]
        m, lo, hi = stats.cluster_bootstrap_ci(d, cl, a.boot)
        mae_a, mae_r = seed_mean(runs, arm, "ct", "name", ["mae_bone"]), seed_mean(runs, ref, "ct", "name", ["mae_bone"])
        kk = mae_a.index.intersection(mae_r.index)
        dm = (mae_r.loc[kk, "mae_bone"] - mae_a.loc[kk, "mae_bone"]).to_numpy()  # positive = arm has lower error
        margin = 0.05 * float(mae_r.loc[kk, "mae_bone"].mean())
        ni = stats.non_inferior(dm, margin, [patient(n) for n in kk], a.boot)
        synth.append(dict(arm=arm, ref=ref, n_sites=len(k), absorption_reduction=m, ci_lo=lo, ci_hi=hi, **{f"wilcoxon_p": stats.paired_tests(sr.loc[k, "absorption"], sa.loc[k, "absorption"])["wilcoxon_p"]},
                          mae_diff_ref_minus_arm=float(dm.mean()), mae_margin=margin, mae_non_inferior=ni["non_inferior"], mae_lower_bound=ni["lower"]))
        for c in ("psnr_cal_bone", "ssim_cal", "psnr_cal_nonbone"):
            xa, xr = seed_mean(runs, arm, "real", "image", [c]), seed_mean(runs, ref, "real", "image", [c])
            kr = xa.index.intersection(xr.index)
            t = stats.paired_tests(xa.loc[kr, c], xr.loc[kr, c])
            mm, l, h = stats.cluster_bootstrap_ci((xa.loc[kr, c] - xr.loc[kr, c]).to_numpy(), None, a.boot)
            real.append(dict(arm=arm, ref=ref, metric=c, n=len(kr), mean_diff_arm_minus_ref=mm, ci_lo=l, ci_hi=h, wilcoxon_p=t["wilcoxon_p"]))
    if synth:
        df = pd.DataFrame(synth)
        df["holm_p"] = stats.holm(df["wilcoxon_p"])  # one family: the lesion-aware contrasts against the reference arm
        df.to_csv(out / "contrasts_synth.csv", index=False)
    if real:
        df = pd.DataFrame(real)
        df["holm_p"] = np.nan
        for _, idx in df.groupby("metric").groups.items():  # one family per metric across arms
            df.loc[idx, "holm_p"] = stats.holm(df.loc[idx, "wilcoxon_p"])
        df.to_csv(out / "contrasts_real.csv", index=False)
    print(pd.DataFrame(rows).round(4).T.to_string())


if __name__ == "__main__":
    main()
