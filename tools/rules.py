"""Apply the pre-declared decision rules of RUNPLAN.md section 4 to development results (SZCH dev-24 bone PSNR_cal, SZCH audit A).

python tools/rules.py r1 outputs/screen_bb.csv            # encoder screen at 100k: which encoders go to 300k x 2 seeds
python tools/rules.py r2 outputs/screen_w1.csv [more.csv]  # 300k candidates vs the v2 reference (mean over seeds)
Run names: <arm>_s<seed>[_suffix]; seeds of one candidate are grouped by removing _s<seed>.
"""
import re
import sys

import pandas as pd

REF = dict(A=0.0667, psnr=21.735, ssim=0.846)  # SwinV2-T twin16 300k, seeds 0-3 (DECISIONS.md, round 4)


def table(paths):
    df = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates("run")
    df["cand"] = df.run.map(lambda r: re.sub(r"_s\d+", "", r))
    g = df.groupby("cand").agg(n=("run", "size"), A=("aud_A", "mean"), psnr=("szch_dev_psnr_cal_bone", "mean"), ssim=("szch_dev_ssim_cal", "mean"))
    return g.round(4)


if __name__ == "__main__":
    rule, paths = sys.argv[1], sys.argv[2:]
    g = table(paths)
    if rule == "r1":  # 100k screen
        g["passes_R1"] = ((g.A <= 0.070) & (g.psnr >= 21.60)) | ((g.psnr >= 21.95) & (g.A <= 0.100))
    else:  # replace the v2 configuration
        g["passes_R2"] = (((g.A <= REF["A"] - 0.01) & (g.psnr >= REF["psnr"] - 0.05)) | ((g.psnr >= REF["psnr"] + 0.15) & (g.A <= REF["A"] + 0.005))) \
            & (g.ssim >= REF["ssim"] - 0.005) & (g.n >= 2)
    print(g.to_string())
