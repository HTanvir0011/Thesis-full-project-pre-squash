"""Gate G2 ("claim check"): does x - predicted_bone behave like soft tissue on DRRs AND on real dual-energy pairs?
Pass rule (fixed before the pilot): on SZCH, paired vs identity, bone-region PSNR_cal and SSIM_cal both improve with Wilcoxon p<0.05
and non-bone PSNR_cal drops by < 1 dB; on synthetic test CTs the bone-image PSNR averages > 25 dB.

python tools/gate_g2.py RUN_RESULTS_DIR IDENTITY_SZCH.csv [OUT.json]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4.stats import paired_tests  # noqa: E402


def col(path, key):
    return {r["image"] if "image" in r else r["name"]: float(r[key]) for r in csv.DictReader(open(path))}


run, ident = Path(sys.argv[1]), sys.argv[2]
res = {}
for k in ("psnr_cal_bone", "ssim_cal", "psnr_cal_nonbone"):
    a, b = col(run / "szch.csv", k), col(ident, k)
    ks = sorted(set(a) & set(b))
    res[k] = paired_tests([a[i] for i in ks], [b[i] for i in ks])
psnr_syn = float(np.mean(list(col(run / "synth_test" / "per_ct.csv", "psnr_bone").values())))
ok = (res["psnr_cal_bone"]["mean_diff"] > 0 and res["psnr_cal_bone"]["wilcoxon_p"] < 0.05 and res["ssim_cal"]["mean_diff"] > 0
      and res["ssim_cal"]["wilcoxon_p"] < 0.05 and res["psnr_cal_nonbone"]["mean_diff"] > -1.0 and psnr_syn > 25.0)
out = dict(gate="G2", passed=bool(ok), synthetic_psnr_bone=psnr_syn, szch_vs_identity=res)
print(json.dumps(out, indent=1))
Path(sys.argv[3] if len(sys.argv) > 3 else run / "gate_g2.json").write_text(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
