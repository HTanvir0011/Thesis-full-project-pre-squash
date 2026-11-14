"""Merge the shard CSVs of the BS-LDM baseline kernel (fidelity_<g>.csv, audit_<g>.csv) into one file each and print the SZCH summary.

python tools/merge_bsldm.py outputs/bsldm/full/out [--dest results/final] [--held]
Writes <dest>/szch/bsldm.csv and <dest>/audit_szch/bsldm.csv (the names tools/final_stats.py reads) and <dir>/merged_*.csv. SZCH numbers are shown for dev-24
(split of tools/szch_split.py); --held adds the held-out-51 and all-75 columns and is for AFTER tools/freeze_v2.py (SAP_v2.md, R4). The audit is development
evidence (all 75 images) like every other SZCH audit; the full run covers sites 0-2 of the 6 per image (225 sites, the same sites as the other methods' first three).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from szch_split import is_dev  # noqa: E402

if __name__ == "__main__":
    d = Path(sys.argv[1])
    dest = Path(sys.argv[sys.argv.index("--dest") + 1]) if "--dest" in sys.argv else None
    fid = pd.concat([pd.read_csv(p) for p in sorted(d.glob("fidelity_*.csv"))]).sort_values("image").reset_index(drop=True)
    aud = pd.concat([pd.read_csv(p) for p in sorted(d.glob("audit_*.csv"))]).sort_values(["image", "site"]).reset_index(drop=True) if list(d.glob("audit_*.csv")) else None
    fid.to_csv(d / "merged_fidelity.csv", index=False)
    print(f"fidelity: {len(fid)} images (expected 75), {fid.image.nunique()} distinct")
    dev = fid.image.map(is_dev)
    cols = [c for c in ("psnr_cal_bone", "psnr_cal", "ssim_cal", "ssim_cal_s4", "msssim_cal", "psnr_raw", "hp_ratio_nonbone") if c in fid]
    tab = {"dev-24": fid[dev][cols].mean()}
    if "--held" in sys.argv:
        tab |= {"held-51": fid[~dev][cols].mean(), "all-75": fid[cols].mean()}
    print(pd.DataFrame(tab).round(4).to_string())
    if aud is not None:
        aud.to_csv(d / "merged_audit.csv", index=False)
        lo, hi = aud.overlap.quantile([1 / 3, 2 / 3])
        print(f"audit: {len(aud)} sites (expected 225 for --sites 0:3): A {aud.absorption.mean():.4f}, low-overlap {aud[aud.overlap <= lo].absorption.mean():.4f}, "
              f"high-overlap {aud[aud.overlap > hi].absorption.mean():.4f}, leak {aud.leak.mean():.4f}")
    if dest:
        (dest / "szch").mkdir(parents=True, exist_ok=True)
        fid.to_csv(dest / "szch" / "bsldm.csv", index=False)
        if aud is not None:
            (dest / "audit_szch").mkdir(parents=True, exist_ok=True)
            aud.to_csv(dest / "audit_szch" / "bsldm.csv", index=False)
        print("written to", dest)
