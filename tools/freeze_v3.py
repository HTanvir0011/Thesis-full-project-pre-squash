"""Write freeze_v3.json once, before phase 1 of v3 (SAP_v3.md freeze rule).

usage: python tools/freeze_v3.py
Records: SHA-256 of SAP_v3.md, DESIGN_v3.md, config.yaml, the v3 code and the local copies of the prep split files (NIH, VinBigData,
BDCXR-3257 folds), the git commit, the screened encoders and the selection rule.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_v2 import sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs" / "prep_dl"
FILES = ["SAP_v3.md", "DESIGN_v3.md", "config.yaml", "s4/cls_hybrid.py", "s4/uq_conformal.py", "s4/xai_multistream.py", "tools/hybrid_stats.py",
         "kernels/down/run_down.py"]
SPLITS = ["ac_6-s4-prep-nih/nih/split.npy", "ac_6-s4-prep-nih/nih/ids.json", "ac_9-s4-prep-vin/vin/split.npy"] + \
         [f"ac_8-s4-prep-bdcxr/bdcxr/split_f{k}.npy" for k in range(5)]

if __name__ == "__main__":
    out = ROOT / "freeze_v3.json"
    if out.exists():
        sys.exit("freeze_v3.json already exists; the freeze is written once (record any change in DEVIATIONS.md)")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("tracked files have uncommitted changes; commit first so the hash identifies the code that runs:\n" + dirty)
    missing = [p for p in FILES if not (ROOT / p).exists()] + [p for p in SPLITS if not (PREP / p).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    rec = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
               files={p: sha(ROOT / p) for p in FILES}, splits={p: sha(PREP / p) for p in SPLITS},
               screen=dict(encoders=["densenet121.ra_in1k", "convnextv2_tiny.fcmae_ft_in22k_in1k", "hf:microsoft/rad-dino",
                                     "swinv2_base_window16_256.ms_in1k"], data="nih", config="orig", seed=0, epochs=10,
                           rule="best validation macro-AUROC goes forward with DenseNet-121; within 0.005 of the best the cheaper (GPU time per epoch) wins"),
               main=dict(configs=["orig", "soft", "early", "hybrid", "ctrl"], epochs=20, seeds=[0, 1, 2, 3, 4], size=256, batch=32, lr=3e-4),
               primary="H6: hybrid > orig in macro-AUROC per data set (NIH, VinBigData, BDCXR-3257), Holm over the three")
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
