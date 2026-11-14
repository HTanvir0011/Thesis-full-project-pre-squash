"""Write freeze_v4.json once, before the first v4 training run (SAP_v4.md freeze rule).

usage: python tools/freeze_v4.py
Records: SHA-256 of SAP_v4.md, DESIGN_v4.md, the v4 code, the CT subset and skip lists, the projection metadata and the local copies of the
NIH and VinBigData split files, the git commit, the training settings and the primary hypothesis. Only the frozen files must be committed
(other tracked files may carry unrelated local edits).
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
FILES = ["SAP_v4.md", "DESIGN_v4.md", "s4/xm.py", "s4/xm_labels.py", "s4/ct_prep.py", "s4/xm_xai.py", "s4/xm_halluc.py", "s4/cls_hybrid.py",
         "s4/runtag.py", "tools/xm_stats.py", "kernels/down/run_down.py", "kernels/xm_prep/run_xm_prep.py", "s4/data/ct_subset.json",
         "s4/data/ct_skip.txt", "results/qp_meta.csv"]
SPLITS = ["ac_6-s4-prep-nih/nih/split.npy", "ac_6-s4-prep-nih/nih/ids.json", "ac_9-s4-prep-vin/vin/split.npy"]

if __name__ == "__main__":
    out = ROOT / "freeze_v4.json"
    if out.exists():
        sys.exit("freeze_v4.json already exists; the freeze is written once (record any change in DEVIATIONS.md)")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *FILES], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("frozen files are not committed; commit first so the hash identifies the code that runs:\n" + dirty)
    missing = [p for p in FILES if not (ROOT / p).exists()] + [p for p in SPLITS if not (PREP / p).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    rec = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
               files={p: sha(ROOT / p) for p in FILES}, splits={p: sha(PREP / p) for p in SPLITS},
               models=dict(arch="swinv2_tiny_window8_256.ms_in1k", seeds=[0, 1, 2, 3, 4], size=256, lr=3e-4, enc_lr=0.25,
                           ct=dict(epochs=20, volumes_per_step=4, slices=24), xray=dict(epochs=15, batch=32, kd=1.0, align=0.5, temp=2.0),
                           nih=dict(config="orig", epochs=10, batch=32, inits=["imagenet", "P0", "P1"])),
               primary="H11: NIH test mean AUROC over effusion, atelectasis, cardiomegaly, consolidation, lesion (mass or nodule) is higher for "
                       "F_p1 than for F_in; patient-cluster bootstrap, 10,000 resamples; Holm with the two H12 contrasts")
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
