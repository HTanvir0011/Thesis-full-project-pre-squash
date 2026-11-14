"""Kaggle GPU kernel (template): SAP_v3 H10 attribution maps and pointing game (s4/xai_multistream.py) for every finished NIH hybrid run in
the mounted outputs of earlier phase-2 kernels, one process per GPU (logic in s4/xai_runs.py; this file only collects paths).
ops/launch_xai.py fills in LIMIT (boxed NIH test images per run; 0 = all) and EXPECT (the NIH hybrid run tags in the job files of its
--sources). Mounts: s4-code; the NIH prep output (folder nih/: split.npy, ids.json, orig.npy, soft.npy); nih-chest-xrays/data
(BBox_List_2017.csv); the phase-2 kernels (results/nih_hyb_hybrid@<arch>_s<seed>/model.pt), all at varying depths under /kaggle/input.
Writes /kaggle/working/xai/<run tag>.json, .maps.npz, .log and /kaggle/working/run_summary.json (s4/xai_runs.summary: jobs, runs, done,
failed, missing = expected runs with no finished model mounted, also in failed).
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

LIMIT = 0
EXPECT = ['nih_hyb_hybrid@densenet121_s2', 'nih_hyb_hybrid@densenet121_s3']
T0 = time.time()
SESSION_H = 11.5  # Kaggle ends a session at 12 h; the rest covers saving the output
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
sys.path.insert(0, CODE)
from s4.xai_runs import find_boxes, run_all, runs, summary  # noqa: E402

if __name__ == "__main__":
    found = runs(glob.glob("/kaggle/input/**/results/nih_hyb_hybrid@*/model.pt", recursive=True))
    data = sorted(glob.glob("/kaggle/input/**/nih/split.npy", recursive=True))
    assert found and data, f"runs {found}, NIH prep {data}; inputs: {sorted(glob.glob('/kaggle/input/*/*'))[:20]}"
    boxes = find_boxes(glob.glob("/kaggle/input/**/BBox_List_2017.csv", recursive=True))
    print(len(found), "runs:", *found, "| data", data[0], "| boxes", boxes, flush=True)
    rc = run_all(found, str(Path(data[0]).parent), boxes, "/kaggle/working/xai", dict(os.environ, PYTHONPATH=CODE), LIMIT,
                 T0 + SESSION_H * 3600)
    s = summary(found, rc, EXPECT)
    json.dump(s, open("/kaggle/working/run_summary.json", "w"))
    print("ALL DONE; failed:", s["failed"], "| expected, no finished model mounted:", s["missing"], flush=True)
