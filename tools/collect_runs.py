"""Merge a downloaded kernel output (RUN_DIR/{results,runs}) into ROOT/{results,weights}, refusing corrupt files.

python tools/collect_runs.py outputs/abl1 outputs/abl_all
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for run in sorted((src / "results").iterdir()):
    for f in run.rglob("*.csv"):
        assert len(pd.read_csv(f)) > 0, f
    w = src / "runs" / run.name
    assert (w / "final_ema.pt").stat().st_size > 50e6, f"corrupt weights {w}"
    shutil.copytree(run, dst / "results" / run.name, dirs_exist_ok=True)
    (dst / "weights" / run.name).mkdir(parents=True, exist_ok=True)
    for n in ("final_ema.pt", "log.jsonl"):
        shutil.copy(w / n, dst / "weights" / run.name / n)
    print("collected", run.name)
