"""Copy downstream run results out of a downloaded kernel output into results/final/down/<dataset>/<tag>/ (tag = <dataset>_<kind>_<input>_s<seed>).

python tools/collect_down.py outputs/down_dl/<kernel> [more ...]
Only finished runs (result.json present) are copied; existing destinations are kept.
"""
import shutil
import sys
from pathlib import Path

DST = Path("results/final/down")
for src in map(Path, sys.argv[1:]):
    for res in sorted((src / "results").glob("*/result.json")):
        tag = res.parent.name
        dst = DST / tag.split("_")[0] / tag
        if dst.exists():
            print("have", dst)
            continue
        shutil.copytree(res.parent, dst)
        print(tag, "->", dst)
