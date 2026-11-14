"""Assemble kdata/s4-bank/bank/ = every finished render (npz + json) + splits.csv + ref_q.npy (Kaggle flattens bank/ to the dataset root).

python tools/package_bank.py SPLITS.csv REF_Q.npy RENDER_DIR [RENDER_DIR ...]
"""
import csv
import shutil
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
splits, ref_q, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
dst = ROOT / "kdata" / "s4-bank" / "bank"
if dst.parent.exists():
    shutil.rmtree(dst.parent)
dst.mkdir(parents=True)
names = {r["name"] for r in csv.DictReader(open(splits)) if r["usable"] == "1"}
n = 0
for d in dirs:
    for p in Path(d).glob("*.npz"):
        if p.stem in names:
            with np.load(p) as z:  # fail loudly on a half-downloaded file (one was an HTML error page)
                assert {'a_bone', 'a_nl', 'a_lung'} <= set(z.files) and z['a_bone'].shape == (512, 512), p
            shutil.copy(p, dst / p.name)
            shutil.copy(p.with_suffix(".json"), dst / (p.stem + ".json"))
            n += 1
shutil.copy(splits, dst / "splits.csv")
shutil.copy(ref_q, dst / "ref_q.npy")
print(f"bank: {n} CTs of {len(names)} usable, {sum(f.stat().st_size for f in dst.iterdir()) / 1e9:.2f} GB")
assert n == len(names), "some usable CTs have no render file"
