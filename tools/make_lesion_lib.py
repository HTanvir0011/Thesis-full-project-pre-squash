"""Lesion library (composite-intensity patches) from one split of the bank.

python tools/make_lesion_lib.py OUT.npz SPLIT.csv WHICH RENDER_DIR [RENDER_DIR ...]
"""
import csv
import sys

import torch

from s4 import compose, lesion_lib
from s4.bank import load_bank

out, split_csv, which, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = [r for r in csv.DictReader(open(split_csv)) if r["usable"] == "1"]
bank = load_bank(dirs, [r["name"] for r in rows], dev)
sp = {r["name"]: r["split"] for r in rows}
idx = [i for i, n in enumerate(bank["names"]) if sp[n] == which]
lib, consts = lesion_lib.build(bank, idx, compose.Physics(dev), dev)
lesion_lib.save(lib, consts, out)
print(f"{len(lib)} lesions from {len(idx)} {which} CTs; consts {consts}; peak x-units {max(p['x_patch'].max() for p in lib):.3f}")
