"""Save the composite-unit lesion library of the TRAIN CTs (what s4.train builds from the render bank for CLIA / twin arms), so a QP-only
arm can run without the 1.2 GB bank (laptop GPU, or a Kaggle account that cannot take the bank).

python tools/make_lib_img.py --bank outputs/shard0/render outputs/shard1/render outputs/shard2/render --split splits/ct_split.csv --out s4/data/lib_train.npz
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4 import compose as C  # noqa: E402
from s4 import lesion_lib  # noqa: E402
from s4.bank import load_bank  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.split)) if r["usable"] == "1"]
    bank = load_bank(a.bank, [r["name"] for r in rows], a.device)
    split = {r["name"]: r["split"] for r in rows}
    tr = [i for i, n in enumerate(bank["names"]) if split[n] == "train"]  # same selection as s4.train (no --limit_ct)
    lib, consts = lesion_lib.build(bank, tr, C.Physics(a.device), a.device)
    lesion_lib.save(lib, consts, a.out)
    print(f"{len(tr)} train CTs -> {len(lib)} lesions, consts {consts}, saved {a.out}")
