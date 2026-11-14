"""Fixed patient-level split + usability flags from the rendered CT sidecars. Written once; its hash goes into freeze.json.

python tools/make_split.py OUT.csv RENDER_DIR [RENDER_DIR ...]
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SEED = 20261026
USABLE = dict(max_pad_rows=50, min_lung_cm3=1500, max_lung_cm3=9000, min_bone_cm3=300)


def split_of(patient):
    h = int(hashlib.sha256(f"{SEED}:{patient}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 80 else "test"


def main(out, dirs):
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if not p.with_suffix(".npz").exists():
                continue  # sidecar of an unfinished render
            m = json.loads(p.read_text())
            why = []
            if m["lung_truncated"]:
                why.append("apex_or_base_cut")
            if m["pad_top_rows"] + m["pad_bottom_rows"] > USABLE["max_pad_rows"]:
                why.append("padded")
            if not USABLE["min_lung_cm3"] <= m["lung_vol_cm3"] <= USABLE["max_lung_cm3"]:
                why.append("lung_volume")
            if m["bone_vol_cm3"] < USABLE["min_bone_cm3"]:
                why.append("bone_volume")
            rows.append(dict(name=p.stem, patient=m["patient"], split=split_of(m["patient"]), usable=int(not why),
                             excluded_because=";".join(why), n_nodules=m["n_nodules"], source=m["source"], dz=m["dz"],
                             manufacturer=m["manufacturer"], lung_cm3=round(m["lung_vol_cm3"]), bone_cm3=round(m["bone_vol_cm3"])))
    rows.sort(key=lambda r: r["name"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("rendered:", len(rows), "usable:", sum(r["usable"] for r in rows))
    print("by split (usable):", dict(Counter(r["split"] for r in rows if r["usable"])))
    print("nodules by split (usable):", {s: sum(r["n_nodules"] for r in rows if r["usable"] and r["split"] == s) for s in ("train", "val", "test")})
    print("excluded:", dict(Counter(x for r in rows for x in r["excluded_because"].split(";") if x)))
    print("sha256:", hashlib.sha256(Path(out).read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
