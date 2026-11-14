"""Index the public Qure ct2xr-projections shards (CC BY-NC-SA 4.0) and recover the exact additive decomposition of every sample.

Each row has translated (T), bone_translated (Bt), soft_translated (St) and lung (L) images (uint8, 512x512). We found T = a*St + b*Bt + c*L + d
holds to ~0.25 grey levels (R^2 = 1.000) with a sample-specific bone weight b (the dataset's component mixing); so the exact additive
bone image of T is b*Bt. This script fits (a, b, c, d) per row on 128x128 copies and writes one metadata row per sample plus a patient-level split.

python tools/qp_prep.py data/qure_proj results/qp_meta.csv
"""
import csv
import hashlib
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

LABELS = ("lung_opacity", "pleural_effusion", "atelectasis", "cardiomegaly", "consolidation", "lung_lesion")
SEED = 20261026


def dec(d, size=128):
    a = cv2.imdecode(np.frombuffer(d["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)


def split_of(patient):
    h = int(hashlib.sha1(f"{SEED}:{patient}".encode()).hexdigest(), 16) % 100
    return "val" if h < 2 else "test" if h < 4 else "train"


def shard_rows(path):
    pf = pq.ParquetFile(path)
    out = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        for i, r in enumerate(t.to_pylist()):
            T, Bt, St, L = (dec(r[k]) for k in ("translated", "bone_translated", "soft_translated", "lung"))
            A = np.stack([St.ravel(), Bt.ravel(), L.ravel(), np.ones(T.size)], 1)
            c, *_ = np.linalg.lstsq(A, T.ravel(), rcond=None)
            res = float(np.abs(T.ravel() - A @ c).mean())
            patient = "_".join(r["volume_name"].split("_")[:2])
            out.append(dict(shard=Path(path).name, rg=rg, row=i, volume_name=r["volume_name"], patient=patient, split=split_of(patient), kvp=r["kvp"],
                            w_soft=c[0], w_bone=c[1], w_lung=c[2], offset=c[3], resid=res, **{k: r[k] for k in LABELS},
                            spacing_x=r["spacing_x"], spacing_z=r["spacing_z"], num_slices=r["num_slices"]))
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    shards = sorted(src.glob("train-*.parquet"))
    assert shards, f"no shards in {src}"
    with Pool(8) as p:
        rows = [r for part in p.map(shard_rows, shards) for r in part]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    b = np.array([r["w_bone"] for r in rows])
    print(f"{len(rows)} rows from {len(shards)} shards; bone weight mean {b.mean():.3f} sd {b.std():.3f} min {b.min():.3f} max {b.max():.3f}; "
          f"resid mean {np.mean([r['resid'] for r in rows]):.3f} max {max(r['resid'] for r in rows):.3f}; "
          f"splits {dict(zip(*np.unique([r['split'] for r in rows], return_counts=True)))}")
