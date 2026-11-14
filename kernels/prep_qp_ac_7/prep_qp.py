"""Kaggle CPU kernel: download the public Qure ct2xr-projections shards (CC BY-NC-SA 4.0, ungated) from Hugging Face and pack the three image
columns we train on into qp.bin + qp_index.npy (+ qp_meta.csv from the s4-code dataset), the layout s4/qp.py reads (same as tools/qp_pack.py).
The output is mounted by the training kernels through kernel_sources, which avoids uploading 6 GB from a slow home connection to every account."""
import csv
import glob
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLS = ("translated", "bone_translated", "lung")
SHARDS, OUT = Path("/tmp/qp_shards"), Path("/kaggle/working/qp")
META = glob.glob("/kaggle/input/**/s4/data/qp_meta.csv", recursive=True)[0]
SHARDS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

for i in range(17):
    n = f"train-{i:05d}-of-00017.parquet"
    p, url = SHARDS / n, f"https://huggingface.co/datasets/qureaiorg/ct2xr-projections/resolve/main/data/{n}"
    want = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    for attempt in range(6):
        try:
            if not (p.exists() and p.stat().st_size == want):
                urllib.request.urlretrieve(url, str(p) + ".part")
                assert os.path.getsize(str(p) + ".part") == want, "short download"
                os.replace(str(p) + ".part", p)
            break
        except Exception as e:  # network hiccup: retry the shard
            print("retry", n, attempt, repr(e), flush=True)
            time.sleep(10)
    else:
        sys.exit(f"{n} failed")
    print("downloaded", n, want, flush=True)

meta = list(csv.DictReader(open(META)))
index, k, pos = np.zeros((len(meta), 3, 2), np.int64), 0, 0
with open(OUT / "qp.bin", "wb") as f:
    for shard in sorted({r["shard"] for r in meta}):
        pf = pq.ParquetFile(SHARDS / shard)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["volume_name", *COLS]).to_pydict()
            for i, name in enumerate(t["volume_name"]):
                assert meta[k]["volume_name"] == name and meta[k]["shard"] == shard, (k, name, meta[k]["volume_name"])
                for c, col in enumerate(COLS):
                    b = t[col][i]["bytes"]
                    f.write(b)
                    index[k, c] = (pos, len(b))
                    pos += len(b)
                k += 1
assert k == len(meta), (k, len(meta))
np.save(OUT / "qp_index.npy", index)
shutil.copy(META, OUT / "qp_meta.csv")
shutil.rmtree(SHARDS, ignore_errors=True)
print(f"packed {k} rows, {pos / 1e9:.2f} GB", flush=True)
assert abs(pos - 6.339e9) < 0.05e9, pos  # the local pack of the same shards is 6.339 GB
print("PREP DONE", flush=True)
