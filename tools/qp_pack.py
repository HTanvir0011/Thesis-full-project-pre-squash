"""Pack the three image columns we train on (translated, bone_translated, lung) of the Qure shards into one file + an offset index.

python tools/qp_pack.py data/qure_proj results/qp_meta.csv OUT_DIR   ->  OUT_DIR/{qp.bin, qp_index.npy, qp_meta.csv}
Row order is the order of qp_meta.csv (shard, row group, row); volume names are asserted to match.
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLS = ("translated", "bone_translated", "lung")

if __name__ == "__main__":
    src, meta_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    meta = list(csv.DictReader(open(meta_csv)))
    index, k, pos = np.zeros((len(meta), 3, 2), np.int64), 0, 0
    with open(out / "qp.bin", "wb") as f:
        for shard in sorted({r["shard"] for r in meta}):
            pf = pq.ParquetFile(src / shard)
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
    np.save(out / "qp_index.npy", index)
    shutil.copy(meta_csv, out / "qp_meta.csv")
    print(f"packed {k} rows, {pos / 1e9:.2f} GB")
