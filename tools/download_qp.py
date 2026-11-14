"""Download the public Qure ct2xr-projections shards (CC BY-NC-SA 4.0, ungated) with retries; skips shards already complete.

python tools/download_qp.py data/qure_proj
"""
import os
import sys
import time
import urllib.request

dst = sys.argv[1]
os.makedirs(dst, exist_ok=True)
for i in range(17):
    n = f"train-{i:05d}-of-00017.parquet"
    p = os.path.join(dst, n)
    url = f"https://huggingface.co/datasets/qureaiorg/ct2xr-projections/resolve/main/data/{n}"
    want = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    if os.path.exists(p) and os.path.getsize(p) == want:
        continue
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, p + ".part")
            assert os.path.getsize(p + ".part") == want
            os.replace(p + ".part", p)
            print("ok", n, want, flush=True)
            break
        except Exception as e:
            print("retry", n, attempt, repr(e), flush=True)
            time.sleep(5)
