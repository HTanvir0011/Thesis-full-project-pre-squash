"""Kaggle CPU kernel (no GPU quota, no internet): sha256 of every prep output mounted here, to check that the prep copies of the
accounts are identical (RUNPLAN_v3 R6). Pushed by ops/launch_hash.py with the account's prep kernels as sources; writes
/kaggle/working/hash.json {<prep name>: {file: sha256}, "timings": {<prep name>: seconds}} (s4/filehash.py).
The accounts are compared locally with tools/compare_hashes.py.
"""
import glob
import json
import sys
import time
from pathlib import Path

code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
sys.path.insert(0, str(Path(code[0]).parent.parent))
from s4.filehash import hash_mounted  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    hashes, secs = hash_mounted("/kaggle/input")
    Path("/kaggle/working/hash.json").write_text(json.dumps(dict(hashes, timings=secs), indent=1))
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)
