"""sha256 of prep outputs, to check that the prep copies of different accounts are identical (RUNPLAN_v3 R6).

hash_mounted('/kaggle/input') -> ({'nih': {'orig.npy': <sha256 hex>, ...}}, {'nih': seconds}); run on Kaggle by kernels/hash/run_hash.py,
compared across accounts by tools/compare_hashes.py.
"""
import glob
import hashlib
import os
import time

FILES = ("orig.npy", "soft.npy", "split.npy", "labels.npy", "patient.npy", "ids.json", "ann.json", "status.json")


def sha256(path, block=64 << 20):
    """Hex sha256 of a file, streamed in blocks (an NIH array is 7.35 GB)."""
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def hash_prep(d):
    """{file: sha256} of the FILES present in the prep dir d."""
    return {f: sha256(os.path.join(d, f)) for f in FILES if os.path.exists(os.path.join(d, f))}


def hash_mounted(root):
    """Hash every prep dir (a folder holding split.npy, at any depth: kernel outputs mount at varying depths) under root
    -> ({prep name: {file: sha256}}, {prep name: seconds}). Of several dirs with one name the first in sorted order is hashed (the one
    kernels/down/run_down.py's find_data trains on); the others are logged as not hashed."""
    dirs = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "split.npy"), recursive=True)):
        d = os.path.dirname(p)
        if dirs.setdefault(os.path.basename(d), d) != d:
            print(f"not hashed: {d} (same name as {dirs[os.path.basename(d)]})", flush=True)
    assert dirs, f"no prep dir (split.npy) under {root}; inputs: {sorted(glob.glob(os.path.join(root, '*', '*')))[:20]}"
    hashes, secs = {}, {}
    for name, d in dirs.items():
        t0 = time.time()
        hashes[name] = hash_prep(d)
        secs[name] = round(time.time() - t0, 1)
        print(name, d, f"{secs[name]} s", hashes[name], flush=True)
    return hashes, secs
