"""SZCH-75 development / held-out split used for v2 decisions: image name hash % 3 == 0 -> dev (24), else held-out (51).

python tools/szch_split.py data/szch  -> prints the dev names; `is_dev(name)` for scripts.
"""
import hashlib
import sys
from pathlib import Path


def is_dev(name: str) -> bool:
    return int(hashlib.sha1(f"szch:{name}".encode()).hexdigest(), 16) % 3 == 0


if __name__ == "__main__":
    names = sorted(p.name for p in (Path(sys.argv[1]) / "CXR").glob("*.png"))
    dev = [n for n in names if is_dev(n)]
    print(len(dev), "dev of", len(names), dev)
