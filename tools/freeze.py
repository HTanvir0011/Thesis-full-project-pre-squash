"""Write freeze.json: hashes of the split, SAP, config and the git commit. Run once, before any locked-set evaluation."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if (ROOT / "freeze.json").exists():
    sys.exit("freeze.json already exists; the freeze is written once (record any change in DEVIATIONS.md instead)")
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
if dirty:
    sys.exit("working tree not clean; commit first so the hash identifies the code that ran:\n" + dirty)
out = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
           text=True).stdout.strip(), splits_sha256=sha(sys.argv[1]), sap_sha256=sha(ROOT / "SAP.md"), config_sha256=sha(ROOT / "config.yaml"))
(ROOT / "freeze.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
