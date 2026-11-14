"""Write freeze_v2.json once, before any held-out / locked-set evaluation of a v2 model (SAP_v2.md, RUNPLAN.md rule R4).

usage: python tools/freeze_v2.py ours=<w.pt+w.pt+w.pt> notwin=<...> clia=<...> [name=<...> ...]
Records: SAP_v2.md, config.yaml and split hashes, the SZCH dev/held-out lists, SHA-256 of every weight file per named model, git commit.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from szch_split import is_dev  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    out = ROOT / "freeze_v2.json"
    if out.exists():
        sys.exit("freeze_v2.json already exists; the freeze is written once (record any change in DEVIATIONS.md)")
    models = dict(a.split("=", 1) for a in sys.argv[1:])
    if "ours" not in models:
        sys.exit("give at least ours=<weights joined by +>")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("tracked files have uncommitted changes; commit first so the hash identifies the code that runs:\n" + dirty)
    names = sorted(p.name for p in (ROOT / "data" / "szch" / "CXR").glob("*.png"))
    rec = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
               sap_v2_sha256=sha(ROOT / "SAP_v2.md"), sap_v1_sha256=sha(ROOT / "SAP.md"), config_sha256=sha(ROOT / "config.yaml"),
               ct_split_sha256=sha(ROOT / "splits" / "ct_split.csv"), qp_meta_sha256=sha(ROOT / "results" / "qp_meta.csv"),
               szch_dev=[n for n in names if is_dev(n)], szch_heldout=[n for n in names if not is_dev(n)],
               models={k: [dict(file=w, sha256=sha(w)) for w in v.split("+")] for k, v in models.items()})
    assert len(rec["szch_dev"]) == 24 and len(rec["szch_heldout"]) == 51, (len(rec["szch_dev"]), len(rec["szch_heldout"]))
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("szch_dev", "szch_heldout")}, indent=1))
