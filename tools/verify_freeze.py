"""Check every file whose SHA-256 a freeze recorded (protocol/freeze*.json) against the file in this repository.

python tools/verify_freeze.py
Per freeze: how many recorded files are unchanged (OK), edited after the freeze (CHANGED: code edits are logged in protocol/DEVIATIONS.md;
config.yaml only gained Kaggle accounts, whose names the public snapshot replaces by labels) or absent (data, split arrays and weights
are not part of the public repository); CHANGED files are listed. Paths are as recorded,
relative to the repository root at freeze time (the plans and freezes moved, unchanged, to protocol/ on 14 November 2026). Exit code 1
when an analysis plan or design document changed or is absent.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAT = {"sap_sha256": "SAP.md", "sap_v1_sha256": "SAP.md", "sap_v2_sha256": "SAP_v2.md", "config_sha256": "config.yaml",
        "splits_sha256": "splits/ct_split.csv", "ct_split_sha256": "splits/ct_split.csv", "qp_meta_sha256": "results/qp_meta.csv"}
HEX = re.compile(r"[0-9a-f]{64}")


def recorded(d):
    """(path, sha256) of every file one freeze recorded (freeze.json and freeze_v2.json name their few files by key)."""
    pairs = [(FLAT[k], v) for k, v in d.items() if k in FLAT]
    pairs += [(p, h) for p, h in d.get("files", {}).items()]
    pairs += [("outputs/prep_dl/" + p, h) for p, h in d.get("splits", {}).items()]  # split files are recorded relative to outputs/prep_dl
    pairs += [(w.get("file"), w.get("sha256")) for ws in d.get("models", {}).values() if isinstance(ws, list) for w in ws
              if isinstance(w, dict)]  # freeze_v2 lists weight files per model; freeze_v4's "models" holds settings
    return [(p, h) for p, h in pairs if isinstance(h, str) and HEX.fullmatch(h)]


def status(rel, sha, root):
    for p in (root / rel, root / "protocol" / rel):
        if p.is_file():
            return "OK" if hashlib.sha256(p.read_bytes()).hexdigest() == sha else "CHANGED"
    return "absent"


def main(root=ROOT):
    bad = 0
    for f in sorted((root / "protocol").glob("freeze*.json")):
        rows = [(rel, status(rel, sha, root)) for rel, sha in recorded(json.loads(f.read_text(encoding="utf-8")))]
        n = Counter(s for _, s in rows)
        print(f"{f.name}: " + ", ".join(f"{n[s]} {s}" for s in ("OK", "CHANGED", "absent") if n[s]))
        for rel, s in rows:
            if s == "CHANGED":
                print(f"  CHANGED {rel}")
        bad += sum(s != "OK" and rel.endswith(".md") for rel, s in rows)
    print("every analysis plan and design is unchanged since its freeze" if not bad else f"{bad} plan document(s) changed or absent")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
