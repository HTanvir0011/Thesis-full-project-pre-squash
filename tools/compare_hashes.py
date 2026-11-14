"""Check that the prep copies of the accounts are identical (RUNPLAN_v3 R6): each file of a prep must have one sha256 in all its copies.

python tools/compare_hashes.py outputs/hash/*/hash.json
Each hash.json is one hash kernel's output (kernels/hash/run_hash.py, fetched to outputs/hash/<ac_N>/) and is labelled by its folder.
Prints identical / DIFFERENT per prep and file; a different file lists which copies share which hash, and a file missing from a copy
counts as different. Exit code 1 if any file differs.
"""
import glob
import json
import sys
from pathlib import Path


def compare(paths):
    """hash.json paths -> (report lines, True when no file differs between the copies of a prep)."""
    copies = {}  # prep -> {label: {file: sha256}}
    for p in map(Path, paths):
        for prep, files in json.loads(p.read_text()).items():
            if prep == "timings":  # seconds per prep, not hashes
                continue
            if p.parent.name in copies.setdefault(prep, {}):
                raise SystemExit(f"two hash.json files labelled {p.parent.name}: give each copy its own folder")
            copies[prep][p.parent.name] = files
    lines, ok = [], True
    for prep, by in sorted(copies.items()):
        lines.append(f"{prep}: {len(by)} {'copies' if len(by) > 1 else 'copy, nothing to compare'} ({', '.join(sorted(by))})")
        for f in sorted({f for files in by.values() for f in files}) if len(by) > 1 else []:
            groups = {}
            for label, files in sorted(by.items()):
                groups.setdefault(files.get(f, "missing"), []).append(label)
            ok &= len(groups) == 1
            lines.append(f"  {f:<12} " + ("identical" if len(groups) == 1 else
                                          "DIFFERENT  " + " | ".join(f"{h[:12]}: {', '.join(ls)}" for h, ls in groups.items())))
    return lines, ok


if __name__ == "__main__":
    paths = [q for a in sys.argv[1:] for q in sorted(glob.glob(a)) or [a]]  # PowerShell passes the wildcard unexpanded
    if not paths:
        raise SystemExit(__doc__)
    lines, ok = compare(paths)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)
