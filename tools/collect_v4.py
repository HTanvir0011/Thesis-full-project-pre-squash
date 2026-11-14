"""Collect the v4 runs from the fetched kernel outputs into results/v4, the layout tools/xm_stats.py reads.

python tools/collect_v4.py [--src outputs/v4] [--dst results/v4]
Every <src>/<slug>/results/<tag>/ with result.json is copied without model.pt and ckpt.pt* to <dst>/xai/<tag> (ct_xmx_*: cross-modal
maps), <dst>/halluc/<tag> (nih_xmh_*) or <dst>/down/<dataset>/<tag> (the rest). ct_xm_ct_s2 (collapsed) and ct_xm_ct_s6 (the unused
spare) stay out, since xm_stats globs the prefix (DEVIATIONS 8f93fec). A copy counts only when every file came home non-empty (else
'incomplete': fetch again). A run in several kernels (--prev copies, reruns) is copied once when all readable result.json agree, else
not at all ('conflicts'). An existing destination run is never overwritten ('dst_differs' when its result.json differs). Missing runs
are reported by xm_stats. Writes <dst>/collect_report.json.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"ct_xm_ct_s2", "ct_xm_ct_s6"}


def where(dst, tag):
    kind = "xai" if tag.startswith("ct_xmx_") else "halluc" if tag.startswith("nih_xmh_") else None
    return dst / kind / tag if kind else dst / "down" / tag.split("_")[0] / tag


def read(d):
    try:
        return json.loads((d / "result.json").read_text())
    except (OSError, ValueError):  # missing, empty or truncated
        return None


def complete(d):
    return all(f.stat().st_size for f in d.rglob("*") if f.is_file())


def collect(src, dst):
    """Copy the finished runs of src into dst -> report dict (module docstring)."""
    if not src.is_dir():
        raise SystemExit(f"{src}: no such folder")
    found = {}
    for d in sorted(p for p in src.glob("*/results/*") if (p / "result.json").exists()):
        found.setdefault(d.name, []).append(d)
    rep = dict(src=str(src), dst=str(dst), copied={}, kept=[], skipped=[], conflicts={}, incomplete=[], dst_differs=[], duplicates={})
    for t, dirs in sorted(found.items()):
        if t in SKIP:
            rep["skipped"].append(t)
            continue
        r = {d.parent.parent.name: read(d) for d in dirs}
        if len({json.dumps(v, sort_keys=True) for v in r.values() if v is not None}) > 1:
            rep["conflicts"][t] = sorted(r)
            continue
        ok = [d.parent.parent.name for d in dirs if r[d.parent.parent.name] is not None and complete(d)]
        if not ok:
            rep["incomplete"].append(t)
            continue
        if len(dirs) > 1:
            rep["duplicates"][t] = sorted(r)
        out = where(dst, t)
        if out.exists():
            (rep["kept"] if read(out) == r[ok[0]] else rep["dst_differs"]).append(t)
            continue
        tmp = out.with_name(f".{t}.tmp")  # copy, then rename: an interrupted copy never looks like a collected run
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src / ok[0] / "results" / t, tmp, ignore=shutil.ignore_patterns("model.pt", "ckpt.pt*"))
        tmp.rename(out)
        rep["copied"][t] = ok[0]
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs/v4"), help="fetched kernel outputs: <slug>/results/<tag>/")
    ap.add_argument("--dst", default=str(ROOT / "results/v4"))
    a = ap.parse_args(argv)
    rep = collect(Path(a.src), Path(a.dst))
    for k in ("skipped", "conflicts", "incomplete", "dst_differs"):
        if rep[k]:
            print(f"{k}: {rep[k]}")
    n = {}
    for t in rep["copied"]:
        k = where(Path(), t).parent.as_posix()
        n[k] = n.get(k, 0) + 1
    print(f"copied {len(rep['copied'])} {n}, already there {len(rep['kept'])}, in several kernels {len(rep['duplicates'])}")
    out = Path(a.dst) / "collect_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
