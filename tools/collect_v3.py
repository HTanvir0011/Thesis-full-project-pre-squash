"""Collect the v3 phase-2 runs from the fetched kernel outputs into results/v3/down (RUNPLAN_v3 R5) and report them against the matrix.

python tools/collect_v3.py [--src outputs/v3/p2] [--dst results/v3/down] [--expect densenet121,<winner>]
Every <src>/<slug>/results/<tag>/ with result.json is copied without model.pt and ckpt.pt* to <dst>/<dataset>/<tag>/, the layout that
tools/hybrid_stats.py gather() and s4/uq_conformal.load_runs read. A copy counts only when every file s4/cls_hybrid.py writes came home
non-empty (else 'incomplete': fetch again). A run finished in several kernels (done_before copies, reruns) is copied once when all
result.json agree on val_macro_auc and test_macro_auc, else not at all ('conflicts'). An existing destination run is never overwritten; one
that differs or is incomplete is reported ('dst_differs'). Both show each copy's validation macro-AUROC and only whether the test values
differ, so a conflict is settled without looking at the test split. Unfinished: a checkpoint (ckpt.pt, or run_summary.json 'unfinished',
since the phase-2 fetch pattern leaves ckpt.pt on Kaggle) and no result; an unreadable run_summary.json is listed in 'bad_summaries'
(fetch again; its unfinished runs show as missing meanwhile). Expected per encoder (--expect, plus any other encoder found):
NIH and VinBigData 5 configurations x 5 seeds, BDCXR-3257 x 5 folds. Prints a table and writes <dst>/collect_report.json; exit code 0
with runs missing.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS, SEEDS, FOLDS = ("orig", "soft", "early", "hybrid", "ctrl"), range(5), {"nih": [None], "vin": [None], "bdcxr": range(5)}
FILES = ("result.json", "log.jsonl", "val_pred.npy", "val_idx.npy", "test_pred.npy", "test_idx.npy", "test_labels.npy", "test_patient.npy",
         "test_routing.npy")  # s4/cls_hybrid.py output without model.pt; test_patient.npy because all three preps have patient.npy
TAG = re.compile(r"(nih|vin|bdcxr)_hyb_[a-z]+@(.+)_s\d+(_f\d+)?")  # s4/runtag.py tag() of a hyb job; group 2 = encoder


def tag(ds, cfg, arch, seed, fold=None):
    return f"{ds}_hyb_{cfg}@{arch}_s{seed}" + ("" if fold is None else f"_f{fold}")


def metrics(d, files=FILES):
    """(val_macro_auc, test_macro_auc) from d/result.json; None when it is unreadable or one of `files` is missing or empty."""
    try:
        r = json.loads((d / "result.json").read_text())
        return (r["val_macro_auc"], r["test_macro_auc"]) if all((d / f).stat().st_size for f in files) else None  # empty: killed download
    except (OSError, ValueError, KeyError):
        return None


def shown(copies):
    """{copy: (val, test) or None} -> the reported part: validation macro-AUROC per copy and whether the test values differ."""
    return dict(val_macro_auc={k: v and v[0] for k, v in copies.items()}, test_differs=len({v[1] for v in copies.values() if v}) > 1)


def collect(src, dst, expect):
    """Copy the finished runs of src into dst -> report dict (module docstring)."""
    if not src.is_dir():
        raise SystemExit(f"{src}: no such folder")
    found, ckpt, ignored, bad = {}, set(), [], []
    for d in sorted(p for p in src.glob("*/results/*") if p.is_dir()):
        if not TAG.fullmatch(d.name):
            ignored.append(d.relative_to(src).as_posix())
        elif (d / "result.json").exists():
            found.setdefault(d.name, []).append(d)
        elif (d / "ckpt.pt").exists():
            ckpt.add(d.name)
    for s in sorted(src.glob("*/run_summary.json")):
        try:
            ckpt.update(json.loads(s.read_text()).get("unfinished", []))
        except (OSError, ValueError):  # empty or truncated: a killed download
            bad.append(s.relative_to(src).as_posix())
    copied, kept, dup, conflicts, differs, incomplete = {}, [], {}, {}, {}, {}
    for t, dirs in sorted(found.items()):
        m = {d.parent.parent.name: metrics(d, ()) for d in dirs}  # every readable result.json must agree, complete copy or not
        if len({v for v in m.values() if v is not None}) > 1:
            conflicts[t] = shown(m)
            continue
        ok = [d.parent.parent.name for d in dirs if metrics(d) is not None]
        if not ok:
            incomplete[t] = sorted(m)
            continue
        if len(m) > 1:
            dup[t] = sorted(m)
        slug, out = ok[0], dst / t.split("_")[0] / t
        if out.exists():
            have = metrics(out)
            if have == m[slug]:
                kept.append(t)
            else:
                differs[t] = shown({"dst": have, slug: m[slug]})
            continue
        tmp = out.with_name(f".{t}.tmp")  # copy, then rename: an interrupted copy never looks like a collected run
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src / slug / "results" / t, tmp, ignore=shutil.ignore_patterns("model.pt", "ckpt.pt*"))
        tmp.rename(out)
        copied[t] = slug
    done = {d.name for ds in FOLDS for d in (dst / ds).glob(f"{ds}_hyb_*") if metrics(d) is not None}
    archs = list(expect) + sorted({g.group(2) for t in done if (g := TAG.fullmatch(t))} - set(expect))
    exp = {(a, ds, c): {tag(ds, c, a, s, k) for s in SEEDS for k in FOLDS[ds]} for a in archs for ds in FOLDS for c in CONFIGS}
    allexp = set().union(*exp.values())
    incomplete = {t: v for t, v in incomplete.items() if t not in done}  # a broken re-fetch of a run collected earlier is no news
    settled = done | set(conflicts) | set(incomplete)
    return dict(src=str(src), dst=str(dst), expect=list(expect), done=len(done & allexp), expected=len(allexp),
                counts={a: {ds: {c: [len(exp[a, ds, c] & done), len(exp[a, ds, c])] for c in CONFIGS} for ds in FOLDS} for a in archs},
                conflicts=conflicts, dst_differs=differs, incomplete=incomplete, bad_summaries=bad, unfinished=sorted(ckpt - settled),
                missing=sorted(allexp - settled - ckpt), unexpected=sorted(done - allexp), copied=copied, kept=kept, duplicates=dup,
                ignored=ignored)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs/v3/p2"), help="fetched phase-2 kernel outputs: <slug>/results/<tag>/")
    ap.add_argument("--dst", default=str(ROOT / "results/v3/down"))
    ap.add_argument("--expect", default="densenet121", help="comma-separated encoders: DenseNet-121 and the screen winner")
    a = ap.parse_args(argv)
    rep = collect(Path(a.src), Path(a.dst), [e for e in a.expect.split(",") if e])
    print(f"{'encoder':<26}{'data':<7}" + "".join(f"{c:>8}" for c in CONFIGS) + f"{'total':>10}")
    for arch, per in rep["counts"].items():
        for ds, c in per.items():
            n, e = map(sum, zip(*c.values()))
            print(f"{arch:<26}{ds:<7}" + "".join(f"{f'{v[0]}/{v[1]}':>8}" for v in c.values()) + f"{f'{n}/{e}':>10}")
    for k in ("conflicts", "dst_differs"):
        for t, v in rep[k].items():
            print(f"{k}: {t} {json.dumps(v)}")
    for k in ("incomplete", "bad_summaries", "unfinished", "missing", "unexpected", "ignored"):
        if rep[k]:
            s = ", ".join(rep[k])
            print(f"{k} ({len(rep[k])}): {s[:600]}" + (" ..." if len(s) > 600 else ""))
    print(f"{rep['done']}/{rep['expected']} expected runs collected; copied {len(rep['copied'])}, already there {len(rep['kept'])}, "
          f"finished in several kernels {len(rep['duplicates'])}")
    out = Path(a.dst) / "collect_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
