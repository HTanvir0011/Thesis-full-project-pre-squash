"""Assemble the public snapshot of the project (code, Kaggle kernels, analysis plans, manuscripts, figures and every result kept as text)
in --out, leaving out what must not or need not be published, and write EXCLUDED.md there.

python tools/build_release.py --out <new folder outside the repository> [--extra FILE ...]
Sources: this repository's working tree (results/ included) and the text outputs of the Kaggle kernels (outputs/). The compiled previews
of the manuscript become manuscripts/manuscript.pdf and manuscripts/supplementary.pdf; --extra files (the thesis PDF) go to manuscripts/.
Rules, in order:
  credentials   never copied (token files, kaggle.json, .env, keys); every copied text file is also scanned for token patterns, and
                one hit aborts the build
  data/weights  raw and staged data (data/, kdata/), archives, DICOM/NIfTI/parquet, model weights and arrays (.pt .npy .npz .pkl ...)
  restricted    in outputs/ and results/, VinDr-RibCXR (incl. the rib benchmark), VinBigData, BDCXR-3257, JSRT, Node21 and CT-RATE-derived
                folders keep only aggregate files (result.json, run_summary.json, log.jsonl, *.log, collect_report.json, hash.json,
                *stats*.json): no per-image or per-site file and no annotation (data use agreements; DESIGN_v4.md section 2)
  superseded    the drafts of the two earlier, separate manuscripts and their Word export (submission/ holds the combined manuscript)
  scratch       smoke, dry, debug, GPU/TPU test and benchmark runs and kernels, data-staging copies, and the launch, queue, watch and
                download logs at the top of outputs/
  size          no file over 95 MB (GitHub refuses files over 100 MB)
Kaggle usernames (the accounts of config.yaml and the ones they replaced) become their labels ac_1 ... ac_9 in every path and text file;
a username left in any file aborts the build, like a token.
"""
import argparse
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_TOP = {"outputs", "kdata", "data"}  # pruned only at the top of the repository (s4/data holds the physics tables the code needs)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "pytest_tmp"}
SKIP_FILES = {"overleaf_submission.zip", ".gitignore"}  # the snapshot is curated already: the repository's ignore rules would drop parts of it
RENAME = {"overleaf_submission_preview.pdf": "manuscripts/manuscript.pdf",
          "overleaf_supplementary_preview.pdf": "manuscripts/supplementary.pdf"}
CRED = re.compile(r"(token|secret|password|credential)[^/]*\.(txt|json|md)$|(^|/)kaggle\.json$|(^|/)\.env$|\.(key|pem)$", re.I)
BINARY = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".npy", ".npz", ".pkl", ".h5", ".hdf5", ".ts", ".parquet", ".dcm", ".nii",
          ".gz", ".zip", ".tar", ".7z", ".mhd", ".raw", ".pack", ".idx", ".xml", ".pyc", ".mat"}
RESTRICTED = re.compile(r"(^|[/_.-])(ribcxr|vindr|vin|vinbig\w*|bdcxr|jsrt|node21|ctprep|xm_prep|xm-teacher|ct-a|ct-b|xmx)([/_.-]|$)"
                        r"|(^|/)final/rib(/|$)|(^|/)ct/|(^|/)ct_xm", re.I)
AGGREGATE = re.compile(r"(^|/)(result\.json|run_summary\.json|log\.jsonl|collect_report\.json|hash\.json|[^/]*stats[^/]*\.json|[^/]+\.log)$",
                       re.I)
SUPERSEDED = re.compile(r"^paper/(main(_cls)?\.tex|[^/]*highlights\.md|refs_todo\.md|[^/]*\.docx|[^/]*_preview\.pdf)$|^tools/docx/")
SCRATCH = re.compile(
    r"^outputs/(smoke|clssmoke|detsmoke|dry|debug|dummy|kinit|tmp_|test_render|render_test|ref_tmp|stats_test|overfit_check|gputest"
    r"|gt_ac_|tpudiag|bench_|qp_ds_ac_|qure_ds|qp_pack_test|qpprep_log|scr1_fail)[^/]*/"
    r"|^outputs/[^/]*(\.(log|out|err|heartbeat|watch|started|sh|png)|state\.json|before[^/]*\.json)$|^outputs/(queue|smoke|dry|\.)[^/]*$"
    r"|^kernels/(gputest|tpu_diag|debug|bench)[^/]*/|^kernels/[^/]*smoke[^/]*/")
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}"
                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|\"key\"\s*:\s*\"[0-9a-f]{32}\"")
TEXT = {".py", ".md", ".txt", ".json", ".jsonl", ".log", ".csv", ".tex", ".bib", ".yaml", ".yml", ".cfg", ".ini", ".sh", ".toml", ".r"}
MAX_BYTES = 95_000_000


def rule(rel):
    """Why a file (posix path in the snapshot) is left out, or None when it is copied."""
    low = rel.lower()
    if CRED.search(low):
        return "credentials"
    if Path(low).suffix in BINARY:
        return "data/weights"
    if low.startswith(("outputs/", "results/")) and RESTRICTED.search(low) and not AGGREGATE.search(low):
        return "restricted"
    if low.startswith("s4/data/ct_"):  # CT-RATE volume lists: nothing derived from CT-RATE leaves but metrics (DESIGN_v4.md section 2)
        return "restricted"
    if SUPERSEDED.search(low):
        return "superseded"
    if SCRATCH.search(low):
        return "scratch"
    return None


def accounts(cfg):
    """Kaggle username -> label: every account of config.yaml and the account it replaced (named in the comment of its line)."""
    names = {}
    for label, user, note in re.findall(r"^\s+(ac_\d+):\s*(\S+)(.*)$", cfg.read_text(encoding="utf-8"), re.M):
        names[user] = label
        names.update((old, label + "_former") for old in re.findall(r"replaced (\w+)", note))
    return names


def walk(base, prefix=""):
    """(source path, snapshot path) for every file under base, pruning SKIP_DIRS."""
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not (Path(dp) == ROOT and d in SKIP_TOP)]
        for f in fn:
            if f not in SKIP_FILES:
                src = Path(dp) / f
                yield src, prefix + src.relative_to(base).as_posix()


def build(out, extra):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("--out must be a new or empty folder (a stale file would survive the build)")
    names = accounts(ROOT / "config.yaml")
    users = re.compile("|".join(re.escape(u) for u in sorted(names, key=len, reverse=True)).encode())
    redact = lambda b: users.sub(lambda m: names[m.group().decode()].encode(), b)  # noqa: E731
    plan = [(s, RENAME.get(r, r)) for s, r in walk(ROOT)]
    plan += list(walk(ROOT / "outputs", "outputs/"))
    plan += [(Path(e), "manuscripts/" + Path(e).name) for e in extra]
    kept, dropped, hits = [], Counter(), []
    for src, rel in plan:
        why = rule(rel)
        if why is None and src.stat().st_size > MAX_BYTES:
            why = "size"
        if why:
            dropped[why] += 1
            continue
        data, rel = src.read_bytes(), redact(rel.encode()).decode()
        if src.suffix.lower() in TEXT:
            data = redact(data)
            if SECRET.search(data.decode("utf-8", errors="ignore")):
                hits.append(rel)  # the matched text itself is never printed
        if users.search(data):
            hits.append(rel)  # a username in a file that cannot be redacted
        kept.append((rel, data))
    if hits:
        raise SystemExit(f"secret or username found, nothing written: {hits}")
    for rel, data in kept:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\noutputs/pytest_tmp/\noverleaf_submission.zip\n", encoding="utf-8",
                                    newline="\n")
    (out / "EXCLUDED.md").write_text(
        "# What this snapshot leaves out\n\nBuilt by `tools/build_release.py` from the project repository. Files left out, by rule:\n\n"
        + "".join(f"- **{k}**: {v}\n" for k, v in sorted(dropped.items()))
        + "\nWhy: credentials are never published. Raw and staged data, model weights and prediction arrays are too large for git and "
          "partly licensed. For VinDr-RibCXR, VinBigData, BDCXR-3257, JSRT, Node21 and CT-RATE-derived outputs only aggregate results are "
          "kept, as their data use agreements and the analysis plans require. Superseded drafts and scratch runs (smoke, dry, debug and "
          "hardware tests, launch and queue logs) carry no result. Kaggle account names are replaced by the labels ac_1 ... ac_9.\n",
        encoding="utf-8", newline="\n")
    total = sum(len(d) for _, d in kept)
    print(f"kept {len(kept)} files ({total / 1e6:.1f} MB); left out {dict(dropped)}")
    return kept, dropped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="further files to publish under manuscripts/ (the compiled thesis)")
    a = ap.parse_args(argv)
    if Path(a.out).resolve().is_relative_to(ROOT):
        raise SystemExit("--out must be outside the repository")
    build(a.out, a.extra)


if __name__ == "__main__":
    main()
