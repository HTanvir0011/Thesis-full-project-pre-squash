"""Download a finished kernel's output and check it: every DONE run must have a loadable final_ema.pt of plausible size.

usage: python ops/fetch.py <ac_N> <kernel_slug> <dest_dir> [--pattern REGEX] [--timeout SECONDS]   (e.g. --pattern "ann.json|status.json|labels.npy" for prep outputs)
Prints one line per run (DONE / missing) and exits non-zero if any downloaded weight file is truncated or unreadable.
Each of the 4 download attempts is killed (with its child processes) after --timeout seconds (default 1800): a stalled connection otherwise hangs for ever.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MIN_BYTES = 30_000_000  # sanity floor (SwinV2-T fp16 EMA is 60.96 MB, DINOv3-S about 49 MB); a truncated zip already fails torch.load

def run_cli(cmd, env, timeout):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=env)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)  # kg.py -> kaggle.EXE -> python: the whole tree
        out, err = p.communicate()
        return subprocess.CompletedProcess(cmd, 1, out, (err or "") + f"\nattempt timed out after {timeout:.0f} s")
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


if __name__ == "__main__":
    label, slug, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    dest.mkdir(parents=True, exist_ok=True)
    pat = sys.argv[sys.argv.index("--pattern") + 1] if "--pattern" in sys.argv else r"^(?!.*ckpt_last).*$"  # resume checkpoints (1+ GB, optimizer state) stay on Kaggle
    to = float(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 1800
    for attempt in range(4):  # home connection drops (10053) and TLS hiccups happen; files already downloaded are skipped by the CLI
        r = run_cli([sys.executable, str(ROOT / "ops" / "kg.py"), label, "kernels", "output", f"{user}/{slug}", "-p", str(dest), "-q", *(["--file-pattern", pat] if pat else [])],
                    dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"), to)
        if not r.returncode:
            break
        for p in dest.rglob("*"):  # a killed attempt leaves an empty file that the CLI may take for a finished download
            if p.is_file() and p.stat().st_size == 0:
                p.unlink()
        time.sleep(30 * (attempt + 1))  # kernel logs can hold progress-bar blocks (U+2588); the CLI writes them with the default (cp1252) encoding unless UTF-8 mode is on
    if r.returncode:
        sys.exit(f"download failed: {(r.stderr or r.stdout)[-500:]}")
    bad = 0
    summary = dest / "run_summary.json"
    jobs = json.loads(summary.read_text()).get("jobs", []) if summary.exists() else []  # spread/xai summaries have no job list
    for run in sorted(p for p in (dest / "runs").glob("*") if p.is_dir()) if (dest / "runs").exists() else []:
        w = run / "final_ema.pt"
        if not w.exists():
            print(f"{run.name}: no final_ema.pt (unfinished: ckpt_last.pt {'present' if (run / 'ckpt_last.pt').exists() else 'absent'})")
            continue
        try:
            st = torch.load(w, map_location="cpu")
            ok = w.stat().st_size >= MIN_BYTES and "ema" in st
        except Exception as e:  # truncated download
            ok, st = False, {"err": repr(e)[:80]}
        bad += not ok
        print(f"{run.name}: {'ok' if ok else 'BAD'} {w.stat().st_size} B steps={st.get('steps')} backbone={st.get('backbone')}")
    res = dest / "results"
    if res.exists():
        for d in sorted(p for p in res.iterdir() if p.is_dir()):
            print(f"  results/{d.name}: " + ", ".join(sorted(q.name for q in d.iterdir())))
    print(f"{len(jobs)} jobs in run_summary.json" if jobs else "no run_summary.json")
    sys.exit(1 if bad else 0)
