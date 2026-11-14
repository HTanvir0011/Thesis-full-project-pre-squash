"""Slot scheduler for Kaggle kernels: launches queued kernels per account as slots and quota allow, fetches finished outputs, logs events.

python ops/jobqueue.py outputs/queue_wave1.json [--dry] [--hb FILE]   (--hb: heartbeat file of a second scheduler running another queue file)
Queue file: JSON list of items {id, ac, kernel, cmd, est_h, needs?: [ids], after_file?: path, after_time?: "YYYY-MM-DD HH:MM" (local),
  live?: true (already launched elsewhere: only monitored), fetch?: dest_dir, fetch_pattern?: regex (ops/fetch.py --pattern), retries?: n}
  kernel = Kaggle slug the command pushes; cmd = shell command that pushes it (ops/launch_*.py ...); est_h = GPU session hours it may use
  (launched only when the account has >= est_h + 0.3 left after the est_h of its running kernels); needs = item ids that must have finished COMPLETE; after_file = launch only once
  that file exists (e.g. freeze_v2.json); fetch = download + size-check the output to that dir when the kernel completes.
rerun_missing?: true (down kernels: after the kernel ends, COMPLETE or not, its output is fetched and every job without a result.json goes
  into a new item <id>-r with only those jobs, which mounts the kernel it replaces (--prev) so unfinished runs resume from their checkpoint;
  appended to the queue file; one rerun level; tools/v3_phase2.py leaves it off on chain segments before the last; RUNPLAN_v3 R1, R3).
Rules: at most 2 running/queued kernels per account (Kaggle limit); an ERROR/CANCEL kernel is retried `retries` times (default 1), then marked
failed and everything that needs it is skipped; a failed push used no quota and is retried up to 5 times whatever `retries` says.
Restart-safe: the phase of every item is written to <queue file>.state.json at each poll and restored at start; before pushing an item the scheduler
asks Kaggle about its slug and ADOPTS a kernel that is already RUNNING/QUEUED/NEW (and, at the first poll only, COMPLETE) instead of pushing it
again; an unreadable status (network) never triggers a push. A failed fetch is retried at the next two polls. Every Kaggle call has a timeout, and
outputs/queue.heartbeat is rewritten each poll so a stalled scheduler can be noticed.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s4.runtag import tag  # noqa: E402
PY = sys.executable
ACC = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"]
POLL = 300
CALL_TIMEOUT = 120  # s per Kaggle CLI call: a hung call would otherwise stall every later launch


def run(cmd, timeout, **kw):
    """subprocess.run with a timeout that also kills the child's children: on Windows subprocess.run(timeout=) kills only the direct child and then waits for its
    orphaned grandchildren (kaggle.EXE) to close the pipe, which hung the scheduler for 20+ minutes on one stalled Kaggle call."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", **kw)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        p.communicate()
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def kg(*a):
    try:
        r = run([PY, str(ROOT / "ops" / "kg.py"), *a], CALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return ""
    return (r.stdout or "") + (r.stderr or "")


def status(ac, slug):
    out = kg(ac, "kernels", "status", f"{ACC[ac]}/{slug}")
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1) if m else "NOTFOUND" if "Cannot access kernel" in out else "UNKNOWN"  # NOTFOUND = never pushed, UNKNOWN = Kaggle unreachable


def left(ac):
    m = re.search(r"^GPU\s+[\d.]+h\s+([\d.]+)h", kg(ac, "-W", "quota"), re.M)
    return float(m.group(1)) if m else 0.0


def missing_runs(it):
    """Jobs of a finished down kernel that left no result.json: jobs from the fetched run_summary.json, else from the job file in the
    command; finished = run_summary 'done', else the result folders found in the fetch dir. None if the item has no job file."""
    m = re.search(r"(outputs/jobs/\S+\.json)", it["cmd"])
    if not m:
        return None
    dest = ROOT / it["fetch"]
    summ = next(iter(sorted(dest.rglob("run_summary.json"))), None) if dest.exists() else None
    s = json.loads(summ.read_text()) if summ else {}
    jobs = s.get("jobs") or json.loads((ROOT / m.group(1)).read_text())
    done = set(s.get("done", [])) | set(s.get("done_before", [])) | ({r.parent.name for r in dest.rglob("result.json")} if dest.exists() else set())
    return [j for j in jobs if tag(j) not in done]


def rerun_item(it, missing):
    """New queue item with only the missing jobs, alternated over the two GPUs, same account and prep; its job file is written.
    Slug: <kernel>-r for the first rerun, <kernel>-r2 for a rerun of a rerun."""
    slug = it["kernel"] + ("2" if it["kernel"].endswith("-r") else "-r")
    jobs = [[i % 2, *j[1:]] for i, j in enumerate(missing)]
    (ROOT / "outputs" / "jobs" / f"{slug}.json").write_text(json.dumps(jobs))
    prep = re.search(r"--prep (\S+)", it["cmd"]).group(1)
    item = dict(id=slug, ac=it["ac"], kernel=slug, est_h=it["est_h"], retries=0, rerun_missing=True,  # fetched even on ERROR; never rerun again
                cmd=f"python ops/launch_down.py {it['ac']} {slug} outputs/jobs/{slug}.json --prep {prep} --prev {it['kernel']}", fetch=it["fetch"] + "-r")
    if it.get("fetch_pattern"):
        item["fetch_pattern"] = it["fetch_pattern"]
    return item


def fetch(it, dry):
    """Download the kernel output to it['fetch'] -> True when the fetch succeeded (logs either way)."""
    if not it.get("fetch") or dry:
        return True
    ok = False
    try:
        r = run([PY, str(ROOT / "ops" / "fetch.py"), it["ac"], it["kernel"], it["fetch"], *(["--pattern", it["fetch_pattern"]] if it.get("fetch_pattern") else [])], 7200, cwd=ROOT)
        ok = r.returncode == 0
        log(f"{it['id']} fetch rc {r.returncode}: " + " | ".join((r.stdout or "").strip().splitlines()[-3:]) + ("" if ok else " || " + re.sub(r"\S{60,}", "<url>", (r.stderr or "").strip())[-200:]))
    except subprocess.TimeoutExpired:
        log(f"{it['id']} fetch timed out")
    return ok


def check_missing(it, state, queue_file, dry):
    """rerun_missing items: log the jobs without a result and queue them once as <id>-r (also written to the queue file, for restarts)."""
    if not it.get("rerun_missing") or dry:  # the generator sets rerun_missing (not on chain segments before the last)
        return
    miss = missing_runs(it)
    if not miss:
        return
    if it["id"].endswith("-r2"):
        log(f"{it['id']} MISSING {len(miss)} runs after two reruns: {', '.join(tag(j) for j in miss)[:400]}; continue by hand")
        return
    if it["id"].endswith("-r"):  # second level: only runs that left a checkpoint (a run that failed twice from scratch is a code problem)
        dest = ROOT / it["fetch"]
        summ = next(iter(sorted(dest.rglob("run_summary.json"))), None) if dest.exists() else None
        unfinished = set(json.loads(summ.read_text()).get("unfinished", [])) if summ else set()
        rest = [j for j in miss if tag(j) not in unfinished]
        if rest:
            log(f"{it['id']} MISSING {len(rest)} runs without a checkpoint: {', '.join(tag(j) for j in rest)[:400]}; continue by hand")
        miss = [j for j in miss if tag(j) in unfinished]
        if not miss:
            return
    new = rerun_item(it, miss)
    log(f"{it['id']} MISSING {len(miss)} runs: {', '.join(tag(j) for j in miss)[:400]} -> queued {new['id']}")
    if new["id"] not in state:
        state[new["id"]] = dict(phase="waiting", tries=0, **new)
        q = json.loads(Path(queue_file).read_text())
        Path(queue_file).write_text(json.dumps(q + [new], indent=1))


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    with open(ROOT / "outputs" / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    items = json.loads(Path(sys.argv[1]).read_text())
    dry = "--dry" in sys.argv
    hb = Path(sys.argv[sys.argv.index("--hb") + 1]) if "--hb" in sys.argv else ROOT / "outputs" / "queue.heartbeat"
    state = {i["id"]: dict(phase="live" if i.get("live") else "waiting", tries=0, **i) for i in items}  # waiting | live | done | failed
    sf = Path(sys.argv[1]).with_suffix(".state.json")  # survives a restart: a COMPLETE kernel of a dependent item would otherwise be pushed (and its quota spent) again
    if sf.exists():
        for k, v in json.loads(sf.read_text()).items():
            if k in state:
                state[k].update(v)
    first = True
    while True:
        hb.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        for it in list(state.values()):  # a failed kernel whose output did not come home yet: fetch again, then check for missing runs
            if it.get("pending_fetch") and not dry:
                if fetch(it, dry):
                    it["pending_fetch"] = 0
                    check_missing(it, state, sys.argv[1], dry)
                else:
                    it["pending_fetch"] -= 1
                    if not it["pending_fetch"]:
                        log(f"{it['id']} fetch gave up; finished runs stay on Kaggle; run: python ops/fetch.py {it['ac']} {it['kernel']} {it['fetch']}")
        for it in list(state.values()):  # refresh launched kernels (a list: check_missing may add rerun items to state)
            if it["phase"] != "live":
                continue
            s = status(it["ac"], it["kernel"])
            if s in ("RUNNING", "QUEUED", "NEW", "UNKNOWN", "NOTFOUND"):
                continue
            if s == "COMPLETE":
                it["phase"] = "done"
                log(f"{it['id']} COMPLETE")
                if it.get("fetch") and not dry:
                    ok = fetch(it, dry)
                    it["fetch_fails"] = it.get("fetch_fails", 0) + (not ok)
                    if not ok and it["fetch_fails"] < 3:
                        it["phase"] = "live"  # still COMPLETE at the next poll: fetch again (the CLI skips files it already has)
                    elif not ok:
                        log(f"{it['id']} fetch gave up; run: python ops/fetch.py {it['ac']} {it['kernel']} {it['fetch']}")
                    if ok:
                        check_missing(it, state, sys.argv[1], dry)
            else:
                it["tries"] += 1
                it["phase"] = "waiting" if it["tries"] <= it.get("retries", 1) else "failed"
                log(f"{it['id']} ended {s}; " + ("will retry" if it["phase"] == "waiting" else "FAILED"))
                if it["phase"] == "failed" and it.get("rerun_missing") and it.get("fetch") and not dry:
                    if fetch(it, dry):  # a failed kernel may still have saved finished runs; keep them and rerun only the rest
                        check_missing(it, state, sys.argv[1], dry)
                    else:
                        it["pending_fetch"] = 3  # an unreachable output is not "nothing finished": try again at the next polls
        live = {ac: sum(1 for i in state.values() if i["ac"] == ac and i["phase"] == "live") for ac in ACC}
        for it in list(state.values()):
            if it["phase"] == "waiting" and any(state[n]["phase"] == "failed" for n in it.get("needs", [])):  # before the not-done test
                it["phase"] = "failed"
                log(f"{it['id']} skipped (a dependency failed)")
                continue
            if it["phase"] != "waiting" or any(state[n]["phase"] != "done" for n in it.get("needs", [])):
                continue
            if it.get("after_file") and not (ROOT / it["after_file"]).exists():
                continue
            if it.get("after_time") and time.strftime("%Y-%m-%d %H:%M") < it["after_time"]:
                continue
            s = status(it["ac"], it["kernel"])
            if s == "UNKNOWN":  # Kaggle unreachable: never push blind, ask again next poll
                continue
            if s in ("RUNNING", "QUEUED", "NEW") or (first and s == "COMPLETE"):  # pushed by an earlier scheduler process
                it["phase"] = "live"
                live[it["ac"]] += 1
                log(f"{it['id']} adopted ({s})")
                continue
            committed = sum(i["est_h"] for i in state.values() if i["ac"] == it["ac"] and i["phase"] == "live")  # what the running kernels may still draw from the same quota
            if live[it["ac"]] >= 2 or left(it["ac"]) - committed < it["est_h"] + 0.3:
                continue
            log(f"launch {it['id']} on {it['ac']}: {it['cmd']}")
            if not dry:
                try:
                    out = run(it["cmd"], 1800, shell=True, cwd=ROOT)
                    msg = (out.stdout or "") + (out.stderr or "")
                except subprocess.TimeoutExpired:
                    msg = "push timed out after 1800 s"
                ok = "successfully pushed" in msg
                log(f"  push {'ok' if ok else 'FAILED: ' + msg[-300:]}")
                if not ok:  # a half-done push shows up as RUNNING at the next poll and is adopted then
                    it["pushfail"] = it.get("pushfail", 0) + 1
                    it["phase"] = "waiting" if it["pushfail"] < 5 else "failed"
                    continue
            it["phase"] = "live"
            live[it["ac"]] += 1
        first = False
        if not dry:
            sf.write_text(json.dumps({k: {f: v[f] for f in ("phase", "tries", "pushfail", "fetch_fails", "pending_fetch") if f in v} for k, v in state.items()}))
        if all(i["phase"] in ("done", "failed") for i in state.values()):
            log("QUEUE FINISHED: " + ", ".join(f"{k}={v['phase']}" for k, v in state.items()))
            break
        if dry:
            break
        time.sleep(POLL)
