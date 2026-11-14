"""Phase-2 schedule for v3 (SAP_v3.md; DESIGN_v3.md §6, §8; RUNPLAN_v3.md): the full matrix packed into Kaggle kernels on the 9 accounts.

python tools/v3_phase2.py --winner {densenet121|convnextv2_tiny|rad-dino|swinv2_base_window16_256} [--warm] [--pre_refill] [--write] [--tag p2a]
--warm        kernels first read the prepared arrays sequentially into the page cache (RUNPLAN_v3 R2; measured 4 Nov: NIH 64 s, VinBigData 49 s)
--pre_refill  let DenseNet units use the quota left before the 2026-11-07 refill (DenseNet runs in every scenario)
--write       write outputs/jobs/<slug>.json and outputs/queue_v3_<tag>_<winner>.json (refuses to overwrite: a pushed queue is frozen)
Rules: a kernel is one 12-h session with 2 GPU lanes, a lane holds runs back to back up to the data set's lane limit (cold start included);
a run longer than that continues in a chain of kernels on the same account (resume, RUNPLAN_v3 R1); one encoder per kernel; DenseNet
units are packed and placed first, so their kernels, slugs and accounts are the same whichever encoder wins; at most 2 kernels at once per
account; weekly quota 30 h per account (29 planned); runs use the account that owns their prep output (NIH: ac_6, ac_7 and fixed copies
on ac_2-ac_5; VinBigData: ac_9, ac_1 (competition rules); BDCXR-3257: ac_8 only (data use agreement)).
The week table is indicative: ops/jobqueue.py launches in queue order whenever an account has quota, so work can start earlier.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPOCHS, SEEDS = 20, range(5)
CONFIGS, MULTI = ("orig", "soft", "early", "hybrid", "ctrl"), ("hybrid", "ctrl")
ENC = ("densenet121", "convnextv2_tiny", "rad-dino", "swinv2_base_window16_256")
SHORT = {"densenet121": "dn", "convnextv2_tiny": "cnx", "rad-dino": "rd", "swinv2_base_window16_256": "swb"}
ENC_LR = {"densenet121": 1.0, "convnextv2_tiny": 0.25, "rad-dino": 0.25, "swinv2_base_window16_256": 0.25}  # DEVIATIONS.md 2026-11-04: x0.25 without batch norm
XAI_H = {"densenet121": 0.35, "convnextv2_tiny": 0.7, "rad-dino": 0.8, "swinv2_base_window16_256": 2.0}  # h per hybrid model on one GPU (~1,000 NIH box targets; estimate)
JOB_OVERHEAD_H, RESUME_H, PREP_COPY_H, WEEK_H, PRE_MARGIN = 1 / 60, 0.15, 1.8, 29.0, 0.5
NIH_COPIES = ["ac_2", "ac_3", "ac_4", "ac_5"]  # fixed (s4-wts3 present; 1.72 h measured on ac_7) so the DenseNet schedule never changes
DATA = {  # sizes from the local split files; cold = first pass over the arrays with random reads; lane = hours per GPU lane incl. cold
    "nih": dict(train=77890, val=8634, test=25596, folds=[None], cold=1.5, warm=0.05, lane=11.0, holders=["ac_6", "ac_7"] + NIH_COPIES),
    "vin": dict(train=10433, val=1469, test=2943, folds=[None], cold=0.75, warm=0.05, lane=11.0, holders=["ac_9", "ac_1"]),
    "bdcxr": dict(train=2344, val=262, test=652, folds=[0, 1, 2, 3, 4], cold=0.05, warm=0.05, lane=9.6, holders=["ac_8"]),  # 3 kernels per week
}
# quota left on 2026-11-05 12:15 (ops/jobqueue.py left()); no kernel running
PRE_QUOTA = {"ac_1": 0.1, "ac_2": 0.59, "ac_3": 3.29, "ac_4": 1.16, "ac_5": 1.34, "ac_6": 3.44, "ac_7": 9.4, "ac_8": 10.78, "ac_9": 5.24}
PERIODS = ("pre", "w1", "w2", "w3", "w4", "w5", "w6")  # quota weeks start 7, 14, 21, 28 Nov, 5, 12 Dec 06:00 local (UTC+6)
PERIOD_START_H = {"pre": -42.0, **{f"w{i}": 168.0 * (i - 1) for i in range(1, 7)}}  # hours after the 7 Nov refill (pre from ~5 Nov 09:00)


def job_hours(ds, arch, cfg, tim, safety):
    t = tim[arch]["multi" if cfg in MULTI else "single"] * (tim.get("vin_factor", 1.3) if ds == "vin" else 1.0)
    d = DATA[ds]
    ev = t / 3 / 32  # s per evaluated image (forward only)
    return (EPOCHS * ((d["train"] // 32) * t + d["val"] * ev) + (d["val"] + d["test"]) * ev) / 3600 * safety + JOB_OVERHEAD_H


def epoch_hours(ds, arch, cfg, tim, safety):
    t = tim[arch]["multi" if cfg in MULTI else "single"] * (tim.get("vin_factor", 1.3) if ds == "vin" else 1.0)
    return ((DATA[ds]["train"] // 32) * t + DATA[ds]["val"] * t / 96) / 3600 * safety


def make_jobs(archs, tim, safety):
    return [dict(ds=ds, arch=arch, cfg=cfg, seed=seed, fold=fold, h=job_hours(ds, arch, cfg, tim, safety), epoch_h=epoch_hours(ds, arch, cfg, tim, safety))
            for arch in archs for ds, d in DATA.items() for cfg in CONFIGS for seed in SEEDS for fold in d["folds"]]


def argv(j, tim):
    a = ["--arch", tim["arch_flag"][j["arch"]]] + (["--enc_lr", str(ENC_LR[j["arch"]])] if ENC_LR[j["arch"]] != 1.0 else [])
    acc = tim[j["arch"]]["accum_multi" if j["cfg"] in MULTI else "accum_single"]
    return a + (["--accum", str(acc)] if acc > 1 else []) + (["--split", f"split_f{j['fold']}.npy"] if j["fold"] is not None else [])


def pack(jobs, cold, lane_max):
    """Fewest kernels whose 2 balanced lanes hold the runs (longest first to the least-loaded lane); longer runs become chains."""
    cap = lane_max - cold
    long_ = sorted([j for j in jobs if j["h"] > cap], key=lambda j: -j["h"])
    short = sorted([j for j in jobs if j["h"] <= cap], key=lambda j: -j["h"])
    kernels, nk = [], max(1, math.ceil(sum(j["h"] for j in short) / (2 * cap)))
    while short:
        lanes, load = [[] for _ in range(2 * nk)], [0.0] * (2 * nk)
        for j in short:
            i = min(range(2 * nk), key=lambda i: load[i])
            if load[i] + j["h"] > cap:
                nk += 1
                break
            lanes[i].append(j)
            load[i] += j["h"]
        else:
            kernels = [dict(lanes=[lanes[2 * i], lanes[2 * i + 1]], chain=None, seg=0, nseg=1) for i in range(nk)]
            break
    chains = []
    for i in range(0, len(long_), 2):
        pair = long_[i:i + 2]
        # a session ends a run only at an epoch end, and starts the next epoch only if it fits (cls_hybrid --deadline_h, run_down's 0.5 h
        # reserve): count whole epochs per session, not hours
        per = [max(1, math.floor((cap - RESUME_H - 0.5) / (1.05 * j["epoch_h"]))) for j in pair]
        nseg = max(math.ceil(EPOCHS / e) for e in per)
        chains.append([dict(lanes=[[dict(j, h=j["h"] / nseg + RESUME_H)] for j in pair] + ([[]] if len(pair) == 1 else []), chain=i // 2, seg=s, nseg=nseg)
                       for s in range(nseg)])
    for k in kernels + [s for c in chains for s in c]:
        k["hours"] = cold + max(sum(x["h"] for x in lane) for lane in k["lanes"])
    return kernels, chains


def carve_pre(jobs, warm, pre_cap):
    """Before the refill: one DenseNet kernel per account, sized to the account's leftover quota (largest leftover first), filled longest
    first into two lanes from the data set that uses most of it; only data the account holds before the refill. pre_cap is reduced.
    -> (kernels, the jobs not carved). DenseNet runs in every scenario, so these kernels do not depend on the screen."""
    held = {"nih": ["ac_6", "ac_7"], "vin": ["ac_9", "ac_1"], "bdcxr": ["ac_8"]}
    pool, out = list(jobs), []
    for ac in sorted(pre_cap, key=lambda a: (-pre_cap[a], a)):
        best = None
        for ds in DATA:
            cold = DATA[ds]["warm" if warm else "cold"]
            cap = min(DATA[ds]["lane"], pre_cap[ac]) - cold
            if ac not in held[ds] or cap <= 0:
                continue
            lanes, load = [[], []], [0.0, 0.0]
            for j in sorted((j for j in pool if j["ds"] == ds), key=lambda j: -j["h"]):
                for i in sorted(range(2), key=lambda i: load[i]):
                    if load[i] + j["h"] <= cap:
                        lanes[i].append(j)
                        load[i] += j["h"]
                        break
            if sum(load) and (best is None or sum(load) > best[1]):
                best = (ds, sum(load), lanes, cold)
        if best:
            ds, _, lanes, cold = best
            k = dict(lanes=lanes, chain=None, seg=0, nseg=1, ds=ds, arch="densenet121", ac=ac, period="pre",
                     hours=cold + max(sum(x["h"] for x in lane) for lane in lanes))
            pre_cap[ac] -= k["hours"]
            out.append(k)
            taken = {id(j) for lane in lanes for j in lane}
            pool = [j for j in pool if id(j) not in taken]
    return out, pool


def assign(units, pre_cap, copies, pre_refill, load0=None):
    """Units in the given order (DenseNet first), each largest first within its encoder group, go to the eligible account with the least
    planned load; on that account its kernels take the earliest period with room (chain segments in order, one account per chain)."""
    cap = {(ac, p): (pre_cap[ac] if p == "pre" else WEEK_H) for ac in PRE_QUOTA for p in PERIODS}
    for c in copies:
        if c["period"] != "pre":
            cap[(c["ac"], c["period"])] -= PREP_COPY_H
    load, plan = {ac: (load0 or {}).get(ac, 0.0) for ac in PRE_QUOTA}, []  # load0: pre-refill kernels already placed
    for u in units:
        periods = list(PERIODS) if (pre_refill and u["arch"] == "densenet121") else list(PERIODS[1:])
        for ac in sorted(DATA[u["ds"]]["holders"], key=lambda ac: (load[ac], DATA[u["ds"]]["holders"].index(ac))):
            trial, pi, ok = [], 0, True
            for k in u["ks"]:
                while pi < len(periods) and cap[(ac, periods[pi])] - sum(t["hours"] for t in trial if t["period"] == periods[pi]) < k["hours"]:
                    pi += 1
                if pi == len(periods):
                    ok = False
                    break
                trial.append(dict(k, ds=u["ds"], arch=u["arch"], ac=ac, period=periods[pi]))
            if ok:
                for t in trial:
                    cap[(ac, t["period"])] -= t["hours"]
                plan += trial
                load[ac] += u["h"]
                break
        else:
            raise SystemExit(f"{u['ds']}/{u['arch']}: does not fit {len(PERIODS) - 1} quota weeks")
    return plan, cap


def simulate(plan, copies):
    """Indicative wall clock: 2 slots per account; a kernel starts at max(free slot, period start, previous chain segment, NIH copy)."""
    end, slots, copy_end = {}, {}, {}
    for c in copies:
        sl = slots.setdefault(c["ac"], [PERIOD_START_H[c["period"]]] * 2)
        sl[0] = max(sl[0], PERIOD_START_H[c["period"]]) + PREP_COPY_H
        copy_end[c["ac"]] = sl[0]
    for k in sorted(plan, key=lambda k: (PERIOD_START_H[k["period"]], k["arch"] != "densenet121", k["chain"] is None, k["seg"])):
        sl = slots.setdefault(k["ac"], [PERIOD_START_H[k["period"]]] * 2)
        i = min(range(2), key=lambda i: sl[i])
        t = max(sl[i], PERIOD_START_H[k["period"]], copy_end.get(k["ac"], -1e9) if k["ds"] == "nih" else -1e9)
        if k["chain"] is not None and k["seg"] > 0:
            t = max(t, end[(k["ds"], k["arch"], k["chain"], k["seg"] - 1)])
        k["start"], k["end"] = t, t + k["hours"]
        sl[i] = k["end"]
        if k["chain"] is not None:
            end[(k["ds"], k["arch"], k["chain"], k["seg"])] = k["end"]
    return {a: max(k["end"] for k in plan if k["arch"] == a) for a in {k["arch"] for k in plan}}


def label(j):
    return f"{SHORT[j['arch']]}-{j['cfg']}-s{j['seed']}" + (f"-f{j['fold']}" if j["fold"] is not None else "")


def build(winner, tim, safety, warm, pre_refill, tag):
    archs = ["densenet121"] + ([winner] if winner != "densenet121" else [])
    jobs = make_jobs(archs, tim, safety)
    pre_cap = {ac: (max(0.0, PRE_QUOTA[ac] - PRE_MARGIN) if pre_refill else 0.0) for ac in PRE_QUOTA}
    copies = []
    for ac in NIH_COPIES:
        copies.append(dict(ac=ac, period="pre" if pre_cap[ac] >= PREP_COPY_H else "w1"))
        if copies[-1]["period"] == "pre":
            pre_cap[ac] -= PREP_COPY_H
    pre_kernels, left = carve_pre([j for j in jobs if j["arch"] == "densenet121"], warm, pre_cap) if pre_refill else ([], jobs)
    left = left + [j for j in jobs if j["arch"] != "densenet121"] if pre_refill else jobs
    units = []
    for arch in archs:  # DenseNet group first, so its placement never depends on the winner
        grp = []
        for ds, d in DATA.items():
            kernels, chains = pack([j for j in left if j["ds"] == ds and j["arch"] == arch], d["warm" if warm else "cold"], d["lane"])
            grp += [dict(ds=ds, arch=arch, ks=[k]) for k in kernels] + [dict(ds=ds, arch=arch, ks=c) for c in chains]
        for u in grp:
            u["h"] = sum(k["hours"] for k in u["ks"])
        units += sorted(grp, key=lambda u: -u["h"])
    plan, cap = assign(units, pre_cap, copies, pre_refill, {ac: sum(k["hours"] for k in pre_kernels if k["ac"] == ac) for ac in PRE_QUOTA})
    plan = pre_kernels + plan
    finish = simulate(plan, copies)
    n = {}
    for k in sorted(plan, key=lambda k: (k["arch"] != "densenet121", k["ds"], k["chain"] is not None, k["chain"] or 0, k["seg"], -k["hours"])):
        base = f"s4-{tag}-{k['ds']}-{SHORT[k['arch']]}"
        if k["chain"] is None:
            n[base] = n.get(base, 0) + 1
            k["slug"] = f"{base}-{n[base]:02d}"
        else:
            k["slug"] = f"{base}-c{k['chain']:02d}-{k['seg']}"
    check(jobs, plan, cap)
    return dict(archs=archs, jobs=jobs, plan=plan, copies=copies, cap=cap, finish=finish)


def check(jobs, plan, cap):
    """Self-check: every run exactly once, lanes within limits, quota within capacity, chains on one account, data on its holders."""
    seen = {}
    for k in plan:
        assert k["ac"] in DATA[k["ds"]]["holders"], k["slug"]
        assert k["hours"] <= DATA[k["ds"]]["lane"] + 1e-6, (k["slug"], k["hours"])
        assert len({j["arch"] for lane in k["lanes"] for j in lane}) == 1, k["slug"]
        for lane in k["lanes"]:
            for j in lane:
                key = (j["ds"], j["arch"], j["cfg"], j["seed"], j["fold"])
                seen.setdefault(key, set()).add((k["ds"], k["arch"], k["chain"]) if k["chain"] is not None else k["slug"])
    assert sorted(seen) == sorted((j["ds"], j["arch"], j["cfg"], j["seed"], j["fold"]) for j in jobs), "a run is missing or duplicated"
    assert all(len(v) == 1 for v in seen.values()), "a run is in two kernels"
    assert all(v >= -1e-6 for v in cap.values()), "quota over capacity"
    chains = {}
    for k in plan:
        if k["chain"] is not None:
            chains.setdefault((k["ds"], k["arch"], k["chain"]), set()).add(k["ac"])
    assert all(len(v) == 1 for v in chains.values()), "a chain spans two accounts"
    assert len(plan) == len({k["slug"] for k in plan}), "duplicate slug"


def report(b, winner, warm, pre_refill, safety):
    plan, jobs = b["plan"], b["jobs"]
    cold = {ds: DATA[ds]["warm" if warm else "cold"] for ds in DATA}
    total = sum(k["hours"] for k in plan) + PREP_COPY_H * len(b["copies"])
    expected = sum(cold[k["ds"]] + (k["hours"] - cold[k["ds"]]) / safety for k in plan) + PREP_COPY_H * len(b["copies"])
    print(f"## Winner = {winner}: {' + '.join(b['archs'])}" + (", warm-up read" if warm else "") + (", pre-refill quota" if pre_refill else ""))
    print(f"\n- runs: {len(jobs)} (" + ", ".join(f"{a}: {sum(1 for j in jobs if j['arch'] == a)}" for a in b["archs"]) + f"); kernels: {len(plan)} "
          f"({sum(1 for k in plan if k['chain'] is not None)} in resume chains); NIH copies: " + ", ".join(f"{c['ac']} ({c['period']})" for c in b["copies"]))
    print(f"- quota: {total:.0f} kernel-hours budgeted (x{safety}; expected {expected:.0f}): " + ", ".join(
        f"{SHORT[a]}-{ds} {sum(k['hours'] for k in plan if k['ds'] == ds and k['arch'] == a):.0f}" for a in b["archs"] for ds in DATA))
    print("- indicative finish (h after the 7 Nov 06:00 refill): " + ", ".join(f"{a} {v:.0f}" for a, v in b["finish"].items()))
    print("\n| account | week | kernel | est h | GPU 0 | GPU 1 |\n|---|---|---|---|---|---|")
    for k in sorted(plan, key=lambda k: (k["ac"], PERIODS.index(k["period"]), k["start"])):
        seg = f" ({k['seg'] + 1}/{k['nseg']})" if k["chain"] is not None else ""
        print(f"| {k['ac']} | {k['period']} | {k['slug']}{seg} | {k['hours']:.1f} | " + " | ".join(", ".join(label(j) for j in lane) or "-" for lane in k["lanes"]) + " |")
    print("\n| account | " + " | ".join(PERIODS) + " |\n|---|" + "---|" * len(PERIODS))
    for ac in sorted(PRE_QUOTA):
        caps = {p: (max(0.0, PRE_QUOTA[ac] - PRE_MARGIN) if p == "pre" else WEEK_H) for p in PERIODS}
        print(f"| {ac} | " + " | ".join(f"{caps[p] - b['cap'][(ac, p)]:.1f}/{caps[p]:.1f}" if (pre_refill or p != "pre") else "-" for p in PERIODS) + " |")


def write(b, winner, tim, tag):
    out = ROOT / f"outputs/queue_v3_{tag}_{winner}.json"
    if out.exists() or out.with_suffix(".state.json").exists():
        raise SystemExit(f"{out} exists: a written queue is frozen; use a new --tag for a new plan")
    queue = [dict(id=f"{tag}-prep-nih-{c['ac']}", ac=c["ac"], kernel="s4-prep-nih", est_h=2.0, retries=0,
                  cmd=f"python ops/launch_prep.py {c['ac']} nih s4-wts3 cnx_t16_s0.pt+cnx_t16_s1.pt+cnx_t16_s2.pt --arrays orig,soft --no-qure",
                  fetch=f"outputs/prep_dl/{c['ac']}-s4-prep-nih", fetch_pattern=r"^.*(\.json|\.log|\.txt|split\.npy|patient\.npy|labels\.npy)$")
             for c in b["copies"]]
    prev = {(k["ds"], k["arch"], k["chain"], k["seg"]): k["slug"] for k in b["plan"] if k["chain"] is not None}
    (ROOT / "outputs/jobs").mkdir(parents=True, exist_ok=True)
    for k in sorted(b["plan"], key=lambda k: (k["arch"] != "densenet121", PERIODS.index(k["period"]), k["chain"] is not None, k["seg"])):
        jl = [[g, "hyb", k["ds"], j["cfg"], j["seed"], EPOCHS, *argv(j, tim)] for g, lane in enumerate(k["lanes"]) for j in lane]
        (ROOT / "outputs/jobs" / f"{k['slug']}.json").write_text(json.dumps(jl))
        p = prev.get((k["ds"], k["arch"], k["chain"], k["seg"] - 1)) if k["chain"] is not None else None
        hybrid_pt = k["ds"] == "nih" and any(j["cfg"] == "hybrid" for lane in k["lanes"] for j in lane)
        item = dict(id=k["slug"], ac=k["ac"], kernel=k["slug"], est_h=round(k["hours"] + 0.3, 1), retries=0 if k["chain"] is None else 1,
                    rerun_missing=k["chain"] is None or k["seg"] == k["nseg"] - 1,  # a chain segment retries itself from its predecessor
                    cmd=f"python ops/launch_down.py {k['ac']} {k['slug']} outputs/jobs/{k['slug']}.json --prep s4-prep-{k['ds']}" + (f" --prev {p}" if p else ""),
                    fetch=f"outputs/v3/p2/{k['slug']}",  # NIH hybrid model.pt files come home for the H10 XAI kernel (RUNPLAN_v3 R5)
                    fetch_pattern=r"^.*(\.json|\.jsonl|\.log|\.txt|\.npy" + (r"|nih_hyb_hybrid@[^/]*/model\.pt" if hybrid_pt else "") + r")$")
        needs = ([f"{tag}-prep-nih-{k['ac']}"] if k["ds"] == "nih" and k["ac"] in NIH_COPIES else []) + ([p] if p else [])
        if needs:
            item["needs"] = needs
        queue.append(item)
    queue += phase3(b, tag)
    out.write_text(json.dumps(queue, indent=1))
    print(f"\nwrote {out} ({len(queue)} items) and {len(b['plan'])} job files")


def phase3(b, tag):
    """Phase-3 kernels (RUNPLAN_v3 R5, R6), queued with the runs so every account works in parallel:
    XAI (H10) on each account that holds NIH hybrid models, mounting those kernels (for a chain, its last segment holds every run);
    the NIH suppressor spread for selective classification (ac_6: prep, s4-wts3 and the raw NIH data; no dependency, so it uses the
    leftover pre-refill quota); CPU hash checks of every prep (holders now, copies after they are made)."""
    items, small = [], r"^.*(\.json|\.log|\.txt|\.csv|\.npz)$"
    for arch in b["archs"]:
        by_ac = {}
        for k in b["plan"]:
            if k["arch"] == arch and k["ds"] == "nih" and (k["chain"] is None or k["seg"] == k["nseg"] - 1) \
                    and any(j["cfg"] == "hybrid" for lane in k["lanes"] for j in lane):
                by_ac.setdefault(k["ac"], []).append(k)
        for ac, ks in sorted(by_ac.items()):
            n = sum(1 for k in ks for lane in k["lanes"] for j in lane if j["cfg"] == "hybrid")
            slug = f"s4-{tag}-xai-{SHORT[arch]}-{ac.replace('_', '')}"
            items.append(dict(id=slug, ac=ac, kernel=slug, est_h=round(0.3 + math.ceil(n / 2) * XAI_H[arch], 1), retries=0, needs=[k["slug"] for k in ks],
                              cmd=f"python ops/launch_xai.py {ac} {slug} --sources {','.join(k['slug'] for k in ks)} --prep s4-prep-nih",
                              fetch=f"outputs/v3/xai_dl/{slug}", fetch_pattern=small))
    slug = f"s4-{tag}-spread-nih"
    items.append(dict(id=slug, ac="ac_6", kernel=slug, est_h=2.5, retries=0, fetch=f"outputs/v3/{slug}", fetch_pattern=small,
                      cmd=f"python ops/launch_spread.py ac_6 {slug} --weights s4-wts3 --files cnx_t16_s0.pt+cnx_t16_s1.pt+cnx_t16_s2.pt --prep s4-prep-nih"))
    for ac, prep, need in [("ac_6", "s4-prep-nih", None), ("ac_7", "s4-prep-nih", None), ("ac_9", "s4-prep-vin", None), ("ac_1", "s4-prep-vin", None)] + \
            [(c["ac"], "s4-prep-nih", f"{tag}-prep-nih-{c['ac']}") for c in b["copies"]]:
        slug = f"s4-{tag}-hash-{ac.replace('_', '')}"
        items.append(dict(id=slug, ac=ac, kernel=slug, est_h=0.0, retries=0, cmd=f"python ops/launch_hash.py {ac} {slug} --prep {prep}",
                          fetch=f"outputs/hash/{ac}", fetch_pattern=small, **({"needs": [need]} if need else {})))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", required=True, choices=ENC)
    ap.add_argument("--timings", default=str(ROOT / "results/v3/timings.json"))
    ap.add_argument("--safety", type=float, default=1.15)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--pre_refill", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="p2a")
    a = ap.parse_args()
    tim = json.loads(Path(a.timings).read_text())
    b = build(a.winner, tim, a.safety, a.warm, a.pre_refill, a.tag)
    report(b, a.winner, a.warm, a.pre_refill, a.safety)
    if a.write:
        write(b, a.winner, tim, a.tag)


if __name__ == "__main__":
    main()
