# RUNPLAN_v3 — phase 2 (full matrix) on the 9 Kaggle accounts from 7 November 2026

Status: **plan, 4 Nov 2026 00:30 local (UTC+6), updated 01:40 with the R2/R4 measurements and 09:30 with screen-b; not launched.
Start confirmed by the author on 4 Nov: 7 November 2026, 06:00 local (the weekly quota refill on all 9 accounts).** Frozen analysis plan: SAP_v3.md (freeze_v3.json). Design and budget:
DESIGN_v3.md §6, §8. The schedule of record is what `tools/v3_phase2.py` generates; the tables here come from it
(results/v3/p2_scenarios.md has every account table for every scenario). The plan was checked by a five-lens adversarial review
(65 agents, 4 Nov); the confirmed problems are fixed in the generator or listed below as prerequisites R1–R8.

Measured on 4 Nov (kernels s4-v3-r24-nih on ac_7 and s4-v3-r24-vin on ac_9): the warm-up read takes 64 s for the NIH arrays and 49 s for
VinBigData, and VinBigData steps cost the same as NIH steps once the resize runs on the GPU. Still **provisional**: the single-stream
step times of ConvNeXt-V2-T and Swin-V2-B (and Swin-V2-B at all), which the phase-1 screen logs give this morning; step R8 regenerates
the plan from them.

## 1. What runs

| | NIH ChestX-ray14 | VinBigData | BDCXR-3257 | per encoder |
|---|---|---|---|---|
| configurations | orig, soft, early, hybrid, ctrl | same | same | 5 |
| seeds | 0–4 | 0–4 | 0–4 × 5 folds | |
| runs | 25 | 25 | 125 | **175** |

20 epochs per run, batch 32, 256 px, the epoch with the best validation macro-AUROC kept (SAP_v3). DenseNet-121 runs in every
scenario. The screen winner (frozen rule: best validation macro-AUROC, ties within 0.005 go to the cheaper encoder) adds another 175 runs
unless DenseNet-121 itself wins.

## 2. Accounts and data (checked 4 Nov 00:10)

| account | GPU-h left now | from 7 Nov 06:00 | data it can run | phase-2 role |
|---|---|---|---|---|
| ac_1 | 0.1 | 30 / week | VinBigData prep (rules accepted); s4-wts3 | VinBigData |
| ac_2 | 0.59 | 30 | s4-wts3 | NIH copy (7 Nov), then NIH |
| ac_3 | 3.29 | 30 | s4-wts3 | NIH copy (before the refill), then NIH |
| ac_4 | 1.16 | 30 | s4-wts3 | NIH copy (7 Nov), then NIH |
| ac_5 | 1.34 | 30 | s4-wts3 | NIH copy (7 Nov), then NIH |
| ac_6 | 6.95 (screen-a running) | 30 | NIH prep | NIH |
| ac_7 | 27.58 (screen-b running) | 30 | NIH prep (copy of 3 Nov; small files byte-identical to ac_6) | NIH; DenseNet before the refill |
| ac_8 | 10.83 | 30 | BDCXR-3257 prep (data use agreement: this account only) | BDCXR-3257; DenseNet before the refill |
| ac_9 | 5.28 | 30 | VinBigData prep | VinBigData |

Limits used: one Kaggle session ≤ 12 h with 2×T4 (lanes planned ≤ 11 h for NIH and VinBigData, 9.6 h for BDCXR so three kernels fit
a week on ac_8); at most 2 kernels at once per account; 29 of the 30 weekly hours planned. A private kernel
output mounts only in its own account, so runs stay with their prep. NIH copies take 1.72 h (measured on ac_7).

## 3. Timings (results/v3/timings.json; seconds per step of 32 samples, warm cache)

| encoder | one stream | three streams | source |
|---|---|---|---|
| DenseNet-121 | 0.21 | 0.63 | both measured (smoke, s4-v3-r24-nih) |
| ConvNeXt-V2-T | 0.45 (provisional) | 1.25 | three streams measured; one stream from screen-a |
| RAD-DINO | 0.44 | 1.23 | one stream: screen-b full epochs; three streams: smoke window |
| Swin-V2-B | 1.26 | 3.9 (estimate) | out of memory at batch 32; runs with --accum 4 / 8; one stream measured in screen-b (3,180 s per epoch) |

Budgets carry a ×1.15 safety factor. Start of a kernel with the warm-up read (R2): 64 s NIH, 49 s VinBigData, planned as 0.05 h
(without it the first epoch loses about 1.5 h / 0.75 h to random reads). VinBigData steps: ×1.1 of NIH (measured; resize on the GPU, R4).

## 4. Scenarios

| winner | kernels (in resume chains) | GPU-h budgeted (expected) | DenseNet done | all done |
|---|---|---|---|---|
| DenseNet-121 | 12 (0) | 109 (96) | 7 Nov ~18:00 | 7 Nov ~18:00 |
| ConvNeXt-V2-T | 36 (10) | 316 (276) | 7 Nov ~18:00 | 14 Nov ~16:00 |
| RAD-DINO | 36 (10) | 317 (277) | 7 Nov ~18:00 | 14 Nov ~20:00 |
| Swin-V2-B (three streams estimated) | 74 (46) | 743 (648) | 7 Nov ~18:00 | 5 Dec ~16:00 |

All rows use the warm-up read (R2, built and measured). The kernels in resume chains are the heavier winner's NIH three-stream runs
(about 20 h each): they need R1.

The nine accounts give 261 planned GPU-h per week, so any heavier winner needs at least a second week. The finish times are indicative:
the scheduler (ops/jobqueue.py) starts a kernel whenever its account has enough quota, so work moves earlier when runs are faster than
budgeted. DenseNet-121 is packed and placed first and its kernels, slugs and accounts are identical in every scenario (checked), so it can
start before the screen decides the second encoder.

### DenseNet-121 schedule (with the warm-up read; same in every scenario)

| account | when | kernel | est h | GPU 0 | GPU 1 |
|---|---|---|---|---|---|
| ac_3 | before 7 Nov | NIH copy | 1.8 | | |
| ac_7 | before 7 Nov | s4-p2a-nih-dn-02 | 10.3 | orig s2, soft s3, early s4 | orig s3, soft s4 |
| ac_8 | before 7 Nov | s4-p2a-bdcxr-dn-01 | 6.4 | 31 fold runs | 31 fold runs |
| ac_2, ac_4, ac_5 | 7 Nov 06:00 | NIH copies | 1.8 each | | |
| ac_6 | week 1 | s4-p2a-nih-dn-01 | 10.3 | orig s0, soft s1, early s2 | orig s1, soft s2, early s3 |
| ac_2 | week 1 | s4-p2a-nih-dn-03, -07 | 10.3 each | hybrid s0; ctrl s3 | hybrid s1; ctrl s4 |
| ac_3 | week 1 | s4-p2a-nih-dn-04, -08 | 10.3, 6.9 | hybrid s2; orig s4, early s0 | hybrid s3; soft s0, early s1 |
| ac_4 | week 1 | s4-p2a-nih-dn-05 | 10.3 | hybrid s4 | ctrl s0 |
| ac_5 | week 1 | s4-p2a-nih-dn-06 | 10.3 | ctrl s1 | ctrl s2 |
| ac_8 | week 1 | s4-p2a-bdcxr-dn-02 | 6.3 | 32 fold runs | 31 fold runs |
| ac_9 | week 1 | s4-p2a-vin-dn-01 | 6.2 | 11 VinBigData runs (both GPUs) | |
| ac_1 | week 1 | s4-p2a-vin-dn-02 | 5.7 | 14 VinBigData runs (both GPUs) | |

The heavier winner's kernels follow in the same queue (tables in results/v3/p2_scenarios.md). ConvNeXt-V2-T or RAD-DINO: 24 kernels,
about 117 GPU-h in week 1 and 94 GPU-h in week 2. Swin-V2-B: 62 kernels over weeks 1–5 (each NIH three-stream run is about 63 h, six chained sessions).

## 5. Prerequisites before any phase-2 kernel (from the review)

| id | change | needed for | test before use |
|---|---|---|---|
| R1 | Resume: cls_hybrid saves an end-of-epoch checkpoint (model, optimiser, scaler, step, epoch, best so far, RNG states) to one file per lane and stops cleanly when the next epoch would pass a deadline (kernel ends COMPLETE); run_down finds the previous segment's checkpoint, passes --resume, and skips runs whose result.json exists; launch_down turns --prev into kernel_sources and rejects unknown arguments; the checkpoint is deleted when result.json is written (20 GB output limit) | every chain: DenseNet NIH hybrid/ctrl without R2, every heavier winner's NIH three-stream runs | CPU test: resumed training equals uninterrupted training at a fixed seed; one 2-segment GPU smoke chain on ac_7 (~0.5 GPU-h) |
| R2 | **Done 4 Nov.** Warm-up read: run_down reads orig.npy and soft.npy sequentially before the lanes start and logs the seconds | every kernel | measured: NIH 64 s, VinBigData 49 s |
| R3 | **Done 4 Nov.** Failure visibility: when a kernel ends (COMPLETE or not) its output is fetched and the jobs without a result become one rerun kernel `<slug>-r`; retries 0; an output that cannot be fetched is retried, not taken as empty; no reruns for chain segments | all | unit tests (tests/test_runtag_rerun.py) |
| R4 | **Done 4 Nov.** VinBigData speed: batches built and resized on the GPU with exact values (lookup table) | VinBigData | measured: steps ×1.0–1.1 of NIH; CPU and CUDA equality tests |
| R5 | Phase-3 paths: NIH hybrid model.pt files are fetched (done in the generator) and uploaded as one private dataset to the XAI account; XAI kernel (s4.xai_multistream + BBox_List_2017.csv); NIH suppressor-spread kernel (s4/spread.py NIH mode) for the second selective-classification score; a collection step into results/v3/down; uq_conformal calibrated on the v3 validation predictions; hybrid_stats writes one output per encoder | phase 3 | unit tests; one XAI kernel on 20 boxes |
| R6 | Identity of the prep copies: CPU kernels (no GPU quota) hash orig.npy and soft.npy on ac_6, ac_7 and each new copy; record in DEVIATIONS.md; if any differ, keep each configuration's seeds on one prep | seeds spread over accounts | hashes |
| R7 | **Decided by the author 4 Nov, before any test evaluation** (DEVIATIONS.md 2026-11-04): (a) H6 one Holm family over encoders × data sets, H7 one family over encoders × data sets × both contrasts; (b) H10 on the DenseNet-121 hybrid models, all 5 seeds, hits averaged per target, focal vs diffuse/boundary with an image-cluster bootstrap; the winner reported the same way as secondary. Implemented in tools/hybrid_stats.py | statistics | tests/test_hybrid_stats.py; dry run on v2 predictions |
| R8 | Refresh: put the screen-a/b step times (and R2, R4) into timings.json, regenerate, read the self-check, write the queue with `--write --tag p2a` (a written queue is frozen; a new plan gets a new tag) | all | the generator's self-check (every run once, lanes within limits, quota within weeks, chains on one account, DenseNet identical across winners) |

R2–R4 were reviewed adversarially (3 lenses, 2 refuters per finding); the 5 confirmed defects were fixed before the measurement kernels
ran. R1 is only needed if a heavier encoder wins (DenseNet-121 runs need no chains with R2). Changes R1–R4 touch files hashed in freeze_v3.json; each is logged in
DEVIATIONS.md as an execution change that leaves the analysis plan as frozen.

## 6. Timeline

| when (local) | step |
|---|---|
| 4 Nov 02:33 | screen-a ended: DenseNet-121 0.840; ConvNeXt-V2-T collapsed at the frozen rate (chance level) |
| 4 Nov 03:31 | screen-c launched on ac_7 (deviation, DEVIATIONS.md 2026-11-04): ConvNeXt-V2-T, RAD-DINO, Swin-V2-B at encoder lr ×0.25 |
| 4 Nov 09:10 / ~12:35 | screen-b ended (frozen rate; reported, not ranked: RAD-DINO 0.677, Swin-V2-B 0.828) / screen-c ends; the selection rule over DenseNet-121 (screen-a) and the three ×0.25 runs names the second encoder; R8 timing refresh |
| 4 Nov 01:30 | R2, R3, R4 done and measured (two short kernels on ac_7 and ac_9, ~0.2 GPU-h) |
| 4–5 Nov | R1 if a heavier encoder wins (unit test + one 2-segment GPU smoke on ac_7); R5, R6 (hash kernels, CPU); R7 decided by the author |
| 5 Nov | queue written (`tools/v3_phase2.py --winner W --warm --pre_refill --write --tag p2a`) and the scheduler started; DenseNet kernels use the quota that expires at the refill (ac_7, ac_8, ac_3: ~19 GPU-h) |
| 7 Nov 06:00 | refill; the remaining DenseNet kernels and the winner's kernels start on all 9 accounts |
| 7 Nov evening | DenseNet-121 matrix complete (all three data sets) |
| 14 Nov (ConvNeXt-V2-T or RAD-DINO) / 21 Nov (Swin-V2-B, estimate) | winner's matrix complete |
| after each encoder | phase 3: collection, statistics (H6–H8), uncertainty, XAI (H10), figures |

Strict 7-Nov start instead: generate without `--pre_refill`; DenseNet then ends about 6 h later and ~19 GPU-h of expiring quota go unused.

## 7. Commands

```
python tools/v3_phase2.py --winner <W> --warm --pre_refill            # read the plan and the self-check
python tools/v3_phase2.py --winner <W> --warm --pre_refill --write --tag p2a
python ops/jobqueue.py outputs/queue_v3_p2a_<W>.json > outputs/queue_v3_p2a.log 2>&1   # local scheduler (background); keep the computer awake
```

## 8. Risks and what happens

| risk | effect | handling |
|---|---|---|
| Swin-V2-B wins | ~743 GPU-h, about four weeks | the frozen rule stands; the author may decide on a logged deviation (e.g. fewer seeds for Swin) before its kernels start |
| a run is slower than budgeted | lane passes 12 h, kernel killed | ×1.15 safety, lanes ≤ 11 h; R3 reruns only the missing runs; R1 makes long runs resumable |
| a kernel fails | its runs are missing | retries 0, R3 rerun items |
| prep copies differ | seeds of one configuration see slightly different arrays | R6 hashes; regroup by prep if needed |
| quota runs short in a week | kernels wait | the scheduler waits for the next refill by itself |
| local computer sleeps | no new launches or fetches; running kernels finish on Kaggle | keep-awake; restart the scheduler (restart-safe) |
| an account is blocked (rules, phone check) | its kernels cannot start | regenerate with that account removed under a new tag |
