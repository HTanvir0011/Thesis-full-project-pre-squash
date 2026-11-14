# SwinTrust v4: run plan from 2026-10-29 to submission

Local time is Dhaka (UTC+6). Kaggle GPU quota refills every Saturday 00:00 UTC = **06:00 local** (next: Oct 31, Nov 7, Nov 14).
Each of the 5 accounts gets 30 GPU-h per week, runs at most 2 sessions at once, and each session is a T4 x2 (two runs in parallel).
Numbers marked *est.* are estimates; everything else comes from saved files. Decisions are recorded in DECISIONS.md; deviations in DEVIATIONS.md.

## 1. Where we are

| | lesion absorption A (lower = lesion kept) | bone PSNR_cal dB | SSIM_cal |
|---|---|---|---|
| v1 `clia` (LIDC-only, pre-registered pick) | 0.300 | 22.34 | 0.853 |
| Qure released model | 0.134 | 21.97 | 0.875 |
| **v2 candidate: SwinV2-T + Qure projections + twin w=16, 300k, 4 seeds** | **0.059-0.071** | 21.71-21.76 | 0.845-0.846 |
| v2 K=3 ensemble (seeds 0-2) | 0.0715 (dev) / 0.0701 (450 sites) | 21.79 | 0.847 |

All of these are development numbers (SZCH dev-24 for paired metrics; the audit used all 75 SZCH images). Held-out and locked sets are untouched.
Open problem: SSIM_cal gap to Qure (about half explained by the smoother DES target; tools/hf_diag.py).

## 2. What the paper still has to show (claim -> evidence -> where it is produced)

| claim | evidence still missing | produced by |
|---|---|---|
| C1 twin lesion-invariance regulariser lowers absorption at no real-image fidelity cost | dose-response at 300k with >= 2 seeds per weight; ablation of its parts; held-out fidelity vs Qure and no-twin | wave 1 + local eval (sections 3, 6) |
| C2 the counterfactual audit measures something real | audit on locked images with an independent bone mask (RibCXR); real-nodule CNR (JSRT, Node21); downstream lesion-overlap subgroup | local eval + wave 3 |
| C3 reliability | ensemble spread vs error; conformal coverage on Qure-projection test and SZCH held-out | local eval |
| C4 benchmark | SZCH held-out 51 vs Qure/controls/v1; rib benchmark; NIH/TBX11K/Node21/VinBigData downstream with 3 seeds | wave 3 + local eval |
| "latest tech" check | modern encoders (ConvNeXt-V2, DINOv3, RAD-DINO) under the same recipe | wave 0 screen (running) |

Hypotheses, margins and tests: SAP.md (v1 family) and **SAP_v2.md** (v2 claims, frozen before any held-out/locked evaluation).

## 3. Calendar and compute

| when | account | kernel | what | hours *est.* | status |
|---|---|---|---|---|---|
| Oct 29 02:30 | ac_4 | s4-bb-1 | twin16 100k: ConvNeXt-V2-T (diverged, NaN from step 6,400), DINOv3-S/16 (A 0.081, PSNR 21.49: no gain) | 3 | done |
| Oct 29 02:30 | ac_5 | s4-bb-2 | twin16 100k: RAD-DINO (chest-X-ray ViT-B/14) seeds 0, 1: A 0.053, PSNR 21.55 (fails R1 on mean PSNR) | 5 | done |
| Oct 29 06:30 | ac_4 | s4-bb-3 | ConvNeXt-V2-T rerun, encoder lr x0.25, 2 seeds, 100k: A 0.066, PSNR 21.86 (passes R1) | 2.5 | done |
| Oct 31 06:05 | ac_4 | s4-w1b-C | ConvNeXt-V2-T twin16 at 300k x 2 seeds (R1 passed) | 8 | queued |
| Oct 29 06:30 | ac_5 | s4-w1b-R1 (then s4-w1b-R2 after the refill, --resume) | RAD-DINO twin16 at 300k x 2 seeds, exploratory | 6.5 + ~4.5 | R1 done (steps 22,000 / 23,400 of 37,500); R2 queued |
| Oct 29 | all 5 | upload | dataset `s4-wts2` = twin16 300k seeds 0-3 (244 MB) | 0.3 | ac_4/ac_5 done, ac_1-3 uploading |
| Oct 29 | ac_5, ac_4 | s4-prep-{nih,node21}-n48 + s4-dn-smoke5 (ac_5); s4-prep-tbx-n48 + s4-dn-smoke4 (ac_4) | 48-image dry runs of prep + 1-epoch cls/det | 1 | all paths verified on Kaggle (48 images, 1 epoch; metrics meaningless at that size): prep NIH (+ separate Qure array)/TBX11K/Node21; NIH cls incl. --qdata; TBX11K and Node21 det |
| **Oct 31 06:05** (auto: ops/jobqueue.py outputs/queue_wave1.json) | ac_1 | s4-w1-1, s4-w1-2 | v1 finals seed 2 (clia, recipe); twin16 stop-grad x2 seeds | 6.5 + 6.5 | ready |
| | ac_2 | s4-w1-3, s4-w1-4 | twin16 dilated support (k=31); whole-image invariance | 13 | ready |
| | ac_3 | s4-w1b-B (+ s4-w1b-B2), s4-w1-5, then s4-w1-6 | SwinV2-B twin16 300k x 2 seeds (met R1; ~11 h, chained); twin16 with 4 lesions/twin; insertion only | 12 + 6.5 + 6.5 | ready |
| | ac_4 | s4-w1-7, s4-w1-8 | no-twin seeds 1, 2; twin w=4 | 13 | ready |
| | ac_5 | s4-w1-9, s4-w1-10 | twin w=8; twin w=64 | 13 | ready |
| Oct 31 13:00 | any | (conditional) | best encoder at 300k x 2 seeds if it passes rule R1 | 6-10 | job file written on result |
| Oct 31 eve / Nov 1 | local | decision + freeze | rules R2-R4, `tools/freeze_v2.py` | - | tool ready |
| Oct 31 ~13:00 | ac_1 | s4-bsldm-full | BS-LDM released SZCH weights: 75-pair fidelity + 225-site lesion audit (first 3 of 6 sites per image), fp32 (~148 s per sample on a T4) | 7.8 | queued after w1-1 frees a slot |
| Nov 1-3 | ac_5 | s4-prep-nih -> s4-dn-nih-a/b | NIH 112k: 5 inputs x 3 seeds DenseNet121 | 3 + 2x5 | job files ready |
| Nov 1-3 | ac_4 | s4-prep-tbx -> s4-dn-tbx-1..4 | TBX11K: 5 inputs x 3 seeds Faster R-CNN | 1 + 4x4 | job files ready |
| Nov 1-3 | ac_2 | s4-prep-node21 -> s4-dn-node21-1..4 | Node21: 5 inputs x 3 seeds | 1 + 4x3 | images already downloaded (CPU kernel s4-node21raw: 4,882 images, 62 min) |
| Nov 1-4 | ac_1 | s4-prep-vin -> s4-dn-vin-1..4 | VinBigData (ac_1 only, competition rules) | 2 + 4x7 | job files ready; spills into week 2 |
| Nov 1-3 | ac_3 | s4-prep-node21 (no-twin ensemble) | CNR of the no-twin control on Node21 | 1 | after w1-7 |
| Nov 1-5 | local | section 6 | all locked-set evaluations, stats, figures | - | partly ready |
| Nov 7 06:00 | all | week 2 | spill-over, failed reruns, optional 1M-sample scale point | <= 150 | - |
| Nov 5-25 | local | writing | paper, thesis, model card, demo | - | skeleton to build |

Budget check *est.*: wave 1 = 10 sessions x 6.5 h = 65 h; downstream = about 60 session-h; wave 0 = 9 h. Week 1 capacity = 150 h. Fits with about 25 h spare.

## 4. Decision rules (written before the results they apply to)

- **R1 encoder screen (100k, development data).** An encoder goes to 300k x 2 seeds if A <= 0.070 with bone PSNR_cal >= 21.60 dB,
  or bone PSNR_cal >= 21.95 dB with A <= 0.100. Reference: SwinV2-T twin16 at 100k, 3 seeds: A 0.073-0.103, PSNR 21.69-21.93.
- **R2 replacing the v2 configuration (300k, mean over >= 2 seeds, development data).** Reference: SwinV2-T twin16 300k, 4 seeds:
  A 0.0667, bone PSNR_cal 21.735, SSIM_cal 0.846. A candidate (encoder or twin variant) replaces it only if
  (A <= 0.0567 and PSNR >= 21.685) or (PSNR >= 21.885 and A <= 0.0717), and SSIM_cal >= 0.841. Otherwise v2 stays as defined in SAP_v2.md.
- **R3 ablation arms** (stop-grad, dilated support, whole image, 4 lesions, insertion only) are reported as an ablation table; they are not
  selection candidates unless they meet R2.
- **R4 freeze.** Before computing any metric of a v2 model on SZCH held-out, RibCXR, JSRT, Node21, TBX11K, VinBigData or NIH test:
  commit the code, then `python tools/freeze_v2.py` (hashes of SAP_v2.md, the SZCH lists, the weights, the commit). Freeze is written once.
- **R5 failures.** A failed kernel is rerun once unchanged; a second failure is diagnosed before anything else on that account. A hypothesis
  that fails is reported as failed.

## 5. Run book (exact commands; run from `swintrust4/`)

Code changes always go out with `python tools/package_code.py` first (the launchers then refresh the account's `s4-code`).

```
# wave 0 dry runs on ac_5 (after s4-wts2 is ready)
python ops/launch_prep.py ac_5 nih s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt --limit 48
python ops/launch_prep.py ac_5 tbx s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt --limit 48
python ops/launch_prep.py ac_5 node21 s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt --limit 48
python ops/launch_down.py ac_5 s4-dn-smoke outputs/jobs/s4-dn-smoke.json --prep s4-prep-nih-n48,s4-prep-tbx-n48,s4-prep-node21-n48

# wave 1 (Oct 31 06:00); ac_1-3 hold the render bank and s4-qproj, ac_4-5 build QP in their s4-qpprep kernel
python ops/launch_train.py ac_1 s4-w1-1 NvidiaTeslaT4 outputs/jobs/s4-w1-1.json
python ops/launch_train.py ac_1 s4-w1-2 NvidiaTeslaT4 outputs/jobs/s4-w1-2.json --qp
python ops/launch_train.py ac_2 s4-w1-3 NvidiaTeslaT4 outputs/jobs/s4-w1-3.json --qp
python ops/launch_train.py ac_2 s4-w1-4 NvidiaTeslaT4 outputs/jobs/s4-w1-4.json --qp
python ops/launch_train.py ac_3 s4-w1-5 NvidiaTeslaT4 outputs/jobs/s4-w1-5.json --qp
python ops/launch_train.py ac_3 s4-w1-6 NvidiaTeslaT4 outputs/jobs/s4-w1-6.json --qp
python ops/launch_train.py ac_4 s4-w1-7 NvidiaTeslaT4 outputs/jobs/s4-w1-7.json --qpk --nobank
python ops/launch_train.py ac_4 s4-w1-8 NvidiaTeslaT4 outputs/jobs/s4-w1-8.json --qpk --nobank
python ops/launch_train.py ac_5 s4-w1-9 NvidiaTeslaT4 outputs/jobs/s4-w1-9.json --qpk --nobank
python ops/launch_train.py ac_5 s4-w1-10 NvidiaTeslaT4 outputs/jobs/s4-w1-10.json --qpk --nobank
# all in one go, with slot/quota handling, dependencies and automatic fetch (running now; log outputs/queue.log):
python ops/jobqueue.py outputs/queue_wave1.json      # restartable; --dry shows what it would launch

# collect a finished kernel (size-checked) and score it on development data
python ops/fetch.py <ac_N> <kernel> outputs/w1_dl/<kernel>
python tools/aggregate_screen.py outputs/screen_w1.csv outputs/w1_dl/*/results qure=outputs/real_eval/szch2_qure.csv

# after the freeze: downstream (weights: s4-wts2 on the account; VinBigData only on ac_1)
python ops/launch_prep.py ac_5 nih s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt     # orig + soft, 14.7 GB
python ops/launch_prep.py ac_5 nihq s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt    # Qure soft only, 7.35 GB (20 GB output limit)
python ops/launch_down.py ac_5 s4-dn-nih-a outputs/jobs/s4-dn-nih-a.json --prep s4-prep-nih,s4-prep-nihq
python ops/launch_down.py ac_5 s4-dn-nih-b outputs/jobs/s4-dn-nih-b.json --prep s4-prep-nih,s4-prep-nihq
python ops/launch_prep.py ac_4 tbx s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt
python ops/launch_down.py ac_4 s4-dn-tbx-1 outputs/jobs/s4-dn-tbx-1.json --prep s4-prep-tbx      # then -2, -3, -4
python ops/push_kernel.py ac_2 kernels/fetch_node21 s4-node21raw -     # CPU: Node21 images once (Zenodo rate-limits bursts; ~40 min)
python ops/launch_prep.py ac_2 node21 s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt
python ops/launch_down.py ac_2 s4-dn-node21-1 outputs/jobs/s4-dn-node21-1.json --prep s4-prep-node21   # then -2, -3, -4
python ops/launch_prep.py ac_1 vin s4-wts2 twin16_s0.pt+twin16_s1.pt+twin16_s2.pt
python ops/launch_down.py ac_1 s4-dn-vin-1 outputs/jobs/s4-dn-vin-1.json --prep s4-prep-vin      # then -2, -3, -4
```
`s4-wts2` must exist on every account that runs a prep kernel (`python ops/push_dataset.py ac_N kdata/stage_w2/ac_N s4-wts2 "..." --new`;
stage with hard links, see section 8). JSRT runs locally (section 6).

## 6. Local evaluation after the freeze (RTX 3050, no quota)

| step | command | output |
|---|---|---|
| SZCH 75 pairs, every method (held-out 51 analysed) | `bash tools/final_local.sh szch` | results/final/szch/*.csv |
| Counterfactual audit on RibCXR (confirmatory) and JSRT | `bash tools/final_local.sh audit` | results/final/audit_*/*.csv |
| Rib benchmark (RibCXR, metrics only, never images) | `bash tools/final_local.sh rib` | results/final/rib/*.csv |
| JSRT arrays + nodule CNR (ours, no-twin, Qure) | `bash tools/final_local.sh jsrt` | results/final/jsrt/ |
| Qure-projection test: bone error + exact-truth audit | `bash tools/final_local.sh qp` | results/final/qp/ |
| LIDC synthetic test (v1 and v2) | `bash tools/final_local.sh synth` | results/final/synth/ |
| Ensemble uncertainty + conformal | `bash tools/final_local.sh unc` | results/final/unc.json |
| Statistics for H1'-H5' and SAP.md H1-H5 | `python tools/final_stats.py` | results/final/stats.json, paper/tables/*.tex |
| Figures | `python tools/fig_pareto.py`, `tools/fig_real_qualitative.py` | paper/figures/ |

Methods evaluated: ours (K=3 ensemble), seeds 0-3 singly, no-twin K=3, v1 clia K=3, the dose-response models (w = 0, 4, 8, 16, 32, 64),
Qure, identity, CLAHE, homomorphic. Licensed rule: RibCXR runs only here, writes metrics only, and no RibCXR image is ever opened in chat.

## 7. Writing and release

- Paper (MedIA / IEEE TMI style): *Do bone-suppression models erase lesions? A counterfactual audit and a lesion-invariance regulariser.*
  Sections: introduction; related work (CT-supervised suppression, diffusion methods, counterfactual evaluation); data and physics; method
  (recipe, twin regulariser); audit; experiments (SZCH, RibCXR, synthetic, real nodules, downstream, reliability); results; limits; reproducibility.
- Figures: A vs bone-PSNR Pareto front (twin weights, Qure, controls); dose-response with seeds; ablation bars; audit maps; qualitative SZCH
  (public, CC BY-NC); downstream forest plots; reliability diagram. Tables generated from result files only.
- Thesis: the paper plus the v1 study (ablation, pre-registered H1), physics/rendering chapter, full negative results, reproducibility chapter.
- Release (only after you say so): model card + weights on the HF Hub (CC BY-NC-SA 4.0, inherited from the Qure projections), HF Space demo
  (no uploads stored; "research use only, not a medical device"), project page. Nothing derived from RibCXR or VinBigData images is released.
- End of project: rotate the 5 Kaggle tokens and the HF tokens (they sat in plain-text files).

## 8. Risks and what we do about them

| risk | effect | plan |
|---|---|---|
| a backbone or ablation kernel crashes (new code) | lost hours | 48-image dry runs and local smoke tests first; R5 |
| quota exhausted mid-run | run killed, outputs lost | never start a run longer than the quota left; long runs checkpoint every 15 min and resume (`--resume`) |
| Node21 Zenodo unreachable or rate-limited | no Node21 | reachable 2026-10-29 (4,882 images, 1,476 nodule rows); bursts get HTTP 429, so a CPU kernel downloads once with 4 threads and back-off (resumable); fallback: JSRT + TBX11K only, reported |
| SSIM gap to Qure persists | H2' SSIM part fails | reported as failed with the smoothing analysis; bone PSNR non-inferiority is the primary |
| audit and regulariser share the insertion mechanism | circular evidence | confirmatory audit on locked RibCXR images with independent rib masks + real-nodule CNR + downstream (H4') |
| SZCH audit was used during development | optimistic SZCH audit | SZCH audit reported as development evidence only (SAP_v2.md) |
| home upload 0.5 MB/s | slow dataset pushes | build big data inside Kaggle (prep kernels, s4-qpprep); upload only weights (60 MB each) |
