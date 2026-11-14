# Deviations from PLAN.md / the reference paper (dated; every one affects how results must be read)

- 2026-10-26: Physics model simplified vs the reference paper: water-equivalent density (rho = 1 + HU/1000) with ICRU-44 cortical-bone / soft-tissue / lung
  attenuation per component; energy-integrating detector (weight E), W-anode 12 deg, 2.5 mm Al (SpekPy). Scatter, noise and blur are applied at training time
  (detector augmentation), not baked into the components.
- 2026-10-26: Suppressor is SwinV2-Tiny at 512 px (reference: SwinV2-Base at 1024 px); the UNet++ nesting covers the Swin stages only, with a plain U-Net tail to
  full resolution. Reason: compute (Kaggle T4, 14.6 img/s measured per GPU).
- 2026-10-26: LIDC source: Kaggle mount holds 606/1018 CT series; patients 0601+ come from the LUNA16 mount (verified voxel-identical to DICOM on 12 series, incl.
  z-location of nodules) or the TCIA API. The frame is a body crop with height/width in [0.75, 1.2] (short scans cannot reach 1.2), squashed to 512x512.
- 2026-10-26: MIMIC-CXR dropped (not needed; credentialed access, not permitted on Kaggle/LLM tools).
- 2026-10-26: Node21 not available as a ready Kaggle copy (official release is a 37 GB Zenodo archive); decision on including it is open.
- 2026-10-27: 38% of rendered CTs needed more than 25 padded frame rows (scans are shorter than 0.75 x body width). Usable limit raised to 50 padded rows (10% of the frame) and padded rows are set to air (zero density) when the bank is loaded, instead of keeping the edge-replicated streaks.
- 2026-10-27: Pilot (60k samples, lr 2e-4) passed gate G2; a learning-rate check (1e-4 / 2e-4 / 4e-4, 60k samples each) on SYNTHETIC validation CTs gave final val MAE 0.0096 / 0.0095 / 0.0100, so lr = 2e-4 is kept for every arm. No real-image result was used for this choice.
- 2026-10-27: The pre-registered H2/H3 comparison with Qure's released weights was looked at on SZCH-75 after the first ablation (arms vs Qure): Qure has higher
  full-resolution SSIM_cal (0.869 vs 0.850), ours higher bone-region PSNR_cal (22.04 vs 21.66); on the real-image audit Qure absorbs less (A 0.134 vs 0.306, 450 sites).
  SZCH-75 therefore counts as a DEVELOPMENT set from here on for decisions about the v2 recipe (data source, regulariser); decisions use the 24-image dev subset
  (`name hash % 3 == 0`, tools/szch_split.py) and final claims use the other 51 images plus RibCXR, JSRT and the downstream sets, which have not been touched.
- 2026-10-27: Secondary real-pair metrics added after seeing the SSIM gap (they do not replace the pre-registered ones): band-limited SSIM (sigma 2/4), fidelity of the
  removed signal to the DES bone signal (correlation / slope after sigma-2 smoothing), high-frequency energy ratio. Diagnosis: the DES soft image has ~9% less
  high-frequency energy than the CXR; at sigma 4 SSIM_cal of ours and Qure is identical (0.9498).
- 2026-10-27: New training source (v2): the public Qure ct2xr-projections (CC BY-NC-SA 4.0; 21,887 CT-RATE volumes, translated components). We verified that each
  composite equals soft + b*bone + c*lung + d to 0.25 grey levels (R^2 = 1.000; b per sample), so b*bone_translated is the exact additive bone image
  (tools/qp_prep.py -> results/qp_meta.csv). Patient-level split by hash (21,051 / 406 / 430). Weights trained on it inherit CC BY-NC-SA 4.0.
- 2026-10-29: v2 confirmatory analysis plan written (SAP_v2.md) after the v2 recipe was developed on SZCH dev-24 paired metrics, synthetic
  validation data and the real-image audit. That audit ran on all 75 SZCH images (450 sites), so SZCH audit numbers count as development
  evidence only; the confirmatory audit moves to VinDr-RibCXR (rib masks give the bone overlap) and JSRT. Round-1 screening printed held-out
  SZCH PSNR/SSIM for v1-family and 100k runs (descriptive, no v2 decision used them).
- 2026-10-29: encoder screen added (exploratory, 100k samples): ConvNeXt-V2-T (weights CC BY-NC 4.0), DINOv3 ViT-S/16 (DINOv3 licence),
  RAD-DINO ViT-B/14 (MSRLA, chest-radiograph pretraining). Plain ViTs feed the UNet++ decoder through a resampled 4-block pyramid; their encoder
  learning rate is 0.25 x the decoder's (fixed before any result). Twin-regulariser ablation arms (stop-gradient, dilated support, whole image,
  4 lesions, w = 8) and secondary paired metrics (MS-SSIM, LPIPS) added.
- 2026-10-29: a dry-run listing of the local evaluation driver started the held-out uncertainty evaluation by mistake; it was stopped after
  ~2 minutes, before it wrote any output, and the driver now guards that step. No held-out number was computed or seen.
- 2026-10-29: ConvNeXt-V2-T diverged under the shared recipe (lr 2e-4, fp16: NaN loss from step 6,400); it is rerun with the encoder learning rate x0.25
  already used for the plain-ViT encoders. The screen is exploratory; the diverged run is reported as such.
- 2026-10-31: BS-LDM released weights run as an extra baseline (see DECISIONS.md). Deviations from the authors' inference code: noise from a seeded generator (paired base /
  lesion-inserted sampling); fp16 autocast only if the smoke rule holds (mean |fp16 - fp32| < 0.01 and bone PSNR_cal within 0.05 dB); our SZCH preprocessing (the released test PNGs are
  1024 px 8-bit, min-max stretched to [0,1]: identical to the authors' /255 for 90% of the images, a <= 10% stretch for the rest). The SZCH audit numbers of every method remain
  development evidence (SAP_v2.md); BS-LDM is not part of the pre-registered hypotheses.
- 2026-10-31: tools/aggregate_screen.py printed the SZCH all-75 and held-out-51 columns next to dev-24 for every run it processed (rounds 1-4, encoder screens). No rule, table or decision used them (R1 / R2 and every
  table use dev-24 and the audit), but held-out numbers of those runs may have been displayed. The tool now computes dev-24 only; `--held` is for after tools/freeze_v2.py. The wave-1 tables were made with the dev-only version.
- 2026-10-31: rule R2 selected RAD-DINO + twin16 (300k) as a replacement of the SwinV2-T configuration (DECISIONS.md). Clarification of SAP_v2.md before the freeze: the no-twin comparator "same data and recipe" is
  taken to include the encoder, i.e. RAD-DINO no-twin seeds 0-2; the SwinV2-T pair (twin16, no-twin) is secondary. This adds four RAD-DINO runs (about 22 session hours) that were not in the original plan.
- 2026-10-31: the pre-registered tie-break between the two R2 passers (ConvNeXt-V2-T, RAD-DINO) selected ConvNeXt-V2-T as "ours" (DECISIONS.md). The RAD-DINO runs started under the earlier provisional choice are kept as the third
  encoder (secondary method); the matched no-twin clarification above now applies to ConvNeXt-V2-T. This adds four ConvNeXt runs (about 15 session hours on ac_4 / ac_1) to the plan.
- 2026-11-01: tie-break timing. ConvNeXt-V2-T was selected as "ours" on the n = 2 vs n = 2 comparison of 2026-10-31 (A difference 0.0037, inside the 0.005 band). With RAD-DINO seed 2 in, the K = 3 vs K = 3 difference is 0.0055 (0.0005 outside the band, RAD-DINO lower), so the same rule applied
  to the final K = 3 means would pick RAD-DINO. The selection was kept as made (DECISIONS.md, notes of 2026-11-01); both encoders are evaluated on the locked sets. No held-out or locked number was involved in this choice.
- 2026-11-01: stored EMA weights carry fp16-overflowed (+inf) BatchNorm running variances in 7 of 27 evaluated checkpoints (DECISIONS.md); results describe the stored models. SZCH audit sites differ between Kaggle and the local machine for most images (lung-mask draw), so the final
  (local) audit table and the Kaggle dev table are not interchangeable at the 0.001-0.01 level; all final contrasts are paired on identical local sites.
- 2026-11-03: the last 45 locked-set outputs (szch metrics, qp audit, unc, JSRT and SZCH real-image audits of 11 methods) are computed on a Kaggle T4 instead of the local RTX 3050 (user request; DECISIONS.md). s4/audit_real.py was changed after the freeze: an optional mask cache (S4_AUDIT_MASKS) so that the Kaggle audits draw exactly the local sites; without the variable the code path is unchanged, and the local check reproduced the local sites exactly. Model outputs still carry the GPU's float differences (max 0.0024 dB measured on dev-24).
- 2026-11-03: the synth step of tools/final_local.sh passed --bank outputs/shard{0,1,2}, but the rendered CTs sit in outputs/shard{0,1,2}/render (s4/bank.py globs one level), so a local run would have loaded no CT. Path fixed after the freeze (no method or data change); the two remaining synth outputs (clia_v1, recipe_v1) run on a Kaggle T4 (ac_3) from s4-bank, which holds all 844 usable CTs (so the CT order and seeds match) and whose 158 test CTs are byte-identical to the local shards.
- 2026-11-03: s4/eval_ribcxr.py also reads the optional mask cache (S4_AUDIT_MASKS) so the 8 Kaggle rib-benchmark outputs use the local lung masks; without the variable the code path is unchanged, and locally the cached and freshly computed masks gave identical output.
- 2026-11-03: the BS-LDM SZCH audit (results/final/audit_szch/bsldm.csv, Kaggle, 2026-10-31) predates the mask pinning: its 225 sites (sites 0-2) use the same lesions and image/site indices as the final local sites but different positions (bone overlap differs at 136 of 225). final_stats.py uses it only as a descriptive mean A (no paired contrast); it is not site-paired with the other methods.
- 2026-11-03 (BDCXR, before any model saw an image): the duplicate rule of SAP_bdcxr.md (64-bit DCT hash of the 256 px image, Hamming <= 6, union of pairs) chained 1,891 of the 3,257 paediatric images into one group (nearest-neighbour hash distances of unrelated images peak at 6), which would have put 2,130 images in one test fold. Replaced by the normalised cross-correlation of the 64 px image (from the 256 px image), pairs with r >= 0.93 united: 0.93 is the lowest of 0.90 / 0.93 / 0.95 / 0.97 at which no group exceeds 2 images (0.90 chains a group of 79). Result: 3,209 groups (48 pairs, 8 of them with different class labels), 5 folds of 650-653 test images with the class mix of the whole set. Only images and folder names were used; no label-dependent outcome was looked at.
- 2026-11-03 (independent recheck of the final statistics; corrections, no new data looked at):
  - Interaction bootstrap (H3' overlap interaction, encoder pairs, BDCXR E2): `set(rng.choice(...))` dropped repeated draws, so each "bootstrap" was a ~63% subsample without replacement and the intervals were ~25% too narrow. Fixed (images resampled with replacement). Main contrasts and p-values are unchanged; every interaction interval widened. One exploratory reading changes: the SwinV2-T twin vs no-twin interaction on VinDr-RibCXR, previously [0.0019, 0.0311], is now [-0.0019, 0.0358] (includes 0).
  - H4'b was only partly implemented. Now as SAP.md H4: NIH macro-AUROC of the 3-seed ensemble with a patient-cluster bootstrap CI against orig (per-class DeLong differences kept with their sign, Holm over classes; DeLong itself ignores the ~9 images per patient, the macro CI does not); detection CPM and mAP@50 as mean +- SD over seeds with an image bootstrap of the difference to orig; recall at 2 FP/image by bone overlap as a difference in differences with an image bootstrap (box hits at the full-set score threshold, re-weighted). TBX11K: every test box has >= 20% bone overlap (all three overlap maps), so its difference in differences is not estimable and is reported as such.
  - Detection metrics (s4/froc.py): 45% of the Node21 detection scores are tied and the default numpy sort orders ties differently across machines, so CPM / mAP depended on where they were computed (largest difference 0.00055 against the values saved on Kaggle). Tied scores now enter the curves together (distinct thresholds); recall_at_fp uses the same rule. All final detection numbers are recomputed from the saved predictions; checked order-independent on all 45 runs.
  - H4'a: Holm within dataset is now applied (image-level Wilcoxon of per-image mean differences; CIs unchanged), and the nodule table shows ours - no-twin.
  - H5': per-item values are now stored by s4/eval_uncertainty.py; the rerun (ac_7) reproduced every stored value exactly and gives 95% CIs for coverage, width and the real-image Spearman. The synthetic Spearman and AUSE pool pixels over CTs and stay without CI.
  - SAP.md H1 (v1 family, synthetic test CTs) is now computed (it existed only for the validation selection): clia vs recipe, the two v1 arms that were trained as finals; Holm over that single contrast.
  - JSRT audit (H3' secondary): the bone overlap of each site came from the Qure bone map alone (`--overlap_ref qure`), not from the consensus of the ours and Qure maps named in SAP_v2. The JSRT interaction therefore uses the comparator's bone map to define overlap; reported with this caveat.
  - Multiplicity, stated explicitly: Holm is applied to the p-values of each family (H3' two contrasts, H4'a within dataset, H4'b per-class DeLong); confidence intervals are unadjusted 95% intervals; the three H2' non-inferiority bounds are not adjusted (all pass by a wide margin).
  - Generated tables escape `_`, `>=` and `<` for main.tex; new generated tables: h2_noninferiority, audit_contrasts, dose_response, downstream_cls, downstream_det, reliability, encoders, synth_v1. ablation_twin.tex stays a placeholder (development data, outside the locked set).
- 2026-11-03 (BDCXR E2): the 2% lung-mask rule is applied to the eroded mask that the sites are drawn from (SAP_bdcxr.md says the PSPNet mask); no image of the 300 was affected.
- 2026-11-03 (correction of the entry above on SAP.md H1): H1 had been reported as planned on 2026-10-27 (results/ablation_test: synthetic test CTs, 200k samples, three seeds; CLIA, lesion-weighted loss and their combination against recipe; CLIA passes, the combination misses the MAE margin). The 2026-11-03 analysis in tools/final_stats.py (H1_v1_synth) is a replication with the 300k finals (CLIA vs recipe), not the planned test; both are reported (paper table v1_h1). The Holm adjustment in results/ablation_test runs over the seven contrasts of the v1 ablation, more conservative than the three named in SAP.md; no decision changes.

## 2026-11-03 - v3 phase 1: gradient accumulation for Swin-V2-B (after freeze_v3)
The smoke runs (kernel s4-v3-smoke, ac_6) showed CUDA out-of-memory for Swin-V2-B (swinv2_base_window16_256) at batch 32 on a 16 GB T4,
already single-stream (14.4 GB allocated). s4/cls_hybrid.py gained --accum K: the batch of 32 is processed in K micro-batches and one
optimiser step is taken, so the effective batch stays 32. For layer-norm encoders (Swin, ConvNeXt-V2, ViT) the gradients equal the
full-batch ones (tests/test_cls_hybrid.py::test_accumulation_matches_full_batch: max difference 5e-8 at gradient scale 0.11); the random
augmentation draws are taken per micro-batch. The Swin-V2-B screen job uses --accum 4; DenseNet-121, ConvNeXt-V2-T and RAD-DINO run
unchanged (accum 1, identical to the frozen code). The logged training loss is now the sum over micro-batches. The file hash differs
from freeze_v3.json for this reason only.

## 2026-11-04 - v3 execution changes R2-R4 (RUNPLAN_v3.md), after freeze_v3
Execution-only changes; the frozen analysis plan (SAP_v3.md) is unchanged. Files hashed in freeze_v3.json that change: kernels/down/run_down.py,
s4/cls_hybrid.py, s4/xai_multistream.py.
- R2 (kernels/down/run_down.py): before the GPU lanes start, orig.npy and soft.npy of each data set are read once sequentially, so the
  first epoch reads them from the page cache; the read time goes to the kernel log and run_summary.json (warm_up, warm_up_s). The job
  folder name now comes from s4/runtag.py (same names as before; tests/test_runtag_rerun.py).
- R4 (s4/cls_hybrid.py load_x, also used by s4/xai_multistream.py): the uint8 batch moves to the GPU before it is converted, the bone
  channel is formed and the 512-px detection-set arrays are area-resized. Values come from a lookup table k -> k/255 built with numpy's
  float32 division, because a plain '/ 255' on CUDA can differ from the CPU path by 1 ulp (found in the 2026-11-04 code review). The
  tensors equal the previous CPU path exactly on CPU and on CUDA, with and without the resize (tests/test_runtag_rerun.py, CUDA test
  run locally).
- R3 (ops/jobqueue.py; tools/v3_phase2.py sets rerun_missing and retries 0): when a phase-2 kernel ends, COMPLETE or not, its output is
  fetched and every job without a result.json is queued once as a new kernel <slug>-r with only those jobs (never for resume-chain
  segments; an output that cannot be downloaded is retried, not taken as 'nothing finished'). Fixed with it, after the code review: the
  scheduler no longer changes its item table while iterating it, an item whose dependency failed is now skipped (before, it waited for
  ever), fetches run in the project folder, and every launch rebuilds the code package (ops/push_kernel.py) so no kernel imports a module
  the upload lacks.

## 2026-11-04 - SAP_v3 clarifications decided by the author before any test evaluation (RUNPLAN_v3 R7)
SAP_v3.md did not say how the tests are corrected when two encoders go forward (DenseNet-121 and the screen winner), nor which model H10
uses. The author chose, on 2026-11-04, before any v3 model was evaluated on a test split:
- H6 (primary): one Holm family over encoders x data sets (up to 2 x 3 = 6 tests). H7 (secondary): one Holm family over encoders x data
  sets x both contrasts (hybrid vs ctrl, hybrid vs early; up to 12 tests). H8 and the per-class DeLong tests stay within encoder and
  data set, as in SAP_v3.md.
- H10: the DenseNet-121 hybrid models of all 5 seeds; for each target (NIH test image x box class) the S-stream minus X-stream
  Grad-CAM++ hit is averaged over the seeds, then focal targets (Mass, Nodule) are compared with diffuse/boundary targets (the other six
  NIH box classes), with an image-cluster bootstrap (10,000 resamples). The winner's hybrid models are reported the same way as secondary.
Implemented in tools/hybrid_stats.py (holm_families, pointing).

## 2026-11-04 - v3 encoder screen: encoder learning-rate rule for encoders without batch normalisation (author's decision, 4 Nov ~03:30 local)
SAP_v3.md screens all encoders with one recipe (AdamW 3e-4 on every parameter). In screen-a ConvNeXt-V2-T collapsed at that rate
(validation macro-AUROC 0.487-0.511 in all 10 epochs); a diagnostic run with the encoder learning rate x0.25 learned normally (0.841
after 2 epochs; DECISIONS.md 2026-11-04). The author decided to apply the v2 rule (the pretrained transformer and ConvNeXt-V2 encoders
of the suppressor trained with the encoder learning rate x0.25) to the classifier screen:
- ConvNeXt-V2-T, RAD-DINO and Swin-V2-B (no batch normalisation) are screened again with --enc_lr 0.25 (decoder and head keep 3e-4;
  otherwise the frozen screen settings: NIH orig, seed 0, 10 epochs, validation only), kernel s4-v3-screen-c on ac_7.
- DenseNet-121 (batch normalisation) keeps its frozen screen-a run (0.8404) at the frozen rate.
- The ranking uses the DenseNet-121 screen-a run and the three x0.25 runs. The runs at the frozen rate (ConvNeXt-V2-T in screen-a,
  RAD-DINO and Swin-V2-B in screen-b) are reported but do not enter the ranking; this was fixed before any screen-b result was seen
  (screen-b was still running on ac_7 and had not been fetched).
- The selected encoder trains in phase 2 with the same encoder learning rate in all five configurations (tools/v3_phase2.py).
The decision was made after seeing validation results only (screen-a, diagnostic); no test split was touched.

## 2026-11-04 - v3 execution change R1: resume across Kaggle sessions (RUNPLAN_v3 R1), after freeze_v3
Execution only; the analysis plan is unchanged. s4/cls_hybrid.py saves a checkpoint at the end of an epoch (model, optimiser, gradient
scaler, step, best-so-far weights, epoch, all random-number states) when the next epoch would pass --deadline_h, exits with code 75, and
continues with --resume; a resumed run equals an uninterrupted one (tests/test_resume.py: identical predictions after two interruptions,
CPU). kernels/down/run_down.py skips runs an earlier kernel (mounted with launch_down --prev) finished and resumes those it left
unfinished, giving each run a deadline inside an 11.5-h session budget. ops/jobqueue.py reruns mount the kernel they replace; the
generator sets reruns only on normal kernels and on the last segment of a chain.
R1 review fixes (independent review, 4 Nov): the checkpoint is now written at every epoch end (a crash or killed session loses at most the
epoch in progress) and copied into the running kernel's output before resuming; a checkpoint made with other training settings is refused;
log lines after the checkpoint epoch are dropped on resume; run_down refuses to start when the --prev kernel is not mounted; a rerun of a
rerun (<slug>-r2) continues only runs that left a checkpoint; chain segments are counted in whole epochs per session (Swin-V2-B three-stream
runs: 7 sessions, ConvNeXt-V2-T / RAD-DINO: 3) and retry once. Kaggle test (s4-v3-r1a/b, ac_6, pre-fix code): a run stopped after epoch 1
and resumed in the next kernel gave validation macro-AUROC 0.541680 and 0.526858 at epochs 1 and 2, identical to six decimals to the
same run uninterrupted (s4-v3-smoke, GPU).

## 2026-11-07 - v3 phase 2: five NIH runs repeated on a consistent soft input (author's decision, 7 Nov ~19:40 local)
The R6 hash check (tools/compare_hashes.py, outputs/hash/*/hash.json) found the NIH soft.npy of the original prep on ac_6 (2 Nov)
different from the five later copies on ac_2, ac_3, ac_4, ac_5 and ac_7 (3-7 Nov), which are byte-identical to each other; every
other NIH file and both VinBigData copies are identical. Same weights (sha256 of the three suppressor files identical on all
accounts) and no change to the NIH prep or inference code between the two dates; the cause is taken to be the Kaggle runtime
(GPU type or library versions) and was not measured further. Each run trains and tests on its own account's copy, so no run mixes
the two versions, but five NIH runs trained on ac_6 (s4-p2a-nih-dn-01) used the old version: soft seeds 0, 3, 4 and early seeds 2, 3.
Their test macro-AUROC (soft 0.8073, 0.8047, 0.8034; early 0.8078, 0.8088) lay within the range of the 20 unaffected NIH runs
(0.7993-0.8101, all configurations), but soft seeds 3 and 4 were below both unaffected soft seeds (0.8065, 0.8082) and early seed 2
below all three unaffected early seeds (0.8083-0.8091). Unaffected: orig and ctrl (no
soft input), all hybrid runs and the XAI kernels (copy version), VinBigData and BDCXR-3257. The author chose to repeat the five runs
on ac_7 (copy version; kernels s4-p2a-nih-rr6a/b, queue outputs/queue_v3_p2a_rr6.json) before any final analysis, so that every NIH
run uses one identical soft input; the ac_6 results are kept in outputs/v3/superseded_ac6_soft/ and are not analysed. The encoder
screen (phase 1) also ran partly on the ac_6 version; its outcome (all four encoders within 0.005, DenseNet-121 cheapest) does not
depend on it.
Outcome (8 Nov): all five reruns finished (rc 0, none unfinished); test macro-AUROC soft s0 0.8057, s3 0.8085, s4 0.8060 and early s2
0.8115, s3 0.8097 (ac_6: 0.8073, 0.8047, 0.8034, 0.8078, 0.8088; mean change +0.0019). Training on GPU is not bitwise repeatable, so
the change is not attributed to the soft input alone. tools/collect_v3.py then found 175/175 runs, none split over several kernels.
Five-seed ensembles (mean probability, as in tools/hybrid_stats.py) with the ac_6 runs in place of the reruns would have scored early
0.8279 and soft 0.8253, against 0.8273 and 0.8243 analysed (check outside the SAP, not used for any test).

## 2026-11-08 - v4: execution-only change before the first CT teacher run
s4/xm.py gains --accum for the CT stage. Each optimiser step still averages the losses of --bs = 4 volumes, but they pass through the
network in micro-batches (4 passes of one volume, 24 slices); each micro-batch loss is weighted by its share of the batch, so the step
equals the full-batch step for this loss (CT labels have no missing values; encoder and decoder use layer norm, no batch statistics);
only the dropout masks are drawn per micro-batch. Reason: v3 needed --accum 4 for Swin-V2-B at 32 images per step on a T4, so 96
Swin-V2-T slices with gradients are not expected to fit. SAP_v4.md and the training settings are unchanged; the hash of s4/xm.py in
freeze_v4.json no longer matches the file. Same day, no code change: kernel s4-ctprep could not read the HF_TOKEN secret in versions
pushed through the API (HTTP 400) and had no internet in a version saved from the editor; it is started from the editor with internet on.

## 2026-11-08 - v4: ac_9 takes ac_5's place in the account plan (author's request)
The author asked to use ac_9 instead of ac_5 (DESIGN_v4.md section 6). ac_9 runs what ac_5 would have run: P1 seeds 0-2, F_p1 seeds 0-2,
their NIH zero-shot transfer and hallucination tests (job list s4-v4-p1-a). ac_9 had no NIH prep and no Qure pack, so it gets its own:
s4-prep-nih (same frozen code, the same three v2 suppressor files, sha256 checked against the ac_7 staging; orig and soft arrays, no Qure)
and s4-qpprep (the public shards, packed as on ac_4, ac_5 and ac_7). The NIH copy is hashed (ops/launch_hash.py) against the common copy of
ac_2, ac_3, ac_4, ac_5 and ac_7 (soft.npy 159adce1...). F_p1 models read only the original image, so a different soft image would not
touch H11-H13; the rib-emphasis test (H15) reads it, so if ac_9's soft.npy differs, the hallucination tests of F_p1 seeds 0-2 run on an
account with the common copy, from the copied model files. Extra cost: the NIH prep on ac_9 (GPU) and the pack (CPU).

## 2026-11-09 - v4: CT teacher outputs leave Kaggle; CT teacher seed 2 collapsed
Data handling. DESIGN_v4.md section 2 says nothing derived from CT-RATE leaves Kaggle except metrics and the 32x32 heat-map frames, but
section 6 of the same document moves the teacher.npz files (logits and pooled decoder outputs of the 2,156 subset volumes; no images) to
other accounts, which Kaggle allows only through a download. They went from ac_1 through the author's computer to ac_7 and ac_9 as the
private dataset s4-xm-teacher (results/<tag>/teacher.npz, all five seeds as trained). The standard fetch (ops/fetch.py) also put the
five teachers' weights (model.pt) and their validation and test predictions on the author's computer (outputs/v4/ct-a, git-ignored).
Nothing of it is public or released.
Collapsed seed. All five CT teacher runs of s4-v4-ct-a ended with rc 0 after 20 epochs (best validation macro-AUROC s0 0.738, s1 0.828,
s2 0.545, s3 0.831, s4 0.803). Seed 2 never learned: validation macro-AUROC 0.46-0.54 in every epoch, logits constant over all 2,156
volumes (per-class SD below 0.0001, at the class priors) and the pooled decoder output constant (mean SD 0.00007 against 0.55-0.89 for
the other seeds). Checked on the saved weights: no non-finite value, input scale gamma near 1, encoder weight norm as in the other seeds,
but the encoder's output no longer depends on the input (features of two different volumes differ by 0.3 %, 112 % for seed 1) and the
decoder attention is uniform. Seed 0 was stuck near 0.60 for 12 epochs before it learned. No code or data fault was found; the recipe
(DESIGN_v4.md section 3) is unchanged. Distilling P1 seed 2 from this teacher would pull it toward one constant target. How seed 2 is
handled is decided by the author before any P1 run of seed 2 starts and recorded below. P1 seeds 3 and 4 (ac_7, s4-v4-p1-b) use
teachers 3 and 4 as planned.
Decision (author, 9 Nov, before any P1 run of seed 2): CT teacher seeds 5 and 6 are trained with the unchanged recipe, in parallel on
ac_1 (kernel s4-v4-ct-b; the jobs of s4-v4-ct-a with seeds 5 and 6). Slot 2 takes seed 5 unless seed 5 collapsed, otherwise seed 6.
Collapsed means constant logits: per-class SD of the teacher logits over the training and validation volumes of the subset below 0.01
in every class (seed 2: at most 0.0001; the weakest working seed, 0: 0.23 to 2.02). The rule reads no label, so it cannot prefer the
better teacher. If both collapse, the author decides again before any P1 run of seed 2. P1 seed 2 then distils from the slot-2 teacher
(s4-v4-p1-a with '--teacher @run:ct_xm_ct_s<5 or 6>/teacher.npz'); H14's seed pair 2 is that teacher with P1 seed 2; the CT teacher's
seed-mean prediction (descriptive) uses seeds 0, 1, slot 2, 3 and 4. The collapsed seed 2 and an unused replacement are reported
separately (test macro-AUROC) and kept out of results/v4/down/ct, so tools/xm_stats.py reads five teacher runs. Extra cost: about
2.7 GPU-h on ac_1; ac_9's kernel starts about 3 h later than planned.

Outcome (recorded 9 Nov, 08:45 UTC, before P1 seed 2 started): neither replacement collapsed (per-class logit SD over the training and
validation volumes: seed 5 0.61 to 3.15, seed 6 0.089 to 1.23), so slot 2 = seed 5 (validation macro-AUROC 0.834, test 0.822). Seed 6 is
the unused replacement (validation 0.714, test 0.730); its validation macro-AUROC fell after epoch 1 (0.68), stayed between 0.49 and
0.60 from epoch 2 to 15 and rose only over the last five epochs. Seed 5's teacher.npz (sha256 734ef33c...) was added to s4-xm-teacher on
ac_9 (new version), and s4-v4-p1-a was launched on ac_9 at 08:40 UTC with P1 seed 2 reading '@run:ct_xm_ct_s5/teacher.npz'.

## 2026-11-09 - v4: last kernels (cross-modal maps, VinBigData zero-shot, pointing game, DenseNet-121 hallucination tests)
Approved by the author on 9 Nov (18:15 UTC), before any of these kernels started.
P1 weights to ac_1. The cross-modal maps (H14) and the VinBigData zero-shot transfer run on ac_1, the only account holding CT-RATE data
(DESIGN_v4.md section 6). The five P1 model files (qp_xm_xraykd_s0-4/model.pt, 113 MB each; distilled from the CT teacher, so derived
from CT-RATE) go from ac_9 (s4-v4-p1-a) and ac_7 (s4-v4-p1-b) through the author's computer to the private ac_1 dataset s4-xm-xray,
together with the five P0 files (fetched with wave 1). Same conflict with section 2 as the teacher hand-off above; nothing is public or
released. maps.npz of s4/xm_xai.py also holds the 32x32 frames of the CT attenuation image and of the projection that decide the
orientation (frame_r, frame_r_flip); they leave Kaggle with the heat-map frames. Job list s4-v4-xmx: xmx with P1 (H14) and P0
(descriptive) for the seed pairs (CT teacher, X-ray model) (0, 0), (1, 1), (5, 2), (3, 3), (4, 4), and the VinBigData zero-shot transfer
of P0 and P1 seeds 0-4 (s4-prep-vin of ac_1).
DenseNet-121 reference. The v3 DenseNet-121 orig NIH runs are on ac_7 (s4-p2a-nih-dn-08, seeds 0-3) and ac_6 (s4-p2a-nih-dn-01, seed 4).
ac_6's NIH copy has another soft image (soft.npy a33d790d... against the common 159adce1...), which the rib-emphasis test reads, so the
five runs (model.pt with their validation and test predictions; not derived from CT-RATE) are copied to ac_9 (common copy, most quota
left) as the private dataset s4-dn121; the hallucination tests (s4-v4-dn121) and the pointing game run there on all five seeds as one group.
Pointing game. kernels/xai/run_xai.py and s4/xai_runs.py found only NIH hybrid runs (v3 H10). ops/launch_xai.py gets --runs REGEX
(default the v3 pattern, so v3 behaviour is unchanged) and --datasets. The v4 kernels take the orig Swin and DenseNet-121 runs, one kernel
per account that holds them: ac_2 F_in s0-2, ac_3 F_in and F_p0 s3-4, ac_4 F_p0 s0-2, ac_9 F_p1 s0-2 and DenseNet-121 s0-4, ac_7 F_p1 s3-4.
Rerun. nih_xm_transfer-xraykd_s3 (ac_7) failed after 7 s: it started on lane 1 when P1 seed 4 ended there, 20 s before P1 seed 3 ended on
lane 0 (kernels/down/run_down.py does not wait for '@run:' paths). It reruns in s4-v4-p1-b-r (--prev s4-v4-p1-b, same job list).
Not run: VinBigData fine-tuning (SAP_v4.md "if quota allows"; about 50 GPU-h, more than the quota left this week).

## 2026-11-10 - v4: results collected (queued rerun, collector, pointing-game summary)
Queued rerun. Version 1 of s4-v4-p1-b-r (ac_7, pushed 9 Nov 18:19 UTC) stayed QUEUED for more than 9 h, while nothing else ran on ac_7
and s4-v4-xai-ac7, pushed after it, had run and finished. Version 2 (the same folder pushed again, 10 Nov about 03:45 UTC) started at once
and ended within 8 min: nih_xm_transfer-xraykd_s3 rc 0 (178 s); its other six runs are the done_before copies of s4-v4-p1-b and match the
fetched originals. Version 1 was still queued then; if it runs, it repeats the same job and its output is not used.
Collection. tools/collect_v4.py (new; it only copies) puts the fetched outputs (results and logs, no model files) into results/v4 in the
layout of tools/xm_stats.py, without ct_xm_ct_s2 and ct_xm_ct_s6 (8f93fec): 67 runs (5 CT teachers, 10 projection models, 15 NIH
fine-tuned models, 10 NIH and 10 VinBigData zero-shot transfers, 10 cross-modal map runs, 7 hallucination seed groups). The 20
pointing-game outputs go to results/v4/pointing with the collect step of s4/xai_runs.py.
CT teacher seed 5 had come home with its teacher files only (9 Nov); its validation and test predictions (no weights) were
fetched on 10 Nov like those of seeds 0-4 (entry of 9 Nov above), so the CT teacher result uses seeds 0, 1, 3, 4 and 5.
Pointing-game summary. SAP_v4.md lists the pointing game as descriptive and does not say how it is summarised. tools/pointing_v4.py
(written on 10 Nov after the hit rates of one run, F_p1 seed 0, had been seen) reports per model and heat map the share of the 984 NIH box
targets hit, averaged over the five seeds, for all targets, the focal classes (Mass, Nodule) and the other six, with a 95 % image-cluster
bootstrap interval; no test.
Per-class DeLong and conformal (SAP_v4.md descriptive; both written or run on 10 Nov after the H11-H15 results had been seen).
tools/delong_v4.py: paired DeLong per class on the seed ensembles, Holm within two families as in v3's per-class tests: the H11 and H12
contrasts on the five shared findings (15 tests) and F_in against DenseNet-121 on the 14 NIH classes (14 tests). Conformal risk control of
the per-image false-negative rate at 0.10: s4/uq_conformal.py cls3 unchanged, --root results/v4/down, once per Swin model (F_in, F_p0,
F_p1; calibrated on the NIH validation predictions; its BDCXR-3257 part is empty, v4 has no BDCXR-3257 runs).
