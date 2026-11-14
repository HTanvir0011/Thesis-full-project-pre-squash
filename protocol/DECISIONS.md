# Decisions taken from pre-registered rules (SAP.md)

## 2026-10-27 — which arm is "ours" (H1 selection rule, synthetic VAL CTs only)
Source: `results/ablation_val/` (3 seeds per arm, 98 val CTs, 177 insertion sites, 200k samples per run; bootstrap over patients, Holm over the 3 lesion-aware arms).

| arm | absorption A (mean) | A reduction vs recipe [95% CI] | Holm p | bone MAE vs recipe (non-inf. margin 5%) |
|---|---|---|---|---|
| recipe | 0.359 | - | - | - |
| clia | 0.326 | 0.033 [0.021, 0.046] | <0.001 | non-inferior |
| nodule_loss | 0.322 | 0.037 [0.025, 0.048] | <0.001 | NOT non-inferior |
| clia_nodule | 0.259 | 0.100 [0.083, 0.119] | <0.001 | NOT non-inferior (MAE ~6% worse) |

Rule: a lesion-aware arm passes H1 only if A is lower (Holm p < 0.05) AND bone MAE is non-inferior. `clia` is the only arm that passes both, so **`clia` is "ours"**.
`clia_nodule` removes most lesion absorption but fails the MAE co-primary; it is reported as a pre-declared trade-off variant ("lesion-priority"), not as the selected model.
Real-image (SZCH) numbers of the ablation runs were computed but played no part in this decision.
Finals: recipe x3 and clia x3 seeds at 300k samples (identical settings); clia_nodule x3 as the secondary variant if compute allows.

## 2026-10-28 - screening round 1 for v2 (exploratory; one seed, 100k samples per run; real-image columns are descriptive, not selection evidence)
Source: `outputs/screen.csv` via `tools/aggregate_screen.py` (table also in `results/screen_round1.md`). SZCH-75 is a development set here (dev = 24 images, held-out = 51; see DEVIATIONS.md).

| run | audit A (450 real sites) | A low-overlap | A high-overlap | leak | PSNR_cal bone, dev-24 | PSNR_cal bone, held-51 | SSIM_cal, held-51 | LIDC-synth bone MAE | QP-test bone MAE |
|---|---|---|---|---|---|---|---|---|---|
| identity |  |  |  |  | 20.33 | 20.21 | 0.8314 |  |  |
| qure |  |  |  |  | 21.97 | 21.51 | 0.866 |  |  |
| clia_s0 | 0.3004 | 0.2403 | 0.3542 | 0.002111 | 22.34 | 21.79 | 0.846 | 0.009521 | 0.0145 |
| mix_s0 | 0.2856 | 0.2472 | 0.3136 | 0.001806 | 21.94 | 21.43 | 0.8463 | 0.01515 | 0.006112 |
| mix_clia_s0 | 0.2687 | 0.2353 | 0.2937 | 0.002073 | 22.03 | 21.53 | 0.8448 | 0.01548 | 0.006386 |
| mix_twin_s0 | 0.2597 | 0.2316 | 0.2824 | 0.002238 | 21.72 | 21.27 | 0.8426 | 0.01495 | 0.006368 |
| qp_s0 | 0.2159 | 0.1978 | 0.2303 | 0.002173 | 21.79 | 21.31 | 0.8429 | 0.02772 | 0.004449 |
| qp_clia_s0 | 0.2176 | 0.1875 | 0.244 | 0.002327 | 21.65 | 21.19 | 0.8429 | 0.02755 | 0.004609 |

Reading: the Qure-projection source (`qp`, `qp_clia`) lowers real-image absorption most (0.30 -> 0.22) and costs ~0.5 dB real bone PSNR; lesion insertion (CLIA) adds nothing on top of it; twin weight 1 on a 50/50 mix is a small step (0.269 -> 0.260). Qure's released weights (SwinV2-B, 1024 px, ~32M training samples per its paper) remain at 0.134. Absorption is nearly independent of lesion amplitude (alpha 0.25-2) and, for QP-trained models, of bone overlap: a uniform linear leak. Round 2 (twin weight sweep on QP, longer training, backbone) was launched from these observations; it is a new exploratory family, labelled as such.

## Round 2 (2026-10-28, exploratory, SZCH dev-24 + audit on 450 real sites; single seed, 100k samples) — ac_1 part
Twin lesion-invariance weight on QP data (outputs/screen_r2.csv, results in outputs/r2_dl/r2-1):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| qp (no twin) | 0.216 | 0.198 | 21.79 | 0.849 | 0.0044 |
| qp_twin4 | 0.151 | 0.140 | 21.73 | 0.846 | 0.0057 |
| qp_twin16 | 0.103 | 0.082 | 21.69 | 0.841 | 0.0082 |
| Qure released | 0.134 | - | 21.97 | 0.875 | - |
Reading: A falls monotonically with twin weight, twin16 is below Qure's A at -0.1 dB bone PSNR_cal vs qp; cost is higher bone MAE on the QP/synthetic test (0.0044 -> 0.0082) and slightly lower SSIM_cal. Still to come: twin1/64 (ac_4), 300k, SwinV2-B, seed 1. Final weight is chosen after those, on dev only; held-out 51 untouched.

Round 2, ac_4 part (bank-free kernel ran correctly; no in-kernel synth test for these): twin weight sweep complete.
| twin w | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| 0 | 0.216 | 0.198 | 21.79 | 0.849 | 0.0044 |
| 1 | 0.186 | 0.171 | 21.69 | 0.847 | 0.0051 |
| 4 | 0.151 | 0.140 | 21.73 | 0.846 | 0.0057 |
| 16 | 0.103 | 0.082 | 21.69 | 0.841 | 0.0082 |
| 64 | 0.020 | 0.016 | 21.27 | 0.830 | 0.0142 |
Reading: A falls monotonically (dose-response); bone PSNR_cal is flat to w=16 and falls 0.5 dB at w=64 (A ~ 0, i.e. lesion fully preserved, bone error grows). Candidate operating points: w=16 (A below Qure, ~no PSNR cost) and w=64 as the "lesion-safe" end of the Pareto front. Awaiting 300k, SwinV2-B, seeds.

Round 2, ac_5 part (bank-free, 100k samples):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| qp twin16 seed 0 | 0.103 | 0.082 | 21.69 | 0.841 | 0.0082 |
| qp twin16 seed 1 | 0.083 | 0.065 | 21.70 | 0.844 | 0.0066 |
| qp, SwinV2-Base (no twin) | 0.189 | 0.162 | 21.83 | 0.850 | 0.0038 |
Reading: seed spread of A at twin16 is ~0.02, so twin16 A in [0.08, 0.10] is reproducible and below Qure's 0.134. SwinV2-B at 100k samples vs SwinV2-T (qp 0.216/21.79/0.849/0.0044): A -0.03, bone PSNR +0.04 dB, SSIM +0.001, MAE -14%: small, consistent gain; not decisive at this budget. Pending: qp 300k + seed 1 (r2-3).

Round 2, ac_4 r2-3 (bank-free): qp longer training and seed repeat.
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| qp 100k seed 0 | 0.216 | 0.198 | 21.79 | 0.849 | 0.0044 |
| qp 100k seed 1 | 0.196 | 0.168 | 21.89 | 0.854 | 0.0039 |
| qp 300k seed 0 | 0.186 | 0.163 | 21.76 | 0.853 | 0.0035 |
Reading (round 2 complete): seed spread (qp s0 vs s1) is ~0.02 in A and ~0.1 dB in bone PSNR, so 3x training (A -0.03, MAE -20%, SSIM +0.004) and SwinV2-B (A -0.03) are within ~1-1.5 seed spreads: small. The big lever on lesion absorption is the twin weight (A 0.21 -> 0.08-0.10 at w=16, 0.02 at w=64), not scale. The SSIM_cal gap to Qure (0.853 best vs 0.875) is not closed by data volume or backbone at this budget. Plan after refill: v2 candidates = SwinV2-T/B + QP + twin w in {16, 32, 64}, longer (>=300k), 3 seeds, K=3 ensemble; report A vs bone-PSNR Pareto front vs Qure; choose the operating point on dev, evaluate once on held-out 51.

Round 3, ac_4 r3-4 (100k, seed 1, bank-free):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| qp_twin32 s1 | 0.057 | 0.049 | 21.51 | 0.841 | 0.0097 |
| qp_twin64 s1 | 0.061 | 0.045 | 20.80 | 0.838 | 0.0124 |
(twin64 s0: A 0.020, PSNR 21.27.) Reading: w=64 is unstable across seeds (A 0.02 vs 0.06, bone PSNR 21.27 vs 20.80), so it is not a usable operating point; w=16-32 is the stable range (A 0.06-0.10, PSNR 21.5-21.7). Pending: twin32 s0 / base twins / 300k twins (r3-1..3).

Round 3, ac_5 r3-1 (100k, seed 0, bank-free):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| SwinV2-T twin32 s0 | 0.076 | 0.044 | 21.34 | 0.839 | 0.0098 |
| SwinV2-B twin16 s0 | 0.063 | 0.045 | 21.79 | 0.844 | 0.0069 |
Reading: SwinV2-B + twin16 is the best trade-off so far (A 0.063 < Qure 0.134; bone PSNR 21.79, -0.18 dB vs Qure, equal to twin-free T); T twin32 s0 vs s1 (A 0.076/0.057, PSNR 21.34/21.51) is consistent. Candidate final config: SwinV2-B, QP, twin w=16 (w=32 for a lesion-safer variant), longer training. Pending r3-2 (B twin64, T twin16 s2), r3-3 (T twin16/32 300k).

Round 3, ac_5 r3-2 (100k, bank-free):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| SwinV2-T twin16 s2 | 0.073 | 0.057 | 21.93 | 0.844 | 0.0070 |
| SwinV2-B twin64 s0 | 0.021 | 0.012 | 21.08 | 0.829 | 0.0145 |
Reading: SwinV2-T twin16 over 3 seeds: A 0.103/0.083/0.073 (mean 0.086), bone PSNR 21.69/21.70/21.93 (mean 21.77) -> reproducibly below Qure's A 0.134 at ~-0.2 dB. Base does not remove the w=64 cost (21.08 dB, SSIM 0.829). Base twin16 s0 (0.063/21.79) is within seed spread of T twin16, so backbone size is NOT shown to help once twin is on; T@512 is the efficient default. Pending r3-3 (T twin16/32 300k).

Round 3, ac_4 r3-3 (300k samples, seed 0, bank-free):
| run | audit A | A low-overlap | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|---|
| T twin16 300k | 0.071 | 0.058 | 21.75 | 0.846 | 0.0061 |
| T twin32 300k | 0.057 | 0.038 | 21.24 | 0.837 | 0.0073 |
Reading: 3x training with twin16 changes little vs 100k (A 0.071 vs mean 0.086; PSNR 21.75 vs 21.77; MAE 0.0061 vs 0.0072) -> within seed spread; twin32 costs ~0.5 dB. Working config for finals: SwinV2-T@512, QP, twin16, >=300k, 3 seeds + K=3 ensemble; twin32 as optional lesion-safer point. Launched round 4 (r4-1 ac_5: twin16 300k s1,s2; r4-2 ac_4: twin32 s1 300k + twin16 s3 300k) to get 3-seed 300k evidence now instead of waiting for the refill.

Round 4, ac_5 r4-1 (300k, bank-free): T twin16 seeds 0/1/2 at 300k (dev-24 / audit 450 sites):
| seed | A | A low-overlap | bone PSNR_cal | SSIM_cal | qp_mae |
|---|---|---|---|---|---|
| 0 | 0.0712 | 0.0581 | 21.749 | 0.8458 | 0.0061 |
| 1 | 0.0666 | 0.0504 | 21.708 | 0.8462 | 0.0055 |
| 2 | 0.0701 | 0.0529 | 21.755 | 0.8461 | 0.0056 |
mean A 0.0693 (spread 0.005), PSNR 21.74, SSIM 0.846: at 300k the result is tight across seeds (vs 0.073-0.103 at 100k). Qure A 0.134. Pending r4-2 (twin32 s1 300k, twin16 s3 300k).

Round 4, ac_4 r4-2 (300k, bank-free):
| run | A | A low-overlap | bone PSNR_cal | SSIM_cal | qp_mae |
|---|---|---|---|---|---|
| T twin16 s3 | 0.0590 | 0.0423 | 21.729 | 0.8448 | 0.0056 |
| T twin32 s1 | 0.0584 | 0.0477 | 21.468 | 0.8396 | 0.0070 |
Twin16 at 300k over 4 seeds (0-3): A 0.0667 mean (0.059-0.071), bone PSNR 21.735, SSIM 0.846. Twin32 at 300k over 2 seeds: A 0.058, PSNR 21.35, SSIM 0.839. Decision: final operating point = twin16 (twin32 costs ~0.4 dB for -0.009 A). Weights: outputs/r3_dl/r3-3, r4_dl/r4-1, r4_dl/r4-2.

SSIM_cal gap diagnosis (2026-10-28, tools/hf_diag.py, SZCH dev-24, same output Gaussian blur applied to every method; exploratory):
| method | sigma 0 | 0.7 | 1.0 | 1.5 |  (SSIM_cal / bone PSNR_cal)
|---|---|---|---|---|
| Qure | .8746 / 21.97 | .8915 / 22.22 | .8877 / 22.33 | .8824 / 22.51 |
| ours T twin16 300k s0 | .8458 / 21.75 | .8682 / 22.10 | .8703 / 22.27 | .8724 / 22.51 |
| identity | .8365 / 20.33 | .8520 / 20.70 | .8529 / 20.93 | .8555 / 21.30 |
Reading: the DES soft target has less high-frequency energy than the CXR (hp ratio: Qure 0.99, ours 1.12, identity 1.11), so raw full-res SSIM_cal partly rewards smoothness; matching HF energy lifts every method (identity +0.02). Under identical smoothing the bone-region PSNR gap to Qure vanishes (22.27 vs 22.33 at sigma 1) and the SSIM gap falls from 0.029 to 0.015-0.023. Remaining gap likely = native 1024 px prediction (ours: 512 prediction bicubic-upsampled). Reporting rule: always report raw and band-limited SSIM, plus identity as the reference; any smoothing is applied identically to all methods and never tuned to favour ours. Next: resolution study (train/infer at 1024).

## 2026-10-29 - rule R1 (RUNPLAN.md section 4) applied to the encoder runs that already exist
`python tools/rules.py r1 outputs/screen_r3.csv`: SwinV2-B + twin16 at 100k (one seed: A 0.063, bone PSNR_cal dev 21.79) meets R1
(A <= 0.070 and PSNR >= 21.60), so SwinV2-B twin16 goes to 300k x 2 seeds (job s4-w1b-B, about 11 h per run, chained over two sessions).
Note: R1 was written after this run was seen (thresholds come from the SwinV2-T 100k range, 0.073-0.103); the 300k comparison under R2 decides.

## 2026-10-29 - encoder screen (exploratory, 100k samples, twin16, QP data; SZCH dev-24 / audit on 450 sites)
Kernels s4-bb-1 (ac_4), s4-bb-2 (ac_5). Reference: SwinV2-T twin16 100k, 3 seeds: A 0.103/0.083/0.073 (mean 0.086), bone PSNR_cal 21.69/21.70/21.93 (mean 21.77), SSIM_cal 0.841/0.844/0.844, QP-test bone MAE 0.0082/0.0066/0.0070.
| encoder | seeds | audit A | A low-overlap | bone PSNR_cal | SSIM_cal | qp_mae |
|---|---|---|---|---|---|---|
| DINOv3 ViT-S/16 (enc lr x0.25) | 1 | 0.081 | 0.064 | 21.49 | 0.847 | 0.0056 |
| RAD-DINO ViT-B/14 (enc lr x0.25) | 2 (0.047, 0.059) | 0.053 | 0.034 | 21.55 | 0.846 | 0.0050 |
| ConvNeXt-V2-T (enc lr x1) | 1 | DIVERGED: loss NaN from step 6,400 (fp16), weights unusable | | | | |
Rule R1 (tools/rules.py r1): neither passes (RAD-DINO: A 0.053 <= 0.070 but mean bone PSNR 21.55 < 21.60; seed 0 alone passes with 21.608, seed 1 does not). Reading: RAD-DINO lowers lesion absorption and bone error (qp_mae 0.0050 vs 0.0072) at ~0.2 dB lower real bone PSNR; at equal absorption it sits above SwinV2-T's twin curve (SwinV2-T w=32 at 100k: A 0.066, PSNR 21.43), but the seed spread (A ~0.015, PSNR ~0.1-0.2 dB) leaves this unproven. DINOv3-S is no better than SwinV2-T.
Follow-ups (both exploratory, not selection candidates unless they meet R2 at 300k): ConvNeXt-V2-T rerun with encoder lr x0.25 (s4-bb-3, ac_4, 2 seeds; deviation: the x0.25 rule was written for plain ViTs, extended after the divergence); RAD-DINO at 300k x 2 seeds (s4-w1b-R1 on ac_5 with the expiring hours, max 6.5 h, then s4-w1b-R2 after the refill with --resume s4-w1b-R1).

## 2026-10-31 - encoder screen, follow-up results (ConvNeXt-V2 rerun, RAD-DINO 300k first chunk)
ConvNeXt-V2-T with encoder lr x0.25 (s4-bb-3, ac_4; 100k samples, twin16, seeds 0/1; both finished, no NaN):
| run | audit A | bone PSNR_cal dev | SSIM_cal dev | qp_mae |
|---|---|---|---|---|
| ConvNeXt-V2-T s0 | 0.059 | 21.93 | 0.845 | 0.0052 |
| ConvNeXt-V2-T s1 | 0.073 | 21.79 | 0.847 | 0.0053 |
| mean | 0.066 | 21.86 | 0.846 | 0.0053 |
Reference SwinV2-T twin16 100k (3 seeds): A 0.086, PSNR 21.77, SSIM 0.843, qp_mae 0.0072. Rule R1 (A <= 0.070 and PSNR >= 21.60): ConvNeXt-V2-T PASSES
(mean A 0.066, PSNR 21.86), so it goes to 300k x 2 seeds (job s4-w1b-C, ac_4, ~8 h; encoder lr x0.25 fixed before the run). Weights licence: CC BY-NC 4.0.
RAD-DINO at 300k (s4-w1b-R1, ac_5): the 6.5 h session budget stopped both seeds at step 22,000 / 23,400 of 37,500 (loss finite, QP val MAE 0.0050 / 0.0046);
resume checkpoints are on Kaggle and continue in s4-w1b-R2 after the refill. Its 100k result (A 0.053, PSNR 21.55) failed R1 only on the mean PSNR.
Both encoder candidates stay exploratory: a replacement of SwinV2-T needs R2 at 300k (RUNPLAN.md section 4) and is decided before the freeze.

## 2026-10-31 - BS-LDM (JBHI 2025) added as an external baseline (user approved download and run; exploratory until the freeze)
What it is: conditional latent diffusion for bone suppression (VQGAN + 383M-parameter UNet, 1000-step DDPM at 1024 px), MIT-licensed code, weights released by the
authors on Google Drive (VQGAN 153 MB, UNet 1.53 GB; trained on SZCH-X-Rays' real dual-energy TRAINING pairs, 592 of the 741 pairs; only the 75 test pairs are public).
It is therefore in-domain for SZCH, whereas our models never see a real dual-energy pair: its fidelity numbers are a reference, not a like-for-like competitor.
Safety: both weight files are full-model pickles (code can run at load time). They are scanned without executing (`s4/bsldm_safe.py`: pickletools walk) and loaded
only through an exact allowlist of the 41 reviewed globals (tensor-rebuild helpers, torch.nn layers, the repo's modules.unet / modules.nn classes, MONAI-generative
VQ-VAE classes); both files pass (19 and 23 globals, none outside). The Kaggle kernel refuses to load if the scan fails (it did stop the first smoke run: `__builtin__.set`
was not yet allowlisted). Code reviewed by grep for os/subprocess/eval/exec/network/file writes: only the logger writes files and is never configured.
Sampler: the authors' ldm_eval.py (same scheduler settings and per-step clip-range schedule); seeded noise (base and lesion-inserted copies share it), optional fp16.
Bone estimate for the lesion audit: B(x) = x - (a S(x) + b) with (a, b) fitted on the base image only (s4/bsldm.py BoneFn).
Run plan: smoke kernel (speed, fp16 agreement rule) now; then the full kernel (75 fidelity pairs + 450-site audit, two GPUs) from the queue.

BS-LDM smoke test (Kaggle T4, kernel s4-bsldm-smoke on ac_4, one SZCH test image, 1000 sampling steps): both pickle scans passed; the weights load once the older VQ-VAE's missing
`use_checkpointing` flag is set (a version difference, set to False). fp32: 147.6 s per image, peak 2.72 GB; fp16 autocast: 161.5 s (no speed-up), outputs agree (mean |diff| 0.0002);
decision: fp32. Single held-out image (107.png; not a v2 comparison): bone PSNR_cal 33.9 dB, PSNR_cal 33.8 dB, SSIM_cal 0.875, SSIM sigma-4 0.990 - far above every CT-supervised model's
dev-24 mean (about 22 / 26 / 0.85 / 0.95), as expected for a model trained on the same hospital's dual-energy pairs. Full run: 75 pairs fidelity + audit sites 0-2 of the 6 per image
(the same sites as the other methods; 225 sites) on ac_1 via the queue (about 7.8 session hours; sites 3-5 can follow with `--sites 3:6`).

## 2026-10-31 - wave 1 results (300k samples, development data) and rule R2
Source: `python tools/aggregate_screen.py outputs/screen_all_dev.csv outputs/{r2,r3,r4,bb,w1}_dl/*/results qure=... identity=...` (dev-24 paired metrics only; the audit ran on all 75
SZCH images = development evidence) and `python tools/rules.py r2 outputs/screen_all_dev.csv`. Mean over seeds; SwinV2-T, QP data, 300k samples unless stated.
Dose-response of the twin weight w (H1' on development data):
| w | seeds | audit A (sd) | A low-overlap | bone PSNR_cal | SSIM_cal | QP-test bone MAE |
|---|---|---|---|---|---|---|
| 0 | 3 | 0.190 (0.005) | 0.170 | 21.77 | 0.8537 | 0.0033 |
| 4 | 2 | 0.120 | 0.106 | 21.79 | 0.8511 | 0.0042 |
| 8 | 2 | 0.094 | 0.076 | 21.79 | 0.8486 | 0.0049 |
| 16 | 4 | 0.067 (0.006) | 0.051 | 21.74 | 0.8457 | 0.0057 |
| 32 | 2 | 0.058 | 0.043 | 21.35 | 0.8385 | 0.0071 |
| 64 | 2 | 0.037 | 0.027 | 21.45 | 0.8400 | 0.0121 |
A falls monotonically (w = 16 vs 0: -65%); real bone PSNR_cal is flat up to w = 8, -0.03 dB at w = 16, -0.4 dB from w = 32; the price at w = 16 is SSIM_cal -0.008 and a
larger error on the Qure-projection test (0.0057 vs 0.0033, still small). The ratio of A in low-overlap to high-overlap sites falls from 0.83 (w = 0) to 0.71 (w = 16) and 0.57 (w = 64): the regulariser removes more absorption where the lesion does not sit on bone.
Ablation arms at w = 16 (R3; 2 seeds): stop-gradient on the base branch A 0.185 / PSNR 22.10 / SSIM 0.844 (no effect: the twin loss needs gradient through both branches);
dilated support (k = 31) A 0.137; whole-image invariance A 0.197; 4 lesions per twin A 0.074 / PSNR 21.84 / SSIM 0.847 (about the same as 1 lesion). Caveat: the loss is a mean over its
support, so the dilated and whole-image arms also dilute the per-pixel weight on the lesion by |lesion| / |support|; they test "support as implemented", not support at equal weight.
v1 finals seed 2 (SAP.md family): clia A 0.297 / PSNR 22.38 / SSIM 0.855; recipe A 0.308 / 22.57 / 0.856 (consistent with seeds 0 and 1).
Encoder at 300k: RAD-DINO ViT-B/14 + twin16 (enc lr x0.25; trained in two sessions with --resume, checkpoints resumed at steps 22,039 / 23,439, all 37,500 steps done):
| run | audit A | A low-overlap | bone PSNR_cal | SSIM_cal | QP-test bone MAE |
|---|---|---|---|---|---|
| RAD-DINO s0 | 0.0453 | 0.032 | 22.06 | 0.8502 | 0.0055 |
| RAD-DINO s1 | 0.0464 | 0.035 | 21.77 | 0.8500 | 0.0044 |
| mean | 0.0459 | 0.034 | 21.92 | 0.8501 | 0.0050 |
| SwinV2-T twin16 reference (4 seeds) | 0.0667 | 0.051 | 21.74 | 0.8457 | 0.0057 |
Rule R2 (A <= 0.0567 and PSNR >= 21.685 and SSIM >= 0.841, n >= 2): **RAD-DINO + twin16 passes**; it is the only finished 300k candidate that does (4 lesions/twin fails: A 0.0736).
Better than the reference on all four metrics (A -31%, bone PSNR +0.18 dB, SSIM +0.004, QP MAE -12%); the seed spread of its bone PSNR is larger (0.29 dB between the two seeds).
Pending 300k candidates: ConvNeXt-V2-T (s4-w1b-C, due about 21:15) and SwinV2-B (s4-w1b-B + B2, the first session stopped at its 11 h budget).
Tie-break fixed now, before those two are read: among candidates that pass R2, "ours" is the one with the lowest mean A; within 0.005 the one with the higher bone PSNR_cal. RAD-DINO is provisional until then.
Consequences (SAP_v2.md Definitions: a replacement keeps the K = 3, seeds 0-2 rule): ours = RAD-DINO + twin16, seeds 0, 1, 2; a third seed is needed. The no-twin comparator must differ from ours only in
the twin loss, otherwise the H1' / H3' contrasts mix encoder and regulariser: RAD-DINO no-twin (arm `qp`, same recipe), seeds 0-2 (clarification of "same data and recipe", before the freeze; DEVIATIONS.md).
SwinV2-T keeps its role as the efficient variant: SwinV2-T twin16 (seeds 0-3) and SwinV2-T no-twin (seeds 0-2) are reported as secondary methods, giving a 2 x 2 design {SwinV2-T, RAD-DINO} x {w = 0, w = 16}.
Runs needed (4 jobs, 2 kernels, each 6.5 h + continuation, about 11 session hours): s4-w2-1 (ac_2: RAD-DINO twin16 seed 2 + no-twin seed 2), s4-w2-2 (ac_5: no-twin seeds 0, 1), continuations s4-w2-1b / s4-w2-2b.
After them ac_2 and ac_5 are nearly out of quota this week; downstream runs after the freeze spill into the week of Nov 7 (accounts with quota left go first). Open checks before release: RAD-DINO licence terms for
redistributing fine-tuned weights (MSRLA), and an inference-cost row for the paper (ViT-B/14 vs SwinV2-T).

## 2026-10-31 - SwinV2-B + twin16 at 300k read against R2
s4-w1b-B (11 h budget, stopped before the end) + s4-w1b-B2 (continuation, finished): 2 seeds, 37,500 steps each. Development data, mean over seeds:
| run | audit A | A low-overlap | bone PSNR_cal | SSIM_cal | QP-test bone MAE |
|---|---|---|---|---|---|
| SwinV2-B s0 | 0.0640 | 0.045 | 21.85 | 0.8469 | 0.0053 |
| SwinV2-B s1 | 0.0589 | 0.047 | 21.90 | 0.8466 | 0.0051 |
| mean | 0.0615 | 0.046 | 21.88 | 0.8467 | 0.0052 |
Rule R2: **fails** (A 0.0615 > 0.0567 for the first branch; PSNR 21.876 is 0.009 dB short of 21.885 for the second). A near miss: A -8% and bone PSNR +0.14 dB against SwinV2-T twin16, at about 3x the parameters.
RAD-DINO + twin16 stays the only R2 passer; ConvNeXt-V2-T (s4-w1b-C) is still to be read.

## 2026-10-31 - RAD-DINO weights through the downstream prep kernel (48-image dry run, ac_4)
`python ops/launch_prep.py ac_4 tbx s4-wts-rd rd_twin16_s0.pt+rd_twin16_s1.pt --limit 48` (weights dataset s4-wts-rd: the two finished RAD-DINO twin16 300k seeds, 171 MB each, uploaded in 32 s each):
the kernel loaded the ViT-B/14 ensemble (the HF config is fetched anonymously, only an "unauthenticated" warning), processed 48 TBX11K images with 0 unreadable and wrote orig / soft / bone / soft_qure arrays
(`DONE tbx ok 48 bad 0 in 162 s`, startup included). The downstream path therefore works with RAD-DINO weights; no code change was needed on Kaggle. Throughput on the full sets is still to be measured by the real prep kernels.

## 2026-10-31 - ConvNeXt-V2-T + twin16 at 300k read against R2; tie-break applied
s4-w1b-C (ac_4, 7.5 session hours): `convnextv2_tiny.fcmae_ft_in22k_in1k`, encoder lr x0.25, 2 seeds, 37,500 steps each. Development data (dev-24 + audit on all 75), mean over seeds:
| run | audit A | A low-overlap | A high-overlap | bone PSNR_cal | SSIM_cal | QP-test bone MAE |
|---|---|---|---|---|---|---|
| ConvNeXt-V2-T s0 | 0.0460 | 0.0394 | 0.0477 | 22.040 | 0.8492 | 0.0043 |
| ConvNeXt-V2-T s1 | 0.0532 | 0.0396 | 0.0593 | 21.977 | 0.8493 | 0.0043 |
| mean | 0.0496 | 0.0395 | 0.0535 | 22.009 | 0.8492 | 0.0043 |
| (RAD-DINO mean, for comparison) | 0.0459 | 0.0335 | 0.0519 | 21.918 | 0.8501 | 0.0050 |
Rule R2: **passes** (A 0.0496 <= 0.0567, bone PSNR_cal 22.009 >= 21.685, SSIM_cal 0.8492 >= 0.841, n = 2). Two candidates now pass (ConvNeXt-V2-T, RAD-DINO). Tie-break (written before ConvNeXt was read):
lowest mean A, and within 0.005 the higher bone PSNR_cal. The mean A values differ by 0.0037 (< 0.005), so bone PSNR decides: ConvNeXt-V2-T 22.009 vs RAD-DINO 21.918 -> **ours = ConvNeXt-V2-T + twin16**.
Caveat stated with the decision: on dev-24 the two encoders are not statistically distinguishable (seed spread of bone PSNR 0.06 dB for ConvNeXt and 0.29 dB for RAD-DINO; of A 0.007 and 0.001);
the rule is followed because it was fixed before the result was seen, not because ConvNeXt was shown to be better. SwinV2-B (fails R2) and SwinV2-T twin16 (the reference) are not candidates.
Consequences (K = 3, seeds 0-2; no-twin comparator differs only in the twin loss): ours = ConvNeXt-V2-T twin16 seeds 0, 1 (done), 2; comparator = ConvNeXt-V2-T no-twin (arm `qp`, same recipe) seeds 0-2.
Four new runs, 2 kernels of about 7.5 session hours: s4-w3-1 (ac_4: twin16 seed 2 + no-twin seed 0) and s4-w3-2 (ac_1: no-twin seeds 1, 2), queue file outputs/queue_wave3.json (second scheduler, own heartbeat file).
The RAD-DINO runs already in flight (s4-w2-1 / s4-w2-2, started under the earlier provisional choice) are kept as the third encoder: RAD-DINO twin16 seeds 0-2 and no-twin seeds 0-2, reported as a secondary method.
Design for the paper: {SwinV2-T, ConvNeXt-V2-T, RAD-DINO} x {w = 0, w = 16}; the headline model is ConvNeXt-V2-T. Quota after the new runs: ac_4 about 1.8 h, ac_1 about 1 h left this week.
Licence: timm lists the ConvNeXt-V2 weights as cc-by-nc-4.0 (RAD-DINO: MSRLA); both restrict commercial use of the weights; check the terms for redistributing fine-tuned weights before any release.
Local check of the ConvNeXt-V2-T weights (61.5 MB each, 37,500 steps): both seeds reproduce Kaggle's dev-24 numbers on the laptop (bone PSNR_cal within 0.0012 dB, SSIM_cal identical; 63-76 s for the 24 images, 0.37 GB GPU memory).
The Kaggle download of the seed-1 weights failed once (the queue logged `fetch rc 1`); it was repeated with the new per-attempt timeout in ops/fetch.py and verified by torch.load.

## 2026-10-31 - BS-LDM baseline, full run (released SZCH weights; development data)
Kernel s4-bsldm-full (ac_1, about 8 session hours, finished 21:03): 75 SZCH pairs for fidelity + audit sites 0-2 of the 6 per image (225 sites, the same sites as the first three of every other method's audit).
Merged with `python tools/merge_bsldm.py outputs/bsldm/full/out --dest results/final` (the tool now prints dev-24 only; `--held` is for after the freeze). Nothing was read for the held-out 51.
Fidelity on dev-24 (mean over the 24 pairs; ours / reference rows are means over seeds):
| method | bone PSNR_cal | PSNR_cal | SSIM_cal |
|---|---|---|---|
| ConvNeXt-V2-T twin16 (2 seeds) | 22.009 | 25.662 | 0.8492 |
| RAD-DINO twin16 (2) | 21.918 | 25.727 | 0.8501 |
| SwinV2-T twin16 (4) | 21.735 | 25.400 | 0.8457 |
| SwinV2-T no twin (3) | 21.766 | 25.223 | 0.8537 |
| Qure released | 21.969 | 25.764 | 0.8746 |
| no suppression | 20.328 | 26.293 | 0.8365 |
| BS-LDM (one released model) | 29.893 | 32.803 | 0.8653 |
Lesion-absorption audit on the same 225 sites (all 75 images = development evidence; overlap tertile cuts 0.031 / 0.259 from the BS-LDM sites):
| method | runs | A | A low-overlap | A high-overlap |
|---|---|---|---|---|
| ConvNeXt-V2-T twin16 | 2 | 0.0512 | 0.0392 | 0.0442 |
| RAD-DINO twin16 | 2 | 0.0457 | 0.0321 | 0.0422 |
| SwinV2-T twin16 | 4 | 0.0670 | 0.0445 | 0.0655 |
| SwinV2-T no twin | 3 | 0.1803 | 0.1573 | 0.1779 |
| Qure released | 1 | 0.1341 | n/a | n/a |
| BS-LDM | 1 | 0.4173 | 0.4503 | 0.3919 |
Reading: BS-LDM is far ahead on bone fidelity (29.9 vs 22.0 dB; it was trained on this hospital's real dual-energy pairs, so this is not a like-for-like contest) and its SSIM_cal sits between ours and Qure's,
but it moves 42% of an inserted lesion into its bone layer (per-site median 0.43, 39% of sites above 0.5) against 5% for ConvNeXt-V2-T + twin16 and 13% for Qure: fidelity to the dual-energy bone image does not protect
lesions. Caveats: synthetic lesions; one stochastic sample per site with seeded paired noise (the change outside the lesion, "leak", is 0.0019, the same as the other methods, so sampling noise does not drive A);
BS-LDM is not one of the pre-registered hypotheses (descriptive); the confirmatory audit on RibCXR and JSRT after the freeze is where such a claim is tested. Audit sites 3-5 can follow (`--sites 3:6`, about 8 more session hours).
Qure's row: outputs/audit_final/qure.csv is the 2026-10-27 audit file; it has the same sites (alpha, diameter) but a different 'overlap' column than the current audit code (60% of sites differ), so only its overall A is used here.
The final local run (tools/final_methods.txt) recomputes Qure's audit with the current code.

## 2026-11-01 - ConvNeXt-V2-T wave 3 (seed 2 + matched no-twin): R2 still passes at K = 3
Kernels s4-w3-1 (ac_4, finished 04:37) and s4-w3-2 (ac_1, finished 04:05) were fetched: 4 runs, final_ema.pt 61.5 MB each, 37,500 steps, torch.load ok. SZCH dev-24 + audit (all 75 images = development evidence), mean (sd) over seeds 0-2:
| arm | runs | A | A low-overlap | A high-overlap | leak | bone PSNR_cal | SSIM_cal | qp_mae |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-V2-T twin16 | 3 | 0.0510 (0.0043) | 0.0406 | 0.0544 | 0.0012 | 21.997 (0.037) | 0.8496 | 0.0043 |
| ConvNeXt-V2-T no twin | 3 | 0.1895 (0.0026) | 0.1620 | 0.2096 | 0.0011 | 21.980 (0.048) | 0.8555 | 0.0031 |
Per seed (twin16): A 0.0460 / 0.0532 / 0.0538, bone PSNR_cal 22.040 / 21.977 / 21.973 (seed 2 agrees with seeds 0-1). Rule R2 at n = 3: passes (A 0.0510 <= 0.0567, PSNR 21.997 >= 21.685, SSIM 0.8496 >= 0.841).
Twin effect for ConvNeXt (development, descriptive): A -73% (0.1895 -> 0.0510) with no bone-PSNR cost (+0.017 dB), SSIM_cal -0.006, qp_mae +0.0012. The no-twin A of ConvNeXt (0.1895) equals that of SwinV2-T (0.1899), so for these two encoders the absorption without the regulariser does not depend on the encoder.
Selection note, written because the tie-break is borderline at K = 3: ConvNeXt was selected on the like-for-like n = 2 vs n = 2 comparison (A difference 0.0037, inside the 0.005 band) before seed 2 existed, and the selection is not reopened. With ConvNeXt at K = 3 its mean A is 0.0510 against
RAD-DINO's 0.0459 at n = 2 (difference 0.0051, 0.0001 outside the band); that is about one seed sd of ConvNeXt's A (0.0043), i.e. not a distinguishable difference. The K = 3 vs K = 3 comparison is reported once RAD-DINO seed 2 is in, as description for the discussion, not as a new choice.
Not available for every run: the synth_test folder exists only for the no-twin seeds 1-2 (s4-w3-2); the final local run (tools/final_local.sh synth) computes it for every method.

## 2026-11-01 - Final evaluation configuration (written before the freeze)
tools/final_methods.txt now follows the ConvNeXt headline: `ours` / `notwin` = ConvNeXt-V2-T twin16 / no twin, K = 3 (seeds 0-2); SwinV2-T pair `swin` / `swin_nt` (seed 3 kept as reserve; the dose-response weights w4-w64 are SwinV2-T); RAD-DINO pair `rd` / `rd_nt`; controls unchanged
(clia_v1, recipe_v1, qure, identity, clahe, homomorphic). Per-seed models run szch, audit and qp; the K = 3 ensembles also run rib, and `ours` / `notwin` / clia_v1 also run jsrt (the CNR evidence of H4'a).
tools/final_stats.py: the dose-response of H1' stays on the SwinV2-T family, the only one with twin weights 0-64 (swin_nt_s0-2 at w = 0, swin_s0-3 at w = 16); the ConvNeXt and RAD-DINO pairs add w = 0 vs 16 contrasts per audit site ("encoder_pairs", secondary, no Holm).
tools/final_local.sh: the H5' uncertainty step uses the three ConvNeXt twin16 models. SAP_v2.md is unchanged: its replacement rule covers the change of encoder.
Smoke test before the freeze: synthetic audit CSVs only (dose-response, contrasts, encoder pairs run); no model was evaluated on any held-out or locked set. Weights through wave 3 are linked under their final names in outputs/final_v2/weights (tools/collect_final.py); only the RAD-DINO seed 2 and no-twin runs are missing.

## 2026-11-01 - Inference cost (tools/inference_cost.py -> results/final/cost.csv; RTX 3050 4 GB laptop GPU, one image)
| model | input | params | GFLOPs | ms per image | peak GPU memory |
|---|---|---|---|---|---|
| ConvNeXt-V2-T + UNet++ (ours) | 512 px, fp16 | 30.7 M | 76.0 | 46 | 0.37 GB |
| SwinV2-T + UNet++ | 512 px, fp16 | 30.4 M | 76.8 | 44 | 0.73 GB |
| RAD-DINO ViT-B/14 + UNet++ | 512 px, fp16 | 89.6 M | 334.2 | 67 | 0.81 GB |
| Qure released (SwinV2-B, TorchScript) | 1024 px, fp32 | 96.1 M | n/a | 604 | 1.16 GB |
The K = 3 ensemble costs three times the single-model row. Latency of ours is single-image, batch 1 (the GPU is not saturated, so the FLOP ratio of RAD-DINO overstates its latency ratio).

## 2026-11-01 - Evaluation-code fix before the freeze
s4/eval_ribcxr.py had no `ens` method, so the `rib` step of every K = 3 ensemble in tools/final_methods.txt would have stopped with KeyError: 'ens' (the other scripts already accepted it). Added (make_ens from s4/eval_real.py). All other steps were checked against the method kinds in the config
(szch: eval_real, audit and qp: audit_real.bone_fn, jsrt: s4.prep with '+'-joined weights, synth and unc: eval_synth.load_model). Ensemble loading smoke-tested on random noise, no real image: ConvNeXt K = 3 170 ms, RAD-DINO K = 2 158 ms per 1024 px image, peak GPU memory 0.63 / 1.17 GB.

## 2026-11-01 - RAD-DINO wave 2 (seed 2 + matched no-twin) read; K = 3 vs K = 3 against ConvNeXt
Kernels s4-w2-1b (ac_2, finished 05:53, fetched 06:01) and s4-w2-2b (ac_5, finished 06:01, fetched 06:09) delivered 4 runs (RAD-DINO twin16 seed 2; RAD-DINO no twin seeds 0-2): final_ema.pt 179.3 MB each, 37,500 steps, torch.load ok, the same 377-tensor layout as seed 0.
Local inference from the stored weights reproduces what Kaggle computed on SZCH dev-24 for all nine checkpoints checked (5 RAD-DINO, 4 ConvNeXt wave 3; max |difference| over 24 images: bone PSNR_cal 0.0024 dB, SSIM_cal 0.0000), so the local final run measures the same models.
Development data (SZCH dev-24 + audit on all 75 images), mean (sd) over seeds 0-2; the audit sites are identical across these 12 runs (checked: the `overlap` column of the 12 audit.csv files agrees at all 450 sites):
| arm | runs | A | A low-overlap | A high-overlap | bone PSNR_cal | SSIM_cal | qp_mae |
|---|---|---|---|---|---|---|---|
| ConvNeXt-V2-T twin16 | 3 | 0.0510 (0.0043) | 0.0406 | 0.0544 | 21.997 (0.037) | 0.8496 | 0.0043 |
| ConvNeXt-V2-T no twin | 3 | 0.1895 (0.0026) | 0.1620 | 0.2096 | 21.980 (0.048) | 0.8555 | 0.0031 |
| RAD-DINO twin16 | 3 | 0.0455 (0.0008) | 0.0331 | 0.0513 | 21.868 (0.168) | 0.8499 | 0.0047 |
| RAD-DINO no twin | 3 | 0.1781 (0.0033) | 0.1547 | 0.1973 | 21.683 (0.028) | 0.8556 | 0.0029 |
RAD-DINO twin16 per seed: A 0.0453 / 0.0464 / 0.0449, bone PSNR_cal 22.061 / 21.774 / 21.768. Rule R2 at n = 3: passes (A 0.0455 <= 0.0567, PSNR 21.868 >= 21.685, SSIM 0.8499 >= 0.841).
Twin effect (A with w = 16 against w = 0; development, descriptive): RAD-DINO -74%, ConvNeXt-V2-T -73%, SwinV2-T -65% (4 vs 3 seeds); bone PSNR_cal +0.185 / +0.017 / -0.03 dB; SSIM_cal -0.006 / -0.006 / -0.008. About the same relative reduction for three encoders.
K = 3 vs K = 3, twin16, ConvNeXt minus RAD-DINO: A +0.0055, bone PSNR_cal +0.129 dB, SSIM_cal -0.0003, qp_mae -0.0004; parameters 30.7 M vs 89.6 M, 46 vs 67 ms per image.
Stated plainly: the A difference is 0.0005 outside the 0.005 band, so applying the tie-break again to these K = 3 means would pick RAD-DINO (lowest A). The selection of 2026-10-31 stands: it was made on the like-for-like n = 2 data that existed then, and the selection note above (commit 1d34aba),
written before this result existed, fixed that the K = 3 vs K = 3 comparison is description, not a new choice. The timing of the rule is logged in DEVIATIONS.md. For the reader: the A difference is 1.3 seed sd of ConvNeXt's A; the seed spread of RAD-DINO's bone PSNR_cal is 0.17 dB (ConvNeXt 0.04);
the two encoders trade A against bone PSNR_cal; the RAD-DINO pair is evaluated next to ours on SZCH held-out, the RibCXR audit and benchmark and the Qure-projection test, so both results are reported. No held-out or locked number of either has been computed.

## 2026-11-01 - Stored EMA weights: fp16 overflow in one BatchNorm running variance (disclosed, not corrected)
final_ema.pt keeps the EMA in fp16. In 7 of the 27 checkpoints named in tools/final_methods.txt the running variance of one decoder BatchNorm (`nodes.2_1.1`, 192 channels) exceeds the fp16 range (65,504) and is +inf in some channels:
ConvNeXt twin16 seed 0 (41/192) and seed 2 (1/192); RAD-DINO twin16 seeds 0 / 1 / 2 (35 / 15 / 7) and RAD-DINO no-twin seed 1 (15); SwinV2-T w = 64 seed 0 (two BatchNorms, 58/96 and 112/192). No NaN anywhere; the other 20 checkpoints are finite.
In eval mode a channel with infinite running variance outputs its bias. Every number in this project, from Kaggle and from the local machine, is computed from the stored weights, and the dev-24 check above reproduces Kaggle's, so the results describe the stored models and nothing is inconsistent.
The BatchNorm statistics of the live fp32 EMA were not saved and cannot be recovered. The overflow appears in the twin runs (and one RAD-DINO no-twin run); the cause was not investigated. Not corrected, because re-estimating the statistics would create a new model after the selection. Reported as a limitation.

## 2026-11-01 - SZCH audit sites differ between Kaggle and the local machine
Local audit of ConvNeXt twin16 seed 2 (all 75 images, 450 sites): mean A 0.0529 against Kaggle's 0.0538 for the same checkpoint; per-site correlation 0.95, 29 of 450 sites differ by more than 0.05 in A (largest 0.33). alpha and diameter of every site are identical, but the bone overlap differs at 259 of 450 sites,
i.e. the lesions sit at different places. The sites are drawn inside the PSPNet lung masks (torchxrayvision); a different mask on another GPU / library version shifts the later draws of that image (likely cause, not verified).
Within one machine the sites are fixed: two local runs, one with another method in a separate process, give identical sites (12 images, 72 sites, max |difference| of overlap 0), and the 12 Kaggle runs above share identical sites.
Consequence: every contrast of the final evaluation (ours vs no twin, vs Qure, dose-response) is paired on identical sites because all methods run locally; its A values are not interchangeable with the Kaggle dev table at the 0.001-0.01 level. The selection was made on the dev table, before the final run.

## 2026-11-01 - Node21 real-nodule contrast (H4'a) read; two boxes with undefined contrast dropped
Source: the prep kernels of the frozen models on Node21 (4,882 images, 1,476 boxes): ours = ConvNeXt-V2-T twin16 K = 3 plus Qure (account ac_2), no-twin K = 3 (account ac_3); files results/final/node21/{ours,notwin}/ann.json. JSRT is not run yet.
Two boxes (images 4268 and 4742, box 0) have an undefined contrast (NaN) in every field. cluster_bootstrap_ci returns NaN for any NaN input, so the first run of tools/final_stats.py gave NaN intervals. The script now drops such boxes jointly for ours and no-twin before the bootstrap and prints how many (2 of 1,476).
Box CNR(soft) - CNR(orig), image-cluster bootstrap 95% CI, paired (the SAP names no margin, so these are estimates, not a pass/fail test):
| subset | boxes | ours | Qure | ours - Qure | ours - no twin |
|---|---|---|---|---|---|
| all | 1,474 | +0.0778 | +0.1028 | -0.0250 [-0.0326, -0.0175] | +0.0116 [+0.0074, +0.0159] |
| bone overlap >= 0.15 | 1,109 | -0.0022 | +0.0149 | -0.0170 [-0.0257, -0.0085] | +0.0074 [+0.0028, +0.0119] |
Plainly: Qure raises the nodule contrast more than ours (both intervals exclude 0), and ours raises it more than the same network without the twin loss. In nodules over bone ours leaves the contrast unchanged (-0.002), Qure adds +0.015, the no-twin model removes 0.010 (derived: ours minus the paired difference).
Not checked: CNR is a ratio, so a suppressor that smooths the background raises it without keeping the lesion signal. ann.json holds only the ratio, so whether this explains Qure's higher value is open; the counterfactual audit (absorption A) is the measurement made for that question. Holm over the two contrasts is not computed by the script; the intervals are reported as they are.
Downstream detection (4 inputs, one seed so far, test split of Node21) landed in outputs/dn_dl/s4-dn-node21-1; seeds 2-4 wait for the quota refill on Nov 7. Nothing is interpreted before all seeds are in.

## 2026-11-02 - Three more Kaggle accounts; Node21 kernels 3-4 and all of NIH move to ac_6
The user added three Kaggle accounts: ac_6 ac_6, ac_7 ac_7_former, ac_8 ac_8_former (config.yaml; tokens only in Kaggle_Token.txt). Each shows 30 GPU-h unused, refill 2026-11-07 00:00 UTC.
A two-line test kernel (kernels/gputest_ac_6..8, logs in outputs/gputest_dl) found 2x T4 and internet on ac_6 only; ac_7 and ac_8 ran on a CPU box without internet, which is what Kaggle does for accounts without phone verification. They get no jobs until the owners verify their phones.
ac_6 received the frozen weights (s4-wts3, the same six files, sha256 checked against ac_1) and now runs, via outputs/queue_wave4.json: its own Node21 download and prep (same code, weights and data as ac_2), Node21 kernels 3 and 4 (soft_qure/clahe seed 1, all of seed 2), and all of NIH (prep-nih, prep-nihq, dn-nih-a, dn-nih-b; 22.4 est GPU-h in total). TBX11K stays on ac_4 (its prep is done there) and VinBigData on ac_1 (only account that accepted the competition rules); both still wait for the Nov 7 refill.
Deviation to disclose: Node21 kernels 1-2 (all of seed 0; orig, soft, both of seed 1) read prep arrays made on ac_2, kernels 3-4 read arrays re-made on ac_6. The prep is the same deterministic inference, but GPU floating point may differ in the last bit before the uint8 rounding; training noise of the detector is far larger. The prep kernels' small outputs (ann.json, labels, split, logs) are now fetched to outputs/prep_dl/<ac>-<kernel> (jobqueue item field fetch_pattern); the image arrays stay on Kaggle.

## 2026-11-02 (later) - ac_8 is now ac_8; TBX11K moves to ac_8
The user replaced the ac_8 token: ac_8 is now ac_8 (ac_8_former is dropped; its second test at 15:27 again had no GPU and no internet). ac_8's test kernel found 2x T4 and internet, 30 GPU-h unused; it received s4-wts3 (same six files). The four TBX11K detection kernels (s4-dn-tbx-1..4, none launched before) move from ac_4 to ac_8 with their own prep kernel s4-prep-tbx-a8 (17 est GPU-h in total), so all TBX11K seeds read the same prep arrays. ac_4's finished s4-prep-tbx is no longer used. ac_7 (ac_7_former) still lacks phone verification and has no jobs. Only VinBigData (ac_1) now waits for the Nov 7 refill.

## 2026-11-02 (later) - ac_9 replaces ac_7; VinBigData moves to ac_9
The user added ac_9 ac_9 to use instead of ac_7 ac_7_former (still unverified, no jobs). ac_9's test kernel found 2x T4 and internet, 30 GPU-h unused; it received s4-wts3 (same six files).
A download of the competition's sample_submission.csv showed that ac_8 and ac_9 have accepted the VinBigData competition rules (ac_6: 403). config.yaml now lists vinbigdata_joined = ac_1, ac_8, ac_9, and ops/launch_prep.py checks that list instead of allowing ac_1 only.
The four VinBigData detection kernels (s4-dn-vin-1..4, none launched before) move from ac_1 to ac_9 with their own prep kernel s4-prep-vin-a9 (29.2 est GPU-h of ac_9's 30; ac_1's prep took 0.8 h of its 1.2 h estimate), so all VinBigData seeds read the same prep arrays. ac_1's finished s4-prep-vin is no longer used. If the last kernel would exceed ac_9's quota, the scheduler holds it until ac_9's Nov 7 refill. No downstream family waits for Nov 7 any more.

## 2026-11-02 (later) - ac_7 is now ac_7; no jobs assigned
The user replaced the ac_7 token: ac_7 is now ac_7 (ac_7_former is dropped). Its test kernel (outputs/gputest_dl/ac_7_v2) found 2x Tesla T4 and a working DNS lookup, 30 GPU-h and 20 TPU-h unused, refill 2026-11-07 00:00 UTC. It has not accepted the VinBigData rules (sample_submission.csv gave 403), so it is not in vinbigdata_joined and cannot run VinBigData.
No wave-4 job is waiting (every job is launched or done), so ac_7 got no weights, no code dataset and no jobs. It is spare capacity: if a downstream kernel fails or overruns another account's quota, it is the place to rerun it (weights and code datasets first).

## 2026-11-03 - The rest of the locked-set evaluation moves to Kaggle; audit sites pinned to the local masks
The local run (`N=3 bash tools/final_parallel.sh`) wrote no result for 11 h (13:11-00:05; its last steps normally took 30-90 min) while the laptop GPU sat at 86 C, thermally throttled to 652 of 2100 MHz. It was restarted with 2 workers, then stopped at the user's request ("no local run, run on Kaggle"); no partial output was left. 64 outputs were missing.
New kernel `kernels/evalk/run_eval.py` + `ops/launch_eval.py`: `tools/final_local.sh` runs unchanged inside a workspace built from the private datasets s4-code (code + evalpack: final_local.sh, final_methods.txt, freeze_v2.json, splits, audit masks) and s4-wte (15 weight files: RAD-DINO twin16 / no-twin s0-2, ConvNeXt twin16 s0-2 for `unc`, clia / recipe s0-2), SZCH and Qure's weights downloaded from Hugging Face in the kernel, JSRT from Kaggle (raddar), and the Qure projection pack built by the CPU kernel s4-qpprep on ac_7 (21,887 rows, 6.34 GB, no GPU quota). `final_local.sh` now skips the RibCXR outputs where data/ribcxr_raw is absent.
A smoke kernel (ac_8) confirmed the 2026-11-01 finding: with Kaggle's own lung masks 259 of 450 SZCH and 988 of 1,482 JSRT sites have a different bone overlap (alpha and diameter identical). Fix: `tools/export_audit_masks.py` exports the local PSPNet lung masks and the bone-overlap masks (SZCH: real DES bone signal; JSRT: the Qure-map rule of s4/audit_real.py) to outputs/audit_masks; `s4/audit_real.py` reads them when S4_AUDIT_MASKS is set (default path unchanged). Local check with the pinned masks: overlap, image, site, alpha and diameter identical to the local final files (ours, rd_s0, notwin) at 450 / 450 SZCH and 1,482 / 1,482 JSRT sites. The Kaggle check is done on every audit file of the Kaggle run before it is merged.
What still differs between machines: the forward pass of each model (Kaggle T4 vs local RTX 3050), measured on 2026-11-01 at max 0.0024 dB on dev-24. Kaggle runs: s4-eval-szch (ac_8, szch for 11 methods), s4-eval-qp (ac_7, qp for 11 methods + unc), s4-eval-audit (ac_8, JSRT + SZCH audits for 11 methods = 22 files). Not on Kaggle: the 17 RibCXR outputs (licensed: ac_1 after its Nov 7 refill, or local) and the 2 synthetic-bank outputs (clia_v1, recipe_v1; the kernel does not read the bank layout yet). The smoke kernel also found that `lpips` is not installed on Kaggle (the kernel now installs it) and that `final_local.sh` hides failures (rc 0, error only in the log; the kernel now flags errors in its summary).
Result of the Kaggle runs (2026-11-03 01:40): all 22 Kaggle audit files (JSRT 1,482 and SZCH 450 sites each, 11 methods) have image, site, alpha, diameter and bone overlap identical to the local final files (0 differences), and so do the 11 Kaggle qp audits (1,720 sites, 430 images each; the qp sites come from the stored pack, no mask pinning needed). The 11 szch files (75 images) and unc_heldout.json came back without errors. All 45 are merged into results/final (no local file overwritten). The first `unc` attempt failed (tools/szch_split.py was not in s4-code; package_code.py now ships it). Still missing: 17 RibCXR outputs (ac_1 after Nov 7 or local) and 2 synthetic-bank outputs.
Update (2026-11-03 10:55): the last 19 outputs are in, so every locked-set output in results/final is complete (DRY lists nothing left). Synthetic bank (clia_v1, recipe_v1, 3 checkpoints each): kernel s4-eval-synth on ac_3, which holds s4-bank (all 844 usable CTs; the 158 test CTs byte-identical to the local shards); the earlier wrong bank path in final_local.sh is in DEVIATIONS.md. Against the end-of-training synth runs of the same checkpoints: same CTs, sites and bone overlap, PSNR within 0.013 dB, absorption median difference 0.0004 (max 0.06 at single small-lesion sites). VinDr-RibCXR (9 audits, 8 rib benchmark): kernel s4-eval-ribcxr on ac_1 only (licence), from the private datasets s4-ribcxr and s4-ribmask (lung masks exported on the local machine by tools/export_audit_masks.py; they never enter s4-code, package_code.py asserts it). All 9 audits have the 1,470 local sites with 0 overlap differences; the 8 rib files cover the same 245 images as the local ones; Kaggle CLAHE equals the local CLAHE to 1e-6 and the Kaggle identity audit equals the local one exactly. First attempt failed in setup (annotation files say data/..., the folder is Data; Linux is case-sensitive) and cost 0.05 h; the second used 0.71 h, leaving ac_1 0.10 h until Nov 7.

## 2026-11-03 - BDCXR-3257 external validation (SAP_bdcxr.md), results
Paediatric chest radiographs from Dr. M R Khan Shishu Hospital & Institute of Child Health, Dhaka (IRB REC-FSIT-2025/No: 11928); downloaded from figshare inside each Kaggle kernel (ac_8), never re-uploaded. Folds: near-duplicate groups by image correlation (DEVIATIONS.md), 5 folds x 3 seeds x 5 inputs = 75 DenseNet121 runs. Numbers from results/bdcxr/stats.json (tools/bdcxr_stats.py).
- E1 (primary): pooled out-of-fold macro-AUROC orig 0.814; soft (ours) 0.816, difference +0.002 [-0.005, 0.010]: non-inferior at the 0.01 margin, not superior. Qure soft +0.001 [-0.008, 0.010] (non-inferior); CLAHE -0.006 [-0.012, 0.001] (not non-inferior); orig+soft -0.001 [-0.008, 0.005].
- E2 (secondary): 300 images, 1,800 sites, no lung-mask failure. A ours 0.084, Qure 0.217, no-twin 0.251; ours - Qure -0.134 [-0.144, -0.124], ours - no-twin -0.167 [-0.175, -0.160] (Holm p < 0.001); the advantage is larger at bone-overlapping sites (interaction -0.137 [-0.154, -0.119]; overlap from the Qure bone map as planned). Inserted lesions are adult nodules: a stress test.
- E3 (descriptive): median lung-field ensemble spread 0.0076 [IQR 0.0067-0.0088] on BDCXR vs 0.0028 [0.0026-0.0033] on SZCH held-out (2.7x); Spearman with age -0.22 (younger children, more disagreement).

## 2026-11-03 - independent recheck of the final analysis
Two fresh-context reviews (statistics vs SAP_v2/SAP.md/SAP_bdcxr; code changes since e18db35) and a result-integrity pass. Every confirmed issue was fixed and is listed in DEVIATIONS.md (2026-11-03 entries); the regenerated stats_held.json keeps every pre-registered verdict (H1', H2', H3' main contrasts and p-values unchanged), widens the interaction intervals, completes H4'b with intervals, adds H5' intervals and SAP.md H1 (clia vs recipe on synthetic test CTs: A -0.024 [-0.034, -0.014], bone MAE non-inferior: passes).

## 2026-11-03 - BDCXR-3257: lab permission
The author confirmed lab permission to use BDCXR-3257 for this work and to complete it. The private Kaggle outputs of the BDCXR kernels on ac_8 (s4-prep-bdcxr: 256 px arrays; s4-dn-bdcxr / s4-bdcxr-eval: predictions and metrics) are therefore kept. Still applies (data use agreement): no raw BDCXR radiograph in any figure or public release without the data provider's written permission; acknowledge Dr. M R Khan Shishu Hospital & Institute of Child Health, Dhaka, and IRB REC-FSIT-2025/No: 11928 in every paper, thesis or talk.

## 2026-11-03 - v3 extension proposed (multi-stream classifier, shift-aware uncertainty, XAI, figure suite)
Research directive from the author. Design review and proposal in DESIGN_v3.md (corrections to the directive: S and B are functions of X, so a
parameter-matched control X x3 is required; no rib-fracture label in NIH or VinBigData; pneumothorax is not focal; no DeLong test for AUPRC;
VinDr-RibCXR has no bone image for conformal calibration; figure panels only from LIDC, SZCH and NIH images). SAP_v3.md is a DRAFT: it is
frozen and hashed (freeze_v3.json) before any v3 model is evaluated on a test split. Before the freeze and without new models, two
analyses ran on existing outputs and are labelled exploratory/post hoc: (1) classification uncertainty from the saved v2 predictions
(results/v3/uq_cls.json); (2) bone-interval recalibration on the 24 SZCH development pairs, tested on the 51 held-out pairs that v2 already
used (results/v3/uq_bone.json; SAP_v3 H9 is therefore post hoc by construction). No v3 GPU job has run; the author approves the plan first.

## 2026-11-03 - v3 plan revised with the author's choices
Primary hypothesis H6 = hybrid vs original (was hybrid vs early fusion); hybrid vs the X x3 control and vs early fusion form the secondary
H7 family. Encoder screen: DenseNet-121, ConvNeXt-V2-T, RAD-DINO, Swin-V2-B (BiomedCLIP dropped); NIH, orig only, seed 0, 10 epochs.
Main runs: 20 epochs; seeds: the author answered "5", read as 5 seeds (0-4) per configuration. The NIH prep (orig + soft) is copied to ac_7.
SAP_v3 is frozen before phase 1 (default; not asked). Correction to the first DESIGN_v3.md: v2 predictions are not reused in v3 contrasts
(different head, 2-channel early fusion); all five configurations are trained with s4/cls_hybrid.py. Budget re-estimated from the v2 logs
(7.6 min per DenseNet epoch on NIH; two jobs per Kaggle session): about 85 GPU-h after phase 1 if DenseNet wins the screen, about 200-315
if a heavier encoder wins; phase 1 about 10 GPU-h.

## 2026-11-04 - v3 screen-a result: ConvNeXt-V2-T did not train at the frozen recipe
Screen-a (ac_6, NIH orig, seed 0, 10 epochs, validation only): DenseNet-121 best validation macro-AUROC 0.8404 (epoch 7; v2 reached 0.835).
ConvNeXt-V2-T stayed at chance in every epoch (0.487-0.511; loss at the prior-only level), i.e. it collapsed early under AdamW 3e-4 on all
parameters, the same instability as in v2 (ConvNeXt-V2-T diverged at the full rate there and was rerun with the encoder lr x0.25). A
diagnostic kernel (s4-v3-diag, ac_6) times one full DenseNet hybrid epoch and runs ConvNeXt-V2-T for 2 epochs with --enc_lr 0.25; it is
not a screen result. Whether the screen is rerun with an encoder lr rule (a deviation from SAP_v3, validation data only) is the
author's decision once screen-b (RAD-DINO, Swin-V2-B) has finished. Full-epoch timings (concurrent lanes): DenseNet one stream 0.25 s
per step, ConvNeXt-V2-T 0.44 s; the first epoch without the warm-up read lost about 1.2 h.
Diagnostic s4-v3-diag (ac_6, 4 Nov 02:39-03:20, not a screen result): ConvNeXt-V2-T single stream with --enc_lr 0.25 reached validation
macro-AUROC 0.818 after epoch 1 and 0.841 after epoch 2 (2-epoch cosine schedule), so the screen-a collapse is a learning-rate failure of
the frozen recipe, not the encoder. One full DenseNet-121 hybrid epoch took 1,482 s with the R4 loader (0.59 s per step); screen-a's slower
single-stream steps (0.247 s against 0.21 s) came from the pre-R4 CPU loader. Warm-up read: 60 s.

## 2026-11-04 - v3 screen-b result (frozen rate; reported, not ranked)
Screen-b (ac_7, NIH orig, seed 0, 10 epochs, encoder lr 1.0; fetched 09:10 after the PC had been off; the kernel itself had completed on
Kaggle): RAD-DINO learned slowly (validation macro-AUROC 0.553 -> 0.677); Swin-V2-B (--accum 4) reached 0.828 (epoch 9). Under the
deviation of 03:30 these runs do not enter the ranking; the x0.25 reruns (screen-c, still running) do. Timings: RAD-DINO one stream
0.44 s per step, Swin-V2-B one stream 1.26 s per step (3,180 s per epoch), so screen-c's Swin lane ends around 12:35.

## 2026-11-05 - v3 encoder screen: result and selection (validation only)
Screen-c (ac_7, encoder lr x0.25, NIH orig, seed 0, 10 epochs; kernel complete 4 Nov 19:24, fetched 19:26), best validation macro-AUROC:
ConvNeXt-V2-T 0.8430 (epoch 3), Swin-V2-B 0.8423 (epoch 4), RAD-DINO 0.8384 (epoch 2); DenseNet-121 (screen-a, frozen rate) 0.8404 (epoch 6).
Rule (SAP_v3.md, freeze_v3.json, DEVIATIONS.md 2026-11-04): the best goes forward with DenseNet-121, and within 0.005 of the best the cheaper
encoder (GPU time per epoch) wins. All four lie within 0.005 of the best (0.8430 - 0.8384 = 0.0046), and DenseNet-121 is the cheapest
(0.22 s per step against 0.42, 0.44 and 1.26 s), so the rule selects DenseNet-121, which goes forward alone. Phase 2 is the DenseNet-121
matrix (175 runs, about 109 GPU-h). Adding a second encoder (e.g. ConvNeXt-V2-T, the best by 0.0026) would be a deviation for the author.
All four runs overfit after epochs 2-6 of a 10-epoch cosine schedule; the main runs keep the best validation epoch of 20.

## 2026-11-08 - publication plan after v3 phase 2: two papers (author's choice)
Paper 1 (lesion audit; paper/main.tex, aimed at Medical Image Analysis) keeps its pre-specified claims. It gains the post-hoc H9 bone-interval
recalibration (SAP_v3.md, exploratory: pooled coverage 0.886 global / 0.895 Mondrian on the 51 held-out SZCH pairs at nominal 0.90, against
0.139 before) and one pointer to paper 2, without paper 2's numbers. Paper 2 (paper/main_cls.tex, an evaluation-study journal) reports the v3
classifier matrix as a null primary result: H6 null on NIH, VinBigData and BDCXR-3257; H7 null (no gain over the X x3 control or channel
stacking). The nodule/mass gain is significant on NIH only (soft - orig +0.0089, Holm p 0.002) with the same direction and size on VinBigData
(+0.0083, Holm p 0.79); the diffuse loss is NIH-only. The v2 per-class result on the same NIH test split (paper 1, results) had shown this
pattern, so NIH is a re-test with new models and VinBigData is the independent test; paper 2 says so. Paper 2 does not attribute the focal
gain to the twin regulariser (v3 has no no-twin suppressor arm). Both papers go into the thesis in full. New text says "pre-specified,
hash-frozen before testing": the plans were frozen by hashes in local commits, not in a public registry.

## 2026-11-08 - wording: "pre-specified" instead of "pre-registered" (author's choice)
The analysis plans were hash-frozen in local commits, not deposited in a public registry. The author chose to replace "pre-registered" with
"pre-specified" in paper 1 and the thesis (paper 2 never used it). Paper 1 (reproducibility, limitations, CLAIM item 34) and the thesis
(background, threats to validity) now also say that the plans were not deposited in a public registry. Project documents (SAP*.md,
RUNPLAN*.md, DECISIONS.md, DEVIATIONS.md) keep their historical wording; README.md line 3 was changed too.

## 2026-11-08 - one combined manuscript instead of two papers (author's choice, supersedes the two-paper plan above)
The author asked for both papers to be combined into one final manuscript for submission (Medical Image Analysis style, one column,
elsarticle), with the relevant equations. It lives in submission/ (main.tex + sections/*.tex, self-contained for Overleaf) and is packaged by
tools/pack_overleaf.py as overleaf_submission.zip (tables and figures are copied from the generated outputs, not versioned twice). The
classifier study becomes Section "Classifier study" plus a results subsection; equations added for the detector model, sample composition,
suppression loss, Qure additivity, ensemble and conformal intervals, the post hoc Mondrian recalibration (H9), calibrated PSNR, rib
benchmark, CNR, dose-response slope, Holm, non-inferiority, the information argument, the multi-stream network, the classifier loss and
ensembles, group contrasts, the cluster-bootstrap p-value, the pointing-game interaction, conformal risk control, class-conditional sets
and AURC, each checked against the code. paper/main.tex, paper/main_cls.tex and the thesis stay as they are.

## 2026-11-08 - v4 cross-modality study: scope and design (author's choices)
The author asked for the second aim of the thesis, an explainable and hallucination-aware Swin framework for cross-modality chest CT and
X-ray disease classification, to be implemented in full, and chose: a Swin CT branch on a CT-RATE subset with CT-to-X-ray distillation;
classification with heat maps, no new box detector; its own study and thesis chapter. Design: DESIGN_v4.md; plan: SAP_v4.md (draft, H11-H15),
frozen with the code and s4/data/ct_subset.json before the first training run. Code: s4/xm.py (2.5D SwinV2-T CT teacher; X-ray on the Qure.ai
projections with binary distillation and decoder alignment; zero-shot transfer), s4/ct_prep.py (2,157 CT-RATE volumes reduced in a CPU Kaggle
kernel), s4/xm_labels.py, s4/xm_xai.py, s4/xm_halluc.py, tools/xm_stats.py, job kinds xm, xmx and xmh in kernels/down/run_down.py, and NIH
fine-tuning from a projection model through s4/cls_hybrid.py --init (folder suffix +xray or +xraykd). Tested on CPU only; no Kaggle job has
run. Before any run the author accepts the CT-RATE terms, attaches a Hugging Face read token as the Kaggle secret HF_TOKEN of s4-ctprep, and
approves SAP_v4.md and about 52 GPU-h.

## 2026-11-08 - v4: CT-RATE checks, skip list and account plan; SAP_v4 frozen (author's go-ahead)
The author answered the approval question with "use multiple account to complete run fast": taken as approval of SAP_v4.md and the
compute plan, with the runs spread over accounts. Checks before the freeze (classic read token of HF_Token.txt, no token printed): CT-RATE
access granted; file layout as s4/ct_prep.py hf_path assumes. The metadata VolumeName ends in .nii.gz, so rescale() had missed every volume
(all would have failed the HU check): fixed. Intercepts are -1024 or -8192 per volume. The NIfTI files carry no scaling and an identity
affine; on train_1_a_1 the array order is DICOM (axis 0 toward the patient's left, 1 posterior, 2 head), so view() showed this data upside
down and mirrored: now a plain transpose, checked on the slice (heart at the top, descending aorta on the image right) and against the Qure
projection (frame correlation 0.62 as is, 0.56 mirrored). CT-RATE lists 789 volumes without chest; 37 of the 2,157 selected volumes were
among them (8 test). s4/data/ct_skip.txt (those plus 61 volumes not scanned head-first supine) is never selected and its projections are
left out of P0/P1 training and evaluation; the subset is now 2,156 volumes. Compute re-estimated from the v3 screen rates: about 81 GPU-h,
not 52. Account plan: DESIGN_v4.md section 6 (CT data only on ac_1, the author's account; projection and NIH runs on ac_2, ac_3, ac_4, ac_5,
ac_7). Hallucination tests run per seed group and tools/xm_stats.py combines them.

## 2026-11-14 - public repository: MIT licence, internal notes left out, account names hidden (author's choices)
For the journal submission the author chose the MIT licence for the code (LICENSE); data, weights, the manuscript and the thesis keep
their own terms (README.md, licence section). The public snapshot (tools/build_release.py) leaves out the internal notes (PLAN.md,
NEW_AGENT_BRIEF.md, notes of the discarded first run), superseded drafts and scratch runs, and shows the Kaggle accounts as ac_1 ... ac_9.
The plans, freezes and these logs moved unchanged to protocol/; tools/verify_freeze.py re-hashes every frozen file. Two flaws of the earlier
snapshots were fixed: the data filter also dropped code, SAP_bdcxr.md and three manuscript tables that only name a restricted data set,
and the folder prune dropped s4/data.
