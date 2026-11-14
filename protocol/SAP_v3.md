# SAP_v3 — multi-stream classifier, shift-aware uncertainty, XAI (DRAFT, not frozen)

Freeze rule: this file, the v3 code commit and the split files are hashed into freeze_v3.json **before phase 1** (the encoder screen),
so the screening rule is fixed too, and in any case before any v3 model is evaluated on a test split (NIH official test list, VinBigData
test split, BDCXR-3257 out-of-fold predictions, SZCH held-out 51). Screening and epoch selection use validation splits only. Every later
change goes to DEVIATIONS.md with its date. Design and reasons: DESIGN_v3.md (author's choices of 2026-11-03 in §8 there).

Already-seen data: the NIH test split, VinBigData test split, BDCXR-3257 folds and the 51 SZCH held-out pairs were used in v2. v3
hypotheses concern models that do not exist yet, so the test sets stay valid for them. The SZCH recalibration (H9) is post hoc on
already-seen pairs and was run before this plan was frozen (DECISIONS.md, 2026-11-03); it is exploratory.

## Configurations and training
orig | soft | early (RGB <- [X, S, B], one stream) | hybrid (streams X, S, B; shared encoder, class-query decoder) | ctrl (hybrid network,
X in all three streams). All five are trained with s4/cls_hybrid.py and identical settings per encoder; v2 predictions (s4/cls.py,
another head, 2-channel early fusion) are not used in any v3 contrast.
- 20 epochs, cosine schedule with 200 warm-up steps, AdamW (lr 3e-4, weight decay 1e-4), batch 32, 256 px; the epoch with the best
  validation macro-AUROC is kept.
- 5 seeds (0-4) per configuration on NIH and VinBigData; BDCXR-3257: 5 folds x 5 seeds. The prediction of a configuration is the mean
  over its seeds.
- A run longer than one 12-hour Kaggle session continues from its end-of-epoch checkpoint (model, optimiser, scaler, step, best-so-far,
  random states) in a follow-up session.

## Hypotheses
- **H6 (primary).** hybrid > orig in macro-AUROC. One test per data set (NIH, VinBigData, BDCXR-3257): two-sided 95 % interval of the
  difference by paired cluster bootstrap (10,000 resamples; clusters: patients for NIH, images for VinBigData, near-duplicate groups for
  BDCXR-3257) and its percentile p-value; Holm over the three data sets.
- **H7 (secondary: why).** hybrid vs ctrl (is the gain due to the bone decomposition or to the extra network passes?) and hybrid vs early
  (does the multi-stream design beat naive channel stacking?). Same test; one Holm family over the six tests (2 contrasts x 3 data sets).
  Reading: H6 passing with hybrid > ctrl attributes the gain to the decomposition; H6 passing with hybrid = ctrl attributes it to capacity.
- **H8 (secondary).** Per taxonomy group (focal, diffuse, boundary; DESIGN_v3.md §3): hybrid - orig and soft - orig in mean per-class
  AUROC. Bootstrap intervals; Holm within data set over groups x contrasts. Per-class AUROC contrasts (hybrid - orig, hybrid - early):
  paired DeLong, Holm within data set. AUPRC contrasts: paired bootstrap only (there is no DeLong test for AUPRC).
- **H9 (exploratory, post hoc).** Bone-image intervals recalibrated on the 24 SZCH development pairs reach pooled coverage in
  [0.85, 0.95] on the 51 held-out pairs at nominal 0.90. Reported with the per-image coverage distribution and interval width. Mondrian
  strata: predicted bone overlap x spread tercile.
- **H10 (secondary).** Pointing game on the NIH expert boxes, hybrid model: the hit-rate difference (S-stream map - X-stream map) is larger
  for focal than for diffuse/boundary findings (interaction; image-cluster bootstrap). Map: Grad-CAM++ per stream; tolerance 15 px at 256 px.

## Encoder screen (phase 1)
NIH, configuration orig, seed 0, 10 epochs (otherwise as above), validation macro-AUROC: DenseNet-121 (densenet121.ra_in1k),
ConvNeXt-V2-T (convnextv2_tiny.fcmae_ft_in22k_in1k), RAD-DINO (microsoft/rad-dino, 224 px), Swin-V2-B (swinv2_base_window16_256).
The best goes forward together with DenseNet-121 (if DenseNet-121 is best, it goes alone); within 0.005 of the best, the cheaper one
(GPU time per epoch, measured in the smoke runs) wins.

## Classification uncertainty (descriptive)
BDCXR-3257: Mondrian conformal sets (calibrated on the other folds' out-of-fold probabilities), coverage per true class at 0.90.
NIH: conformal risk control of the per-image false-negative rate at 0.10, calibrated on the validation predictions, tested on the test
split. Selective classification: risk-coverage and AURC with 1 - entropy of the seed ensemble, and with entropy + lung-field suppressor
spread.

## Reporting
All primary and secondary results are reported whatever their direction. Figures: tools/generate_all_thesis_figures.py.
Statistics: results/final_hybrid_stats.json (tools/hybrid_stats.py), deterministic seeds.
