# SAP_v4: cross-modality Swin framework (frozen 2026-11-08, freeze_v4.json)

Freeze rule: this file, the v4 code commit and the CT subset list are hashed into freeze_v4.json before the first v4 training run. Every
later change goes to DEVIATIONS.md with its date. Design and reasons: DESIGN_v4.md. Model selection uses validation splits only.

Already-seen data: the NIH and VinBigData test splits were used in v2 and v3 with other models; the v4 hypotheses concern models that do
not exist yet. The projection test split (430 projections) and the CT subset test volumes have not been used for any classification.

CT subset: the 2,156 volumes of s4/data/ct_subset.json, hashed with this file together with s4/data/ct_skip.txt (CT-RATE's no-chest
volumes and the volumes not scanned head-first supine: never selected; their projections are left out of the projection models' training
and evaluation, 518 training, 10 validation and 15 test projections). Volumes that fail to download or fail the HU check of s4/ct_prep.py
are excluded from every analysis and counted in the report.

## Models and seeds
CT teacher (CTNet, SwinV2-T 2.5D), X-ray on projections without (P0) and with (P1) distillation, NIH fine-tuning from ImageNet (F_in),
from P0 (F_p0) and from P1 (F_p1); 5 seeds (0-4) each; a model's prediction is the mean of its seeds' probabilities. Training settings as
in DESIGN_v4.md section 3; NIH fine-tuning uses 10 epochs and keeps the best validation epoch.

## Hypotheses
- **H11 (primary).** On the NIH test split, the mean AUROC over the five exactly mapped shared findings (effusion, atelectasis, cardiomegaly,
  consolidation, lesion = mass or nodule) is higher for F_p1 than for F_in. Paired patient-cluster bootstrap (10,000 resamples), two-sided
  95 % interval and percentile p-value.
- **H12 (secondary).** The same contrast for F_p1 against F_p0 (does distillation from CT add to CT-derived labels alone?), and for F_p0
  against F_in. One Holm family with H11's companion contrasts (three tests).
- **H13 (secondary).** NIH macro-AUROC over all 14 diseases: F_in (Swin) against the v3 DenseNet-121 orig seed ensemble. Same test.
- **H14 (secondary).** Cross-modal heat-map agreement on CT subset test scans positive for a finding (s4/xm_xai.py: CT teacher seed k with
  P1 seed k; 32x32 lung-box frames; each map normalised to sum 1 and averaged over the five seed pairs): the Pearson correlation of the
  matched CT and X-ray maps exceeds the mean correlation of the CT map with the X-ray maps of the other scans positive for the same finding.
  Per (scan, finding) difference, scan-cluster bootstrap. Orientation: the X-ray maps are mirrored when the attenuation frames correlate
  better mirrored (frame_r_flip > frame_r; decided from the frames, no heat map involved). P0 is reported the same way, descriptively.
- **H15 (secondary).** Hallucination under rib emphasis (DESIGN_v4.md section 5) on NIH test images without any finding: per image, whether
  some class goes from at most to above its threshold (seed-mean probability; threshold = 90 % specificity on the validation split). F_p1
  against F_in: mean of the per-image differences, patient-cluster bootstrap. Noise and gamma rates, counterfactual-nodule sensitivity
  (images whose Mass and Nodule scores are at most their thresholds before insertion; score change by nodule-size tertile) and the
  off-target share of Grad-CAM++ mass are reported for F_in, F_p0, F_p1 and the v3 DenseNet-121 orig model. Seed groups of one model
  tested on different accounts are combined first (probabilities averaged with the number of seeds as weights, thresholds from the
  averaged validation probabilities); the off-target pairs are the positives of each group's own seed mean.

## Descriptive
CT teacher macro-AUROC on the CT subset test volumes; zero-shot transfer of P0 and P1 to NIH and VinBigData shared findings; per-class
DeLong tests; VinBigData fine-tuning if quota allows; conformal false-negative risk control at 0.10 for the Swin models; pointing game.

## Reporting
All hypotheses are reported whatever their direction. Statistics: tools/xm_stats.py (10,000 resamples; two-sided percentile p-values with
the +1 rule); figures from saved outputs only.
