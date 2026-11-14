# Statistical analysis plan, version 2 (v2 recipe): addendum to SAP.md

Written 2026-10-29, before any held-out or locked-set evaluation of a v2 model. Frozen with `tools/freeze_v2.py` (hashes of this file, the
SZCH dev/held-out lists, the weights and the code commit go to `freeze_v2.json`). SAP.md (v1) stays in force for the v1 family (`clia`, selected
on synthetic validation CTs); its results are reported as planned. Anything not listed here is exploratory and labelled so.

## Why a second plan
After the v1 ablation the real SZCH pairs were looked at (DEVIATIONS.md, 2026-10-27). SZCH was split into a development part (24 images,
`tools/szch_split.py`) and a held-out part (51 images). The v2 recipe (Qure ct2xr-projections as training source, twin lesion-invariance
regulariser) was developed on the dev part of the paired metrics, the synthetic validation data, and the real-image audit. **The audit used
during development ran on all 75 SZCH images (450 sites, seed 7)**, so SZCH held-out images are NOT clean for audit claims: SZCH audit numbers
are reported as development evidence only. The confirmatory audit runs on VinDr-RibCXR (245 images, never opened by any model run so far; bone
overlap from its radiologist rib masks) and JSRT (247 images). Paired-fidelity claims use the SZCH held-out 51, whose paired metrics were never
used for a v2 decision (round-1 screening printed held-out PSNR/SSIM of v1-family and 100k runs as description; DEVIATIONS.md).

## Definitions
- **Ours (v2)** = deep ensemble (mean bone image) of K = 3 models: SwinV2-T, 512 px, UNet++ + CBAM, 300k samples, trained on the Qure
  ct2xr-projections train split (21,051 CT-RATE volumes), twin weight 16 (arm `qp_twin16`), seeds 0, 1, 2. Seed 3 is a reserve: single-model
  results of seeds 0-3 are reported as secondary (seed variability). If the replacement rule in RUNPLAN.md (section 4) selects another
  configuration before the freeze, that configuration replaces this one with the same K = 3 / seeds rule, and DECISIONS.md records why.
- **Comparators:** Qure released weights (SwinV2-B, 1024 px); **no-twin** control (arm `qp`, same data and recipe, 300k, seeds 0-2, K = 3);
  v1 `clia` (LIDC-only, 300k, seeds 0-2); identity, CLAHE, homomorphic filtering.
- **Sets used for claims:** SZCH held-out 51 pairs; synthetic: Qure-projection test split (430 CTs, exact additive bone) and LIDC test CTs (158);
  real lesions: JSRT (154 nodules), Node21 (1,476 nodule boxes), TBX11K, VinBigData boxes; NIH ChestX-ray14 official test list; VinDr-RibCXR (245).

## Hypotheses (confirmatory)
- **H1' (dose-response, real audit).** Lesion absorption A on VinDr-RibCXR (245 images x 6 sites, lesions from LIDC test CTs, alpha = 1,
  sites drawn once with a fixed seed and shared by every model), per-site mean over seeds, falls with twin weight w in {0, 4, 8, 16, 32, 64}:
  slope of A on log2(1 + w), image-cluster bootstrap 95% CI; passes if the upper bound is < 0. Same analysis on the Qure-projection test split
  (exact truth, lesions inserted into the soft component) as secondary.
- **H2' (fidelity, SZCH held-out 51).** Bone-region PSNR_cal of ours vs Qure: non-inferiority, margin 0.5 dB (one-sided 95% lower bound of the
  paired mean difference > -0.5 dB, image-level bootstrap). SSIM_cal: margin 0.01 (kept from SAP.md; dev data show a gap of about 0.03, so this
  part is expected to fail and is reported either way). Ours vs no-twin on bone PSNR_cal: non-inferiority margin 0.25 dB (cost of the regulariser).
  Secondary: whole-image PSNR_cal, band-limited SSIM (sigma 2, 4), MS-SSIM, LPIPS, raw (uncalibrated) metrics, high-frequency energy ratio.
- **H3' (lesion absorption, confirmatory audit).** On VinDr-RibCXR sites: A(ours) < A(Qure) and A(ours) < A(no-twin): paired site-level
  differences, image-cluster bootstrap 95% CI excluding 0; Holm over the two contrasts. Interaction: (ours - Qure) in sites overlapping an
  annotated rib (>= 15% of the lesion support) vs sites that do not. JSRT (overlap from the consensus of the ours and Qure bone maps) and the
  SZCH audit are secondary.
- **H4' (evidence that does not use the insertion mechanism).**
  (a) Real lesions: box CNR(soft) - CNR(orig) on JSRT and Node21 nodules, ours vs Qure and ours vs no-twin, all lesions and lesions with
      >= 15% bone overlap (consensus bone map); paired, image-cluster bootstrap; Holm within dataset.
  (b) Downstream, as SAP.md H4: NIH macro-AUROC per input {orig, ours, Qure, CLAHE, orig+ours}, 3 seeds; TBX11K / Node21 / VinBigData
      detection FROC CPM and mAP@50; recall at 2 FP/image for lesions with >= 15% vs < 15% bone overlap, difference-in-differences of
      (ours - orig) with bootstrap CI.
- **H5' (reliability).** Spearman between ensemble spread and absolute bone error on SZCH held-out (pseudo-truth bone = x - DES soft after least
  squares); split-conformal intervals calibrated on Qure-projection validation CTs, coverage on the Qure-projection test split (nominal 90%);
  coverage on SZCH held-out reported as measured under shift.

## Rules
- No tuning, threshold choice or model selection on any set listed above. The ensemble, the seeds and every hyper-parameter are fixed by this
  file and freeze_v2.json; downstream settings are the ones in s4/cls.py and s4/det.py.
- Multiplicity: Holm within each hypothesis family. Effect sizes with 95% CIs are always reported, p-values never alone.
- A hypothesis that fails is reported as failed in the paper and thesis, with the numbers.
- All tables and figures are produced by scripts from saved result files (`tools/final_stats.py`, `tools/fig_*.py`).
