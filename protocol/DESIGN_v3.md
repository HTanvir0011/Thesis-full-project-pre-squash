# v3 extension: multi-stream classifier, shift-aware uncertainty, multi-stream XAI, figure suite

Status: **proposal, 2026-11-03, revised the same day with the author's choices (§8).** No GPU job for v3 has run. SAP_v3.md (draft) is
frozen and hashed before phase 1. This file answers step 1 of the research directive (critical review and proposal before code).

## 1. Corrections to the directive (checked against the code, the data and our results)

| Directive says | What holds | Consequence |
|---|---|---|
| Three streams X, S, B̂ give the classifier more to work with | S = X − B̂ and B̂ = f(X) for a fixed suppressor f. The streams are a deterministic function of X: no new information enters. | Any gain must come from the CT-supervised prior in f or from the per-stream inductive bias, not from extra signal. A control with the same network fed X in all three streams ("X×3") is mandatory. Without it a gain could just be extra capacity or compute. |
| orig + soft gives +0.006 AUROC on NIH | Correct: +0.006 [0.003, 0.008]. Soft alone is lower for effusion, infiltration and pleural thickening (−0.004, −0.007, −0.013) and higher for nodule (+0.011), Holm-adjusted DeLong p < 0.05. | The ceiling for any fusion gain on NIH is small; the matrix must be powered by the per-class analysis, not by macro-AUROC alone. |
| Conformal intervals 90 % → 14 % | Correct: 0.139 [0.134, 0.145] at nominal 0.90 on the 51 held-out SZCH pairs. | Target of Module C. |
| Calibrate on real pairs from "SZCH / VinDr-RibCXR" | VinDr-RibCXR has rib masks, no bone image, so no bone error can be computed there. Real bone pairs exist only in SZCH (75 images). The 51 held-out pairs were already used for the confirmatory fidelity claim. | Calibrate on the 24 SZCH development images, test on the 51 held-out. This is post hoc and is reported as such. |
| Taxonomy: Pneumothorax focal; Rib fractures structural | Pneumothorax is a pleural boundary finding, not focal. Neither NIH ChestX-ray14 nor VinBigData has a rib-fracture label. The full VinDr-CXR (PhysioNet, credentialed) is not in our data use. | Taxonomy fixed below (§3). Rib fracture is dropped and the reason is reported. |
| Paired DeLong for AUROC **and AUPRC** | DeLong's variance is specific to the AUROC (a U-statistic); there is no DeLong test for AUPRC. | AUROC uses DeLong; AUPRC uses the paired patient-cluster bootstrap (10,000 resamples). |
| VinDr-CXR, Node21 boxes in XAI figures | Node21 is CC BY-NC-ND 4.0, and a heatmap overlay is adapted material. VinBigData, VinDr-RibCXR and BDCXR-3257 images never appear in our figures (licence or data use agreement). | Figure panels use NIH ChestX-ray14 images with the NIH expert boxes (BBox_List_2017.csv, about 1,000 boxes in 8 classes, count to be confirmed on download), plus LIDC-IDRI (CC BY 3.0) and SZCH (CC BY-NC 4.0). Pointing-game numbers may also use Node21 and VinBigData boxes, as numbers only. |
| Run the four input configurations | v2 DenseNet-121 predictions exist for orig, soft and orig+soft, but from s4/cls.py: another head (global pooling) and a 2-channel early fusion. | All five v3 configurations are trained with s4/cls_hybrid.py for every encoder, so a contrast changes only the input streams. (The first version of this file said DenseNet could reuse v2 runs; that was wrong and is corrected here.) |
| FROC for the hybrid | The detectors are single-stream Faster R-CNN. A multi-branch detector is a separate project. | FROC curves are drawn for the existing detection inputs only (not for the hybrid). |

## 2. Architecture: shared-encoder streams + class-query cross-attention

**Choice.** One encoder with shared weights processes each stream separately. Each stream has its own learned input normalisation and a
stream embedding. Fusion is a class-query cross-attention decoder (Query2Label / ML-Decoder style) over the union of the spatial tokens of
the three streams.

**Why this and not the alternatives.**
- *Separate encoders per stream* triple the parameters, so a gain over early fusion could be capacity. They also do not fit three ViT-B on
  a T4 at a useful batch size. A shared encoder keeps the parameter count of the single-stream model (+ a 2-layer decoder), at 3× the compute.
- *Pooled gated fusion* (gate × pooled feature per stream) throws away where the evidence is. Nodules are a few tokens. The decoder keeps
  spatial tokens.
- *Cross-covariance (XCiT-style) gating* attends over channels, not positions. It mixes the streams globally and gives no per-class,
  per-position readout. *Capsule-style dynamic routing* has no track record on chest radiographs and is unstable to train. Neither is
  better on the two things we need: locality and a per-class routing readout.
- The decoder gives the routing for free. The attention mass of class query c on the tokens of stream s,
  ρ_{c,s} = Σ_n a_{c,(s,n)}, is the per-class stream routing that Module D asks about. It is measured, not assumed.

**Per-stream normalisation.** The bone image spans a small intensity range; X and S span the full range. Each stream gets a learned affine
(γ_s, β_s) after a per-image standardisation, then is replicated to 3 channels for the pretrained stem. This is the fix for "conflicting
intensity statistics" in early fusion.

**Tensor shapes (DenseNet-121, 256 px, batch B, C classes, D = 256):**

```
X, S, B̂            (B,1,256,256) each        B̂ = clip(X − S) from the stored uint8 arrays (no new prep)
per-stream norm    (B,1,256,256) → (B,3,256,256)
shared encoder     (3B,3,256,256) → (3B,1024,8,8)
1×1 projection     (3B,D,8,8) → tokens (B,3·64,D) + stream embedding e_s + 2-D sine position
decoder (2 layers, 8 heads): queries (C,D) → cross-attn over (B,192,D) → FFN → (B,C,D)
head               group linear (C,D) → logits (B,C)
attention          (B,8,C,192) → mean over heads, last layer → (B,C,3,8,8): ρ (B,C,3) and per-stream maps
```
ConvNeXt-V2-T: encoder output (3B,768,8,8). RAD-DINO (ViT-B/14) at 224 px: (3B,256,768) → 16×16 tokens per stream.

**Configurations (all with identical training settings):** orig | soft | early-fused (RGB ← [X, S, B̂], one stream: the naive stacking
the directive criticises) | hybrid (X, S, B̂) | control X×3 (hybrid network, X in all three streams).

## 3. Disease taxonomy (fixed before any v3 run)

| Group | NIH ChestX-ray14 | VinBigData |
|---|---|---|
| Focal / nodular | Nodule, Mass | Nodule/Mass |
| Diffuse / fluid | Effusion, Infiltration, Consolidation, Edema, Pneumonia | Pleural effusion, Infiltration, Consolidation, Lung Opacity, ILD |
| Boundary / structural | Cardiomegaly, Atelectasis, Pneumothorax, Pleural thickening, Emphysema, Fibrosis, Hernia | Cardiomegaly, Atelectasis, Pneumothorax, Pleural thickening, Aortic enlargement, Pulmonary fibrosis, Calcification |

"Other lesion" (VinBigData) is reported but in no group. BDCXR-3257 is three-class pneumonia (normal, bacterial, viral) and is analysed
separately.

## 4. Uncertainty (Module C)

- **Bone-image intervals under shift.** Per-pixel score r = |b − b̂| / σ̂ (ensemble spread). (a) The current split-conformal on synthetic
  data, as the reference. (b) Real-data recalibration: calibrate on the 24 SZCH development images and test on the 51 held-out ones.
  (c) Mondrian by stratum: bone-overlap vs. other lung pixels × spread tercile. Images, not pixels, are the exchangeable units, so the
  quantile is taken over per-image quantiles. The guarantee is then per image, which matches how a reader sees one radiograph. Coverage
  is reported per image (median, 10th percentile) and pooled. With n = 24 calibration images the attainable levels are k/25.
- **Classification.** BDCXR-3257: Mondrian (class-conditional) conformal prediction sets from the out-of-fold probabilities (calibrate on
  the other folds), for coverage per true class. NIH: conformal risk control of the false-negative rate per image (Angelopoulos et al.
  2024), calibrated on the validation split and tested on the test split.
- **Selective classification.** Risk–coverage curves and the area under them (AURC), with the confidence score = 1 − entropy of the
  5-seed ensemble. A second score adds the suppressor spread inside the lung field (high where bone removal is unsure). Reported for NIH and
  BDCXR-3257.

## 5. XAI (Module D)

Grad-CAM++ per stream on the last encoder block (the shared encoder runs per stream, so each stream has its own map). Integrated Gradients
on the three inputs jointly (blurred-image baseline, sigma 8 px, 32 steps: after the per-image standardisation a black baseline has no contrast); the per-stream share of the total attribution is reported. The decoder routing
ρ_{c,s} is reported next to both. Pointing game: hit if the arg-max of the map lies in a ground-truth box of that class (± 15 px
tolerance, stated in advance). Data: NIH boxes (primary), Node21 and VinBigData boxes (numbers only). Contrast tested: the hit rate of
the S stream minus the X stream, focal vs. diffuse/boundary groups.

## 6. Compute and schedule

**Basis (v2 logs).** DenseNet-121, one stream, NIH: 7.6 min per epoch on a T4 (447–468 s). The first job of a session also waits about
45 min while the arrays are read. A Kaggle session has 2×T4 and runs two jobs at once, and quota counts session hours, so one run costs
about half its wall time in quota. Three streams ≈ 3× the encoder work. The factors for the other encoders are estimates; the phase-1 smoke
runs measure them before phase 2 is budgeted.

**Where the data are.** The full NIH arrays exist only on ac_6 (ac_2/ac_5 hold 48-image test preps). VinBigData: ac_1, ac_9. BDCXR-3257:
ac_8 only (data use agreement). A run must use the account that owns its prep output. Quota now: ac_6 8.6 h, ac_7 29.2 h, ac_8 10.8 h,
≈ 60 h in total until the refill on 2026-11-07, then 30 h per account per week.

| Phase | Work | GPU-h (estimate) |
|---|---|---|
| now (no GPU) | design, SAP_v3 draft, code + CPU tests, figures from existing outputs | 0 |
| freeze | SAP_v3 hashed into freeze_v3.json | 0 |
| P1 (needs approval) | NIH prep copy (orig + soft) on ac_7; smoke runs of the 4 encoders × {orig, hybrid} (step time, finite loss); encoder screen: NIH orig, seed 0, 10 epochs, 4 encoders | ≈ 3 + 1 + 6 |
| P2 (after refill; budget re-measured from P1) | NIH, 5 configurations × 5 seeds × 20 epochs: DenseNet ≈ 60; the screen winner, if not DenseNet, 1.5–3× that | ≈ 60 (+ 90–180) |
| P2 | VinBigData classification, same matrix (training set 13 % of NIH's) | ≈ 8 (+ 12–24) |
| P2 (ac_8 only) | BDCXR-3257: 5 folds × 5 seeds × 5 configurations | ≈ 10 per encoder (+ 15–30) |
| P3 | XAI maps + pointing game, conformal, selective classification, figures | ≈ 5 |

Total P2–P3: ≈ 85 GPU-h if DenseNet wins the screen, ≈ 200–315 if a heavier encoder wins. NIH is limited to ac_6 + ac_7 (60 h/week), so
NIH alone takes about 1 week (DenseNet) or 3–4 weeks (heavier winner); a third NIH prep copy (≈ 3 h) would cut that. BDCXR-3257 is limited
to ac_8 (30 h/week). Three-stream runs of ConvNeXt-V2-T, RAD-DINO or Swin-V2-B at 20 epochs likely exceed the 12-hour Kaggle session and
resume from their end-of-epoch checkpoint (to be added to s4/cls_hybrid.py and kernels/down/run_down.py before P2, only if the P1 timings
show it is needed).

## 7. What is out of scope (named, not built)

A multi-branch detector (FROC for the hybrid); rib-fracture classification (no labels in our data); new suppressor training;
BiomedCLIP (dropped from the screen by the author; the loader stays in s4/cls_hybrid.py).

## 8. The author's choices (2026-11-03)

| Question | Choice | Effect on the plan |
|---|---|---|
| Primary hypothesis | hybrid vs original | SAP_v3 H6 = hybrid > orig; hybrid vs X×3 control and vs early fusion are the secondary H7 family. |
| Encoders screened next to DenseNet-121 | ConvNeXt-V2-T, RAD-DINO, Swin-V2-B | BiomedCLIP dropped. |
| Training | 20 epochs; seeds: "5", read as 5 seeds (0–4) | v2 NIH runs peaked at epoch 4 of 10 and then overfit, so the extra epochs cost GPU time; the best validation epoch is kept, so test results stay valid. |
| Encoder screen | 10 epochs | ≈ 6 GPU-h. |
| NIH account | copy the NIH prep to ac_7 | NIH runs on ac_6 and ac_7. |
| Freeze time | not asked; default: before phase 1 | The screening rule is fixed before any screen result is seen. |
