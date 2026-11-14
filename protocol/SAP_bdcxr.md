# SAP addendum: external validation on BDCXR-3257 (frozen 2026-11-03)

Status: **frozen 2026-11-03** (approved by the author; committed before any model of this project saw a BDCXR image).
Nothing below may change after the freeze except through DEVIATIONS.md. All models are the frozen ones of freeze_v2.json; nothing is
trained or tuned on BDCXR except the downstream classifiers, with the fixed settings of s4/cls.py.

## Data
- BDCXR-3257 (Durjoy et al., figshare, doi 10.6084/m9.figshare.32086362.v1, CC BY 4.0 on figshare): 3,257 paediatric chest radiographs (JPEG),
  Dr. M R Khan Shishu Hospital & Institute of Child Health, Dhaka; IRB REC-FSIT-2025/No: 11928. Classes: Normal 880, Bacterial pneumonia 1,477,
  Viral pneumonia 900. Age mostly < 5 years (1,132 under one year; 3 images of adults); sex M 2,006 / F 1,251. Image-level labels only: no patient
  ID, no lesion boxes, no dual-energy pairs, no rib masks.
- Handling: the lab's data use agreement forbids redistribution, so no copy is uploaded anywhere. Kaggle kernels download the public figshare file
  themselves (as SZCH from Hugging Face), or the work runs locally. Raw images never appear in a figure unless the data provider agrees in writing.
  Every paper/thesis text acknowledges the hospital and the IRB number.
- Duplicates: with no patient ID, several images of one child may exist. Before any split, near-duplicates are grouped (perceptual hash of the
  256 px image, Hamming distance <= 6); a group never spans two folds. The number of groups and their sizes are reported.

## Questions
- **E1 (primary): does bone suppression change pneumonia classification on an unseen paediatric population?**
  Task: 3 classes, one-vs-rest. Inputs (as H4'b): orig, soft (ours), soft_qure, CLAHE, orig+soft. DenseNet121, s4/cls.py settings unchanged.
  Stratified 5-fold cross-validation over duplicate groups (fold seed 0); inside each training part 10% (stratified, grouped) is the validation
  split for epoch selection; 3 seeds per fold and input (75 runs). Out-of-fold probabilities, averaged over seeds, give one prediction per image.
  Primary metric: macro-AUROC of the pooled out-of-fold predictions. Primary contrast: soft (ours) vs orig, paired DeLong per class with Holm over
  the 3 classes, and the macro-AUROC difference with an image-group bootstrap 95% CI. Non-inferiority margin 0.01 macro-AUROC
  (pass: lower CI bound > -0.01). Superiority is claimed only if the lower bound is > 0. Every other input vs orig: same statistics, descriptive.
  Secondary: pneumonia vs normal AUROC, bacterial vs viral AUROC, balanced accuracy at the argmax, per-fold mean +- SD, age < 1 vs >= 1 year.
- **E2 (secondary): lesion absorption on paediatric images.** s4/audit_real.py as for JSRT: a fixed subset of 300 images (stratified by class,
  seed 7), 6 sites per image, LIDC test-CT lesions, bone overlap from the Qure bone map. Methods: ours, no-twin, Qure, identity. Contrasts as
  H3' (ours - Qure, ours - no-twin; image-cluster bootstrap; Holm). Images whose PSPNet lung mask is empty or under 2% of the image are skipped
  and counted (the segmenter was trained on adults). Inserted lesions are adult nodules, so this is a stress test, not a clinical claim.
- **E3 (descriptive): domain-shift signal.** Per-image ensemble spread (mean over the lung field) on BDCXR vs SZCH held-out; Spearman of the
  spread with age. No error can be computed (no bone truth); this reports only whether the uncertainty rises out of the training domain.

## Not done (and why)
- No bone-suppression fidelity metric: there is no dual-energy truth. No rib benchmark: no rib masks. No fine-tuning of the suppressor.
- No patient-level split is possible; the duplicate grouping above is the mitigation, stated as a limitation.

## Compute (estimate)
Prep (orig, soft, soft_qure, bone at 512 px; 3,257 images): about 0.3 GPU-h. E1: 75 classifier runs, about 4-6 GPU-h. E2: about 0.5 GPU-h.
E3 comes out of the prep pass. Code to add: a `bdcxr` loader in s4/prep.py (figshare download, labels, duplicate groups, fold files), a
`--split` option in s4/cls.py, JPEG support in s4/audit_real.py, and the statistics in tools/bdcxr_stats.py.
