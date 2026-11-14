# v4: SwinTrust-XM, an explainable and hallucination-aware Swin framework for cross-modality chest CT and X-ray classification

Status: **frozen with SAP_v4.md on 2026-11-08 (freeze_v4.json); later changes go to DEVIATIONS.md.** Scope chosen by the author on 2026-11-08: a Swin CT branch on a CT-RATE
subset with CT-to-X-ray distillation; classification with heat maps (no new box detector); its own study and thesis chapter. SAP_v4.md (draft)
is frozen and hashed before the first training run, as for v1 to v3.

## 1. What exists and what v4 adds

| Already in the project | v4 adds |
|---|---|
| 14-disease NIH ChestX-ray14 and 14-finding VinBigData classifiers (DenseNet-121, s4/cls_hybrid.py) | Swin (SwinV2-T) X-ray classifiers with the same class-query decoder |
| Grad-CAM++, Integrated Gradients, pointing game on NIH boxes (s4/xai_multistream.py) | the same for Swin, and agreement between CT and X-ray heat maps of the same scan |
| Lesion-absorption audit of the suppressor; conformal intervals and sets | counterfactual hallucination tests of the classifiers (inserted nodules, finding-free perturbations, off-target attribution) |
| CT-supervised bone suppression (CT used only to make training images) | CT disease classification (2.5D Swin) and CT-to-X-ray knowledge transfer |

## 2. Data

- **Paired CT and X-ray: Qure.ai ct2xr-projections + CT-RATE.** Each of the 21,887 projections (results/qp_meta.csv) is one CT-RATE volume
  (volume_name) and carries six findings taken from that scan: lung_opacity, pleural_effusion, atelectasis, cardiomegaly, consolidation,
  lung_lesion. Split by patient as in v2 (21,051 / 406 / 430 projections). The CT volumes come from CT-RATE (Hugging Face
  ibrahimhamamci/CT-RATE, gated "auto", CC BY-NC-SA 4.0).
- **CT subset.** One reconstruction per patient: every validation and test patient of the v2 split and 1,600 training patients drawn
  with a fixed seed (s4/ct_prep.py select), never a volume of s4/data/ct_skip.txt (CT-RATE's own no-chest lists, 789 volumes, and the 61
  volumes not scanned head-first supine): 2,156 volumes (1,600 training, 259 validation, 297 test patients; s4/data/ct_subset.json).
  Downloaded and reduced inside a CPU Kaggle kernel (kernels/xm_prep): 24 axial slices spread over the lung extent, three HU windows as
  channels (lung -1350..150, mediastinum -160..240, bone -500..1300), body-cropped to 256 px, uint8 (4.7 MB per volume, about 10 GB); plus
  the lung box and an AP attenuation projection for the cross-modal heat maps. The projections of skip-list volumes (518 training,
  10 validation, 15 test) are left out of the projection models as well.
- **CT-RATE facts checked on 2026-11-08** (metadata CSVs and volume train_1_a_1): HU = stored value x RescaleSlope + RescaleIntercept of
  the metadata, whose VolumeName ends in .nii.gz (slope 1; intercept -1024 for 33,358 and -8192 for 13,791 training volumes); the NIfTI
  header has no scaling and an identity affine, so the array is used as stored (axis 0 toward the patient's left, 1 toward posterior,
  2 toward the head; ImageOrientationPatient [1,0,0,0,1,0] for every selected volume). The reduced frontal frame of that volume correlates
  0.62 with its Qure projection as is and 0.56 mirrored. Volume files are 140 to 530 MB (gzip).
  Nothing derived from CT-RATE leaves Kaggle except metrics and the 32x32 heat-map frames of s4/xm_xai.py.
- **Real X-rays.** NIH ChestX-ray14 (14 diseases, official test list, 984 expert boxes) and VinBigData (14 findings), the v3 preps.
- **Label map (fixed now, s4/xm_labels.py).** pleural_effusion -> NIH Effusion / VinBigData Pleural effusion; atelectasis -> Atelectasis;
  cardiomegaly -> Cardiomegaly; consolidation -> Consolidation; lung_lesion -> NIH Mass or Nodule / VinBigData Nodule/Mass; lung_opacity ->
  VinBigData Lung Opacity (exact) and NIH Infiltration (approximate: reported, not tested).

## 3. Models (one Swin family for both modalities)

- **X-ray:** s4/cls_hybrid.py MultiStream, configuration orig, encoder swinv2_tiny_window8_256 (timm, ImageNet-1k), class-query decoder.
- **CT (s4/xm.py CTNet, 2.5D):** the same SwinV2-T runs on every slice (three HU windows as RGB); each slice gives 8x8 tokens with a 2-D sine
  code and a learned slice-depth embedding; the class-query decoder attends over the tokens of all 24 slices. A full 3D Swin (Swin UNETR)
  needs MONAI and does not fit a T4 at a useful batch; 2.5D keeps ImageNet weights and the X-ray architecture.
- **Cross-modal transfer (s4/xm.py stage xray):** the X-ray model learns the six findings on the projections; on rows whose CT is in the
  subset it also matches the CT teacher (binary distillation at temperature 2) and aligns its pooled decoder output to the teacher's
  through a linear head (cosine). Half of every batch is drawn from these paired rows.
- **Real-X-ray fine-tuning:** s4/cls_hybrid.py --init loads the encoder and decoder of a stage-xray model (heads are re-initialised) and
  trains on the 14 NIH diseases, as the v3 runs did.

Training stages: (1) CT teacher on the CT subset (5 seeds); (2) X-ray on projections, without and with distillation (5 seeds each);
(3) NIH fine-tuning from ImageNet, from (2) without distillation and from (2) with distillation (5 seeds each), 10 epochs (v2 and v3 peaked
at epochs 2 to 6). The DenseNet-121 orig runs of v3 are the CNN reference.

## 4. Explainability

- Swin X-ray: Grad-CAM++ on the last stage, Integrated Gradients and decoder attention (s4/xai_multistream.py works unchanged), pointing
  game on the NIH boxes.
- CT: Grad-CAM++ per slice (s4/xm_xai.py), assembled into a slice x row x column saliency volume and summed along the anteroposterior axis,
  which gives a frontal map in the geometry of the paired projection.
- **Cross-modal agreement:** for test scans positive for a finding, the projected CT map and the X-ray map of the paired projection are
  cropped to their own lung boxes, resized to 32x32 and correlated; the null pairs the CT map of one scan with the X-ray map of another
  positive scan of the same finding.

## 5. Hallucination awareness (s4/xm_halluc.py)

A classifier "hallucinates" when it reports a finding the image does not contain, or misses one it does. On NIH test images with no finding:
- **Finding-free perturbations:** rib emphasis (x + 0.5 * bone image from our suppressor; anatomy, not disease), Gaussian noise
  (sigma 0.02) and a tone change (gamma 0.8). Hallucination rate = share of images in which any class moves from below to above its
  operating threshold (90 % specificity on the validation split).
- **Counterfactual lesions:** a LIDC test nodule from the audit library is pasted into a lung field; sensitivity = share of images in which the
  nodule or mass score crosses its threshold, with the score change by nodule size.
- **Off-target attribution:** share of Grad-CAM++ mass outside the lung field for positive predictions of lung findings.
- Uncertainty: conformal risk control of the false-negative rate (s4/uq_conformal.py) for the Swin models, and CT-X-ray disagreement on
  paired scans as an abstention signal.

## 6. Compute (Kaggle, two T4 per session; approved by the author on 2026-11-08, spread over accounts)

Rates from the v3 screen (one stream, 32 images per step, about 0.4 s for a tiny encoder): a 10-epoch NIH run takes about 3.3 h, a
15-epoch projection run about 1.2 h, a 20-epoch CT teacher run (4 volumes x 24 slices per step) about 3 h.

| Step | Where | Estimate |
|---|---|---|
| CT subset download and reduction | ac_1, CPU kernel s4-ctprep, HF token as a Kaggle secret | 3 to 5 h, no GPU |
| CT teacher, 5 seeds x 20 epochs | ac_1 (the only account with CT data) | about 15 GPU-h |
| X-ray on projections, P0 and P1 x 5 seeds x 15 epochs | ac_4, ac_3 (P0); ac_5, ac_7 (P1) | about 12 GPU-h |
| NIH fine-tuning, F_in, F_p0, F_p1 x 5 seeds x 10 epochs | ac_2 (F_in); ac_4, ac_3 (F_p0); ac_5, ac_7 (F_p1) | about 50 GPU-h |
| Zero-shot transfer, cross-modal maps, hallucination tests | with the models; maps on ac_1 | about 4 GPU-h |

About 81 GPU-h (the first estimate, 52, predates the measured rates). Each session runs two T4 lanes, so about 41 session-hours, within
the quota left in the week of 8 November (ac_1 24 h, ac_2 19 h, ac_3 12 h, ac_4 19 h, ac_5 19 h, ac_7 12 h).

Account plan (kernels/down/run_down.py job lists pushed with ops/launch_down.py; A = swinv2_tiny_window8_256.ms_in1k):
- ac_1 (the author's account, the only one holding CT-RATE data): s4-ctprep (`python tools/package_code.py`, then `python
  ops/launch_xm_prep.py ac_1`), then the CT teacher `[g, "xm", "ct", "ct", k, 20, "--stage", "ct", "--arch", A]` for k = 0..4. The
  teacher.npz files go to ac_5 and ac_7 as the small private dataset s4-xm-teacher (layout results/<tag>/teacher.npz, which '@run:'
  finds). Later the P0 and P1 model files come to ac_1 the same way for the cross-modal maps `[g, "xmx", "ct", "xraykd", k, 0, "--qp",
  "@data:qp", "--ct_run", "@run:ct_xm_ct_s<k>", "--xr_run", "@run:qp_xm_xraykd_s<k>"]` (and xray) and the VinBigData zero-shot transfer.
- ac_2: F_in `[g, "hyb", "nih", "orig", k, 10, "--arch", A, "--enc_lr", "0.25"]` for k = 0..4, then the hallucination tests
  `[g, "xmh", "nih", "fin", 0, 0, "--runs", "@run:nih_hyb_orig@swinv2_tiny_window8_256_s0", ...]`.
- ac_4 (k = 0..2) and ac_3 (k = 3, 4): P0 `[g, "xm", "qp", "xray", k, 15, "--stage", "xray", "--arch", A]`, then F_p0 (the F_in job with
  `"--init", "@run:qp_xm_xray_s<k>/model.pt"`), the NIH zero-shot transfer `[g, "xm", "nih", "transfer-xray", k, 0, "--stage", "transfer",
  "--run", "@run:qp_xm_xray_s<k>", "--dataset", "nih"]` and the hallucination tests of those seeds.
- ac_5 (k = 0..2) and ac_7 (k = 3, 4), after the CT teacher: the same with P1 (variant "xraykd" with `"--teacher",
  "@run:ct_xm_ct_s<k>/teacher.npz"`) and F_p1 (`"--init", "@run:qp_xm_xraykd_s<k>/model.pt"`).
- dn121: hallucination tests of the v3 DenseNet-121 orig runs on the accounts that hold them; the pointing game with ops/launch_xai.py.
Folders: nih_hyb_orig@swinv2_tiny_window8_256[+xray|+xraykd]_s<k> (s4/runtag.py). Hallucination tests run per account on its seeds and
tools/xm_stats.py combines the seed groups (probabilities averaged with the number of seeds as weights, thresholds from the averaged
validation probabilities). v4 stages have no checkpoint: a run that would outlast the session is cut so the kernel still saves the others.
tools/xm_stats.py runs locally on the fetched outputs (results/v4/...).

## 7. Author actions

1. Done: the CT-RATE terms are accepted (checked on 2026-11-08 with the classic read token of HF_Token.txt).
2. Attach that classic read token as the Kaggle secret HF_TOKEN of ac_1's kernel s4-ctprep (Kaggle editor: Add-ons, Secrets) and run it;
   the code reads it only through kaggle_secrets and never writes or prints it.
3. Done: SAP_v4.md and the compute plan approved on 2026-11-08 ("use multiple account to complete run fast").

## 8. Out of scope (named, not built)

A box detector with a Swin backbone (author's choice); full CT-RATE (multi-terabyte); 3D Swin UNETR; the 18-label CT-RATE classification
(needs the gated label files; the six projection labels suffice for the cross-modal question).
