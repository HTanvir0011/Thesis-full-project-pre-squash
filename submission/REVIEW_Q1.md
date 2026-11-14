# Internal pre-submission review (Q1 reviewer perspective)

Manuscript: *Explainable and hallucination-aware cross-modality learning from chest CT for X-ray disease classification: lesion-faithful
bone suppression and Swin transformer knowledge transfer* (target: Medical Image Analysis, full-length article).

This is a self-review written in the voice of a demanding reviewer, followed by what was changed in response. It is not a substitute
for review by co-authors or colleagues.

## Summary of the submission
Three pre-specified, hash-frozen studies of CT-supervised models of chest radiographs: (I) a counterfactual lesion audit and a twin
lesion-invariance regulariser for CT-supervised bone suppression; (II) a capacity-controlled test of bone-suppressed classifier inputs
(175 runs, three datasets); (III) distillation of a 2.5D Swin CT teacher (CT-RATE) into Swin radiograph classifiers through paired
projections, with heat-map agreement, pointing game, hallucination tests and conformal risk control.

## Strengths
- Pre-specified hypotheses, hash-frozen plans, disclosed deviations and negative results reported as prominently as positive ones.
- Paired designs with cluster bootstraps, seed ensembles and capacity-matched controls; intervals reported for every contrast.
- The audit addresses a real gap (lesion erasure by suppressors) with an additive, exactly known signal.
- Study III asks a timely question (can CT teach radiograph classifiers?) and evaluates trustworthiness, not only accuracy.

## Major concerns and responses
1. **Cohesion of three studies in one paper.** The reader needs one thread. *Response:* the introduction now frames all three studies around
   one question (what CT-supervised models erase, invent and highlight); the discussion opens with all failed hypotheses across studies,
   and the conclusion answers the question for each study.
2. **Title over-claims a "Swin framework".** The final suppressor ensemble uses ConvNeXt-V2-T. *Response:* the title now names
   "cross-modality learning from chest CT" and ties "Swin transformer" to the knowledge transfer (Study III).
3. **H14 conclusion depends on the attribution method.** The CT maps are slice-dominated bands and the Swin X-ray Grad-CAM++ maps show
   border artefacts. *Response:* every statement now says that the CT and X-ray *Grad-CAM++* maps did not agree; the mean-map figure is
   used to explain why, and the limitation names agreement of Integrated Gradients or attention as untested.
4. **H13 confound.** The DenseNet-121 reference was trained for 20 epochs at the full encoder learning rate, the Swin classifiers for 10
   epochs at a quarter of it. *Response:* disclosed in the methods (Section 6.2) and the limitations; the per-class analysis showing that
   hernia carries the difference is labelled descriptive.
5. **The null result of H11 may reflect a weak teacher.** *Response:* the discussion states that the teacher was hardly more accurate on
   CT (0.821) than P0 on the projections (0.814), weakest for lung lesions, and seed-unstable, and restricts the conclusion to this
   teacher and this amount of paired CT.
6. **Multiplicity and post hoc analyses.** *Response:* H13-H15 are reported as secondary and uncorrected (H15 p = 0.044 is called
   uncorrected wherever it appears); the DeLong grouping, the pointing-game summary and the conformal analysis are flagged as fixed after
   the hypothesis results were known.
7. **Length.** *Response:* the main text was kept to 90 pages by moving the deviations log, the per-seed results and secondary figures to
   the supplementary material, and by condensing the introduction, discussion and limitations; the CLAIM 2024 checklist is submitted as
   the journal's own form.

## Minor concerns and responses
- Notation of the distillation loss: the label symbol was undefined (now defined).
- The abstract exceeded 250 words (now within the limit).
- "Integrated Gradients localised as well as" over-stated a small difference (now "about as well").
- The example heat maps contain no mass or nodule box (stated in the caption and the limitations).
- Data handling of CT-RATE-derived files deviated from the design's own rule (stated in the limitations and the supplementary
  deviations, with the statement that nothing derived from CT-RATE was released).

## Second audit (external inconsistency report, 13 Nov 2026)
Each of the 17 items was checked against the result files, the code and the decision log before any change:
- 1.1 SZCH sites: the contrasts use the 51 held-out images (306 sites); the text and the Table 8 and 15 captions now say so.
- 1.2 Node21: two boxes narrower than 4 px at 512 px have no defined contrast (s4/prep.py box_cnr); stated in the data section.
- 1.3 Table 3 now prints the stored raw PSNR, raw SSIM and high-frequency ratio (tools/final_stats.py; old columns unchanged).
- 1.4 K = 4 only where the dose-response averages all four SwinV2-T seeds at w = 16; stated in Tables 5 and A.29.
- 1.5 The per-class figure uses 1,000 resamples for display; the caption says so (the tests use 10,000).
- 1.6 16 of 25,596 NIH images have an empty lung mask and no spread; stated in the text and Supplementary Table S11.
- 1.7 The abstract, highlights and conclusion now restrict the null result to separate streams and report the small channel-stacking gains.
- 2.1 Section 5.1 gives the Qure composer's actual order (s4/qp.py) with distinct symbols; 2.2 the Table 2 caption explains the base-image
  and twin lesions (s4/train.py); 2.3 the insertion figure is called an illustration on an LIDC test CT; 2.4 "both" is defined.
- 3.1 R1 only gated the full-budget screen; R2 applies to every full-budget candidate (RUNPLAN.md section 4). The text now gives the R2
  numbers from the decision log and states that ConvNeXt-V2-T, the only encoder passing both rules, would have been chosen anyway.
- 3.2 The DenseNet-121 reference score was known about ten hours before the fourth plan was frozen (commit times); disclosed.
- 3.3 The unequal training budgets of H13 are now stated in the results and discussion, not only in the limitations.
- 4.1 The CT-RATE data-handling sentence of Section 9.1 now matches the limitation; 4.2 the BDCXR-3257 licence (public, CC BY 4.0) and
  the data use agreement the authors follow are both stated.
- 5.1 Author block, CRediT, competing interests, funding and the repository link still need the authors' input.
To keep the main PDF within 90 pages, the Study II uncertainty table, selective-classification table and figure moved to the supplementary
(Tables S10, S11, Fig. S13).

## Remaining items for the authors
- Author block, CRediT statement, competing interests, funding (red TODO markers in main.tex).
- Confirm the generative-AI declaration matches the actual use; journals require this statement to be accurate.
- Repository link, release commit and checksums after release approval.
- Fill the official CLAIM 2024 form (sections/claim_checklist.tex lists where each item is addressed).
- Read every cited paper once more for the exact claim made about it before submission.
