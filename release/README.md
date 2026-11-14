# Release folder: what may and may not be released

Status: **nothing here is published.** Release only after the owner says so (RUNPLAN.md, section 7). On the Hugging Face Hub the model card
(`MODEL_CARD.md`) becomes the `README.md` of the model repository.

## May be released (after the go-ahead)
| item | condition |
|---|---|
| Weights of the final K = 3 ensemble: ConvNeXt-V2-T twin16 seeds 0-2 (`outputs/final_v2/weights/qp_twin16_s{0,1,2}_cnx2_300k/final_ema.pt`; the SwinV2-T `twin16_s*` files are no longer the release; DECISIONS.md 2026-10-31) | CC BY-NC-SA 4.0 (inherited from the Qure.ai `ct2xr-projections` training data); model card with attribution to Qure.ai, CT-RATE and LIDC-IDRI (CC BY 3.0, data DOI) |
| Inference code (`s4/infer.py`, `s4/model.py` and what they import) | MIT (chosen by the author, DECISIONS.md 2026-11-14) |
| Training, evaluation and table/figure scripts, `SAP.md`, `SAP_v2.md`, `DECISIONS.md`, `DEVIATIONS.md`, `freeze*.json` | contain no data; check them for paths, user names and tokens first |
| Aggregate result files and the paper tables | set-level statistics only; for VinDr-RibCXR and VinBigData only the aggregates printed in the paper |
| Figures | only from permitted images (below) or synthetic renders |
| Demo Space (`site/app.py`) | `REPO` set to the published model repository; the Space stores no uploads |
| Project page | text and permitted figures |

## May not be released
- **Anything derived from VinDr-RibCXR or VinBigData (VinDr-CXR) images**: images, crops, thumbnails, rib or lung masks, predictions, bone or soft-tissue
  maps, audit maps, per-image or per-site metrics, file names or identifiers, embeddings, and any model trained on them (for example the downstream
  detectors trained on VinBigData). Not even as demo samples. These data are licensed (a data use agreement); only metrics computed by the licence
  holder leave the machine.
- Real radiographs from JSRT, NIH ChestX-ray14, TBX11K and Node21 (link to the providers instead), unless the licence page was read and allows
  redistribution. Do not copy the two JSRT example images that ship with the Qure.ai repository: JSRT has its own terms.
- **Anything derived from BDCXR-3257 images** (the lab's data use agreement forbids redistribution and public release of the radiographs): images, arrays, crops, per-image predictions, file names, and the downstream classifiers trained on them. Aggregate results may be published, with the acknowledgement of the hospital and IRB REC-FSIT-2025/No: 11928.
- Qure.ai's released weights and the Qure.ai projections themselves (users download them from Qure.ai under its licence); CT-RATE volumes or any
  data derived from them (gated, redistribution not allowed by the dataset card).
- Downstream classifiers and detectors (not part of the release); rendered banks and other large intermediate files (not planned).
- Credentials of any kind (token files, API keys, `.env`), machine paths, anything that identifies a person.

## Demo samples
Only from licences that allow them. SZCH-X-Rays (CC BY-NC 4.0: attribution, non-commercial) or synthetic renders, with the licence line next to each
image and a selection rule that is not hand-picking. No sample from JSRT, VinDr-RibCXR, VinBigData, NIH, TBX11K or Node21.

## Checklist before publishing
- [ ] The owner's explicit go-ahead.
- [ ] Rotate the Kaggle and Hugging Face tokens that sat in plain-text files (RUNPLAN.md, section 7).
- [ ] Freeze files exist; the code commit is tagged; `python -m pytest` passes on CPU.
- [ ] SHA-256 of the three checkpoints written into the model card; repository revision pinned.
- [ ] Checkpoints inspected: no paths, user names or tokens inside (`torch.load`, look at every key).
- [ ] ConvNeXt V2 pretrained weights are CC BY-NC 4.0 (timm): confirm the combined licence of the released weights.
- [ ] Licence text of every data set read; Table data of the paper filled; model card TODOs resolved.
- [ ] Space: `REPO` set, its own copy of `s4/` included, analytics off, cache cleanup on (`site/app.py`), tested with a synthetic image only.
- [ ] Scan the release tree for image files; none may come from `data/`.
