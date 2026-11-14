# Analysis plans, hash freezes and decision logs

Every hypothesis in the manuscript and the thesis was written into an analysis plan before the data it governs were evaluated. The plan was
then hash-frozen: its SHA-256 went into a freeze file together with the hashes of the split files and the code (or the code commit) and,
for Study I, of the model weights, and the freeze file was committed. The plans are pre-specified and hash-frozen but were not deposited in
a public registry. Every change after a freeze is logged with its date and reason in `DEVIATIONS.md`.

| Plan | Governs | Freeze file | Frozen (UTC+6) | Code commit recorded |
|---|---|---|---|---|
| `SAP.md` | first plan: CT-supervised suppressor and lesion audit | `freeze.json` | 27 Oct 2026, 06:14 | `8e9e3be` |
| `SAP_v2.md` | Study I: twin regulariser, locked audits and fidelity | `freeze_v2.json` (with `SAP.md`, splits, 52 weight files) | 1 Nov 2026, 06:27 | `6b3df86` |
| `SAP_bdcxr.md` | BDCXR-3257 external validation (addendum) | committed before any model saw an image; not hashed | 3 Nov 2026 | |
| `SAP_v3.md`, `DESIGN_v3.md` | Study II: bone-suppressed classifier inputs | `freeze_v3.json` | 3 Nov 2026, 21:51 | `5e7e69f` |
| `SAP_v4.md`, `DESIGN_v4.md` | Study III: CT-to-X-ray distillation | `freeze_v4.json` | 8 Nov 2026, 14:46 | `2ebaebe` |

The commit hashes refer to the private development repository; the SHA-256 values in the freeze files can be checked against the files
here. `SAP_v3.md` still carries the heading "DRAFT, not frozen" because it was frozen byte for byte as it stood.

| Log | Content |
|---|---|
| `RUNPLAN.md`, `RUNPLAN_v3.md` | run plans: screening and selection rules, seeds, compute budget |
| `DECISIONS.md` | dated decisions taken during development, with the evidence each one used |
| `DEVIATIONS.md` | every change after a freeze, dated, with its reason and its effect on the results |

The logs were written while the work was under way, so they are terse. Two wording notes: where they say "pre-registered", read
"pre-specified and hash-frozen" (no plan was deposited in a public registry); and `PLAN.md`, the initial project plan they mention, is an
internal note that is not part of this repository.

## Checking the freezes

```bash
python tools/verify_freeze.py
```

The script re-hashes every file a freeze recorded. In this repository it reports:

- every analysis plan and design unchanged since its freeze;
- the code files edited after a freeze as `CHANGED` (each edit is logged in `DEVIATIONS.md`);
- `config.yaml` as `CHANGED`: Kaggle accounts were added after the first two freezes, and this public copy shows their names as the labels
  `ac_1` to `ac_9`;
- the model weights, split arrays and CT-RATE volume lists as absent, because they are not published.

These files lived at the top of the repository when they were frozen; they moved here unchanged on 14 November 2026, so the paths inside
the freeze files are relative to the repository root. `tools/freeze*.py` are kept as they ran.
