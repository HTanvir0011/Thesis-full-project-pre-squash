# What this snapshot leaves out

Built by `tools/build_release.py` from the project repository. Files left out, by rule:

- **data/weights**: 5478
- **restricted**: 194
- **scratch**: 402
- **superseded**: 19

Why: credentials are never published. Raw and staged data, model weights and prediction arrays are too large for git and partly licensed. For VinDr-RibCXR, VinBigData, BDCXR-3257, JSRT, Node21 and CT-RATE-derived outputs only aggregate results are kept, as their data use agreements and the analysis plans require. Superseded drafts and scratch runs (smoke, dry, debug and hardware tests, launch and queue logs) carry no result. Kaggle account names are replaced by the labels ac_1 ... ac_9.
