"""Kaggle CPU kernel: per-series manifest of the LIDC copy + integrity manifest of the private RibCXR dataset.
Outputs: /kaggle/working/lidc_series.csv, lidc_summary.json, ribcxr_manifest.json (hashes only, no images)."""
import csv
import hashlib
import json
import os
import re
import time

import pydicom

t0 = time.time()
rows = []
for root, _dirs, files in os.walk("/kaggle/input"):
    if "ribcxr" in root.lower():
        continue
    dcm = sorted(f for f in files if f.endswith(".dcm"))
    if not dcm:
        continue
    m = re.search(r"LIDC-IDRI-(\d{4})(?![\d-])", root)
    chunk = re.search(r"LIDC-IDRI-\d{4}-\d{4}", root)
    row = dict(path=root, patient=m.group(1) if m else "", chunk=chunk.group(0) if chunk else "",
               n_dcm=len(dcm), xml=";".join(f for f in files if f.endswith(".xml")))
    try:
        ds = pydicom.dcmread(os.path.join(root, dcm[0]), stop_before_pixels=True)
        for k in ("Modality", "SliceThickness", "KVP", "Manufacturer", "ConvolutionKernel", "Rows", "Columns",
                  "PixelSpacing", "SeriesInstanceUID"):
            row[k] = str(getattr(ds, k, ""))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    rows.append(row)

keys = sorted({k for r in rows for k in r})
with open("/kaggle/working/lidc_series.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)

ct = [r for r in rows if r.get("Modality") == "CT"]
summary = dict(
    n_series=len(rows), n_ct_series=len(ct), n_patients_all=len({r["patient"] for r in rows}),
    n_patients_ct=len({r["patient"] for r in ct}), chunks=sorted({r["chunk"] for r in rows}),
    modalities={m: sum(r.get("Modality") == m for r in rows) for m in {r.get("Modality") for r in rows}},
    ct_with_xml=sum(bool(r["xml"]) for r in ct), ct_ge_100_slices=sum(r["n_dcm"] >= 100 for r in ct),
    unreadable=[r["path"] for r in rows if "error" in r][:20], seconds=round(time.time() - t0))
json.dump(summary, open("/kaggle/working/lidc_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

man = []
for root, _d, files in os.walk("/kaggle/input"):
    if "ribcxr" not in root.lower():
        continue
    for fn in files:
        p = os.path.join(root, fn)
        man.append(dict(name=os.path.relpath(p, root.split("s4-ribcxr")[0] + "s4-ribcxr") if "s4-ribcxr" in p else fn,
                        size=os.path.getsize(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()))
json.dump(man, open("/kaggle/working/ribcxr_manifest.json", "w"))
print("ribcxr files:", len(man))
