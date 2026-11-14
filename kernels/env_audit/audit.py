"""Kaggle CPU kernel: record the runtime and audit the LIDC-IDRI copy. Writes /kaggle/working/env_audit.json."""
import collections
import json
import os
import platform
import re
import shutil
import sys
import time
import urllib.request

t0 = time.time()
out = {"python": sys.version.split()[0], "cpus": os.cpu_count(), "platform": platform.platform()}
out["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
out["working_free_gb"] = round(shutil.disk_usage("/kaggle/working").free / 2**30, 1)
for m in ("torch", "timm", "numpy", "scipy", "pandas", "sklearn", "skimage", "cv2", "pydicom", "SimpleITK",
          "lpips", "torchmetrics", "statsmodels", "totalsegmentator"):
    try:
        out[m] = getattr(__import__(m), "__version__", "ok")
    except Exception as e:
        out[m] = f"MISSING ({type(e).__name__})"
try:
    with urllib.request.urlopen("https://huggingface.co/api/models/qureaiorg/bone-suppression", timeout=15) as r:
        out["internet_hf"] = r.status
except Exception as e:
    out["internet_hf"] = f"FAIL {type(e).__name__}: {e}"

out["input_top"] = {d: sorted(os.listdir(f"/kaggle/input/{d}"))[:8] for d in sorted(os.listdir("/kaggle/input"))}

patients, series, xmls = set(), {}, 0
for root, _dirs, files in os.walk("/kaggle/input"):
    dcm = [f for f in files if f.endswith(".dcm")]
    xmls += sum(f.endswith(".xml") for f in files)
    if dcm:
        series[root] = len(dcm)
        m = re.search(r"LIDC-IDRI-(\d{4})", root)
        if m:
            patients.add(m.group(1))
out.update(n_patients=len(patients), n_series=len(series), n_xml=xmls, walk_s=round(time.time() - t0))
cnt = sorted(series.values())
out["slices_per_series"] = {"min": cnt[0], "median": cnt[len(cnt) // 2], "max": cnt[-1]} if cnt else None
out["series_lt_100_slices"] = sum(c < 100 for c in cnt)

try:
    import pydicom
    meta = collections.Counter()
    thick = []
    for path in list(series)[::max(1, len(series) // 60)][:60]:
        f = sorted(x for x in os.listdir(path) if x.endswith(".dcm"))[0]
        ds = pydicom.dcmread(os.path.join(path, f), stop_before_pixels=True)
        thick.append(float(getattr(ds, "SliceThickness", "nan")))
        meta[(str(getattr(ds, "Manufacturer", "?")), str(getattr(ds, "KVP", "?")))] += 1
    out["sample_thickness_mm"] = sorted(thick)
    out["sample_manufacturer_kvp"] = {f"{k[0]}|{k[1]}": v for k, v in meta.items()}
except Exception as e:
    out["pydicom_sample"] = f"FAIL {type(e).__name__}: {e}"

json.dump(out, open("/kaggle/working/env_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
