"""Kaggle CPU kernel: do LUNA16 .mhd volumes match the DICOM volumes of the same series (axis order/flips, z location)?"""
import csv
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = Path(code[0]).parent.parent
sys.path.insert(0, str(CODE))
from s4.ct import load_mhd, load_series  # noqa: E402
from s4.nodule import consensus_nodules, xml_series_uid  # noqa: E402

DATA = CODE / "s4" / "data"
LIDC = glob.glob("/kaggle/input/datasets/washingtongold/lidcidri30")[0]
LUNA = glob.glob("/kaggle/input/datasets/vafaeii/luna16")[0]
mount = {r["SeriesInstanceUID"]: r["path"].split("lidcidri30/", 1)[1]
         for r in csv.DictReader(open(DATA / "lidc_mount_series.csv")) if r["Modality"] == "CT" and "lidcidri30/" in r["path"]}
luna = {Path(p).stem: p for p in glob.glob(os.path.join(LUNA, "subset*", "subset*", "*.mhd"))}
xml_by_uid = {}
for p in glob.glob(str(DATA / "lidc_xml" / "**" / "*.xml"), recursive=True):
    u = xml_series_uid(p)
    if u:
        xml_by_uid.setdefault(u, []).append(p)
common = sorted(set(mount) & set(luna))
print("series in both:", len(common), flush=True)
rep = []
for uid in common[:: max(1, len(common) // 12)][:12]:
    try:
        a = load_series(sorted(glob.glob(os.path.join(LIDC, mount[uid], "*.dcm"))))
        b = load_mhd(luna[uid])
        r = dict(uid=uid[-8:], shape_dicom=a["hu"].shape, shape_mhd=b["hu"].shape, spacing_dicom=a["spacing"], spacing_mhd=b["spacing"],
                 transform=b["meta"]["transform"])
        if a["hu"].shape == b["hu"].shape:
            best = None
            for fz, fy, fx in itertools.product([1, -1], repeat=3):
                bb = b["hu"][::fz, ::fy, ::fx]
                d = float(np.abs(a["hu"][::8].astype(np.float32) - bb[::8].astype(np.float32)).mean())
                if best is None or d < best[0]:
                    best = (d, dict(flip_z=fz < 0, flip_y=fy < 0, flip_x=fx < 0))
            r["best_mean_abs_hu_diff"], r["best_flip"] = best
            r["identity_mean_abs_hu_diff"] = float(np.abs(a["hu"][::8].astype(np.float32) - b["hu"][::8].astype(np.float32)).mean())
        xmls = xml_by_uid.get(uid, [])
        if xmls:
            za = [sorted(n["mask"])[0] for n in consensus_nodules(xmls, a["locate"], a["hu"].shape[1:], a["hu"].shape[0], a["spacing"])]
            zb = [sorted(n["mask"])[0] for n in consensus_nodules(xmls, b["locate"], b["hu"].shape[1:], b["hu"].shape[0], b["spacing"])]
            r["nodule_first_slice_dicom"], r["nodule_first_slice_mhd"] = sorted(za), sorted(zb)
        rep.append(r)
    except Exception as e:
        rep.append(dict(uid=uid[-8:], error=f"{type(e).__name__}: {e}"))
    print(rep[-1], flush=True)
json.dump(rep, open("/kaggle/working/orient.json", "w"), indent=1, default=str)
