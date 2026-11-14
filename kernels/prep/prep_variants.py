"""Kaggle GPU kernel (template): build downstream image arrays for one dataset with the selected suppressor.

Placeholders: DATASET ('nih'|'tbx'|'vin'), WEIGHTS (final_ema.pt path), LIMIT (0 = all).
Output in /kaggle/working/<dataset>/: orig.npy, soft.npy (+ bone.npy for detection sets), uint8 [N,S,S]; labels.npy (NIH) or ann.json
(boxes in S px, labels, bone overlap per box); split.npy; patient.npy; ids.json; status.json. Unreadable images are logged, never fatal.
"""
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

DATASET, WEIGHTS, LIMIT, QURE = __DATASET__, __WEIGHTS__, __LIMIT__, __QURE__  # QURE: path of the released TorchScript model or None
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
sys.path.insert(0, str(Path(code[0]).parent.parent))
from s4.infer import load_suppressor, prep512, suppress_batch, to_u8  # noqa: E402

OUT = Path("/kaggle/working") / DATASET
OUT.mkdir(parents=True, exist_ok=True)
NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
       "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def first(patterns):
    for p in patterns:
        hit = glob.glob(p)
        if hit:
            return hit[0]
    raise FileNotFoundError(patterns)


def bucket(key, seed=20261026):
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 80 else "test"


def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    return (cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im).astype(np.float32)


def overlap(bone512, box, size):
    """Fraction of the box covered by strong predicted bone (> 25% of the image's bone p99); box in `size` px."""
    s = 512 / size
    x1, y1, x2, y2 = (int(round(v * s)) for v in box)
    sub = bone512[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    return float((sub > 0.25 * np.percentile(bone512, 99)).mean()) if sub.size else float("nan")


def nih():
    root = first(["/kaggle/input/datasets/nih-chest-xrays/data", "/kaggle/input/data"])
    df = pd.read_csv(os.path.join(root, "Data_Entry_2017.csv"))
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "images_*", "images", "*.png"))}
    tv, te = (set(open(os.path.join(root, f)).read().split()) for f in ("train_val_list.txt", "test_list.txt"))
    rows = []
    for r in df.itertuples():
        name, pid = r[1], str(r[4])
        if name not in files:
            continue
        sp = "test" if name in te else ("val" if bucket(pid) == "val" else "train") if name in tv else None
        if sp:  # official test list is the only test set; patients are split by hash inside train_val
            rows.append((name, files[name], sp, pid, np.array([c in str(r[2]).split("|") for c in NIH], np.float32)))
    return dict(rows=rows, size=256, with_bone=False, kind="cls")


def tbx():
    root = first(["/kaggle/input/datasets/omareldash75/tbx11k-original-dataset-with-bounding-boxes/TBX11K",
                  "/kaggle/input/tbx11k-original-dataset-with-bounding-boxes/TBX11K"])
    rows, ann, n_classes = [], {}, 0
    for fn, official in (("all_train.json", "train"), ("all_val.json", "test")):
        coco = json.load(open(os.path.join(root, "annotations", "json", fn)))
        cats = {c["id"]: i for i, c in enumerate(coco["categories"])}
        n_classes = len(cats)
        per = {}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            per.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], cats[a["category_id"]]))
        for im in coco["images"]:
            try:
                p = first([os.path.join(root, "imgs", im["file_name"]), os.path.join(root, "imgs", "*", os.path.basename(im["file_name"]))])
            except FileNotFoundError:
                print("missing image", im["file_name"], flush=True)
                continue
            sp = "test" if official == "test" else ("val" if bucket(im["file_name"]) == "val" else "train")
            rows.append((im["file_name"], p, sp, im["file_name"], None))
            sx, sy = 512 / im["width"], 512 / im["height"]
            ann[im["file_name"]] = [([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], c) for b, c in per.get(im["id"], [])]
    return dict(rows=rows, size=512, with_bone=True, kind="det", ann=ann, n_classes=n_classes)


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def consensus(boxes):
    """(rad_id, box) of one class -> (boxes marked by >= 2 distinct radiologists, any single-reader cluster left over)."""
    clusters = []
    for rad, b in boxes:
        for c in clusters:
            if iou(np.mean([m[1] for m in c], 0), b) > 0.4 and rad not in {m[0] for m in c}:
                c.append((rad, b))
                break
        else:
            clusters.append([(rad, b)])
    return [np.mean([m[1] for m in c], 0).tolist() for c in clusters if len({m[0] for m in c}) >= 2], any(len(c) == 1 for c in clusters)


def vin():
    comp = first(["/kaggle/input/vinbigdata-chest-xray-abnormalities-detection", "/kaggle/input/competitions/vinbigdata-chest-xray-abnormalities-detection"])
    png = first(["/kaggle/input/datasets/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024", "/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024"])
    tr = pd.read_csv(os.path.join(comp, "train.csv"))
    meta = pd.read_csv(os.path.join(png, "train_meta.csv")).set_index("image_id")
    rows, ann = [], {}
    for iid, g in tr.groupby("image_id"):
        h0, w0 = meta.loc[iid, "dim0"], meta.loc[iid, "dim1"]
        items, dropped = [], False
        for cid, gc in g[g.class_id != 14].groupby("class_id"):
            keep, lone = consensus([(r.rad_id, [r.x_min, r.y_min, r.x_max, r.y_max]) for r in gc.itertuples()])
            dropped |= lone and not keep
            items += [([b[0] * 512 / w0, b[1] * 512 / h0, b[2] * 512 / w0, b[3] * 512 / h0], int(cid)) for b in keep]
        if dropped and not items:
            continue  # only single-reader findings: ambiguous, excluded
        rows.append((iid, os.path.join(png, "train", f"{iid}.png"), bucket(iid), iid, None))
        ann[iid] = items
    return dict(rows=rows, size=512, with_bone=True, kind="det", ann=ann, n_classes=14)


def main():
    ds = {"nih": nih, "tbx": tbx, "vin": vin}[DATASET]()
    rows = ds["rows"][: LIMIT or None]
    S, N = ds["size"], len(rows)
    print(DATASET, "images:", N, flush=True)
    global WEIGHTS
    if not os.path.exists(WEIGHTS):  # datasets may be mounted under /kaggle/input/datasets/<owner>/
        WEIGHTS = glob.glob("/kaggle/input/datasets/*/" + WEIGHTS.split("/kaggle/input/", 1)[1])[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_suppressor(WEIGHTS, dev)
    names = ["orig", "soft"] + (["bone"] if ds["with_bone"] else []) + (["soft_qure"] if QURE else [])
    qm = None
    if QURE:
        if not os.path.exists(QURE):
            QURE = glob.glob("/kaggle/input/datasets/*/" + QURE.split("/kaggle/input/", 1)[1])[0]
        qm = torch.jit.load(QURE, map_location=dev).eval()
    arrs = {k: np.lib.format.open_memmap(OUT / f"{k}.npy", mode="w+", dtype=np.uint8, shape=(N, S, S)) for k in names}
    ok, bad, ov, t0 = np.zeros(N, bool), [], {}, time.time()
    with ThreadPoolExecutor(6) as pool:
        for s in range(0, N, 32):
            chunk = rows[s:s + 32]
            ims = list(pool.map(lambda r: read_gray(r[1]), chunk))
            good = [i for i, im in enumerate(ims) if im is not None]
            bad += [chunk[i][0] for i in range(len(chunk)) if i not in good]
            if not good:
                continue
            x = np.stack([prep512(ims[i]) for i in good])
            b = suppress_batch(model, x, dev)
            v = dict(orig=to_u8(x, S), soft=to_u8(x - b, S), bone=to_u8(np.clip(b, 0, 1), S))
            if qm is not None:  # released CT-supervised model at its native 1024 px input; soft = x - bone at 1024, then resized to S
                sq = []
                for j in range(0, len(good), 4):
                    xs = np.stack([(lambda a: (a - a.min()) / (a.max() - a.min() + 1e-10))(cv2.resize(ims[i], (1024, 1024), interpolation=cv2.INTER_AREA)) for i in good[j:j + 4]])
                    with torch.no_grad():
                        o = qm(torch.from_numpy(xs.astype(np.float32))[:, None].to(dev))
                    o = o[0] if isinstance(o, (tuple, list)) else o
                    sq.append(xs - o.float()[:, 0].cpu().numpy())
                v["soft_qure"] = to_u8(np.concatenate(sq), S)
            for j, i in enumerate(good):
                for k in names:
                    arrs[k][s + i] = v[k][j]
                ok[s + i] = True
                if ds["kind"] == "det":
                    ov[chunk[i][0]] = [overlap(b[j], box, S) for box, _ in ds["ann"].get(chunk[i][0], [])]
            if (s // 32) % 100 == 0:
                print(s, N, f"{time.time() - t0:.0f}s", flush=True)
    for a in arrs.values():
        a.flush()
    np.save(OUT / "split.npy", np.array([r[2] for r in rows]))
    np.save(OUT / "patient.npy", np.array([r[3] for r in rows]))
    json.dump([r[0] for r in rows], open(OUT / "ids.json", "w"))
    json.dump(dict(ok=ok.tolist(), unreadable=bad), open(OUT / "status.json", "w"))
    if ds["kind"] == "cls":
        np.save(OUT / "labels.npy", np.stack([r[4] for r in rows]))
    else:
        json.dump([dict(boxes=[b for b, _ in ds["ann"].get(r[0], [])], labels=[c for _, c in ds["ann"].get(r[0], [])],
                        overlap=ov.get(r[0], [])) for r in rows], open(OUT / "ann.json", "w"))
    print("DONE", DATASET, "ok", int(ok.sum()), "bad", len(bad), flush=True)


if __name__ == "__main__":
    main()
