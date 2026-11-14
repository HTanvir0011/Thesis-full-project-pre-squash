"""Downstream image arrays for one dataset: run the frozen suppressor (deep ensemble) and, optionally, the released Qure model over a dataset and keep the
input variants used downstream at one fixed resolution. Runs on a Kaggle GPU kernel (kernels/prep/prep_variants.py) or locally.

python -m s4.prep --dataset nih|tbx|vin|node21|jsrt|bdcxr --weights a.pt+b.pt+c.pt --qure PATH|none --arrays orig,soft[,bone,soft_qure] --tag NAME [--limit N]

Output in <S4_PREP_OUT or /kaggle/working>/<tag>/: <array>.npy uint8 [N,S,S]; labels.npy (NIH) or ann.json (detection sets: boxes in S px, labels, overlap of each box
with strong predicted bone by ours / Qure / their consensus, box CNR of orig / soft / soft_qure / clahe); split.npy; patient.npy; ids.json; status.json.
Unreadable images are logged, never fatal. The parent builds the files and starts one worker per GPU (rows i % n == g); workers fill the shared memmaps.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("S4_PREP_OUT", "/kaggle/working"))
NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
       "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
VARIANTS = ("orig", "soft", "soft_qure", "clahe")  # images whose box CNR is stored for detection sets
NODE21_URL = "https://zenodo.org/api/records/5548363/files/dataset_node21.zip/content"  # CC BY-NC-ND 4.0: evaluation only, derived images stay private


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
    if str(path).endswith(".mha"):
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    return (cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im).astype(np.float32)


def overlap(m512, box, size):
    """Fraction of the box covered by strong bone (> 25% of the map's p99); box in `size` px, map at 512."""
    s = 512 / size
    x1, y1, x2, y2 = (int(round(v * s)) for v in box)
    sub = m512[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    return float((sub > 0.25 * np.percentile(m512, 99)).mean()) if sub.size else float("nan")


def box_cnr(img, box, pad=0.5):
    """(mean inside the box - mean of the surrounding ring) / std of the ring. img float [0,1] at the box's pixel scale; ring = box grown by `pad` per side."""
    H, W = img.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return float("nan")
    X1, Y1, X2, Y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    ex, ey = int(round(pad * (x2 - x1))), int(round(pad * (y2 - y1)))
    RX1, RY1, RX2, RY2 = max(0, X1 - ex), max(0, Y1 - ey), min(W, X2 + ex), min(H, Y2 + ey)
    mask = np.ones((RY2 - RY1, RX2 - RX1), bool)
    mask[Y1 - RY1:Y2 - RY1, X1 - RX1:X2 - RX1] = False
    ring = img[RY1:RY2, RX1:RX2][mask]
    if ring.size < 16 or X2 <= X1 or Y2 <= Y1:
        return float("nan")
    return float((img[Y1:Y2, X1:X2].mean() - ring.mean()) / (ring.std() + 1e-3))


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
    return dict(rows=rows, size=256, kind="cls")


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
    return dict(rows=rows, size=512, kind="det", ann=ann, n_classes=n_classes)


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
    return dict(rows=rows, size=512, kind="det", ann=ann, n_classes=14)


def node21_fetch(d, threads=4):
    """Node21 (Zenodo 5548363): download the 4,882 processed 1024 px radiographs (1,134 with 1,476 nodule boxes) and metadata.csv into d,
    straight out of the 37 GB zip by HTTP range requests (its 30 GB of originals and CT patches are never fetched). Zenodo answers bursts
    with 429 (16 threads gave ~1,900 retries), so few threads and long back-off; files already present are skipped (resumable)."""
    from remotezip import RemoteZip
    pre = "cxr_images/proccessed_data/"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "metadata.csv").exists():
        with RemoteZip(NODE21_URL) as z:
            (d / "metadata.csv").write_bytes(z.read(pre + "metadata.csv"))
    meta = pd.read_csv(d / "metadata.csv")
    todo = [n for n in sorted(meta.img_name.unique()) if not (d / n).exists()]
    print("node21: fetching", len(todo), "of", meta.img_name.nunique(), "images", flush=True)

    def fetch(chunk):
        with RemoteZip(NODE21_URL) as z:  # one handle per thread
            for n in chunk:
                for attempt in range(10):
                    try:
                        (d / (n + ".part")).write_bytes(z.read(pre + "images/" + n))
                        os.replace(d / (n + ".part"), d / n)
                        break
                    except Exception as e:  # rate limit or network hiccup: back off and retry the image
                        print("retry", n, attempt, repr(e)[:80], flush=True)
                        time.sleep((30 if "429" in repr(e) else 5) * (attempt + 1))
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(fetch, [todo[i::threads] for i in range(threads)]))
    print(f"node21 fetch done in {time.time() - t0:.0f}s", flush=True)
    return meta


def node21():
    """Rows from a mounted copy made by kernels/prep/fetch_node21.py (CPU kernel, output dir node21_raw), else downloaded here."""
    raw = sorted(glob.glob("/kaggle/input/**/node21_raw/metadata.csv", recursive=True))
    d = Path(raw[0]).parent if raw else Path("/tmp/node21")
    meta = pd.read_csv(d / "metadata.csv") if raw else node21_fetch(d)
    names = sorted(meta.img_name.unique())
    rows, ann = [], {}
    for n in names:
        if not (d / n).exists():
            print("missing image", n, flush=True)
            continue
        g = meta[(meta.img_name == n) & (meta.label == 1) & (meta.width > 0) & (meta.height > 0)]
        rows.append((n, str(d / n), bucket(n), n, None))
        ann[n] = [([r.x / 2, r.y / 2, (r.x + r.width) / 2, (r.y + r.height) / 2], 0) for r in g.itertuples()]  # 1024 px -> 512 px frame
    assert len(rows) > 0.99 * len(names), f"only {len(rows)} of {len(names)} images present"
    return dict(rows=rows, size=512, kind="det", ann=ann, n_classes=1)


def jsrt():
    """JSRT (247 images, 154 nodules; Kaggle copy raddar/nodules-in-chest-xrays-jsrt): evaluation only (contrast of real nodules), every row is 'test'."""
    root = Path(os.environ.get("S4_JSRT", "")) if os.environ.get("S4_JSRT") else Path(first(["/kaggle/input/nodules-in-chest-xrays-jsrt", "/kaggle/input/datasets/raddar/nodules-in-chest-xrays-jsrt"]))
    meta = pd.read_csv(root / "jsrt_metadata.csv")
    rows, ann = [], {}
    for r in meta.itertuples():
        p = first([str(root / "images" / "images" / r.study_id), str(root / "images" / r.study_id)])
        rows.append((r.study_id, p, "test", r.study_id, None))
        ann[r.study_id] = []
        if r.state in ("malignant", "benign") and not np.isnan(r.x):
            rad = float(r.size) / 0.175 / 2  # nodule diameter (mm) -> radius in 2048 px (0.175 mm/px)
            ann[r.study_id] = [([(r.x - rad) / 4, (r.y - rad) / 4, (r.x + rad) / 4, (r.y + rad) / 4], 0)]  # 2048 px -> 512 px frame
    return dict(rows=rows, size=512, kind="det", ann=ann, n_classes=1)


BDCXR_URL = "https://ndownloader.figshare.com/files/64149946"  # doi 10.6084/m9.figshare.32086362 (CC BY 4.0); the lab's DUA forbids re-uploading: always fetched
BDCXR = ["Normal", "Pneumonia Bacterial", "Pneumonia Viral"]


def bdcxr_fetch():
    """-> folder with one sub-folder per class. S4_BDCXR_ZIP = a local copy of the zip (else downloaded), S4_BDCXR_DIR = where to unpack."""
    import urllib.request
    import zipfile
    d = Path(os.environ.get("S4_BDCXR_DIR", "/tmp/bdcxr"))
    if not (d / "BDCXR-3257" / BDCXR[0]).exists():
        zp = Path(os.environ.get("S4_BDCXR_ZIP", "/tmp/BDCXR-3257.zip"))
        if not zp.exists():
            urllib.request.urlretrieve(BDCXR_URL, zp)
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.startswith("BDCXR-3257/")]
            for n in names:
                assert os.path.abspath(d / n).startswith(os.path.abspath(d)), n
            z.extractall(d, names)
    return d / "BDCXR-3257"


def dup_groups(paths, min_ncc=0.93):
    """Near-duplicate groups: union of image pairs whose normalised cross-correlation at 64 px (from the 256 px image) is >= min_ncc.
    DEVIATIONS.md 2026-11-03: the pre-registered 64-bit DCT hash (Hamming <= 6) chained 1,891 of the 3,257 images into one group; 0.93 is the
    lowest of 0.90 / 0.93 / 0.95 / 0.97 at which no group exceeds 2 images (0.90 chains 79). Chosen before any model saw the images."""
    v = np.zeros((len(paths), 64 * 64), np.float32)
    for i, p in enumerate(paths):
        im = read_gray(p)
        if im is None:
            continue  # unreadable: zero vector, correlation 0 with everything, its own group
        x = cv2.resize(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA), (64, 64), interpolation=cv2.INTER_AREA).ravel()
        v[i] = (x - x.mean()) / (x.std() + 1e-6) / 64
    near = v @ v.T >= min_ncc
    parent = list(range(len(paths)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(near, 1))):
        parent[root(i)] = root(j)
    return np.array([root(i) for i in range(len(paths))])


def grouped_folds(y, g, k=5, val_frac=0.1, seed=0):
    """Stratified, grouped k folds -> list of k split arrays ('train'/'val'/'test'). A group takes its majority class; groups are dealt to the
    fold with the fewest images of that class. Validation: whole groups of the training part, per class until >= val_frac of its images."""
    rng = np.random.default_rng(seed)
    gs = np.unique(g)
    gcls = {x: np.bincount(y[g == x], minlength=y.max() + 1).argmax() for x in gs}
    gsize = {x: int((g == x).sum()) for x in gs}
    fold_of, load = {}, np.zeros((k, y.max() + 1))
    for c in range(y.max() + 1):
        for x in rng.permutation([x for x in gs if gcls[x] == c]):
            f = int(load[:, c].argmin())
            fold_of[x], load[f, c] = f, load[f, c] + gsize[x]
    fold = np.array([fold_of[x] for x in g])
    out = []
    for f in range(k):
        s = np.where(fold == f, "test", "train").astype(object)
        for c in range(y.max() + 1):
            need, got = val_frac * ((fold != f) & (y == c)).sum(), 0
            for x in rng.permutation([x for x in gs if gcls[x] == c and fold_of[x] != f]):
                if got >= need:
                    break
                s[g == x], got = "val", got + gsize[x]
        out.append(s.astype(str))
    return out


def bdcxr():
    """BDCXR-3257 (SAP_bdcxr.md): one-hot Normal / Bacterial / Viral; 'patient' = near-duplicate group; split_f<k>.npy = the 5 folds."""
    root = bdcxr_fetch()
    rows = [(f"{c}/{p.name}", str(p), "cv", None, np.eye(len(BDCXR), dtype=np.float32)[k])
            for k, c in enumerate(BDCXR) for p in sorted((root / c).glob("*.jpg"))]
    n = np.bincount([r[4].argmax() for r in rows], minlength=3).tolist()
    assert n == [880, 1477, 900], f"BDCXR incomplete or changed: {n} images per class (expected 880, 1477, 900)"
    g = dup_groups([r[1] for r in rows])
    rows = [(n, p, s, f"g{gi}", y) for (n, p, s, _, y), gi in zip(rows, g)]
    folds = grouped_folds(np.array([r[4].argmax() for r in rows]), g)
    print("bdcxr:", len(rows), "images,", len(np.unique(g)), "duplicate groups, largest", np.bincount(g).max(), flush=True)
    return dict(rows=rows, size=256, kind="cls", folds=folds)


LOADERS = {"nih": nih, "tbx": tbx, "vin": vin, "node21": node21, "jsrt": jsrt, "bdcxr": bdcxr}


def load_qure(path, dev):
    import torch
    if path and not os.path.exists(path):
        hit = glob.glob("/kaggle/input/datasets/*/" + path.split("/kaggle/input/", 1)[-1]) + glob.glob(path.replace("/kaggle/input/", "/kaggle/input/datasets/*/"))
        path = hit[0] if hit else None
    if not path:  # public, ungated (CC BY-NC-SA 4.0); the kernel has internet
        import urllib.request
        path = "/tmp/bone_suppression.ts"
        if not os.path.exists(path):
            urllib.request.urlretrieve("https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts", path)
    return torch.jit.load(path, map_location=dev).eval()


def resolve_weights(spec):
    out = []
    for w in spec.split("+"):
        if os.path.exists(w):
            out.append(w)
            continue
        hit = glob.glob("/kaggle/input/**/" + os.path.basename(w), recursive=True)
        assert hit, f"weights {w} not found"
        out.append(hit[0])
    return "+".join(out)


def worker(a, ds, rows, g, n):
    import torch
    from .infer import load_suppressor, prep512, suppress_batch, to_u8
    S, N, out = ds["size"], len(rows), OUT_ROOT / a.tag
    names = a.arrays.split(",")
    det = ds["kind"] == "det"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_suppressor(resolve_weights(a.weights), dev) if (det or "soft" in names or "bone" in names) else None
    qm = load_qure(a.qure if a.qure != "none" else None, dev) if (det or "soft_qure" in names) and a.qure != "off" else None
    arrs = {k: np.load(out / f"{k}.npy", mmap_mode="r+") for k in names}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    part = dict(ok=[], bad=[], ann={})
    mine, t0 = list(range(g, N, n)), time.time()
    with ThreadPoolExecutor(6) as pool:
        for s in range(0, len(mine), 32):
            ids = mine[s:s + 32]
            ims = list(pool.map(lambda i: read_gray(rows[i][1]), ids))
            good = [j for j, im in enumerate(ims) if im is not None]
            part["bad"] += [rows[ids[j]][0] for j in range(len(ids)) if j not in good]
            if not good:
                continue
            x = np.stack([prep512(ims[j]) for j in good])
            b = suppress_batch(model, x, dev) if model is not None else None
            v = dict(orig=to_u8(x, S))
            if b is not None:
                v.update(soft=to_u8(x - b, S), bone=to_u8(np.clip(b, 0, 1), S))
            sq, bq = None, None
            if qm is not None:  # released CT-supervised model at its native 1024 px input (fp16 autocast: soft images agree with fp32 to 78 dB)
                sq, bq = [], []
                for j0 in range(0, len(good), 4):
                    xs = np.stack([(lambda q: (q - q.min()) / (q.max() - q.min() + 1e-10))(cv2.resize(ims[j], (1024, 1024), interpolation=cv2.INTER_AREA))
                                   for j in good[j0:j0 + 4]]).astype(np.float32)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                        o = qm(torch.from_numpy(xs)[:, None].to(dev))
                    o = (o[0] if isinstance(o, (tuple, list)) else o).float()[:, 0].cpu().numpy()
                    sq.append(xs - o)
                    bq.append(np.stack([cv2.resize(m, (512, 512), interpolation=cv2.INTER_AREA) for m in o]))
                sq, bq = np.concatenate(sq), np.concatenate(bq)
                v["soft_qure"] = to_u8(sq, S)
            for k in names:
                for jj, j in enumerate(good):
                    arrs[k][ids[j]] = v[k][jj]
            for jj, j in enumerate(good):
                i = ids[j]
                part["ok"].append(i)
                if not det:
                    continue
                boxes = [bx for bx, _ in ds["ann"].get(rows[i][0], [])]
                rec = dict(overlap=[overlap(b[jj], bx, S) for bx in boxes])
                if bq is not None:
                    cons = 0.5 * (b[jj] / max(np.percentile(b[jj], 99), 1e-6) + bq[jj] / max(np.percentile(bq[jj], 99), 1e-6))
                    rec["overlap_qure"] = [overlap(bq[jj], bx, S) for bx in boxes]
                    rec["overlap_cons"] = [overlap(cons, bx, S) for bx in boxes]
                img = dict(orig=x[jj], soft=np.clip(x[jj] - b[jj], 0, 1), clahe=clahe.apply((x[jj] * 255 + 0.5).astype(np.uint8)).astype(np.float32) / 255)
                if sq is not None:
                    img["soft_qure"] = np.clip(cv2.resize(sq[jj], (512, 512), interpolation=cv2.INTER_AREA), 0, 1)
                for k, im512 in img.items():
                    rec[f"cnr_{k}"] = [box_cnr(im512, bx) for bx in boxes]
                part["ann"][str(i)] = rec
            if (s // 32) % 50 == 0:
                print(f"[w{g}] {s}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    for arr in arrs.values():
        arr.flush()
    json.dump(part, open(out / f"part_{g}.json", "w"))
    print(f"[w{g}] done {len(part['ok'])} ok, {len(part['bad'])} unreadable", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--weights", required=True, help="final_ema.pt files joined by '+' (ensemble); names are resolved under /kaggle/input")
    ap.add_argument("--qure", default="none", help="TorchScript path, 'none' = download from Hugging Face when needed, 'off' = never run Qure")
    ap.add_argument("--arrays", default="orig,soft")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    a.tag = a.tag or a.dataset
    out = OUT_ROOT / a.tag
    ds = LOADERS[a.dataset]()
    rows = ds["rows"][: a.limit or None]
    if a.worker >= 0:
        return worker(a, ds, rows, a.worker, a.nworkers)
    import torch
    S, N, names = ds["size"], len(rows), a.arrays.split(",")
    out.mkdir(parents=True, exist_ok=True)
    print(a.dataset, "images:", N, "arrays:", names, flush=True)
    for k in names:
        m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.uint8, shape=(N, S, S))
        m.flush()
        del m
    np.save(out / "split.npy", np.array([r[2] for r in rows]))
    np.save(out / "patient.npy", np.array([r[3] for r in rows]))
    json.dump([r[0] for r in rows], open(out / "ids.json", "w"))
    if ds["kind"] == "cls":
        np.save(out / "labels.npy", np.stack([r[4] for r in rows]))
    for k, s in enumerate(ds.get("folds", [])):  # cross-validation folds (BDCXR): s4.cls --split split_f<k>.npy
        np.save(out / f"split_f{k}.npy", s[:N])
    n = max(1, torch.cuda.device_count())
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, "-m", "s4.prep", *sys.argv[1:], "--worker", str(g), "--nworkers", str(n)],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)) if torch.cuda.is_available() else os.environ) for g in range(n)]
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), f"worker failed: {rcs}"
    ok, bad, per = np.zeros(N, bool), [], {}
    for g in range(n):
        part = json.load(open(out / f"part_{g}.json"))
        ok[part["ok"]] = True
        bad += part["bad"]
        per.update(part["ann"])
        os.remove(out / f"part_{g}.json")
    json.dump(dict(ok=ok.tolist(), unreadable=bad, n_classes=ds.get("n_classes")), open(out / "status.json", "w"))
    if ds["kind"] == "det":
        recs = []
        for i, r in enumerate(rows):
            boxes = ds["ann"].get(r[0], [])
            rec = per.get(str(i), {})
            recs.append(dict(boxes=[bx for bx, _ in boxes], labels=[c for _, c in boxes], **rec))
        json.dump(recs, open(out / "ann.json", "w"))
        for k in VARIANTS:
            v = [c for rec in recs for c in rec.get(f"cnr_{k}", []) if c == c]
            if v:
                print(f"box CNR {k}: n={len(v)} mean={np.mean(v):.3f}", flush=True)
    print(f"DONE {a.dataset} ok {int(ok.sum())} bad {len(bad)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
