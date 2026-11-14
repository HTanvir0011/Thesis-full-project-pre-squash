"""CT-RATE subset for the cross-modality study (DESIGN_v4.md section 2): select one reconstruction per patient whose Qure.ai projection
is in results/qp_meta.csv, download it from Hugging Face inside a Kaggle kernel, and reduce it to 2.5D model input.

python -m s4.ct_prep select --meta results/qp_meta.csv --n_train 1600 --out s4/data/ct_subset.json
python -m s4.ct_prep build --subset ct_subset.json --meta results/qp_meta.csv --out DIR [--metadata train_metadata.csv validation_metadata.csv]
  [--limit N] [--workers 4]
  (build reads the Hugging Face token from the environment variable named by --token_env, default HF_TOKEN; it is never written or printed)

Volumes in s4/data/ct_skip.txt (CT-RATE's no-chest lists; not head-first supine) are never selected. CT-RATE NIfTI files keep the DICOM
pixel order under an identity affine, so the array is used as stored: axis 0 runs toward the patient's left, axis 1 toward posterior,
axis 2 toward the head (checked on train_1_a_1 against its Qure projection; ImageOrientationPatient is [1,0,0,0,1,0] for every selected
volume). Per volume (images in radiological view: anterior at the top, the patient's right on the left): K axial slices spread evenly
from the top to the bottom of the lung extent, three HU windows as uint8
channels, cropped to a square around the body and resized to SIZE px; the lung box (first and last column holding lung in any kept slice)
and the anteroposterior attenuation profile of the kept slices [K, SIZE], the frontal frame in which CT heat maps are compared with the
paired projection. Outputs in DIR: ct.npy [N,K,3,S,S] uint8, ap.npy [N,K,S] float16, lung_box.npy [N,2], labels.npy [N,6], split.npy,
names.json, rows.npy (row of qp_meta.csv), status.json (ok flag and message per volume).
"""
import argparse
import csv
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .xm_labels import from_meta

REPO = "ibrahimhamamci/CT-RATE"
WINDOWS = ((-1350.0, 150.0), (-160.0, 240.0), (-500.0, 1300.0))  # lung, mediastinum, bone
K, SIZE = 24, 256
SKIP = Path(__file__).parent / "data" / "ct_skip.txt"


def skip_set(path=SKIP):
    """Volume names of s4/data/ct_skip.txt (first word of each line; '#' lines are comments)."""
    return {line.split()[0] for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")}


def select(meta_rows, n_train, seed=0, skip=()):
    """One reconstruction per patient (the first volume name not in skip): every val/test patient and n_train random training patients."""
    first = {}
    for i, r in enumerate(meta_rows):
        p = r["patient"]
        if r["volume_name"] not in skip and (p not in first or r["volume_name"] < meta_rows[first[p]]["volume_name"]):
            first[p] = i
    rng = np.random.default_rng(seed)
    train = sorted(p for p, i in first.items() if meta_rows[i]["split"] == "train")
    keep = set(rng.choice(train, size=min(n_train, len(train)), replace=False).tolist()) | {p for p, i in first.items() if meta_rows[i]["split"] != "train"}
    return [dict(row=first[p], volume_name=meta_rows[first[p]]["volume_name"], patient=p, split=meta_rows[first[p]]["split"]) for p in sorted(keep)]


def hf_path(name):
    """CT-RATE file of a volume: train_1_a_1 -> dataset/train/train_1/train_1_a/train_1_a_1.nii.gz (valid_* volumes likewise)."""
    p = name.split("_")
    return f"dataset/{p[0]}/{'_'.join(p[:2])}/{'_'.join(p[:3])}/{name}.nii.gz"


def load_hu(path, slope=1.0, intercept=0.0):
    """CT-RATE NIfTI -> float32 HU array as stored (axis 0 toward the patient's left, 1 toward posterior, 2 toward the head); the
    identity affine of these files does not describe the anatomy, so nothing is reoriented."""
    import nibabel as nib  # in the Kaggle image; not needed for the array functions below
    return np.clip(np.asarray(nib.load(str(path)).dataobj, np.float32) * slope + intercept, -1024, 3071)


def view(a):
    """[axis 0, axis 1] slice -> radiological image [rows anterior..posterior, cols patient right..left]."""
    return np.asarray(a).T


def check_hu(hu):
    """Raise unless the volume looks like HU: air around the body near -1000 and soft tissue near 0."""
    air, body = np.mean(hu < -900), hu[hu > -500]
    if air < 0.05 or body.size == 0 or not -300 < np.median(body) < 300:
        raise ValueError(f"implausible HU: air fraction {air:.3f}, body median {np.median(body) if body.size else float('nan'):.0f}")


def process(hu, k=K, size=SIZE):
    """HU volume (load_hu axes) -> (slices [k,3,size,size] uint8, ap [k,size] float16, lung_box [x0, x1] in output px)."""
    from scipy.ndimage import binary_fill_holes
    check_hu(hu)
    body = hu > -500
    lung = np.stack([binary_fill_holes(body[:, :, z]) for z in range(hu.shape[2])], 2) & (hu < -400)
    area = lung.sum((0, 1))
    zs = np.nonzero(area > 0.005 * hu.shape[0] * hu.shape[1])[0]
    lo, hi = (zs.min(), zs.max()) if len(zs) else (0, hu.shape[2] - 1)
    zk = np.round(np.linspace(hi, lo, k)).astype(int)  # superior first: slice order = rows of the frontal frame
    bmask = view(body[:, :, zk].any(2))
    rows, cols = np.nonzero(bmask)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    half = int(np.ceil(0.525 * max(rows.max() - rows.min(), cols.max() - cols.min())))
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half

    def crop(img, fill):
        out = np.full((2 * half, 2 * half), fill, img.dtype)
        sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(img.shape[0], y0 + 2 * half), min(img.shape[1], x0 + 2 * half)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)

    slices = np.zeros((k, 3, size, size), np.uint8)
    ap = np.zeros((k, size), np.float32)
    lcols = np.zeros(size, bool)
    for j, z in enumerate(zk):
        s = crop(view(hu[:, :, z]), np.float32(-1024))
        for c, (a, b) in enumerate(WINDOWS):
            slices[j, c] = np.clip(np.round((s - a) / (b - a) * 255), 0, 255).astype(np.uint8)
        ap[j] = np.clip((s + 1000) / 1000, 0, 3).sum(0)
        lcols |= crop(view(lung[:, :, z]).astype(np.float32), np.float32(0)).max(0) > 0.5
    xs = np.nonzero(lcols)[0]
    box = np.array([xs.min(), xs.max()] if len(xs) else [0, size - 1], np.int16)
    return slices, ap.astype(np.float16), box


def rescale(metadata_csvs):
    """{volume name: (slope, intercept)} from CT-RATE metadata CSVs (train and validation; columns found case-insensitively)."""
    out = {}
    for f in metadata_csvs or ():
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        key = {c.lower().replace("_", ""): c for c in rows[0]} if rows else {}
        n, s, i = key.get("volumename"), key.get("rescaleslope"), key.get("rescaleintercept")
        out.update({r[n].removesuffix(".nii.gz"): (float(r[s] or 1), float(r[i] or 0)) for r in rows} if n and s and i else {})  # names end in .nii.gz
    return out


def build(subset, out, meta_rows, metadata_csvs=None, token=None, workers=4, k=K, size=SIZE, fetch=None, deadline=None):
    """Download (fetch(name) -> local path; default: Hugging Face) and reduce every volume of `subset`; resumable through status.json.
    Volumes not started by `deadline` (time.time() value) are marked 'deadline' and left for a resumed build."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(subset)
    mode = "r+" if (out / "ct.npy").exists() else "w+"
    ct = np.lib.format.open_memmap(out / "ct.npy", mode, np.uint8, (n, k, 3, size, size))
    ap = np.lib.format.open_memmap(out / "ap.npy", mode, np.float16, (n, k, size))
    box = np.lib.format.open_memmap(out / "lung_box.npy", mode, np.int16, (n, 2))
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else [None] * n
    np.save(out / "labels.npy", from_meta([meta_rows[s["row"]] for s in subset]))
    np.save(out / "split.npy", np.array([s["split"] for s in subset]))
    np.save(out / "rows.npy", np.array([s["row"] for s in subset]))
    (out / "names.json").write_text(json.dumps([s["volume_name"] for s in subset]))
    scale = rescale(metadata_csvs)
    tmp = Path(tempfile.mkdtemp(prefix="ctrate_"))
    if fetch is None:
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return hf_hub_download(REPO, hf_path(name), repo_type="dataset", token=token, local_dir=tmp)

    def one(j):
        name = subset[j]["volume_name"]
        if deadline and time.time() > deadline:  # a Kaggle session that times out saves no output at all
            return j, None, "deadline"
        try:
            p = fetch(name)
            try:
                return j, process(load_hu(p, *scale.get(name, (1.0, 0.0))), k, size), "ok"
            finally:
                if str(p).startswith(str(tmp)):
                    os.remove(p)
        except Exception as e:  # one bad volume must not end a 4-hour kernel: it is marked and skipped by every later step
            return j, None, f"{type(e).__name__}: {str(e)[:200]}"

    todo = [j for j in range(n) if not (status[j] and status[j]["ok"])]
    with ThreadPoolExecutor(workers) as pool:
        for c, (j, res, msg) in enumerate(pool.map(one, todo)):
            if res is not None:
                ct[j], ap[j], box[j] = res
            status[j] = dict(ok=res is not None, msg=msg)
            if c % 50 == 49 or c == len(todo) - 1:
                ct.flush(), ap.flush(), box.flush()
                (out / "status.json").write_text(json.dumps(status))
                print(f"{c + 1}/{len(todo)} volumes, {sum(bool(s and s['ok']) for s in status)} ok", flush=True)
    (out / "status.json").write_text(json.dumps(status))
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--meta", required=True)
    s.add_argument("--n_train", type=int, default=1600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--skip", default=str(SKIP))
    s.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--subset", required=True)
    b.add_argument("--meta", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--metadata", nargs="*", default=None, help="CT-RATE metadata CSVs with RescaleSlope/RescaleIntercept per volume")
    b.add_argument("--token_env", default="HF_TOKEN")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--hours", type=float, default=0, help="start no volume after this many hours (0: no limit)")
    a = ap.parse_args(argv)
    meta = list(csv.DictReader(open(a.meta, encoding="utf-8")))
    if a.cmd == "select":
        sel = select(meta, a.n_train, a.seed, skip_set(a.skip))
        Path(a.out).write_text(json.dumps(sel))
        print(len(sel), "volumes:", {sp: sum(x["split"] == sp for x in sel) for sp in ("train", "val", "test")})
        return
    subset = json.loads(Path(a.subset).read_text())[: a.limit or None]
    st = build(subset, a.out, meta, a.metadata, os.environ.get(a.token_env), a.workers, deadline=time.time() + 3600 * a.hours if a.hours else None)
    print(sum(bool(x and x["ok"]) for x in st), "of", len(st), "volumes ok")


if __name__ == "__main__":
    main()
