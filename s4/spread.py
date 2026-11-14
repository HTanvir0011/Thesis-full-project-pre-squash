"""Per-image ensemble spread (SAP_bdcxr.md E3): SD over the K member bone predictions at 512 px (s4/uncertainty.py), averaged over the lung
field (the eroded PSPNet mask of s4/audit_real.py, nearest-resized to 512) and over the whole image. There is no truth: this shows only whether
the ensemble disagrees more away from the training domain. NIH (second score of the SAP_v3 selective classification, entropy + spread): the
test split of a prep output, read from the raw images, with the extra column 'index' = prep index (joins test_idx of the runs).

python -m s4.spread --weights a.pt+b.pt+c.pt --dataset bdcxr --root DIR/BDCXR-3257 --out spread_bdcxr.csv      (szch: --root data/szch)
python -m s4.spread --weights a.pt+b.pt+c.pt --dataset nih --root NIH_RAW --prep PREP/nih --out spread_nih.csv
    [--limit N (smoke test)] [--worker G --nworkers N (one process per GPU, whole batches dealt out; s4.spread.merge joins the parts)]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .audit_real import lung_masks
from .eval_real import load_pairs
from .infer import load_suppressor, prep512
from .prep import read_gray
from .uncertainty import ensemble

BATCH = 32


def images(dataset, root, prep=None, limit=0, worker=0, nworkers=1):
    """-> (name, image, extra columns). nih: the test split of the prep output `prep` in index order (the first `limit`; of those, every
    nworkers-th batch from batch `worker`: one process per GPU), read from root/images_*/images/<name>; extra = {index: prep index}."""
    if dataset == "nih":
        ids = json.loads((Path(prep) / "ids.json").read_text())
        files = {p.name: p for p in Path(root).glob("images_*/images/*.png")}
        assert files, f"no images_*/images/*.png under {root}"  # a wrong root stops here, not as one 'missing' line per test image
        test = np.flatnonzero(np.load(Path(prep) / "split.npy") == "test")[: limit or None]
        # whole batches dealt out: with no file missing, each part forms the batches a single process would (same fp16 values)
        for i in test[np.arange(len(test)) // BATCH % nworkers == worker]:
            p = files.get(ids[i])
            im = read_gray(p) if p else None
            if im is None:
                print("missing" if p is None else "unreadable", ids[i], flush=True)
                continue
            yield ids[i], im, dict(index=int(i))
        return
    if dataset == "szch":
        yield from ((name, x, {}) for name, x, _ in load_pairs(root))
        return
    for p in sorted(Path(root).glob("*/*.jpg")):
        im = read_gray(p)
        if im is None:
            print("unreadable", p.name, flush=True)
            continue
        yield f"{p.parent.name}/{p.name}", im, {}


def write(rows, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def merge(parts, out):
    """NIH part CSVs (one per --worker) -> one CSV in prep index order, values copied as written -> number of rows."""
    rows = []
    for p in parts:
        with open(p, newline="") as f:
            rows += csv.DictReader(f)
    rows.sort(key=lambda r: int(r["index"]))
    assert len({r["index"] for r in rows}) == len(rows), "an image is in two parts"
    write(rows, out)
    return len(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True, choices=["bdcxr", "szch", "nih"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--prep", help="nih: prep output folder (ids.json, split.npy)")
    ap.add_argument("--limit", type=int, default=0, help="nih: first N test images only (smoke test)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nih = a.dataset == "nih"
    if nih != bool(a.prep) or not 0 <= a.worker < a.nworkers or (not nih and (a.limit or a.nworkers > 1)):
        ap.error("--prep (required), --limit and --worker G --nworkers N (0 <= G < N) belong to --dataset nih only")
    members = load_suppressor(a.weights, "cuda").members
    assert len(members) > 1, "spread needs an ensemble (weights joined by '+')"
    rows, batch = [], []

    def flush():
        x = np.stack([prep512(im) for _, im, _ in batch])
        with torch.autocast("cuda", dtype=torch.float16):
            preds = torch.stack([m(torch.from_numpy(x)[:, None].cuda()).float()[:, 0] for m in members]).cpu().numpy()  # [K,B,512,512]
        lungs = lung_masks(list(x), "cuda")
        for j, (name, _, extra) in enumerate(batch):
            sd = ensemble(preds[:, j])[1]
            lung = cv2.resize(lungs[j].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
            rows.append(dict(dataset=a.dataset, image=name, lung_frac=float(lung.mean()),
                             spread_lung=float(sd[lung].mean()) if lung.any() else float("nan"), spread_all=float(sd.mean()), **extra))
        batch.clear()
    for item in images(a.dataset, a.root, a.prep, a.limit, a.worker, a.nworkers):
        batch.append(item)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    write(rows, a.out)
    print(a.dataset, len(rows), "images, median lung spread", float(np.nanmedian([r["spread_lung"] for r in rows])), flush=True)


if __name__ == "__main__":
    main()
