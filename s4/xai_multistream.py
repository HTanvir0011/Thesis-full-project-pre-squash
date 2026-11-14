"""Multi-stream attribution and the pointing game (DESIGN_v3.md §5, SAP_v3.md H10).

python -m s4.xai_multistream --run RUN_DIR --data NIH_PREP_DIR --boxes BBox_List_2017.csv --out xai.json [--maps 24] [--limit N]
RUN_DIR holds model.pt from s4/cls_hybrid.py. Only boxed images of the TEST split are used (a training image could be memorised).
Per (image, box class c): Grad-CAM++ on the last encoder map of every stream, the decoder attention of class c (summed over streams),
Integrated Gradients on the X, S, B channels (blurred-image baseline: the model standardises each image, so a black baseline has no
contrast and its path carries no signal), and the routing rho[c, :]. A map hits when its arg-max lies inside a box of class c widened by
--tol px (at the prep size). --maps saves the first N maps (NIH images only; licence allows figures) for tools/generate_all_thesis_figures.py.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cls_hybrid import MultiStream, load_x
from .prep import NIH

BOX_NAME = {"Infiltrate": "Infiltration"}  # BBox_List_2017.csv spells one class differently from Data_Entry_2017.csv


def load_model(run, dev):
    st = torch.load(Path(run) / "model.pt", map_location="cpu")
    m = MultiStream(st["arch"], st["n_cls"], st["config"], size=st["size"], pretrained=False)
    m.load_state_dict(st["state"])
    return m.to(dev).eval(), st


def nih_boxes(csv_path, ids, size, native=1024):
    """{prep index: [(class index, x0, y0, x1, y1) in prep px]} from BBox_List_2017.csv (x, y, w, h at the native 1024 px)."""
    pos, out, k = {n: i for i, n in enumerate(ids)}, {}, size / native
    for r in csv.reader(open(csv_path)):
        if r[0] not in pos:
            continue  # header row and images outside the prepared set
        c = NIH.index(BOX_NAME.get(r[1], r[1]))
        x, y, w, h = (float(v) * k for v in r[2:6])
        out.setdefault(pos[r[0]], []).append((c, x, y, x + w, y + h))
    return out


def gradcam_pp(feat, grad):
    """Grad-CAM++ (Chattopadhyay et al. 2018) for one map: feat, grad [ch, g, g] -> cam [g, g] >= 0."""
    g2, g3 = grad ** 2, grad ** 3
    a = g2 / (2 * g2 + feat.sum((1, 2), keepdim=True) * g3 + 1e-7)
    a = torch.where(grad != 0, a, torch.zeros_like(a))
    w = (a * F.relu(grad)).sum((1, 2))
    return F.relu((w[:, None, None] * feat).sum(0))


def blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def integrated_gradients(model, x, c, steps=32, sigma=8.0):
    """IG of logit c for x [1,3,S,S] from a blurred baseline -> attribution [3,S,S] (channels X, S, B)."""
    base = blur(x, sigma)
    total = torch.zeros_like(x)
    for a in (torch.arange(steps, dtype=x.dtype) + 0.5) / steps:  # midpoint rule
        xi = (base + a * (x - base)).requires_grad_(True)
        total += torch.autograd.grad(model(xi)[0, c], xi)[0]
    return ((x - base) * total / steps)[0]


def explain(model, x, c):
    """x [1,3,S,S] -> dict of maps at the prep size and the routing of class c."""
    S, nS = x.shape[-1], len(model.streams)
    feats = []
    hook = model.enc.register_forward_hook(lambda m, i, o: feats.append(o[-1]))
    try:
        logit = model(x)[0, c]
    finally:
        hook.remove()
    grad = torch.autograd.grad(logit, feats[0])[0]
    up = lambda m: F.interpolate(m[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]  # noqa: E731
    maps = {f"cam_{i}": up(gradcam_pp(feats[0][i].detach(), grad[i])) for i in range(nS)}
    maps["attn"] = up(model.attn[0, c].sum(0).detach().float())
    ig = integrated_gradients(model, x, c).detach()
    maps.update({f"ig_{k}": ig[j].abs() for j, k in enumerate("XSB")})
    rho = model.routing()[0, c].detach().float().cpu().numpy()
    return {k: v.float().cpu().numpy() for k, v in maps.items()}, rho


def hit(m, boxes, tol):
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return any(x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol for x0, y0, x1, y1 in boxes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--maps", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, st = load_model(a.run, dev)
    d = Path(a.data)
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("orig", "soft")}
    split = np.load(d / "split.npy", allow_pickle=True)
    boxes = nih_boxes(a.boxes, json.load(open(d / "ids.json")), st["size"])
    items = sorted(i for i in boxes if split[i] == "test")[: a.limit or None]
    rows, saved = [], {}
    for i in items:
        x = load_x(arrays, [i], st["size"], dev)
        for c in sorted({b[0] for b in boxes[i]}):
            bx = [b[1:] for b in boxes[i] if b[0] == c]
            maps, rho = explain(model, x, c)
            ig_tot = maps["ig_X"] + maps["ig_S"] + maps["ig_B"]
            rows.append(dict(image=int(i), cls=NIH[c], rho=rho.tolist(), hits={k: bool(hit(v, bx, a.tol)) for k, v in maps.items()} | {"ig_all": bool(hit(ig_tot, bx, a.tol))},
                             ig_share=[float(maps[f"ig_{k}"].sum() / (ig_tot.sum() + 1e-12)) for k in "XSB"]))
            if len(saved) < a.maps:
                saved[f"{i}_{c}"] = dict(maps, boxes=np.array(bx, np.float32), x=x[0].cpu().numpy())
    res = dict(run=str(a.run), config=st["config"], arch=st["arch"], tol=a.tol, n_targets=len(rows), rows=rows)
    for k in rows[0]["hits"] if rows else []:
        res[f"hit_rate_{k}"] = float(np.mean([r["hits"][k] for r in rows]))
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez_compressed(Path(a.out).with_suffix(".maps.npz"), **{f"{k}__{m}": v for k, d_ in saved.items() for m, v in d_.items()})
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
