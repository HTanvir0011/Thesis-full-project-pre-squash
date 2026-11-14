"""Cross-modal heat-map agreement (DESIGN_v4.md section 4, SAP_v4.md H14).

python -m s4.xm_xai --data CT_PREP --qp QP_PACK --ct_run RUNS/ct_xm_ct_s0 --xr_run RUNS/qp_xm_xraykd_s0 --out DIR [--limit N]

CT subset test scans, and for each the shared findings it is positive for: Grad-CAM++ of the CT teacher (s4/xm.py CTNet) on every slice,
summed along the anteroposterior axis (the rows of a radiological slice) into a frontal map [slices, columns] cropped to the CT lung box;
Grad-CAM++ of an xray-stage model on the paired projection, cropped to the box of the projection's lung component L (pixels above 10 % of
its maximum). Both are resized to G x G. Per scan also the same crops of the CT attenuation frame (ap.npy) and of the projection: their
correlation (frame_r; with the projection mirrored, frame_r_flip) shows whether the two frames line up before any map is compared.
Writes DIR/maps.npz and DIR/result.json; tools/xm_stats.py correlates matched and mismatched maps.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .cls_hybrid import MultiStream
from .xai_multistream import gradcam_pp
from .xm import CTNet, as3, ct_batch, qp_load

G = 32  # side of the common frame


def load(run, dev):
    """model.pt of s4/xm.py (CT stage: CTNet; xray stage) or s4/cls_hybrid.py -> model in eval mode, its saved settings."""
    st = torch.load(Path(run) / "model.pt", map_location="cpu")
    m = CTNet(st["arch"], st["n_cls"], st["k"], st["size"], pretrained=False) if "k" in st else \
        MultiStream(st["arch"], st["n_cls"], st["config"], size=st["size"], pretrained=False)
    m.load_state_dict(st["state"])
    return m.to(dev).eval(), st


def cams(model, x, c):
    """Grad-CAM++ of logit c of the one case in x on the last encoder map, one map per encoder image (CTNet: per slice) -> [n, g, g]."""
    feats = []
    hook = model.enc.register_forward_hook(lambda m, i, o: feats.append(o[-1]))
    try:
        logit = model(x)[0, c]
    finally:
        hook.remove()
    grad = torch.autograd.grad(logit, feats[0])[0]
    return torch.stack([gradcam_pp(f.detach(), g) for f, g in zip(feats[0], grad)])


def upsampled(model, x, c):
    """cams() bilinearly resized to the input size -> numpy [n, S, S]."""
    S = x.shape[-1]
    return F.interpolate(cams(model, x, c)[:, None], size=(S, S), mode="bilinear", align_corners=False)[:, 0].detach().cpu().numpy()


def to_frame(m):
    """Cropped map [rows, cols] -> [G, G]: area averaging along an axis that shrinks, linear along one that grows."""
    m = np.ascontiguousarray(m, np.float32)
    m = cv2.resize(m, (G, m.shape[0]), interpolation=cv2.INTER_AREA if m.shape[1] > G else cv2.INTER_LINEAR)
    return cv2.resize(m, (G, G), interpolation=cv2.INTER_AREA if m.shape[0] > G else cv2.INTER_LINEAR)


def lung_box(L, frac=0.1):
    """(r0, r1, c0, c1), inclusive, of the pixels of a lung component image above frac of its maximum (whole image when empty)."""
    r, c = np.nonzero(L > frac * L.max())
    return (r.min(), r.max(), c.min(), c.max()) if len(r) else (0, L.shape[0] - 1, 0, L.shape[1] - 1)


def pearson(a, b):
    a, b = np.ravel(a) - np.mean(a), np.ravel(b) - np.mean(b)
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="s4/ct_prep.py output")
    ap.add_argument("--qp", required=True, help="packed Qure projections (qp.bin, qp_index.npy)")
    ap.add_argument("--ct_run", required=True)
    ap.add_argument("--xr_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ct, att_all = np.load(d / "ct.npy", mmap_mode="r"), np.load(d / "ap.npy", mmap_mode="r")
    box, Y, rows = np.load(d / "lung_box.npy"), np.load(d / "labels.npy"), np.load(d / "rows.npy")
    split = np.load(d / "split.npy", allow_pickle=True)
    ok = np.array([bool(s and s["ok"]) for s in json.loads((d / "status.json").read_text())]) if (d / "status.json").exists() else np.ones(len(Y), bool)
    test = np.nonzero((split == "test") & ok & (Y.max(1) > 0))[0][: a.limit or None]
    ct_m, _ = load(a.ct_run, dev)
    xr_m, st = load(a.xr_run, dev)
    img, L = qp_load(a.qp, rows[test], st["size"]), qp_load(a.qp, rows[test], st["size"], col=2)
    maps, frames = dict(ct=[], xr=[], scan=[], cls=[]), dict(att=[], prj=[], frame_r=[], frame_r_flip=[])
    for n, j in enumerate(test):
        r0, r1, c0, c1 = lung_box(L[n])
        x0, x1 = box[j]
        att, prj = to_frame(att_all[j][:, x0:x1 + 1]), to_frame(img[n][r0:r1 + 1, c0:c1 + 1])
        frames["att"].append(att)
        frames["prj"].append(prj)
        frames["frame_r"].append(pearson(att, prj))
        frames["frame_r_flip"].append(pearson(att, prj[:, ::-1]))
        v, x = ct_batch(ct, [j], dev), as3(img, [n], dev)
        for c in np.nonzero(Y[j] > 0)[0]:
            maps["ct"].append(to_frame(upsampled(ct_m, v, c).sum(1)[:, x0:x1 + 1]))  # sum over the anteroposterior rows of every slice
            maps["xr"].append(to_frame(upsampled(xr_m, x, c)[0][r0:r1 + 1, c0:c1 + 1]))
            maps["scan"].append(int(j))
            maps["cls"].append(int(c))
    np.savez_compressed(out / "maps.npz", test=test, names=np.array(json.loads((d / "names.json").read_text()))[test],
                        **{k: np.array(v, np.float32 if k in ("ct", "xr", "att", "prj", "frame_r", "frame_r_flip") else np.int64)
                           for k, v in (maps | frames).items()})
    res = dict(ct_run=str(a.ct_run), xr_run=str(a.xr_run), n_scans=int(len(test)), n_maps=len(maps["cls"]),
               frame_r=float(np.mean(frames["frame_r"])) if len(test) else None,
               frame_r_flip=float(np.mean(frames["frame_r_flip"])) if len(test) else None)
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
