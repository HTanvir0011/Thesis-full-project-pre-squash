"""Load rendered CTs (npz + json) onto the GPU as one tensor bank; build lesion masks and the lesion-patch library."""
import json
from pathlib import Path

import numpy as np
import torch


def load_bank(render_dirs, names, device="cuda"):
    """-> dict(a=[N,3,512,512] fp16 (bone, nl, lung g/cm^2), lesion=[N,512,512] uint8, patches=list[float32 tensors], names, meta)."""
    index = {}
    for d in render_dirs:
        for p in Path(d).glob("*.npz"):
            index[p.stem] = p
    names = [n for n in names if n in index]
    a = torch.empty((len(names), 3, 512, 512), dtype=torch.float16, device=device)
    lesion = torch.zeros((len(names), 512, 512), dtype=torch.uint8, device=device)
    patches, metas = [], []
    for i, n in enumerate(names):
        z = np.load(index[n])
        a[i] = torch.from_numpy(np.stack([z["a_bone"], z["a_nl"], z["a_lung"]]).astype(np.float16)).to(device)
        meta = json.loads(index[n].with_suffix(".json").read_text())
        pt, pb = meta["pad_top_rows"], meta["pad_bottom_rows"]  # rows beyond the scan were edge-replicated at render time: make them air
        a[i, :, :pt] = 0
        if pb:
            a[i, :, 512 - pb:] = 0
        for nod in meta["nodules"]:
            p = z[f"nod{nod['k']}"]
            h, w = p.shape
            sl = (i, slice(nod["y0"], nod["y0"] + h), slice(nod["x0"], nod["x0"] + w))
            lesion[sl] |= torch.from_numpy((p > 0.02).astype(np.uint8)).to(device)
            patches.append(dict(patch=torch.from_numpy(p).to(device), diam_mm=nod["diam_mm"], ct=i))
        metas.append(meta)
    return dict(a=a, lesion=lesion, patches=patches, names=names, meta=metas)
