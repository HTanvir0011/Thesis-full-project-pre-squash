"""Image-level multi-label classifier on uint8 image arrays (DenseNet121). Identical settings for every input variant.

python -m s4.cls --data DIR --variants orig soft --seed 0 --out runs/x
DIR holds <variant>.npy uint8 [N,S,S], labels.npy float32 [N,C] (NaN = not annotated), split.npy ('train'/'val'/'test'), patient.npy.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .infer import Clahe


def batch_aug(x, gen):
    """x [B,C,S,S] float in [0,1]: random scale/rotation/shift + brightness/contrast, identical over channels of a sample."""
    B = x.shape[0]
    r = lambda: torch.rand(B, device=x.device, generator=gen)  # noqa: E731
    ang = (r() * 2 - 1) * math.radians(7)
    s = 1.0 / (0.9 + 0.2 * r())
    theta = torch.stack([torch.stack([torch.cos(ang) * s, -torch.sin(ang) * s, (r() * 2 - 1) * 0.06], 1),
                         torch.stack([torch.sin(ang) * s, torch.cos(ang) * s, (r() * 2 - 1) * 0.06], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), padding_mode="border", align_corners=False)
    return ((x - 0.5) * (0.85 + 0.3 * r()).view(B, 1, 1, 1) + 0.5 + (r() - 0.5).view(B, 1, 1, 1) * 0.15).clamp(0, 1)


def masked_bce(logit, y):
    m = ~torch.isnan(y)
    return F.binary_cross_entropy_with_logits(logit[m], y[m])


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        m = ~np.isnan(y[:, c])
        if m.sum() and 0 < np.nansum(y[m, c]) < m.sum():
            aucs.append(roc_auc_score(y[m, c], p[m, c]))
        else:
            aucs.append(float("nan"))
    return np.array(aucs)


@torch.no_grad()
def predict(model, X, idx, dev, bs=128):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.concatenate([X[v][idx[s:s + bs]][:, None] for v in X], 1)).to(dev).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--qdata", default=None, help="dir with soft_qure.npy when it lives in a separate prep output (NIH: 20 GB kernel output limit)")
    ap.add_argument("--variants", nargs="+", required=True, help="channels, e.g. orig  |  soft  |  orig soft  |  soft_qure  |  clahe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="densenet121.ra_in1k")
    ap.add_argument("--split", default="split.npy", help="split file in --data (BDCXR: split_f<k>.npy, one per cross-validation fold)")
    a = ap.parse_args()
    d, out, dev = Path(a.data), Path(a.out), "cuda"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    src = lambda v: Path(a.qdata) if v == "soft_qure" and a.qdata else d  # noqa: E731
    X = {v: Clahe(np.load(d / "orig.npy", mmap_mode="r")) if v == "clahe" else np.load(src(v) / f"{v}.npy", mmap_mode="r") for v in a.variants}
    Y = np.load(d / "labels.npy")
    split = np.load(d / a.split, allow_pickle=True)
    if (d / "status.json").exists():  # images that could not be read at prep time never enter any split
        split = np.where(np.array(json.loads((d / "status.json").read_text())["ok"]), split, "skip")
    tr, va, te = (np.nonzero(split == s)[0] for s in ("train", "val", "test"))
    model = timm.create_model(a.arch, pretrained=True, num_classes=Y.shape[1], in_chans=len(a.variants)).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    steps = a.epochs * (len(tr) // a.bs)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    Yt = torch.from_numpy(Y)
    step, best, t0, log = 0, (-1.0, None), time.time(), open(out / "log.jsonl", "w")
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            idx = np.sort(perm[s:s + a.bs])
            x = torch.from_numpy(np.concatenate([X[v][idx][:, None] for v in a.variants], 1)).to(dev).float() / 255
            y = Yt[idx].to(dev)
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = masked_bce(model(batch_aug(x, gen).contiguous(memory_format=torch.channels_last)).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
        pv = predict(model, X, va, dev)
        vauc = float(np.nanmean(macro_auc(Y[va], pv)))
        log.write(json.dumps(dict(epoch=ep, val_macro_auc=vauc, sec=round(time.time() - t0))) + "\n")
        log.flush()
        if best[1] is None or vauc > best[0]:  # model selection on the validation split only (NaN AUC: keep the first epoch until a real one)
            best = (vauc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    pt = predict(model, X, te, dev)
    np.save(out / "test_pred.npy", pt.astype(np.float16))
    np.save(out / "test_idx.npy", te)
    np.save(out / "test_labels.npy", Y[te])  # with the patient ids, the result dir is self-contained for the final statistics
    if (d / "patient.npy").exists():
        np.save(out / "test_patient.npy", np.load(d / "patient.npy", allow_pickle=True)[te])
    auc = macro_auc(Y[te], pt)
    json.dump(dict(val_macro_auc=best[0], test_macro_auc=float(np.nanmean(auc)), per_class=auc.tolist(), variants=a.variants,
                   seed=a.seed, epochs=a.epochs, split=a.split), open(out / "result.json", "w"))
    print(json.dumps(dict(val=best[0], test=float(np.nanmean(auc)))))


if __name__ == "__main__":
    main()
