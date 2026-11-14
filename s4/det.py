"""Faster R-CNN (ResNet50-FPN v2) on uint8 image arrays; identical settings for every input variant.

python -m s4.det --data DIR --variant orig|soft|both --seed 0 --epochs 12 --out runs/x
DIR: orig.npy / soft.npy / bone.npy / soft_qure.npy uint8 [N,S,S]; ann.json (per image {boxes:[[x1,y1,x2,y2]], labels:[0-based], overlap*:[frac]});
split.npy ('train'/'val'/'test'). Channels: orig=(o,o,o), soft=(s,s,s), soft_qure=(q,q,q), clahe=(c,c,c) (on the fly from orig), both=(o,s,b).
"""
import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import froc
from .infer import Clahe


SINGLE = ("orig", "soft", "soft_qure", "clahe")


def make_input(arrs, idx, variant, dev):
    g = lambda k: torch.from_numpy(np.asarray(arrs[k][idx])).to(dev).float() / 255  # noqa: E731
    if variant in SINGLE:
        x = g(variant)
        return torch.stack([x, x, x], 0)
    return torch.stack([g("orig"), g("soft"), g("bone")], 0)


def build(n_classes):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                                                                 min_size=512, max_size=512, box_detections_per_img=100, box_score_thresh=0.01)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_classes + 1)
    return m


@torch.no_grad()
def predict(model, arrs, idx, variant, dev):  # dev: "cuda" (or "cpu" for local smoke tests)
    model.eval()
    out = []
    for i in idx:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            r = model([make_input(arrs, i, variant, dev)])[0]
        out.append((r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy() - 1))
    return out


def evaluate(preds, ann, idx, n_classes):
    gts = [(np.array(ann[i]["boxes"], float).reshape(-1, 4), np.array(ann[i]["labels"])) for i in idx]
    return dict(cpm=froc.froc(preds, gts)["cpm"], map50=froc.average_precision(preds, gts, 0.5, n_classes)), gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", required=True, choices=["orig", "soft", "both", "soft_qure", "clahe"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n_classes", type=int, default=0, help="0 = read from status.json written by s4.prep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    d, out, dev = Path(a.data), Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    need = {"orig", "soft", "bone"} if a.variant == "both" else {"orig", a.variant} - {"clahe"}
    arrs = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in need}
    if a.variant == "clahe":
        arrs["clahe"] = Clahe(arrs["orig"])
    ann = json.loads((d / "ann.json").read_text())
    split = np.load(d / "split.npy", allow_pickle=True)
    if (d / "status.json").exists():  # images that could not be read at prep time never enter any split
        split = np.where(np.array(json.loads((d / "status.json").read_text())["ok"]), split, "skip")
    tr, va, te = (np.nonzero(split == s)[0] for s in ("train", "val", "test"))
    if not a.n_classes:
        a.n_classes = json.loads((d / "status.json").read_text())["n_classes"]
    model = build(a.n_classes).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    steps = a.epochs * (len(tr) // a.bs)
    step, best, t0, log = 0, (-1.0, None), time.time(), open(out / "log.jsonl", "w")
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            imgs, tgts = [], []
            for i in perm[s:s + a.bs]:
                x = make_input(arrs, i, a.variant, dev)
                b = torch.tensor(ann[i]["boxes"], dtype=torch.float32, device=dev).reshape(-1, 4)
                if rng.random() < 0.5:  # horizontal flip
                    x = x.flip(-1)
                    W = x.shape[-1]
                    b = torch.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1) if len(b) else b
                x = ((x - 0.5) * float(rng.uniform(0.85, 1.15)) + 0.5 + float(rng.uniform(-0.07, 0.07))).clamp(0, 1)
                imgs.append(x)
                tgts.append(dict(boxes=b, labels=torch.tensor(ann[i]["labels"], dtype=torch.int64, device=dev) + 1))
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * step / steps))
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            step += 1
        m, _ = evaluate(predict(model, arrs, va, a.variant, dev), ann, va, a.n_classes)
        log.write(json.dumps(dict(epoch=ep, loss=float(loss), val=m, sec=round(time.time() - t0))) + "\n")
        log.flush()
        if best[1] is None or m["cpm"] > best[0]:  # model selection on validation only (NaN score: keep the first epoch until a real one)
            best = (m["cpm"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    preds = predict(model, arrs, te, a.variant, dev)
    m, gts = evaluate(preds, ann, te, a.n_classes)
    pickle.dump(dict(test_idx=te, preds=preds, gts=gts, **{k: [ann[i].get(k, []) for i in te] for k in ("overlap", "overlap_qure", "overlap_cons")}),
                open(out / "test_preds.pkl", "wb"))
    json.dump(dict(val_cpm=best[0], test=m, variant=a.variant, seed=a.seed, epochs=a.epochs), open(out / "result.json", "w"))
    print(json.dumps(dict(val_cpm=best[0], test=m)))


if __name__ == "__main__":
    main()
