"""Multi-stream classifier (DESIGN_v3.md §2): one shared encoder runs on each stream, a class-query cross-attention decoder reads the
spatial tokens of all streams. The attention mass of class c on stream s is the measured routing rho[c, s].

python -m s4.cls_hybrid --data DIR --config hybrid --arch densenet121.ra_in1k --seed 0 --out runs/x
DIR holds orig.npy, soft.npy uint8 [N,S,S], labels.npy float32 [N,C] (NaN = not annotated), split.npy, patient.npy (as s4/cls.py).
The bone stream is B = clip(orig - soft): prep stores soft = x - b and orig = x on the same uint8 scale, so no extra array is needed.
--config: orig | soft | early (RGB <- [X, S, B], one stream) | hybrid (streams X, S, B) | ctrl (hybrid network, X in all three streams).
--arch: a timm model name, hf:microsoft/rad-dino, or bclip (BiomedCLIP ViT-B/16 vision tower; needs open_clip_torch).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls import batch_aug, macro_auc, masked_bce
from .model import HFViT

CONFIGS = {"orig": [["X"] * 3], "soft": [["S"] * 3], "early": [["X", "S", "B"]],
           "hybrid": [["X"] * 3, ["S"] * 3, ["B"] * 3], "ctrl": [["X"] * 3, ["X"] * 3, ["X"] * 3]}
SOURCES = ("X", "S", "B")


class BClip(nn.Module):
    """BiomedCLIP vision trunk (timm ViT-B/16, 224 px) -> last-layer patch grid, NCHW."""

    def __init__(self, pretrained):
        super().__init__()
        import open_clip
        name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.t = (open_clip.create_model_from_pretrained(name)[0] if pretrained else open_clip.create_model(name)).visual.trunk
        self.num_features = self.t.num_features

    def forward(self, x):
        tok = self.t.forward_features(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))[:, self.t.num_prefix_tokens:]
        return [tok.transpose(1, 2).reshape(len(x), -1, 14, 14)]


class NCHW(nn.Module):
    """timm Swin features_only returns NHWC maps (as s4/model.py notes): permute to NCHW so hooks and the projection see one layout."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return [f.permute(0, 3, 1, 2).contiguous() for f in self.m(x)]


def encoder(arch, size, pretrained):
    """-> (module returning a list of NCHW maps whose last is used, channels of that map)."""
    if arch.startswith("hf:"):
        m = HFViT(arch[3:], pretrained, layers=(12,), size=224)
        return m, m.feature_info.channels()[-1]
    if arch == "bclip":
        m = BClip(pretrained)
        return m, m.num_features
    m = timm.create_model(arch, pretrained=pretrained, features_only=True, **(dict(img_size=size) if "swin" in arch else {}))
    return (NCHW(m) if "swin" in arch else m), m.feature_info.channels()[-1]


def sine_pos(g, d, device):
    """2-D sine position code [g*g, d] (DETR style)."""
    y, x = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing="ij")
    f = 1.0 / 10000 ** (torch.arange(d // 4, device=device) / (d // 4))
    p = [torch.sin(x.flatten()[:, None] * f), torch.cos(x.flatten()[:, None] * f), torch.sin(y.flatten()[:, None] * f), torch.cos(y.flatten()[:, None] * f)]
    return torch.cat(p, 1)


class XLayer(nn.Module):
    """Query cross-attention + FFN (pre-norm). No self-attention between class queries (ML-Decoder)."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * d, d))

    def forward(self, q, mem):
        a, w = self.att(self.n1(q), mem, mem, need_weights=True, average_attn_weights=True)
        q = q + a
        return q + self.ff(self.n2(q)), w


class MultiStream(nn.Module):
    def __init__(self, arch, n_cls, config, size=256, pretrained=True, d=256, heads=8, layers=2):
        super().__init__()
        self.streams = CONFIGS[config]
        self.enc, ch = encoder(arch, size, pretrained)
        self.proj = nn.Conv2d(ch, d, 1)
        self.gamma = nn.Parameter(torch.ones(len(SOURCES)))  # per-source affine after per-image standardisation (bone range is small)
        self.beta = nn.Parameter(torch.zeros(len(SOURCES)))
        self.emb = nn.Parameter(torch.zeros(len(self.streams), d))
        self.query = nn.Parameter(torch.randn(n_cls, d) * 0.02)
        self.mem_norm = nn.LayerNorm(d)
        self.dec = nn.ModuleList([XLayer(d, heads) for _ in range(layers)])
        self.head_w = nn.Parameter(torch.randn(n_cls, d) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_cls))
        self.attn = None  # last-layer attention [B, C, n_streams, g, g] of the latest forward

    def forward(self, x):
        """x [B,3,S,S] in [0,1], channels X, S, B -> logits [B,C]."""
        B = x.shape[0]
        z = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)
        z = z * self.gamma.view(1, 3, 1, 1) + self.beta.view(1, 3, 1, 1)
        idx = {s: i for i, s in enumerate(SOURCES)}
        inp = torch.cat([z[:, [idx[c] for c in st]] for st in self.streams])  # streams stacked on the batch axis: one encoder pass
        f = self.proj(self.enc(inp)[-1])  # [nS*B, d, g, g]
        g, d, nS = f.shape[-1], f.shape[1], len(self.streams)
        tok = f.flatten(2).transpose(1, 2).reshape(nS, B, g * g, d) + sine_pos(g, d, x.device)[None, None] + self.emb[:, None, None]
        mem = self.mem_norm(tok.permute(1, 0, 2, 3).reshape(B, nS * g * g, d))
        q = self.query[None].expand(B, -1, -1)
        for layer in self.dec:
            q, w = layer(q, mem)
        self.attn = w.reshape(B, -1, nS, g, g)
        self.emb_out = q.mean(1)  # pooled decoder output: the cross-modal alignment target of s4/xm.py
        return (q * self.head_w).sum(-1) + self.head_b

    def routing(self):
        """[B, C, n_streams]: attention mass of each class query per stream (sums to 1 over streams)."""
        return self.attn.flatten(3).sum(-1)


_LUTS = {}


def _lut(dev):
    if str(dev) not in _LUTS:
        _LUTS[str(dev)] = torch.from_numpy(np.arange(256, dtype=np.float32) / 255).to(dev)
    return _LUTS[str(dev)]


def load_x(arrays, idx, size=None, dev="cpu"):
    """uint8 orig/soft arrays -> [n,3,size,size] float channels X, S, B with B = clip(X - S) on `dev` (area-resized when the arrays are
    larger). The uint8 batch moves to the device first, so the resize of 512-px detection-set arrays runs on the GPU (RUNPLAN_v3 R4)."""
    u = torch.from_numpy(np.stack([arrays["orig"][idx], arrays["soft"][idx]], 1)).to(dev).long()
    x = _lut(dev)[torch.cat([u, (u[:, :1] - u[:, 1:]).clamp_min(0)], 1)]  # k -> k/255 exactly as numpy float32 (CUDA '/ 255' can differ by 1 ulp)
    return x if not size or x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="area")


def labels_from_ann(path, n_cls):
    """Detection sets (ann.json: per image dict with 'labels'): image-level label = any kept (consensus) box of that class."""
    ann = json.loads(Path(path).read_text())
    y = np.zeros((len(ann), n_cls), np.float32)
    for i, a in enumerate(ann):
        y[i, a["labels"]] = 1
    return y


@torch.no_grad()
def predict(model, arrays, idx, dev, bs=64, size=None):
    model.eval()
    out, rho = [], []
    for s in range(0, len(idx), bs):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(load_x(arrays, idx[s:s + bs], size, dev))).float().cpu())
        rho.append(model.routing().float().cpu())
    return torch.cat(out).numpy(), torch.cat(rho).numpy()


RESUME_KEYS = ("config", "arch", "seed", "epochs", "bs", "lr", "enc_lr", "split", "pretrained", "size", "n_cls", "max_steps", "max_eval", "accum")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--arch", default="densenet121.ra_in1k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="decoder and head; the pretrained encoder gets lr * --enc_lr")
    ap.add_argument("--enc_lr", type=float, default=1.0)
    ap.add_argument("--split", default="split.npy")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--size", type=int, default=256, help="training size; larger stored arrays (detection sets: 512) are area-resized")
    ap.add_argument("--n_cls", type=int, default=14, help="classes when labels come from ann.json (VinBigData: 14)")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke tests: stop each epoch after this many steps")
    ap.add_argument("--max_eval", type=int, default=0, help="smoke tests: evaluate on the first N validation and test images only")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step (the batch stays --bs); for encoders that do not fit a T4")
    ap.add_argument("--deadline_h", type=float, default=0.0, help="stop cleanly (exit 75) when the next epoch would end after this many hours")
    ap.add_argument("--resume", default=None, help="checkpoint (ckpt.pt) of an unfinished run, e.g. from the previous Kaggle session")
    ap.add_argument("--stop_after_epochs", type=int, default=0, help="tests and smoke runs: checkpoint and exit 75 after this many epochs in this process")
    ap.add_argument("--init", default=None, help="model.pt (e.g. s4/xm.py stage xray): start from its tensors whose names and shapes match")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d, out = Path(a.data), Path(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("orig", "soft")}
    Y = np.load(d / "labels.npy") if (d / "labels.npy").exists() else labels_from_ann(d / "ann.json", a.n_cls)
    split = np.load(d / a.split, allow_pickle=True)
    if (d / "status.json").exists():
        split = np.where(np.array(json.loads((d / "status.json").read_text())["ok"]), split, "skip")
    tr, va, te = (np.nonzero(split == s)[0] for s in ("train", "val", "test"))
    if a.max_eval:
        va, te = va[:a.max_eval], te[:a.max_eval]
    model = MultiStream(a.arch, Y.shape[1], a.config, size=a.size, pretrained=bool(a.pretrained)).to(dev)
    if a.init:  # cross-modal pretraining (DESIGN_v4.md): encoder and decoder carry over, class queries and heads of another label set do not
        src, own = torch.load(a.init, map_location="cpu")["state"], model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"init from {a.init}: {len(keep)} of {len(own)} tensors", flush=True)
    enc = list(model.enc.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([dict(params=enc, scale=a.enc_lr), dict(params=rest, scale=1.0)], lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    per_ep = len(tr) // a.bs if not a.max_steps else min(a.max_steps, len(tr) // a.bs)
    steps = a.epochs * per_ep
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    Yt = torch.from_numpy(Y)
    step, best, start_ep, sec0, t0 = 0, (-1.0, None), 0, 0.0, time.time()
    if a.resume:  # everything that decides the rest of the run, so a resumed run equals an uninterrupted one
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        diff = {k: (ck["args"].get(k), v) for k, v in vars(a).items() if k in RESUME_KEYS and ck["args"].get(k) != v}
        if diff:
            raise SystemExit(f"checkpoint {a.resume} was made with other settings: {diff}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, start_ep, sec0 = ck["step"], ck["best"], ck["epoch"], ck["sec"]
        rng.bit_generator.state = ck["rng"]
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        if dev == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        del ck  # up to ~1 GB of CPU tensors (Swin-V2-B) that would otherwise take page cache from the image arrays
        print(f"resumed from {a.resume} at epoch {start_ep}", flush=True)
        if (out / "log.jsonl").exists():  # a session that died after the checkpoint may have logged later epochs: they are redone
            keep = [x for x in (out / "log.jsonl").read_text().splitlines() if x.strip() and json.loads(x)["epoch"] < start_ep]
            (out / "log.jsonl").write_text("".join(x + "\n" for x in keep))
    log = open(out / "log.jsonl", "a" if a.resume else "w")
    for ep in range(start_ep, a.epochs):
        t_ep = time.time()
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, per_ep * a.bs, a.bs):
            idx = np.sort(perm[s:s + a.bs])
            for gp in opt.param_groups:
                gp["lr"] = a.lr * gp["scale"] * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for part in np.array_split(idx, a.accum):  # --accum > 1: same batch in micro-batches (equivalent for layer-norm encoders)
                x, y = load_x(arrays, part, a.size, dev), Yt[part].to(dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    loss = masked_bce(model(batch_aug(x, gen)).float(), y) * (len(part) / len(idx))  # one geometry for X, S and B of a sample
                assert torch.isfinite(loss), f"non-finite loss at step {step}"
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.step(opt)
            scaler.update()
            step += 1
        pv, _ = predict(model, arrays, va, dev, size=a.size)
        vauc = float(np.nanmean(macro_auc(Y[va], pv)))
        log.write(json.dumps(dict(epoch=ep, val_macro_auc=vauc, loss=step_loss, sec=round(sec0 + time.time() - t0))) + "\n")
        log.flush()
        if best[1] is None or vauc > best[0]:  # model selection on the validation split only
            best = (vauc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if ep + 1 == a.epochs:
            break
        tmp = out / "ckpt.pt.tmp"  # every epoch: a crash or a killed session loses at most the epoch in progress
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, best=best, epoch=ep + 1,
                        sec=sec0 + time.time() - t0, rng=rng.bit_generator.state, gen=gen.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if dev == "cuda" else None, args=vars(a)), tmp)
        os.replace(tmp, out / "ckpt.pt")  # never a truncated checkpoint if the session ends during the save
        late = a.deadline_h and (time.time() - t0) + 1.05 * (time.time() - t_ep) > a.deadline_h * 3600  # the next epoch would not fit
        if late or (a.stop_after_epochs and ep + 1 - start_ep >= a.stop_after_epochs):
            log.close()
            print(f"checkpoint after epoch {ep + 1}; exit 75 (resume with --resume {out / 'ckpt.pt'})", flush=True)
            sys.exit(75)
    model.load_state_dict(best[1])
    torch.save(dict(state=best[1], arch=a.arch, config=a.config, n_cls=Y.shape[1], size=a.size), out / "model.pt")
    pv, _ = predict(model, arrays, va, dev, size=a.size)
    np.save(out / "val_pred.npy", pv.astype(np.float16))  # validation predictions: calibration data for the conformal methods
    np.save(out / "val_idx.npy", va)
    pt, rho = predict(model, arrays, te, dev, size=a.size)
    np.save(out / "test_pred.npy", pt.astype(np.float16))
    np.save(out / "test_routing.npy", rho.astype(np.float16))
    np.save(out / "test_idx.npy", te)
    np.save(out / "test_labels.npy", Y[te])
    if (d / "patient.npy").exists():
        np.save(out / "test_patient.npy", np.load(d / "patient.npy", allow_pickle=True)[te])
    auc = macro_auc(Y[te], pt)
    json.dump(dict(val_macro_auc=best[0], test_macro_auc=float(np.nanmean(auc)), per_class=auc.tolist(), config=a.config, arch=a.arch,
                   seed=a.seed, epochs=a.epochs, split=a.split, routing_mean=rho.mean(0).tolist()), open(out / "result.json", "w"))
    (out / "ckpt.pt").unlink(missing_ok=True)  # finished: the checkpoint would only fill the 20 GB kernel output
    print(json.dumps(dict(val=best[0], test=float(np.nanmean(auc)))))


if __name__ == "__main__":
    main()
