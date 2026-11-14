"""Cross-modality Swin models and training (DESIGN_v4.md section 3).

python -m s4.xm --stage ct --data CT_PREP --seed 0 --epochs 20 --out runs/ct_s0
python -m s4.xm --stage xray --data QP_PACK --seed 0 --epochs 15 --out runs/xr_s0 [--teacher runs/ct_s0/teacher.npz --kd 1 --align 0.5]
python -m s4.xm --stage transfer --run runs/xr_s0 --data NIH_PREP --dataset nih --out runs/nih_transfer_s0

ct: CTNet on s4/ct_prep.py output (labels: the six shared findings); writes model.pt, val/test predictions, result.json and teacher.npz
(logits and pooled decoder output of every volume) for distillation. xray: s4/cls_hybrid.py MultiStream (configuration orig) on the Qure.ai
projections (qp.bin pack, labels from qp_meta.csv; s4/data/ct_skip.txt volumes left out); with --teacher, half of every batch comes from projections whose CT is in the teacher,
which add binary distillation at temperature --temp and a cosine alignment of the pooled decoder output through a linear head. Its
model.pt loads into s4/cls_hybrid.py --init for NIH fine-tuning. transfer: zero-shot AUROC of an xray model on the shared findings of a
real X-ray test split (s4/xm_labels.py map), written to result.json in --out.
"""
import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls import batch_aug, macro_auc, masked_bce
from .cls_hybrid import MultiStream, XLayer, encoder, labels_from_ann, load_x, sine_pos
from .ct_prep import skip_set
from .xm_labels import SHARED, from_meta, to_shared

ARCH = "swinv2_tiny_window8_256.ms_in1k"


class CTNet(nn.Module):
    """2.5D Swin: the encoder runs on every axial slice (three HU windows as channels); each slice gives g x g tokens with a 2-D sine code and a
    learned depth embedding (slices sit at fixed relative depths of the lung extent); the class-query decoder of s4/cls_hybrid.py reads all."""

    def __init__(self, arch, n_cls, k, size=256, pretrained=True, d=256, heads=8, layers=2):
        super().__init__()
        self.enc, ch = encoder(arch, size, pretrained)
        self.proj = nn.Conv2d(ch, d, 1)
        self.gamma, self.beta = nn.Parameter(torch.ones(3)), nn.Parameter(torch.zeros(3))
        self.zemb = nn.Parameter(torch.zeros(k, d))
        self.query = nn.Parameter(torch.randn(n_cls, d) * 0.02)
        self.mem_norm = nn.LayerNorm(d)
        self.dec = nn.ModuleList([XLayer(d, heads) for _ in range(layers)])
        self.head_w, self.head_b = nn.Parameter(torch.randn(n_cls, d) * 0.02), nn.Parameter(torch.zeros(n_cls))
        self.attn = self.emb_out = None

    def forward(self, v):
        """v [B,K,3,S,S] in [0,1] -> logits [B,C]; keeps the last-layer attention [B,C,K,g,g] and the pooled decoder output [B,d]."""
        B, Kk = v.shape[:2]
        x = (v.flatten(0, 1) - 0.5) / 0.25 * self.gamma.view(1, 3, 1, 1) + self.beta.view(1, 3, 1, 1)
        f = self.proj(self.enc(x)[-1])
        g, d = f.shape[-1], f.shape[1]
        tok = f.flatten(2).transpose(1, 2).reshape(B, Kk, g * g, d) + sine_pos(g, d, v.device)[None, None] + self.zemb[None, :, None]
        mem = self.mem_norm(tok.reshape(B, Kk * g * g, d))
        q = self.query[None].expand(B, -1, -1)
        for layer in self.dec:
            q, w = layer(q, mem)
        self.attn, self.emb_out = w.reshape(B, -1, Kk, g, g), q.mean(1)
        return (q * self.head_w).sum(-1) + self.head_b


def cosine_lr(opt, base, step, steps, warm=200):
    for gp in opt.param_groups:
        gp["lr"] = base * gp["scale"] * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))


def optimiser(model, lr, enc_lr):
    enc = list(model.enc.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    return torch.optim.AdamW([dict(params=enc, scale=enc_lr), dict(params=rest, scale=1.0)], lr=lr, weight_decay=1e-4)


# ---- CT stage ------------------------------------------------------------------------------------------------------------
def ct_batch(ct, idx, dev):
    return torch.from_numpy(np.ascontiguousarray(ct[np.sort(idx)])).to(dev).float() / 255  # sorted: memmap reads in file order


@torch.no_grad()
def ct_predict(model, ct, idx, dev, bs=4):
    model.eval()
    out, emb = [], []
    for s in range(0, len(idx), bs):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(ct_batch(ct, idx[s:s + bs], dev)).float().cpu())
        emb.append(model.emb_out.float().cpu())
    return torch.cat(out).numpy(), torch.cat(emb).numpy()


def train_ct(a, dev):
    d = Path(a.data)
    ct = np.load(d / "ct.npy", mmap_mode="r")
    Y = np.load(d / "labels.npy")
    split = np.load(d / "split.npy", allow_pickle=True)
    if (d / "status.json").exists():  # volumes that failed to download or convert are left out everywhere
        split = np.where(np.array([bool(s and s["ok"]) for s in json.loads((d / "status.json").read_text())]), split, "skip")
    tr, va, te = (np.nonzero(split == s)[0] for s in ("train", "val", "test"))
    model = CTNet(a.arch, Y.shape[1], ct.shape[1], ct.shape[-1], bool(a.pretrained)).to(dev)
    opt = optimiser(model, a.lr, a.enc_lr)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    per_ep = len(tr) // a.bs if not a.max_steps else min(a.max_steps, len(tr) // a.bs)
    steps, step, best, Yt = a.epochs * per_ep, 0, (-1.0, None), torch.from_numpy(Y)
    out = Path(a.out)
    log = open(out / "log.jsonl", "w")
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, per_ep * a.bs, a.bs):
            idx = np.sort(perm[s:s + a.bs])
            cosine_lr(opt, a.lr, step, steps)
            opt.zero_grad(set_to_none=True)
            v = ct_batch(ct, idx, dev)
            Bn, Kk, _, S, _ = v.shape
            v = batch_aug(v.reshape(Bn, Kk * 3, S, S), gen).reshape(Bn, Kk, 3, S, S)  # one geometry for all slices of a volume
            tot = 0.0
            for part in np.array_split(np.arange(len(idx)), a.accum):  # micro-batches for memory; the step still averages all --bs volumes
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    loss = masked_bce(model(v[part]).float(), Yt[idx[part]].to(dev)) * len(part) / len(idx)
                assert torch.isfinite(loss), f"non-finite loss at step {step}"
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.step(opt)
            scaler.update()
            step += 1
        pv, _ = ct_predict(model, ct, va, dev)
        vauc = float(np.nanmean(macro_auc(Y[va], torch.sigmoid(torch.from_numpy(pv)).numpy())))
        log.write(json.dumps(dict(epoch=ep, val_macro_auc=vauc, loss=tot)) + "\n")
        log.flush()
        if best[1] is None or vauc > best[0]:
            best = (vauc, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save(dict(state=best[1], arch=a.arch, n_cls=Y.shape[1], k=ct.shape[1], size=ct.shape[-1]), out / "model.pt")
    ok = np.nonzero(split != "skip")[0]
    logit, emb = ct_predict(model, ct, ok, dev)
    names = np.array(json.loads((d / "names.json").read_text()))
    np.savez(out / "teacher.npz", names=names[ok], logits=logit.astype(np.float32), emb=emb.astype(np.float32), split=split[ok])
    p = torch.sigmoid(torch.from_numpy(logit)).numpy()
    pos = {i: j for j, i in enumerate(ok)}
    for name, ix in (("val", va), ("test", te)):
        np.save(out / f"{name}_pred.npy", p[[pos[i] for i in ix]].astype(np.float16))
        np.save(out / f"{name}_idx.npy", ix)
    np.save(out / "test_labels.npy", Y[te])
    auc = macro_auc(Y[te], p[[pos[i] for i in te]])
    json.dump(dict(stage="ct", val_macro_auc=best[0], test_macro_auc=float(np.nanmean(auc)), per_class=dict(zip(SHARED, map(float, auc))),
                   arch=a.arch, seed=a.seed, epochs=a.epochs), open(out / "result.json", "w"))
    print(json.dumps(dict(val=best[0], test=float(np.nanmean(auc)))))


# ---- X-ray stage on the projections ------------------------------------------------------------------------------------------
def qp_load(pack, rows, size, workers=8, col=0):
    """Translated projections (PNG in qp.bin, column 0 of qp_index.npy; 2 = lung component L) of `rows` -> uint8 [n, size, size], decoded
    once into memory."""
    buf, index = np.memmap(Path(pack) / "qp.bin", dtype=np.uint8, mode="r"), np.load(Path(pack) / "qp_index.npy")

    def one(r):
        o, n = index[r][col]
        return cv2.resize(cv2.imdecode(np.asarray(buf[o:o + n]), cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)

    with ThreadPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(one, rows)))


def as3(img, idx, dev):
    """uint8 [n,S,S] rows -> [n,3,S,S] float: channel X; the S and B channels are unused by configuration orig."""
    x = torch.from_numpy(np.ascontiguousarray(img[idx])).to(dev).float()[:, None] / 255
    return torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1)


@torch.no_grad()
def xr_predict(model, img, idx, dev, bs=64):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(as3(img, idx[s:s + bs], dev))).float().cpu())
    return torch.cat(out).numpy()


def kd_loss(student, teacher, temp):
    """Binary (per-finding) distillation: cross-entropy of the student against the teacher's tempered probabilities, scaled by temp^2."""
    return F.binary_cross_entropy_with_logits(student / temp, torch.sigmoid(teacher / temp)) * temp ** 2


def train_xray(a, dev):
    d = Path(a.data)
    meta = list(csv.DictReader(open(d / "qp_meta.csv", encoding="utf-8")))
    Y = from_meta(meta)
    split = np.array([r["split"] for r in meta])
    keep = ~np.isin([r["volume_name"] for r in meta], list(skip_set()))  # no chest or not head-first (s4/data/ct_skip.txt)
    tr, va, te = (np.nonzero((split == s) & keep)[0] for s in ("train", "val", "test"))
    if a.max_eval:
        va, te = va[:a.max_eval], te[:a.max_eval]
    use = np.concatenate([tr, va, te])
    img = np.zeros((len(meta), a.size, a.size), np.uint8)
    img[use] = qp_load(d, use, a.size)
    model = MultiStream(a.arch, len(SHARED), "orig", size=a.size, pretrained=bool(a.pretrained)).to(dev)
    teacher, paired = None, np.array([], int)
    if a.teacher:
        t = np.load(a.teacher, allow_pickle=True)
        row = {n: i for i, n in enumerate(t["names"])}
        name = np.array([r["volume_name"] for r in meta])
        paired = np.array([i for i in tr if name[i] in row and t["split"][row[name[i]]] == "train"], int)  # never a val/test CT
        tl, te_ = np.zeros((len(meta), len(SHARED)), np.float32), np.zeros((len(meta), t["emb"].shape[1]), np.float32)
        tl[paired] = t["logits"][[row[name[i]] for i in paired]]
        te_[paired] = t["emb"][[row[name[i]] for i in paired]]
        teacher = (torch.from_numpy(tl), torch.from_numpy(te_))
        model.align = nn.Linear(model.query.shape[1], te_.shape[1]).to(dev)
        print(f"distillation on {len(paired)} paired training projections", flush=True)
    opt = optimiser(model, a.lr, a.enc_lr)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    per_ep = len(tr) // a.bs if not a.max_steps else min(a.max_steps, len(tr) // a.bs)
    steps, step, best, Yt = a.epochs * per_ep, 0, (-1.0, None), torch.from_numpy(Y)
    out = Path(a.out)
    log = open(out / "log.jsonl", "w")
    half = a.bs // 2 if len(paired) else 0  # same number of steps with and without distillation (equal training budget)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(per_ep):
            new = perm[i * (a.bs - half):(i + 1) * (a.bs - half)]
            idx = np.concatenate([new, rng.choice(paired, half) if half else np.array([], int)]).astype(int)
            cosine_lr(opt, a.lr, step, steps)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logit = model(batch_aug(as3(img, idx, dev), gen)).float()
                loss = masked_bce(logit, Yt[idx].to(dev))
                if half:
                    k = slice(len(idx) - half, None)
                    loss = loss + a.kd * kd_loss(logit[k], teacher[0][idx[k]].to(dev), a.temp)
                    loss = loss + a.align * (1 - F.cosine_similarity(model.align(model.emb_out[k].float()), teacher[1][idx[k]].to(dev))).mean()
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
        vauc = float(np.nanmean(macro_auc(Y[va], xr_predict(model, img, va, dev))))
        log.write(json.dumps(dict(epoch=ep, val_macro_auc=vauc, loss=loss.item())) + "\n")
        log.flush()
        if best[1] is None or vauc > best[0]:
            best = (vauc, {k: t.detach().cpu().clone() for k, t in model.state_dict().items() if not k.startswith("align.")})
    model.load_state_dict(best[1], strict=False)
    torch.save(dict(state=best[1], arch=a.arch, config="orig", n_cls=len(SHARED), size=a.size, labels=list(SHARED)), out / "model.pt")
    pt = xr_predict(model, img, te, dev)
    np.save(out / "test_pred.npy", pt.astype(np.float16))
    np.save(out / "test_idx.npy", te)
    np.save(out / "test_labels.npy", Y[te])
    auc = macro_auc(Y[te], pt)
    json.dump(dict(stage="xray", teacher=a.teacher, kd=a.kd, align=a.align, paired=int(len(paired)), val_macro_auc=best[0],
                   test_macro_auc=float(np.nanmean(auc)), per_class=dict(zip(SHARED, map(float, auc))), arch=a.arch, seed=a.seed,
                   epochs=a.epochs), open(out / "result.json", "w"))
    print(json.dumps(dict(val=best[0], test=float(np.nanmean(auc)))))


# ---- zero-shot transfer to real X-rays -------------------------------------------------------------------------------------------
@torch.no_grad()
def transfer(run, data, dataset, dev, out, bs=64):
    """AUROC per shared finding of an xray-stage model on the test split of a real X-ray prep (NIH or VinBigData); predictions, indices and
    shared labels go to `out` as in the other stages (seed ensembles in tools/xm_stats.py)."""
    st = torch.load(Path(run) / "model.pt", map_location="cpu")
    model = MultiStream(st["arch"], st["n_cls"], "orig", size=st["size"], pretrained=False)
    model.load_state_dict(st["state"], strict=False)
    model = model.to(dev).eval()
    d = Path(data)
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("orig", "soft")}
    te = np.nonzero(np.load(d / "split.npy", allow_pickle=True) == "test")[0]
    p = np.concatenate([torch.sigmoid(model(load_x(arrays, te[s:s + bs], st["size"], dev))).float().cpu().numpy() for s in range(0, len(te), bs)])
    Y = np.load(d / "labels.npy") if (d / "labels.npy").exists() else labels_from_ann(d / "ann.json", 14)  # VinBigData: boxes only
    y = to_shared(Y[te], dataset)
    np.save(Path(out) / "test_pred.npy", p.astype(np.float16))
    np.save(Path(out) / "test_idx.npy", te)
    np.save(Path(out) / "test_labels.npy", y)
    auc = macro_auc(y, p)
    return dict(run=str(run), dataset=dataset, n=int(len(te)), per_class=dict(zip(SHARED, map(float, auc))), mean=float(np.nanmean(auc)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("ct", "xray", "transfer"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=ARCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=None, help="volumes per step for ct (default 4), images per step for xray (default 32)")
    ap.add_argument("--accum", type=int, default=1, help="ct: micro-batches per step (same step, less memory; a T4 needs 4 for 4 x 24 slices)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=0.25, help="encoder lr factor (the v3 rule for encoders without batch normalisation)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--teacher", default=None, help="teacher.npz of a ct-stage run (xray stage: distillation and alignment)")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--align", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--run", default=None, help="transfer: an xray-stage run folder")
    ap.add_argument("--dataset", default="nih", choices=("nih", "vin"))
    ap.add_argument("--max_steps", type=int, default=0, help="smoke tests: steps per epoch")
    ap.add_argument("--max_eval", type=int, default=0, help="smoke tests: validation/test images")
    a = ap.parse_args(argv)
    a.bs = a.bs or (4 if a.stage == "ct" else 32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "transfer":
        res = transfer(a.run, a.data, a.dataset, dev, a.out)
        (Path(a.out) / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    (train_ct if a.stage == "ct" else train_xray)(a, dev)


if __name__ == "__main__":
    main()
