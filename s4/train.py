"""Train one suppressor arm. Everything except the arm flags is identical across arms (R7).

python -m s4.train --arm recipe --seed 0 --samples 200000 --out runs/recipe_s0 --bank DIR [DIR..] --split splits.csv
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import compose as C
from . import lesion_lib, qp
from .bank import load_bank
from .losses import ssim, suppression_loss
from .model import SwinUNetPP

# one factor changes per arm relative to `recipe`
ARMS = {
    "recipe": dict(),
    "nofreq": dict(detector=False),
    "nohist": dict(tone="gamma"),
    "nomark": dict(w_roi=0.0),
    "nodule_loss": dict(w_nod=1.0),
    "clia": dict(n_insert=3),
    "clia_nodule": dict(n_insert=3, w_nod=1.0),
    "rot0": dict(rot_deg=0.0),
    "rot30": dict(rot_deg=30.0),
    # extra data source: Qure ct2xr-projections (exactly additive, translated); qp_frac = share of steps drawn from it
    "qp": dict(qp_frac=1.0),
    "qp_clia": dict(qp_frac=1.0, n_insert=3),
    "mix": dict(qp_frac=0.5),
    "mix_clia": dict(qp_frac=0.5, n_insert=3),
    "mix_clia_nodule": dict(qp_frac=0.5, n_insert=3, w_nod=1.0),
    # counterfactual twins: every base sample also enters with a lesion pasted into the image (bone target unchanged) and the bone estimates
    # of the pair are tied on the lesion support (lesion-invariance regulariser); half of each batch are the twins, so the cost per step is unchanged
    "mix_twin": dict(qp_frac=0.5, n_insert=3, w_inv=1.0),
    "mix_twin4": dict(qp_frac=0.5, n_insert=3, w_inv=4.0),
    "twin": dict(n_insert=3, w_inv=1.0),
    # round 2: the Qure projections alone were the best source in screening, so push on them (twin weight sweep, lesion count, LIDC share)
    "qp_twin": dict(qp_frac=1.0, n_insert=3, w_inv=1.0),
    "qp_twin4": dict(qp_frac=1.0, n_insert=3, w_inv=4.0),
    "qp_twin16": dict(qp_frac=1.0, n_insert=3, w_inv=16.0),
    "qp_twin64": dict(qp_frac=1.0, n_insert=3, w_inv=64.0),
    "qp_twin32": dict(qp_frac=1.0, n_insert=3, w_inv=32.0),
    "qp_twin8": dict(qp_frac=1.0, n_insert=3, w_inv=8.0),
    # round 5: which part of the twin regulariser matters (all at w_inv = 16): stop-gradient on the un-pasted branch, support dilated by k px
    # (inv_k > 0) or the whole image (inv_k = -1), more pasted lesions per twin
    "qp_twin16_sg": dict(qp_frac=1.0, n_insert=3, w_inv=16.0, inv_sg=True),
    "qp_twin16_wide": dict(qp_frac=1.0, n_insert=3, w_inv=16.0, inv_k=31),
    "qp_twin16_all": dict(qp_frac=1.0, n_insert=3, w_inv=16.0, inv_k=-1),
    "qp_twin16_n4": dict(qp_frac=1.0, n_insert=3, w_inv=16.0, n_twin=4),
    "qp_clia6": dict(qp_frac=1.0, n_insert=6),
    "qp75_clia": dict(qp_frac=0.75, n_insert=3),
}
DEFAULTS = dict(detector=True, tone="hist", w_roi=2.5, w_nod=0.0, n_insert=0, rot_deg=15.0, qp_frac=0.0, w_inv=0.0, n_twin=2, inv_sg=False, inv_k=0)


def lesion_contrast(S, mask):
    """Mean intensity inside the lesion minus a surrounding ring, per sample with a lesion; NaN otherwise."""
    m = mask.float()
    ring = (torch.nn.functional.max_pool2d(m, 15, 1, 7) - torch.nn.functional.max_pool2d(m, 5, 1, 2)).clamp(0, 1)
    inner = (S * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp_min(1)
    outer = (S * ring).sum((1, 2, 3)) / ring.sum((1, 2, 3)).clamp_min(1)
    return torch.where(m.sum((1, 2, 3)) > 20, inner - outer, torch.full_like(inner, float("nan")))


@torch.no_grad()
def validate(model, bank, idx, phys, dev, bs=8, seed=1234):
    """Canonical composition (gamma tone, no detector effects, no rotation/insertion) so arms are comparable."""
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    acc = {k: [] for k in ("mae", "psnr", "ssim", "lcr")}
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        a4 = torch.cat([bank["a"][j].float(), bank["lesion"][j][:, None].float()], 1)
        out = C.compose(a4, torch.full((len(j),), 3, device=dev), phys, gen=gen, tone="gamma", detector=False)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(out["X"]).float()
        err = pred - out["B"]
        acc["mae"].append(err.abs().mean((1, 2, 3)))
        acc["psnr"].append(10 * torch.log10(1.0 / (err**2).mean((1, 2, 3)).clamp_min(1e-10)))
        acc["ssim"].append(ssim(pred.clamp(0, 1), out["B"].clamp(0, 1)))
        c_hat = lesion_contrast(out["X"] - pred, out["nod"])
        c_true = lesion_contrast(out["S"], out["nod"])
        acc["lcr"].append(c_hat / c_true)
    model.train()
    r = {k: torch.cat(v) for k, v in acc.items()}
    return {k: float(torch.nanmean(v)) for k, v in r.items()}


@torch.no_grad()
def validate_qp(model, qpv, dev, n=64, bs=8):
    """Canonical QP samples (no augmentation): bone MAE / PSNR against the exact additive bone image."""
    model.eval()
    mae, psnr = [], []
    for s in range(0, min(n, len(qpv)), bs):
        X, B, _ = qpv.raw(bs, idx=list(range(s, min(s + bs, n, len(qpv)))))
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(X).float()
        err = pred - B
        mae.append(err.abs().mean((1, 2, 3)))
        psnr.append(10 * torch.log10(1.0 / (err**2).mean((1, 2, 3)).clamp_min(1e-10)))
    model.train()
    return dict(mae=float(torch.cat(mae).mean()), psnr=float(torch.cat(psnr).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ref_q", default=None, help=".npy [R,257] quantiles of real radiographs (train only)")
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--max_hours", type=float, default=11.0)
    ap.add_argument("--backbone", default="swinv2_tiny_window8_256.ms_in1k")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--limit_ct", type=int, default=0)
    ap.add_argument("--qp_pack", default=None, help="dir with qp.bin / qp_index.npy / qp_meta.csv (tools/qp_pack.py); needed by qp/mix arms")
    ap.add_argument("--qp_workers", type=int, default=4)
    ap.add_argument("--lib_file", default=None, help="lesion library of the TRAIN CTs (tools/make_lib_img.py); a QP-only arm then runs without the render bank")
    ap.add_argument("--enc_lr_mult", type=float, default=1.0, help="learning-rate multiplier of the pretrained encoder (decoder keeps --lr)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (bs is the per-step batch; the effective batch is bs * accum)")
    args = ap.parse_args()
    cfg = {**DEFAULTS, **ARMS[args.arm]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import csv
    if args.lib_file:  # bank-free: only the Qure projections are sampled, lesions come from the saved TRAIN-CT library
        if cfg["qp_frac"] < 1.0:
            raise SystemExit("--lib_file replaces the render bank, so the arm must draw every sample from the Qure projections (qp_frac = 1)")
        bank = phys = None
        tr = va = lib = []
        print(f"arm={args.arm} cfg={cfg} bank-free (lesion library from {args.lib_file})", flush=True)
    else:
        rows = [r for r in csv.DictReader(open(args.split)) if r["usable"] == "1"]
        bank = load_bank(args.bank, [r["name"] for r in rows], dev)
        split = {r["name"]: r["split"] for r in rows}
        tr = [i for i, n in enumerate(bank["names"]) if split[n] == "train"][: args.limit_ct or None]
        va = [i for i, n in enumerate(bank["names"]) if split[n] == "val"][:64]
        lib = [p["patch"] for p in bank["patches"] if p["ct"] in set(tr)]
        print(f"arm={args.arm} cfg={cfg} train CT={len(tr)} val CT={len(va)} lesion library={len(lib)}", flush=True)
        phys = C.Physics(dev)
    qtr = qva = lib_img = None
    if cfg["qp_frac"] > 0:
        if not args.qp_pack:
            raise SystemExit("arm draws from the Qure projections but no --qp_pack was given")
        qtr = qp.QPBank(args.qp_pack, "train", dev, args.qp_workers, seed=args.seed)
        qva = qp.QPBank(args.qp_pack, "val", dev, 1, seed=0, limit=64)
        print(f"QP source: train rows {len(qtr)} val rows {len(qva)}", flush=True)
    if (cfg["qp_frac"] > 0 and cfg["n_insert"] > 0) or cfg["w_inv"] > 0:
        lib_img = [torch.from_numpy(p["x_patch"]) for p in (lesion_lib.load(args.lib_file) if args.lib_file else lesion_lib.build(bank, tr, phys, dev)[0])]  # composite-unit lesions of TRAIN CTs
        print(f"image-space lesion library {len(lib_img)}", flush=True)
    cpu_gen = torch.Generator().manual_seed(args.seed * 7919 + 1)
    ref_q = torch.from_numpy(np.load(args.ref_q)).float().to(dev) if args.ref_q else None
    if cfg["tone"] == "hist" and ref_q is None and cfg["qp_frac"] < 1.0:  # QP-only arms use their own gamma tone
        raise SystemExit("arm needs histogram matching but no --ref_q was given (refusing to silently switch it off)")
    model = SwinUNetPP(args.backbone, 512, pretrained=not args.no_pretrained).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) in enc_ids], lr_mult=args.enc_lr_mult),
                             dict(params=[p for p in model.parameters() if id(p) not in enc_ids], lr_mult=1.0)], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    steps = args.samples // (args.bs * args.accum)
    sched = lambda s: min(1.0, (s + 1) / 500) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.99 + 0.01)  # noqa: E731
    step, t0 = 0, time.time()
    ck = out / "ckpt_last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        step = st["step"]
        print("resumed at step", step, flush=True)
    gen = torch.Generator(device=dev).manual_seed(args.seed * 1000 + step)
    log = open(out / "log.jsonl", "a")
    last_ck = time.time()
    model.train()
    twin = cfg["w_inv"] > 0
    nb = args.bs // 2 if twin else args.bs  # base samples per step (twins double them)
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * g["lr_mult"] * sched(step)
        opt.zero_grad(set_to_none=True)
        for _micro in range(args.accum):  # gradient accumulation: several small batches act as one (BatchNorm still sees the small batch)
            if qtr is not None and float(torch.rand(1, generator=cpu_gen)) < cfg["qp_frac"]:
                o = qp.qp_compose(*qtr.raw(nb), library=lib_img, n_insert=cfg["n_insert"], rot_deg=cfg["rot_deg"], detector=cfg["detector"], gen=gen)
            else:
                j = torch.randint(len(tr), (nb,), device=dev, generator=gen)
                j = torch.tensor(tr, device=dev)[j]
                a4 = torch.cat([bank["a"][j].float(), bank["lesion"][j][:, None].float()], 1)
                C.insert_nodules(a4, lib, cfg["n_insert"], gen)
                a4 = C.random_affine(a4, rot_deg=cfg["rot_deg"], gen=gen)
                kvp = torch.randint(1, 7, (nb,), device=dev, generator=gen)  # 80..130 kVp
                o = C.compose(a4, kvp, phys, gen=gen, tone=cfg["tone"], ref_q=ref_q, detector=cfg["detector"])
            if twin:
                n_les = torch.zeros_like(o["X"])
                les = F.max_pool2d(qp.insert_lesions(n_les, qp.lung_field(o["roi"], 0.5), lib_img, cfg["n_twin"], gen, n_min=1), 7, 1, 3)
                o = dict(X=torch.cat([o["X"], (o["X"] + n_les).clamp(max=1.0)]), B=torch.cat([o["B"]] * 2), roi=torch.cat([o["roi"]] * 2),
                         nod=torch.cat([o["nod"], torch.maximum(o["nod"], les)]))
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(o["X"].contiguous(memory_format=torch.channels_last))
            loss, parts = suppression_loss(pred, o["B"], o["roi"], o["nod"], cfg["w_roi"], cfg["w_nod"])
            if twin:  # lesion-invariance: the bone estimate must not change where a lesion was pasted
                p0, p1 = pred.float().chunk(2)
                if cfg["inv_sg"]:
                    p0 = p0.detach()
                sup = les if cfg["inv_k"] == 0 else torch.ones_like(les) if cfg["inv_k"] < 0 else F.max_pool2d(les, cfg["inv_k"], 1, cfg["inv_k"] // 2)
                parts["inv"] = (((p1 - p0).abs() * sup).sum((1, 2, 3)) / sup.sum((1, 2, 3)).clamp_min(1.0)).mean()
                loss = loss + cfg["w_inv"] * parts["inv"]
            scaler.scale(loss / args.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, loss=float(loss), **{k: float(v) for k, v in parts.items()}, sec=round(time.time() - t0))) + "\n")
            log.flush()
        if step % args.val_every == 0 or step == steps:
            v = validate(ema, bank, va, phys, dev) if bank is not None else {}
            if qva is not None:
                v["qp"] = validate_qp(ema, qva, dev)
            print(step, f"{time.time() - t0:.0f}s", f"loss {float(loss):.4f}", v, flush=True)
            log.write(json.dumps(dict(step=step, val=v)) + "\n")
            log.flush()
        over = (time.time() - t0) / 3600 > args.max_hours
        if time.time() - last_ck > 900 or step == steps or over:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, cfg=cfg), ck)
            last_ck = time.time()
        if over:
            print("time budget reached; checkpoint saved, rerun to resume", flush=True)
            return
    torch.save(dict(ema={k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()}, cfg=cfg, arm=args.arm,
                    seed=args.seed, steps=step, backbone=args.backbone), out / "final_ema.pt")
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
