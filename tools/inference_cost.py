"""Inference cost of the suppressors: parameters, GFLOPs, latency and peak GPU memory for one image (batch 1) on the local GPU.

python tools/inference_cost.py [--out results/final/cost.csv]
Ours run at 512 px in fp16 autocast (as in every evaluation script); the released Qure TorchScript model runs at its own 1024 px in fp32 (as audit_real calls it).
GFLOPs = 2 x multiply-adds counted by torch.utils.flop_counter for the first forward; the TorchScript model is not traceable by it (nan).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s4.model import SwinUNetPP  # noqa: E402

W = ROOT / "outputs" / "final_v2" / "weights"
MODELS = {"ConvNeXt-V2-T (ours)": "qp_twin16_s0_cnx2_300k", "RAD-DINO ViT-B/14": "qp_twin16_s0_raddino300k", "SwinV2-T": "twin16_s0"}


def bench(m, size, amp, n=30):
    x = torch.rand(1, 1, size, size).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            m(x)
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000, torch.cuda.max_memory_allocated() / 1e9


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "results" / "final" / "cost.csv"
    rows = []
    for name, d in MODELS.items():
        st = torch.load(W / d / "final_ema.pt", map_location="cpu")
        m = SwinUNetPP(st["backbone"], 512, pretrained=False)
        m.load_state_dict({k: v.float() for k, v in st["ema"].items()})
        m = m.cuda().eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(torch.rand(1, 1, 512, 512).cuda())
        ms, gb = bench(m, 512, True)
        rows.append(dict(model=name, backbone=st["backbone"], input_px=512, params_M=sum(p.numel() for p in m.parameters()) / 1e6, gflops=fc.get_total_flops() / 1e9, ms_per_image=ms, peak_gpu_gb=gb))
        del m
        torch.cuda.empty_cache()
    q = torch.jit.load(str(ROOT / "data" / "qure" / "weights" / "bone_suppression.ts"), map_location="cuda").eval()
    ms, gb = bench(q, 1024, False, 10)
    rows.append(dict(model="Qure released", backbone="SwinV2-B (TorchScript)", input_px=1024, params_M=sum(p.numel() for p in q.parameters()) / 1e6, gflops=float("nan"), ms_per_image=ms, peak_gpu_gb=gb))
    df = pd.DataFrame(rows).round(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{torch.cuda.get_device_name(0)}\n" + df.to_string(index=False))
