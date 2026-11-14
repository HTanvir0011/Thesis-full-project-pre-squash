"""Hard-link finished weights from downloaded kernel outputs into the names tools/final_methods.txt uses (outputs/final_v2/weights/<name>/).

python tools/collect_final.py outputs/w1_dl/s4-w1-7 [more download dirs ...]
Kernel run names -> final names: qp_twin<W>_s<S>_300k -> twin<W>_s<S>; qp_s<S>_300k -> qp_s<S>_300k; other arms keep their run name
(e.g. qp_twin16_sg_s0_300k). LIDC-bank v1 finals (clia_s2, recipe_s2) go to outputs/fin_all/weights/. Refuses files that do not load.
"""
import os
import re
import sys
from pathlib import Path

import torch

W2, V1 = Path("outputs/final_v2/weights"), Path("outputs/fin_all/weights")


def final_name(run):
    m = re.fullmatch(r"qp_twin(\d+)_s(\d+)_300k", run)
    if m:
        return W2, f"twin{m.group(1)}_s{m.group(2)}"
    if re.fullmatch(r"(clia|recipe)_s\d+", run):
        return V1, run
    return W2, run


if __name__ == "__main__":
    for src in map(Path, sys.argv[1:]):
        for run in sorted(p for p in (src / "runs").glob("*") if (p / "final_ema.pt").exists()):
            root, name = final_name(run.name)
            dst = root / name
            dst.mkdir(parents=True, exist_ok=True)
            st = torch.load(run / "final_ema.pt", map_location="cpu")
            assert "ema" in st and st.get("steps"), f"{run}: not a finished checkpoint"
            for f in ("final_ema.pt", "log.jsonl"):
                if (run / f).exists() and not (dst / f).exists():
                    os.link(run / f, dst / f)
            print(f"{run.name} -> {dst} (arm {st.get('arm')}, seed {st.get('seed')}, steps {st.get('steps')})")
