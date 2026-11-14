"""Push a GPU training kernel for one account.

usage: python ops/launch_train.py <ac_N> <kernel_name> <shape NvidiaTeslaT4|NvidiaL4> <jobs.json> [--extra "--limit_ct 20"] [--qp | --qpk] [--nobank] [--resume slug[,slug]]
--qp mounts the private dataset s4-qproj (packed Qure projections; tools/qp_pack.py) for the qp/mix arms; --qpk mounts the output of the account's s4-qpprep kernel instead
--resume mounts the output of earlier kernels of this account (kernel_sources) so unfinished runs continue from their ckpt_last.pt
jobs.json: [[gpu, arm, seed, samples, bs, lr, optional name suffix, optional extra s4.train argv], ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import KG, ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

if __name__ == "__main__":
    label, name, shape, jobs_file = sys.argv[1:5]
    extra = sys.argv[sys.argv.index("--extra") + 1].split() if "--extra" in sys.argv else []
    resume = sys.argv[sys.argv.index("--resume") + 1].split(",") if "--resume" in sys.argv else []
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"train_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kernels" / "train" / "run_train.py").read_text()
    src = src.replace("__JOBS__", repr(json.loads(Path(jobs_file).read_text()))).replace("__EXTRA__", repr(extra))
    (kdir / "run_train.py").write_text(src, encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_train.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape=shape,
        dataset_sources=[*([] if "--nobank" in sys.argv else [f"{user}/s4-bank"]), f"{user}/s4-code", *([f"{user}/s4-qproj"] if "--qp" in sys.argv else [])], competition_sources=[], kernel_sources=[f"{user}/{k}" for k in [*(["s4-qpprep"] if "--qpk" in sys.argv else []), *resume]], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
