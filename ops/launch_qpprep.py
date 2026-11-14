"""Push the CPU kernel that downloads + packs the Qure projections on one account (output used later via kernel_sources).

usage: python ops/launch_qpprep.py <ac_N>
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

if __name__ == "__main__":
    label = sys.argv[1]
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"prep_qp_{label}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "prep_qp.py").write_text((ROOT / "kernels" / "prep_qp" / "prep_qp.py").read_text(), encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/s4-qpprep", title="s4-qpprep", code_file="prep_qp.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=True, dataset_sources=[f"{user}/s4-code"],
        competition_sources=[], kernel_sources=[], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
