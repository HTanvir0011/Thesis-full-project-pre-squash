"""Push the CPU kernel that downloads and reduces the CT-RATE subset (kernels/xm_prep/run_xm_prep.py) on one account: s4-ctprep.

usage: python ops/launch_xm_prep.py <ac_N> [--limit N]
Before it can run: tools/package_code.py (s4/data/ct_subset.json ships in s4-code), the CT-RATE terms accepted on Hugging Face by the
account whose read token is used, and that token attached as the Kaggle secret HF_TOKEN of this kernel (editor: Add-ons, Secrets). A
first push without the secret fails at get_secret: attach it in the editor and run the kernel again.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

if __name__ == "__main__":
    label = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"xm_prep_{label}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "run_xm_prep.py").write_text((ROOT / "kernels" / "xm_prep" / "run_xm_prep.py").read_text().replace("__LIMIT__", str(limit)), encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/s4-ctprep", title="s4-ctprep", code_file="run_xm_prep.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=True, dataset_sources=[f"{user}/s4-code"],
        competition_sources=[], kernel_sources=[], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
