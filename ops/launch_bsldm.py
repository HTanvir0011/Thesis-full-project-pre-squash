"""Push the BS-LDM baseline kernel.  usage: python ops/launch_bsldm.py <ac_N> <smoke|full|audit> [--fp16] [--sites A:B]   (kernel slug s4-bsldm-<mode>)
full = fidelity on the 75 SZCH pairs + audit; audit = audit only; --sites A:B evaluates sites A..B-1 of the 6 drawn per image (default 0:3; 3:6 completes the 450).
One 1000-step sampling takes ~148 s on a T4 (fp16 is not faster), so sites 0:3 + fidelity is ~7.8 session hours on a T4 x2 session."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

if __name__ == "__main__":
    label, mode = sys.argv[1], sys.argv[2]
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"bsldm_{label}"
    kdir.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kernels" / "bsldm" / "run_bsldm.py").read_text().replace("__MODE__", repr(mode)).replace("__FP16__", repr("--fp16" in sys.argv)).replace(
        "__SITES__", repr(sys.argv[sys.argv.index("--sites") + 1] if "--sites" in sys.argv else "0:3"))
    (kdir / "run_bsldm.py").write_text(src, encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/s4-bsldm-{mode}", title=f"s4-bsldm-{mode}", code_file="run_bsldm.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
        dataset_sources=[f"{user}/s4-code"], competition_sources=[], kernel_sources=[], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
