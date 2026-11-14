"""Push a CPU kernel (no GPU quota) that hashes the prep outputs of the same account (kernels/hash/run_hash.py; RUNPLAN_v3 R6).

usage: python ops/launch_hash.py <ac_N> <kernel_name> --prep s4-prep-nih[,s4-prep-vin,...]
A private kernel output mounts only in its own account, so every account with a prep copy runs its own hash kernel. Then:
python ops/fetch.py <ac_N> <kernel_name> outputs/hash/<ac_N>; python tools/compare_hashes.py outputs/hash/*/hash.json
"""
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402


def parse(argv):
    """argv after the script name -> (label, name, prep list); any other argument is an error."""
    if len(argv) != 4 or argv[2] != "--prep" or not all(argv[3].split(",")):
        raise SystemExit(f"launch_hash: expected <ac_N> <kernel_name> --prep LIST, got {argv}")
    return argv[0], argv[1], argv[3].split(",")


if __name__ == "__main__":
    label, name, prep = parse(sys.argv[1:])
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)  # the kernel imports s4.filehash from the account's s4-code
    kdir = ROOT / "kernels" / f"hash_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "kernels" / "hash" / "run_hash.py", kdir / "run_hash.py")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_hash.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=False, machine_shape="",
        dataset_sources=[f"{user}/s4-code"], competition_sources=[], kernel_sources=[f"{user}/{p}" for p in prep], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
