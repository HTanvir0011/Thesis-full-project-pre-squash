"""Push kdata/s4-code as the account's private dataset (create or new version), wait until ready, then push a kernel dir.

usage: python ops/push_kernel.py <ac_N> <kernel_dir> <kernel_name> <dataset_source,...> [--gpu] [--no-internet]
The kernel dir must hold the script named in `code_file`; metadata is generated here (private, script kernel).
"""
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402


def ensure_code_dataset(label, user):
    # per-account staging copy: the metadata file names the owner, so launches for several accounts must not share one folder
    import shutil
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools" / "package_code.py")], check=True)  # current working tree, never a stale package
    src = ROOT / "kdata" / "s4-code"
    code = ROOT / "kdata" / f"s4-code-{label}"
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(src / "code", code / "code")
    assert not [p for p in (code / "code").rglob("*") if "ribcxr" in p.name.lower() and p.suffix != ".py"], "licensed RibCXR data in s4-code: not uploaded"
    write_json(code / "dataset-metadata.json", dict(id=f"{user}/s4-code", title="s4-code", licenses=[{"name": "CC0-1.0"}], isPrivate=True))
    if run(label, "datasets", "version", "-p", str(code), "-m", "refresh", "--dir-mode", "zip") != 0:
        run(label, "datasets", "create", "-p", str(code), "--dir-mode", "zip")
    else:
        time.sleep(90)  # status says "ready" for the OLD version until the new one registers (a kernel pushed in that window ran stale code)
    for _ in range(30):
        if "ready" in _status(label, user):
            return
        time.sleep(10)
    sys.exit("s4-code dataset never became ready")


def _status(label, user):
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "ops" / "kg.py"), label, "datasets", "status", f"{user}/s4-code"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "").strip()


if __name__ == "__main__":
    label, kdir, name, sources = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4].replace("-", "").split(",") if sys.argv[4] == "-" else sys.argv[4].split(",")
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    code_file = next(p.name for p in kdir.glob("*.py"))
    shape = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shape=")), "")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file=code_file, language="python", kernel_type="script", is_private=True,
        enable_gpu="--gpu" in sys.argv or bool(shape), enable_tpu=False, enable_internet="--no-internet" not in sys.argv,
        machine_shape=shape, dataset_sources=[s.replace("{user}", user) for s in sources if s] + [f"{user}/s4-code"],
        competition_sources=[], kernel_sources=[], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
