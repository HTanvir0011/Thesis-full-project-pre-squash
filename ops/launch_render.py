"""Create/refresh the private s4-code dataset for one account and push a render-shard kernel.

usage: python ops/launch_render.py <ac_1|ac_2|ac_3> <shard> <nshard> [limit] [--test]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KG = [sys.executable, str(ROOT / "ops" / "kg.py")]


def run(label, *args):
    r = subprocess.run([*KG, label, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print((r.stdout or "").strip(), "\n".join(l for l in (r.stderr or "").splitlines() if "outdated" not in l).strip())
    return r.returncode


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")  # no BOM: kaggle's json parser rejects one


if __name__ == "__main__":
    label, shard, nshard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    test = "--test" in sys.argv
    rest = [a for a in sys.argv[4:] if not a.startswith("--")]
    limit = int(rest[0]) if rest else None
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    code = ROOT / "kdata" / "s4-code"
    write_json(code / "dataset-metadata.json", dict(id=f"{user}/s4-code", title="s4-code", licenses=[{"name": "CC0-1.0"}], isPrivate=True))
    if run(label, "datasets", "version", "-p", str(code), "-m", "refresh", "--dir-mode", "zip") != 0:
        run(label, "datasets", "create", "-p", str(code), "--dir-mode", "zip")
    for _ in range(40):  # a kernel pushed before the dataset is ready silently drops it as a source
        st = subprocess.run([*KG, label, "datasets", "status", f"{user}/s4-code"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace").stdout
        if "ready" in st:
            break
        time.sleep(8)
    name = f"s4-render-{'test' if test else shard}"
    kdir = ROOT / "kernels" / f"render_{label}"
    kdir.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kernels" / "render" / "render_shard.py").read_text()
    for k, v in dict(__SHARD__=shard, __NSHARD__=nshard, __LIMIT__=limit, __TEST__=test).items():
        src = src.replace(k, repr(v))
    (kdir / "render_shard.py").write_text(src, encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="render_shard.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=["washingtongold/lidcidri30", "vafaeii/luna16", f"{user}/s4-code"], competition_sources=[], kernel_sources=[],
        model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
