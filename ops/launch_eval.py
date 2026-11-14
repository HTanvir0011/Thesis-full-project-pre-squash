"""Push the Kaggle evaluation kernel (kernels/evalk): tools/final_local.sh steps for a job list, one process per GPU.

usage: python ops/launch_eval.py <ac_N> <kernel_name> <jobs.json> [--qp] [--ribcxr]
jobs.json: [[gpu, step, method...], ...] (see kernels/evalk/run_eval.py). The account needs the dataset s4-wte (eval weights, ops/push_dataset.py)
and, with --qp, the kernel s4-qpprep (ops/launch_qpprep.py). --ribcxr mounts the licensed VinDr-RibCXR (s4-ribcxr, s4-ribmask): ac_1 only.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

if __name__ == "__main__":
    label, name, jobs_file = sys.argv[1:4]
    jobs = json.loads(Path(jobs_file).read_text())
    assert all(j[1] in ("szch", "audit", "qp", "synth", "rib", "unc") for j in jobs), "step must be szch, audit, qp, synth, rib or unc"
    rib = "--ribcxr" in sys.argv
    assert rib or not any(j[1] == "rib" for j in jobs), "rib jobs need --ribcxr"
    assert not rib or label == "ac_1", "VinDr-RibCXR is licensed to ac_1 only"
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    bank = [f"{user}/s4-bank"] if any(j[1] == "synth" for j in jobs) else []  # only ac_1, ac_2, ac_3 hold s4-bank (1.2 GB)
    bank += [f"{user}/s4-ribcxr", f"{user}/s4-ribmask"] if rib else []  # licensed data + its local lung masks (tools/export_audit_masks.py)
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"eval_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "run_eval.py").write_text((ROOT / "kernels" / "evalk" / "run_eval.py").read_text().replace("__JOBS__", repr(jobs)), encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_eval.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
        dataset_sources=[f"{user}/s4-code", f"{user}/s4-wte", "raddar/nodules-in-chest-xrays-jsrt", *bank], competition_sources=[],
        kernel_sources=[f"{user}/s4-qpprep"] if "--qp" in sys.argv else [], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
