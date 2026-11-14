"""Push a downstream training kernel (classification / detection) that reads arrays prepared by s4-prep-* kernels of the same account.

usage: python ops/launch_down.py <ac_N> <kernel_name> <jobs.json> --prep s4-prep-nih[,s4-prep-tbx,...] [--prev <kernel slug>] [--datasets s4-qproj]
--prev mounts an earlier kernel of the same account, whose finished runs are skipped and unfinished runs resumed (RUNPLAN_v3 R1).
--prep lists every kernel output to mount (preps, s4-ctprep, s4-qpprep, earlier v4 kernels whose runs '@run:' arguments name);
--datasets adds private datasets of the account next to s4-code (v4: s4-qproj when the account has no s4-qpprep output).
jobs.json: [[gpu, kind, data_dir_name, variant, seed, epochs, extra argv...], ...]  (see kernels/down/run_down.py)
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

def parse(argv):
    """argv after the script name -> (label, name, jobs_file, prep list, prev slug or None, dataset list); any other option is an error."""
    label, name, jobs_file, rest = argv[0], argv[1], argv[2], argv[3:]
    opts = dict(zip(rest[0::2], rest[1::2]))
    if len(rest) % 2 or not set(opts) <= {"--prep", "--prev", "--datasets"} or "--prep" not in opts:
        raise SystemExit(f"launch_down: expected --prep LIST [--prev SLUG] [--datasets LIST], got {rest}")
    return label, name, jobs_file, opts["--prep"].split(","), opts.get("--prev"), [d for d in opts.get("--datasets", "").split(",") if d]


def render(template, jobs, prev):
    """The kernel script: job list and --prev slug filled in (the module docstring also names __JOBS__)."""
    assert "__JOBS__" in template and "PREV_SLUG = __PREV__" in template, "run_down.py placeholders"
    return template.replace("__JOBS__", repr(jobs)).replace("__PREV__", repr(prev))


if __name__ == "__main__":
    label, name, jobs_file, prep, prev, datasets = parse(sys.argv[1:])
    jobs = json.loads(Path(jobs_file).read_text())
    assert all(j[1] in ("cls", "det", "hyb", "xm", "xmx", "xmh") for j in jobs), "job kind must be cls, det, hyb, xm, xmx or xmh"
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"down_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "run_down.py").write_text(render((ROOT / "kernels" / "down" / "run_down.py").read_text(), jobs, prev), encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_down.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
        dataset_sources=[f"{user}/s4-code", *(f"{user}/{d}" for d in datasets)], competition_sources=[], kernel_sources=[f"{user}/{p}" for p in prep] + ([f"{user}/{prev}"] if prev else []), model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
