"""Push the H10 XAI kernel (kernels/xai/run_xai.py): s4/xai_multistream.py on every finished NIH hybrid run in the outputs of earlier
phase-2 kernels of the same account, with the NIH expert boxes of the Kaggle data set nih-chest-xrays/data.

usage: python ops/launch_xai.py <ac_N> <kernel_name> --sources slugA[,slugB,...] [--prep s4-prep-nih] [--limit N] [--runs REGEX] [--datasets d1,...]
--sources: phase-2 kernels of the account holding results/nih_hyb_hybrid@*/model.pt; a run only a rerun kernel <slug>-r finished needs
that kernel listed too. Their job files outputs/jobs/<slug>.json name the expected runs: the kernel lists those it cannot find as failed.
--limit N: first N boxed test images per run (smoke).
--runs REGEX: run tags to take (default s4/xai_runs.py TAG, the v3 hybrid runs); --datasets: private datasets of the account with
more results/<tag>/ runs (v4: s4-dn121).
"""
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

sys.path.insert(0, str(ROOT))
from s4.xai_runs import TAG, expected  # noqa: E402


def parse(argv):
    """argv after the script name -> (label, name, source slugs, prep slug, limit, run pattern, dataset slugs); any other option is an error."""
    label, name, rest = argv[0], argv[1], argv[2:]
    opts = dict(zip(rest[0::2], rest[1::2]))
    sources = opts.get("--sources", "").split(",")
    if len(rest) % 2 or not set(opts) <= {"--sources", "--prep", "--limit", "--runs", "--datasets"} or not all(sources):
        raise SystemExit(f"launch_xai: expected --sources LIST [--prep SLUG] [--limit N] [--runs REGEX] [--datasets LIST], got {rest}")
    return (label, name, sources, opts.get("--prep", "s4-prep-nih"), int(opts.get("--limit", 0)), opts.get("--runs", TAG.pattern),
            [d for d in opts.get("--datasets", "").split(",") if d])


def render(template, limit, expect, pattern):
    """The kernel script with LIMIT, EXPECT and PATTERN filled in."""
    assert all(f"{k} = __{k}__" in template for k in ("LIMIT", "EXPECT", "PATTERN")), "run_xai.py placeholders"
    return template.replace("__LIMIT__", repr(limit)).replace("__EXPECT__", repr(expect)).replace("__PATTERN__", repr(pattern))


if __name__ == "__main__":
    label, name, sources, prep, limit, pattern, datasets = parse(sys.argv[1:])
    expect = expected((ROOT / "outputs" / "jobs" / f"{s}.json" for s in sources), re.compile(pattern))  # before the code upload: a wrong source costs nothing
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"xai_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "run_xai.py").write_text(render((ROOT / "kernels" / "xai" / "run_xai.py").read_text(), limit, expect, pattern), encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_xai.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
        dataset_sources=[f"{user}/s4-code", "nih-chest-xrays/data", *(f"{user}/{d}" for d in datasets)], competition_sources=[],
        kernel_sources=[f"{user}/{s}" for s in [prep, *sources]], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
