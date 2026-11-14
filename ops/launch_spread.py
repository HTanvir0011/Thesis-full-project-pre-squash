"""Push the NIH suppressor-spread kernel (kernels/spread/run_spread.py): s4/spread.py on the NIH test split, the second score of the SAP_v3
selective classification. Reads the raw images of nih-chest-xrays/data and the NIH prep output of the same account.

usage: python ops/launch_spread.py <ac_N> <kernel_name> --weights <dataset slug> [--files] <w1.pt+w2.pt+w3.pt> [--prep s4-prep-nih]
The weights data set is a private data set of the account (s4-wts3: cnx_t16_s{0,1,2}.pt; s4-wte: v2__qp_twin16_s{0,1,2}_cnx2_300k.pt, the
same weights): three files (the frozen K = 3 ensemble), found by name under /kaggle/input. tools/v3_phase2.py queues the --files form.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402


def parse(argv):
    """argv after the script name -> (label, name, weights slug, '+'-joined weight files, prep slug); any other form is an error."""
    a = argv[:4] + argv[5:] if argv[4:5] == ["--files"] else list(argv)
    files = a[4].split("+") if len(a) > 4 else []
    if len(a) not in (5, 7) or a[2] != "--weights" or a[5:6] not in ([], ["--prep"]) \
            or any("/" in s or s.startswith("-") for s in a[3:4] + a[6:7]) or len(files) != 3 or not all(f.endswith(".pt") for f in files):
        raise SystemExit(f"launch_spread: expected <ac_N> <name> --weights SLUG [--files] W1.pt+W2.pt+W3.pt [--prep SLUG], got {argv}")
    return a[0], a[1], a[3], a[4], a[6] if len(a) == 7 else "s4-prep-nih"


if __name__ == "__main__":
    label, name, wslug, weights, prep = parse(sys.argv[1:])
    user = yaml.safe_load(open(ROOT / "config.yaml"))["accounts"][label]
    ensure_code_dataset(label, user)
    kdir = ROOT / "kernels" / f"spread_{label}_{name}"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "run_spread.py").write_text((ROOT / "kernels" / "spread" / "run_spread.py").read_text().replace("__WEIGHTS__", repr(weights)),
                                        encoding="utf-8")
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_spread.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",  # internet: pip, PSPNet weights
        dataset_sources=[f"{user}/s4-code", f"{user}/{wslug}", "nih-chest-xrays/data"], competition_sources=[],
        kernel_sources=[f"{user}/{prep}"], model_sources=[]))
    run(label, "kernels", "push", "-p", str(kdir))
