"""Push a downstream-array preparation kernel (s4.prep with the frozen suppressor ensemble; Qure is downloaded from Hugging Face in the kernel).

usage: python ops/launch_prep.py <ac_N> <nih|nihq|tbx|vin|node21|jsrt> <weights_slug> <w1.pt+w2.pt+...> [--limit N] [--arrays orig,soft,...] [--no-qure]
The weights dataset is a private dataset of the same account (ops/push_dataset.py); files are found by name under /kaggle/input.
Kernel slug: s4-prep-<dataset>[-n<limit>]; its output is mounted by the downstream kernels of the same account (ops/launch_down.py --prep).
Node21 first needs the account's CPU download kernel: python ops/push_kernel.py <ac_N> kernels/fetch_node21 s4-node21raw -
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import ROOT, run, write_json  # noqa: E402
from push_kernel import ensure_code_dataset  # noqa: E402

# Kaggle sources per dataset: (datasets, competitions). Node21 comes from Zenodo inside the kernel (internet), so it has none.
SOURCES = {"nih": (["nih-chest-xrays/data"], []), "nihq": (["nih-chest-xrays/data"], []),
           "tbx": (["omareldash75/tbx11k-original-dataset-with-bounding-boxes"], []),
           "vin": (["xhlulu/vinbigdata-chest-xray-resized-png-1024x1024"], ["vinbigdata-chest-xray-abnormalities-detection"]),
           "node21": ([], []),
           "jsrt": (["raddar/nodules-in-chest-xrays-jsrt"], []),
           "bdcxr": ([], [])}  # BDCXR comes from figshare inside the kernel (the lab's DUA forbids re-uploading it)
# detection sets keep the bone image too (input variant 'both' = orig, soft, bone)
# NIH at 256 px is 7.35 GB per array: orig+soft (s4-prep-nih) and soft_qure (s4-prep-nihq) go to separate kernels (20 GB output limit)
ARRAYS = {"nih": "orig,soft", "nihq": "soft_qure", "tbx": "orig,soft,bone,soft_qure", "vin": "orig,soft,bone,soft_qure",
          "node21": "orig,soft,bone,soft_qure", "jsrt": "orig,soft,bone,soft_qure", "bdcxr": "orig,soft,soft_qure"}


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    label, ds, wslug, weights = sys.argv[1:5]
    limit = int(opt("--limit", 0))
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    user = cfg["accounts"][label]
    if ds == "vin" and label not in cfg["vinbigdata_joined"]:
        sys.exit(f"VinBigData needs the competition rules accepted: {label} is not in vinbigdata_joined (config.yaml)")
    ensure_code_dataset(label, user)
    name = f"s4-prep-{ds}" + (f"-n{limit}" if limit else "")
    kdir = ROOT / "kernels" / f"prep_{label}_{ds}"
    kdir.mkdir(parents=True, exist_ok=True)
    arrays = opt("--arrays", ARRAYS[ds])
    args = ["--dataset", "nih" if ds == "nihq" else ds, "--weights", weights, "--qure", "off" if "--no-qure" in sys.argv else "none", "--arrays", arrays, "--tag", ds,
            *(["--limit", str(limit)] if limit else [])]
    src = (ROOT / "kernels" / "prep" / "run_prep.py").read_text().replace("__ARGS__", repr(args))
    (kdir / "run_prep.py").write_text(src, encoding="utf-8")
    for old in kdir.glob("prep_variants.py"):  # the older inline template must not be picked up as the code file
        old.unlink()
    dsrc, csrc = SOURCES[ds]
    write_json(kdir / "kernel-metadata.json", dict(
        id=f"{user}/{name}", title=name, code_file="run_prep.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
        dataset_sources=dsrc + [f"{user}/{wslug}", f"{user}/s4-code"], competition_sources=csrc,
        kernel_sources=[f"{user}/s4-node21raw"] if ds == "node21" else [], model_sources=[]))  # Node21 images: CPU kernel kernels/fetch_node21
    run(label, "kernels", "push", "-p", str(kdir))
