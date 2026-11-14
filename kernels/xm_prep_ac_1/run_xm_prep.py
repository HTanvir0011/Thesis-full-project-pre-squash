"""Kaggle CPU kernel (internet on): the CT-RATE subset of the cross-modality study (DESIGN_v4.md section 2, s4/ct_prep.py build).

Reads the volume list s4/data/ct_subset.json and the projection metadata s4/data/qp_meta.csv from the s4-code dataset and the Hugging Face
read token from the Kaggle secret HF_TOKEN (attach it in the kernel editor: Add-ons, Secrets; the token's account must have accepted the
CT-RATE terms). The token reaches the build process through its environment only; it is never written or printed. Output
/kaggle/working/ct (about 10 GB): the s4/ct_prep.py arrays, which the training kernels mount as kernel_sources s4-ctprep (data name 'ct').
No volume starts after HOURS, so the session ends with its output saved; LIMIT > 0 builds only the first LIMIT volumes (smoke run).
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

LIMIT = 0
HOURS = 10.5
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
DATA = Path(CODE) / "s4" / "data"

from kaggle_secrets import UserSecretsClient  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

token = UserSecretsClient().get_secret("HF_TOKEN")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nibabel"], check=False)
meta = [hf_hub_download("ibrahimhamamci/CT-RATE", f"dataset/metadata/{f}", repo_type="dataset", token=token, local_dir="/tmp/ctrate_meta")
        for f in ("train_metadata.csv", "validation_metadata.csv")]
cmd = [sys.executable, "-m", "s4.ct_prep", "build", "--subset", str(DATA / "ct_subset.json"), "--meta", str(DATA / "qp_meta.csv"),
       "--out", "/kaggle/working/ct", "--metadata", *meta, "--workers", "4", "--hours", str(HOURS)] + (["--limit", str(LIMIT)] if LIMIT else [])
rc = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=CODE, HF_TOKEN=token, HF_HUB_DISABLE_SYMLINKS_WARNING="1")).returncode
print("build rc", rc, flush=True)
