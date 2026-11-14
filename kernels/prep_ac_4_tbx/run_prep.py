"""Kaggle GPU kernel (template): build downstream image arrays with s4.prep (frozen suppressor ensemble + Qure from Hugging Face).

Placeholder ARGS (filled by ops/launch_prep.py) = the s4.prep argv. Output: /kaggle/working/<tag>/ (see s4/prep.py).
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

ARGS = ['--dataset', 'tbx', '--weights', 'cnx_t16_s0.pt+cnx_t16_s1.pt+cnx_t16_s2.pt', '--qure', 'none', '--arrays', 'orig,soft,bone,soft_qure', '--tag', 'tbx']
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
CODE = str(Path(code[0]).parent.parent)
if "node21" in ARGS:  # Node21 is read straight from its Zenodo zip by HTTP range requests
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip"], check=False)
print("s4.prep", ARGS, flush=True)
rc = subprocess.run([sys.executable, "-m", "s4.prep", *ARGS], env=dict(os.environ, PYTHONPATH=CODE)).returncode
print("PREP rc", rc, flush=True)
sys.exit(rc)
