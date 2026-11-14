"""Kaggle CPU kernel: download the processed Node21 radiographs once into /kaggle/working/node21_raw (no GPU quota). The GPU prep kernel
of the same account mounts this output (ops/launch_prep.py adds it as a kernel source for node21). Node21: CC BY-NC-ND 4.0, evaluation only."""
import glob
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip"], check=False)
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
sys.path.insert(0, str(Path(code[0]).parent.parent))
from s4.prep import node21_fetch  # noqa: E402

out = Path("/kaggle/working/node21_raw")
meta = node21_fetch(out)
n = sum(1 for _ in out.glob("*.mha"))
assert n >= meta.img_name.nunique(), f"{n} of {meta.img_name.nunique()} images"
print("FETCH DONE", n, "images", flush=True)
