"""Create or version a private Kaggle dataset from a folder and wait until it is ready.

usage: python ops/push_dataset.py <ac_N> <folder> <slug> [message] [--new]
--new skips the version attempt (a failed version call still uploads every file once, which hurts for multi-GB datasets). Slugs need >= 6 characters.
Note: Kaggle flattens a top-level sub-directory (dir-mode zip), so put the files you want at the dataset root directly in <folder>.
"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from launch_render import KG, ROOT, run, write_json  # noqa: E402

if __name__ == "__main__":
    label, folder, slug = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
    msg = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "update"
    new = "--new" in sys.argv
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    if "rib" in (slug + folder.name).lower() and label != cfg["licensed_only_on"]:  # VinDr-RibCXR and anything derived from it (e.g. s4-ribmask)
        sys.exit(f"refused: {slug} looks like licensed RibCXR data; only {cfg['licensed_only_on']} may hold it")
    user = cfg["accounts"][label]
    write_json(folder / "dataset-metadata.json", dict(id=f"{user}/{slug}", title=slug, licenses=[{"name": "other"}], isPrivate=True))
    if new or run(label, "datasets", "version", "-p", str(folder), "-m", msg, "--dir-mode", "zip") != 0:
        run(label, "datasets", "create", "-p", str(folder), "--dir-mode", "zip")
    else:
        time.sleep(90)  # status says "ready" for the OLD version until the new one registers; a kernel pushed too early mounts the old files
    for _ in range(120):
        st = subprocess.run([*KG, label, "datasets", "status", f"{user}/{slug}"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace").stdout
        if "ready" in st:
            print(f"{user}/{slug} ready")
            break
        time.sleep(10)
    else:
        sys.exit(f"{user}/{slug} not ready after 20 min")
