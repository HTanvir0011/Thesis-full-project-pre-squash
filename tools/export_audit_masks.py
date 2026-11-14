"""Export the lung masks and bone-overlap masks that s4.audit_real draws its lesion sites from, so that another machine reproduces this
machine's sites exactly (the PSPNet lung mask and Qure's bone map differ slightly between GPUs, which moves the sites).

python tools/export_audit_masks.py OUT_DIR [dataset ...]  ->  OUT_DIR/<dataset>.npz ; then S4_AUDIT_MASKS=OUT_DIR python -m s4.audit_real ...
ribcxr is licensed: export it only to a private folder (kdata/s4-ribmask -> ac_1), never to outputs/audit_masks, which ships in s4-code.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4.audit_real import bone_fn, load_images, lung_masks  # noqa: E402

SETS = (("szch", "data/szch", None), ("jsrt", "data/jsrt/images", "qure:data/qure/weights/bone_suppression.ts"),
        ("ribcxr", "data/ribcxr_raw", None))  # as in tools/final_local.sh

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for ds, root, ref in SETS:
        if ds not in (sys.argv[2:] or ("szch", "jsrt")):
            continue
        pairs = load_images(ds, root)
        lungs = lung_masks([x for _, x, _ in pairs], "cuda")
        f = bone_fn(ref, "cuda") if ref else None
        bones = []
        for _, x, b in pairs:
            if b is None:  # no bone truth (jsrt): same rule as s4/audit_real.py main
                r = f(x)
                b = r > 0.25 * np.percentile(r, 99)
            bones.append(b)
        np.savez_compressed(out / f"{ds}.npz", names=np.array([n for n, _, _ in pairs]),
                            lung=np.stack([np.packbits(m.ravel()) for m in lungs]), bone=np.stack([np.packbits(b.ravel()) for b in bones]))
        print(ds, len(pairs), "images")
