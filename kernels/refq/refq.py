"""Kaggle CPU kernel: 257-point intensity quantiles of real chest radiographs (NIH train_val_list only) for histogram matching."""
import glob
import json
import os
import random

import cv2
import numpy as np

root = next(p for p in glob.glob("/kaggle/input/datasets/nih-chest-xrays/data") + glob.glob("/kaggle/input/data") if os.path.isdir(p))
train = set(open(os.path.join(root, "train_val_list.txt")).read().split())
paths = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "images_*", "images", "*.png"))}
ids = sorted(train & set(paths))
print("train_val images found:", len(ids), "of", len(train), flush=True)
random.Random(20261026).shuffle(ids)
q, used = [], []
for n in ids:
    if len(q) >= 4000:
        break
    im = cv2.imread(paths[n], cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2:
        continue
    x = cv2.resize(im.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    q.append(np.quantile(x, np.linspace(0, 1, 257)))
    used.append(n)
np.save("/kaggle/working/ref_q.npy", np.array(q, np.float32))
json.dump(used, open("/kaggle/working/ref_q_ids.json", "w"))
print("saved", np.array(q).shape, "median of medians", float(np.median([r[128] for r in q])))
