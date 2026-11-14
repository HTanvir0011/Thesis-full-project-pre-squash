import csv

import cv2
import numpy as np
import torch

from s4 import qp


def _fake_pack(d, n=6, size=128):
    rng = np.random.default_rng(0)
    d.mkdir(parents=True, exist_ok=True)
    index, pos, rows = np.zeros((n, 3, 2), np.int64), 0, []
    with open(d / "qp.bin", "wb") as f:
        for i in range(n):
            soft = cv2.GaussianBlur(rng.uniform(60, 200, (size, size)).astype(np.float32), (0, 0), 9)
            bone = np.zeros((size, size), np.float32)
            bone[:, 30:40] = 90
            bone[:, 70:80] = 90
            lung = np.zeros((size, size), np.float32)
            lung[20:100, 20:100] = 150
            w = 0.5
            for c, img in enumerate((np.clip(soft + w * bone, 0, 255), bone, lung)):
                b = cv2.imencode(".png", img.astype(np.uint8))[1].tobytes()
                f.write(b)
                index[i, c] = (pos, len(b))
                pos += len(b)
            rows.append(dict(volume_name=f"v{i}", split="train" if i < 4 else "val", w_bone=w))
    np.save(d / "qp_index.npy", index)
    with open(d / "qp_meta.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)


def test_components_stay_exact_and_lesions_never_touch_bone(tmp_path):
    _fake_pack(tmp_path)
    bank = qp.QPBank(tmp_path, "train", "cpu", workers=1)
    assert len(bank) == 4
    X, B, L = bank.raw(4)
    assert X.shape == (4, 1, 128, 128) and float((X - B).min()) > -0.05  # the bone image is part of X
    lib = [torch.full((9, 9), 0.05)]
    g = torch.Generator().manual_seed(0)
    o = qp.qp_compose(X, B, L, library=lib, n_insert=3, detector=False, gen=g)
    assert float((o["X"] - o["B"] - o["S"]).abs().max()) < 1e-5  # X' = S' + B' exactly, whatever the tone curve
    assert float(o["B"].min()) >= -1e-6
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    clean = qp.qp_compose(X, B, L, library=lib, n_insert=0, detector=False, gen=g1)
    with_les = qp.qp_compose(X, B, L, library=lib, n_insert=3, detector=False, gen=g2)
    assert float(with_les["nod"].sum()) > 0 and float(clean["nod"].sum()) == 0
    assert float((with_les["X"] - clean["X"]).abs().max()) > 0.01  # the lesion changed the input
    assert float(with_les["B"].sum()) > 0.9 * float(clean["B"].sum())  # ...while the bone target stays (up to the tone curve)


def test_noise_and_blur_only_touch_what_they_should(tmp_path):
    _fake_pack(tmp_path)
    bank = qp.QPBank(tmp_path, "train", "cpu", workers=1)
    X, B, L = bank.raw(4)
    o = qp.qp_compose(X, B, L, detector=True, rot_deg=0.0, gen=torch.Generator().manual_seed(3))
    assert (o["X"] - o["B"] - o["S"]).abs().max() > 1e-4  # noise/scatter are added to the input only
    assert o["B"].min() >= -1e-6
