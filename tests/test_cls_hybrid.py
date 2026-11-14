import json

import numpy as np
import torch

from s4.cls_hybrid import MultiStream, load_x, main


def test_shapes_and_routing():
    m = MultiStream("resnet18", 5, "hybrid", size=64, pretrained=False).eval()
    logits = m(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2, 5)
    rho = m.routing()
    assert rho.shape == (2, 5, 3) and torch.allclose(rho.sum(-1), torch.ones(2, 5), atol=1e-5)


def test_ctrl_streams_are_identical_at_init():
    """X in all three streams and zero stream embeddings: the decoder cannot tell the streams apart, so each gets 1/3."""
    m = MultiStream("resnet18", 4, "ctrl", size=64, pretrained=False).eval()
    m(torch.rand(2, 3, 64, 64))
    assert torch.allclose(m.routing(), torch.full((2, 4, 3), 1 / 3), atol=1e-4)


def test_bone_stream_is_orig_minus_soft():
    arr = dict(orig=np.array([[[200, 10]]], np.uint8), soft=np.array([[[150, 30]]], np.uint8))
    x = load_x(arr, [0])
    assert torch.allclose(x[0, 2], torch.tensor([[50.0, 0.0]]) / 255)


def test_train_end_to_end_cpu(tmp_path):
    rng = np.random.default_rng(0)
    n, s = 48, 64
    np.save(tmp_path / "orig.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    np.save(tmp_path / "soft.npy", rng.integers(0, 255, (n, s, s), dtype=np.uint8))
    y = (rng.random((n, 4)) < 0.5).astype(np.float32)
    y[0, 1] = np.nan
    np.save(tmp_path / "labels.npy", y)
    np.save(tmp_path / "split.npy", np.array(["train"] * 24 + ["val"] * 12 + ["test"] * 12))
    out = tmp_path / "run"
    main(["--data", str(tmp_path), "--config", "hybrid", "--arch", "resnet18", "--pretrained", "0", "--epochs", "1", "--max_steps", "2",
          "--bs", "8", "--size", "64", "--out", str(out)])
    r = json.loads((out / "result.json").read_text())
    assert r["config"] == "hybrid" and len(r["routing_mean"]) == 4
    assert np.load(out / "test_routing.npy").shape == (12, 4, 3)
    assert (out / "model.pt").exists()


def test_labels_from_detection_boxes(tmp_path):
    from s4.cls_hybrid import labels_from_ann
    (tmp_path / "ann.json").write_text('[{"boxes": [[1, 2, 3, 4]], "labels": [2]}, {"boxes": [], "labels": []}]')
    y = labels_from_ann(tmp_path / "ann.json", 4)
    assert y.tolist() == [[0, 0, 1, 0], [0, 0, 0, 0]]


def test_accumulation_matches_full_batch():
    """--accum splits the batch; with a layer-norm encoder (no BatchNorm) and dropout off the gradients are those of the full batch."""
    from s4.cls import masked_bce
    arr = dict(orig=np.random.default_rng(0).integers(0, 255, (8, 64, 64), dtype=np.uint8),
               soft=np.random.default_rng(1).integers(0, 255, (8, 64, 64), dtype=np.uint8))
    y = torch.from_numpy((np.random.default_rng(2).random((8, 3)) < .5).astype(np.float32))
    grads = []
    for accum in (1, 4):
        torch.manual_seed(0)
        m = MultiStream("convnextv2_atto", 3, "hybrid", size=64, pretrained=False).eval()
        idx = np.arange(8)
        for part in np.array_split(idx, accum):
            (masked_bce(m(load_x(arr, part, 64)), y[part]) * (len(part) / len(idx))).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert (grads[0] - grads[1]).abs().max() < 1e-4 * grads[0].abs().max() + 1e-7
