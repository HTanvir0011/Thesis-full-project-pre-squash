import numpy as np
import torch

from s4.cls_hybrid import MultiStream
from s4.xai_multistream import blur, explain, gradcam_pp, hit, integrated_gradients, nih_boxes


def tiny():
    torch.manual_seed(0)
    return MultiStream("resnet18", 14, "hybrid", size=64, pretrained=False).eval()


def test_explain_maps_and_routing():
    maps, rho = explain(tiny(), torch.rand(1, 3, 64, 64), 5)
    assert set(maps) == {"cam_0", "cam_1", "cam_2", "attn", "ig_X", "ig_S", "ig_B"}
    assert all(v.shape == (64, 64) and (v >= 0).all() for v in maps.values())
    assert rho.shape == (3,) and abs(rho.sum() - 1) < 1e-4


def test_integrated_gradients_completeness():
    m, x = tiny(), torch.rand(1, 3, 64, 64)
    attr = integrated_gradients(m, x, 2, steps=64)
    with torch.no_grad():
        gap = (m(x)[0, 2] - m(blur(x, 8.0))[0, 2]).item()
    assert abs(attr.sum().item() - gap) <= 0.05 * abs(gap) + 1e-3


def test_gradcam_pp_is_nonnegative():
    cam = gradcam_pp(torch.rand(8, 4, 4), torch.randn(8, 4, 4))
    assert cam.shape == (4, 4) and (cam >= 0).all()


def test_boxes_and_pointing(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,\nimg1.png,Infiltrate,512,256,128,64\nimg9.png,Mass,1,1,1,1\n")
    b = nih_boxes(p, ["img0.png", "img1.png"], 256)
    assert b == {1: [(3, 128.0, 64.0, 160.0, 80.0)]}  # Infiltrate -> Infiltration (index 3), scaled 1024 -> 256
    m = np.zeros((256, 256))
    m[70, 150] = 1
    assert hit(m, [b[1][0][1:]], 0) and not hit(m, [(0, 0, 10, 10)], 15)
