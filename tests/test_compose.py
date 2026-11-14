import numpy as np
import pytest
import torch

from s4 import compose as C
from s4 import xray
from s4.losses import ssim, suppression_loss
from s4.model import SwinUNetPP

DEV = "cpu"


@pytest.fixture(scope="module")
def phys():
    return C.Physics(DEV)


def maps(B=2, H=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(B, 4, H, H)
    a[:, 0] = torch.rand(B, H, H, generator=g) * 4  # bone g/cm^2
    a[:, 1] = 10 + torch.rand(B, H, H, generator=g) * 20
    a[:, 2] = torch.rand(B, H, H, generator=g) * 2
    return a


def test_lut_matches_numpy_physics(phys):
    a = maps()[:, :3]
    idx = torch.tensor([3, 3])  # 100 kVp
    got = phys(a, idx).numpy()
    for c, m in enumerate(xray.MATERIALS):
        np.testing.assert_allclose(got[:, c], xray.to_pixel(a[:, c].numpy(), 100, m), atol=2e-4)


def test_identity_affine_is_identity():
    x = torch.rand(2, 4, 32, 32)
    y = C.random_affine(x, rot_deg=0, zoom=(1, 1), shift=0, flip=False)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=0)


def test_insertion_never_touches_bone_and_adds_lesions():
    a = maps(B=3, H=96)
    before = a.clone()
    lib = [torch.ones(8, 8) * 0.5]
    C.insert_nodules(a, lib, n_max=3, gen=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a[:, :2], before[:, :2])
    assert (a[:, 2] >= before[:, 2]).all() and a[:, 3].sum() > 0 and a[:, 2].sum() > before[:, 2].sum()


def test_compose_components_are_exact_and_nonnegative(phys):
    a = maps(B=4, H=64)
    out = C.compose(a, torch.tensor([1, 3, 5, 7]), phys, gen=torch.Generator().manual_seed(2), detector=False)
    torch.testing.assert_close(out["X"], out["S"] + out["B"], atol=1e-6, rtol=0)
    assert out["B"].min() >= -1e-6 and out["B"].max() > 0.01


def test_hist_tone_matches_reference_quantiles(phys):
    a = maps(B=2, H=64)
    ref = torch.sort(torch.rand(1, 257) ** 2, dim=1).values
    ref[:, 0], ref[:, -1] = 0.0, 1.0
    out = C.compose(a, torch.tensor([3, 3]), phys, gen=torch.Generator().manual_seed(3), tone="hist", ref_q=ref, detector=False)
    med = torch.quantile(out["X"].flatten(1), 0.5, dim=1)
    assert torch.allclose(med, ref[0, 128].expand(2), atol=0.02)
    assert out["B"].min() >= -1e-6  # monotone curve keeps the bone component non-negative


def test_model_and_loss_backward():
    m = SwinUNetPP(pretrained=False, img_size=256)
    x = torch.rand(1, 1, 256, 256)
    y = m(x)
    assert y.shape == x.shape
    roi = (torch.rand_like(x) > 0.8).float()
    loss, parts = suppression_loss(y, torch.rand_like(x), roi, roi, w_nod=1.0)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-4)
