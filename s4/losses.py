"""Bone-image loss: 0.5 L1 + 0.5 MSE + 0.5 (1 - SSIM) + w_roi (L1 + MSE on the lung-structure ROI) + w_nod (same on lesions)."""
import torch
import torch.nn.functional as F


def _gauss(win=11, sigma=1.5, device=None):
    g = torch.exp(-((torch.arange(win, dtype=torch.float32, device=device) - win // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, data_range=1.0):
    k = _gauss(device=x.device).to(x.dtype)
    mu = lambda t: F.conv2d(t, k, padding=5)  # noqa: E731
    mx, my = mu(x), mu(y)
    vx, vy, cxy = mu(x * x) - mx**2, mu(y * y) - my**2, mu(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))).mean((1, 2, 3))


def masked(pred, target, mask):
    m = mask.to(pred.dtype)
    n = m.sum((1, 2, 3)).clamp_min(1.0)
    return ((pred - target).abs() * m).sum((1, 2, 3)) / n + (((pred - target) ** 2) * m).sum((1, 2, 3)) / n


def suppression_loss(pred, target, roi, nodule=None, w_roi=2.5, w_nod=0.0):
    """-> (scalar, dict of per-term values). roi/nodule: [B,1,H,W] {0,1}. pred/target in bone-image units."""
    pred, target = pred.float(), target.float()
    parts = dict(l1=(pred - target).abs().mean(), mse=((pred - target) ** 2).mean(), ssim=(1 - ssim(pred, target)).mean())
    total = 0.5 * parts["l1"] + 0.5 * parts["mse"] + 0.5 * parts["ssim"]
    if w_roi:
        parts["roi"] = masked(pred, target, roi).mean()
        total = total + w_roi * parts["roi"]
    if w_nod and nodule is not None:
        parts["nod"] = masked(pred, target, nodule).mean()
        total = total + w_nod * parts["nod"]
    return total, parts
