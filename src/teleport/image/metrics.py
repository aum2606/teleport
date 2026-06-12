"""MS-SSIM (Wang, Simoncelli & Bovik 2003) — hand-rolled with torch only, no
new dependencies.

Used by `scripts/eval_image.py` to check the literal Phase 3 acceptance
criterion (IMPLEMENTATION_PLAN.md section 3: "beat JPEG on MS-SSIM at low
bitrates"), which PSNR alone does not establish.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MSSSIM_WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, data_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    channels = img1.shape[1]
    window_size = window.shape[-1]
    kernel = window.expand(channels, 1, window_size, window_size).contiguous()

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, kernel, groups=channels)
    mu2 = F.conv2d(img2, kernel, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=channels) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    return ssim_map.mean(), cs_map.mean()


def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """`img1`, `img2`: (1, C, H, W) float64 tensors. Returns a scalar, ~[0, 1]."""
    window = _gaussian_window(window_size, sigma)
    weights = _MSSSIM_WEIGHTS
    levels = weights.shape[0]

    mcs = []
    x, y = img1, img2
    for level in range(levels):
        ssim_val, cs_val = _ssim(x, y, window, data_range)
        if level < levels - 1:
            mcs.append(torch.relu(cs_val))
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
        else:
            mcs.append(torch.relu(ssim_val))

    mcs_tensor = torch.stack(mcs)
    return float(torch.prod(mcs_tensor**weights).item())
