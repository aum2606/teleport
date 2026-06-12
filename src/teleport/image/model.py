"""Phase 3 — convolutional autoencoder + spatial entropy context model
(IMPLEMENTATION_PLAN.md section 3.1).

Architecture (Balle-style, deliberately small per CLAUDE.md "don't train
large models"):

  Encoder: 3 stride-2 conv layers, /8 spatial downsample, -> latent
           (latent_channels, H/8, W/8)
  Decoder: mirrored stride-2 transposed convs, sigmoid output in [0, 1]

Quantization: additive uniform noise U(-0.5, 0.5) during training (the
standard differentiable proxy for rounding), hard `round()` at test time.

Entropy model: `SpatialContextModel` predicts a per-position Gaussian
(mean, scale) for each latent symbol from its causal context — the
already-decoded value immediately to its left and immediately above it
(within the same channel) plus its channel index. Symbols are coded
channel-major, raster order within each channel
(`codec.compress_image`'s `z_hat.reshape(-1)`), so both context values are
always already decoded by the time a symbol is coded — this is the
"spatial entropy context" the Phase 3 acceptance gap called for.
`gaussian_prob_table()` discretizes a (mean, scale) pair into the
256-symbol distribution the Phase 1 arithmetic coder needs (reusing the
frozen coder — CLAUDE.md rule: never modify the coder, adapt the
predictor).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

_SQRT2 = math.sqrt(2.0)


class Encoder(nn.Module):
    def __init__(self, latent_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, latent_channels, 5, stride=2, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def gaussian_prob_table(mean: float, scale: float) -> np.ndarray:
    """256-symbol probability table for a Gaussian(mean, scale) over
    integer-quantized values, byte = value + 128.

    Used both to estimate rate during training (via `rate_bits`, vectorized
    in torch) and, parametrized per-position, by `ContextLatentPredictor` at
    inference — the same discretization either way so encoder and decoder
    agree.
    """
    scale = max(scale, 1e-6)

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf((x - mean) / (scale * _SQRT2)))

    probs = np.empty(256, dtype=np.float64)
    for byte in range(256):
        value = byte - 128
        if byte == 0:
            probs[byte] = cdf(value + 0.5)
        elif byte == 255:
            probs[byte] = 1.0 - cdf(value - 0.5)
        else:
            probs[byte] = cdf(value + 0.5) - cdf(value - 0.5)

    probs = np.maximum(probs, 1e-12)
    return probs / probs.sum()


def shift_context(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each position in `z` (B, C, H, W), the value immediately to its
    left and immediately above it (within the same channel), 0 at the
    image edges — the same causal context `ContextLatentPredictor` builds
    incrementally from decoded symbols at inference."""
    left = torch.zeros_like(z)
    left[:, :, :, 1:] = z[:, :, :, :-1]
    top = torch.zeros_like(z)
    top[:, :, 1:, :] = z[:, :, :-1, :]
    return left, top


class SpatialContextModel(nn.Module):
    """Predicts a per-position Gaussian(mean, scale) for each latent symbol
    from (left neighbor, top neighbor, channel index) — a small causal
    "PixelCNN-lite" context model, evaluated position-by-position at
    inference (`ContextLatentPredictor`) and in one parallel pass over
    shifted tensors during training (`rate_bits`).
    """

    def __init__(self, channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def _features(self, left: torch.Tensor, top: torch.Tensor, channel_idx: torch.Tensor) -> torch.Tensor:
        channel_norm = channel_idx.float() / max(self.channels - 1, 1)
        return torch.stack([left.float(), top.float(), channel_norm], dim=-1)

    def forward(
        self, left: torch.Tensor, top: torch.Tensor, channel_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(self._features(left, top, channel_idx))
        return out[..., 0], out[..., 1]

    def rate_bits(
        self, z: torch.Tensor, left: torch.Tensor, top: torch.Tensor, channel_idx: torch.Tensor
    ) -> torch.Tensor:
        """Total estimated bits to code `z` (B, C, H, W) under the model."""
        mean, log_scale = self(left, top, channel_idx)
        scale = torch.exp(log_scale).clamp(min=1e-6)
        upper = _standard_normal_cdf((z + 0.5 - mean) / scale)
        lower = _standard_normal_cdf((z - 0.5 - mean) / scale)
        likelihood = (upper - lower).clamp(min=1e-9)
        return (-torch.log2(likelihood)).sum()


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 4, hidden: int = 32, context_hidden: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(latent_channels, hidden)
        self.decoder = Decoder(latent_channels, hidden)
        self.context_model = SpatialContextModel(latent_channels, context_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass: returns (x_hat, z, z_tilde)."""
        z = self.encoder(x)
        noise = torch.rand_like(z) - 0.5
        z_tilde = z + noise
        x_hat = self.decoder(z_tilde)
        return x_hat, z, z_tilde

    def rate_bits(self, z_tilde: torch.Tensor) -> torch.Tensor:
        """Estimated bits to code `z_tilde` (B, C, H, W) under the spatial
        context model, using causal neighbors of `z_tilde` itself as
        context (matching what `ContextLatentPredictor` sees at inference:
        the already-coded, quantized neighbor values)."""
        b, c, h, w = z_tilde.shape
        left, top = shift_context(z_tilde)
        channel_idx = torch.arange(c).view(1, c, 1, 1).expand(b, c, h, w)
        return self.context_model.rate_bits(z_tilde, left, top, channel_idx)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: encode and hard-quantize, clamped to the byte range."""
        z = self.encoder(x)
        return torch.clamp(torch.round(z), -128, 127)

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_hat)
