"""Phase 3 image codec: ConvAutoencoder latent + Phase 1 arithmetic coder.

Wire format: a 10-byte header (orig height, width, latent channels) followed
by the arithmetic-coded latent symbols (channel-major, raster order within
each channel). The decoder reconstructs the (padded) latent grid from the
header and the model's known /8 downsampling, runs the frozen decoder
network, and crops back to the original size.

This is lossy end to end (the point of Phase 3): `decompress(compress(x))
!= x` in general. What must stay exact is the entropy coding of the
quantized latent — `tests/test_image.py` checks that round trip.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from teleport.coder import compress as coder_compress
from teleport.coder import decompress as coder_decompress
from teleport.image.model import ConvAutoencoder, SpatialContextModel, gaussian_prob_table

_HEADER = struct.Struct(">IIH")  # height, width, latent_channels
_DOWNSAMPLE = 8  # 3 stride-2 conv layers


class ContextLatentPredictor:
    """Adaptive Predictor: at each position, predicts a Gaussian(mean, scale)
    from the spatial context model conditioned on the already-decoded
    left/top neighbors (within the same channel) and the channel index, then
    discretizes it into a 256-symbol table (`gaussian_prob_table`).

    Symbols are visited channel-major, raster order within each channel
    (matching `compress_image`'s `z_hat.reshape(-1)`), so `update()` always
    fills in left/top neighbors before they're needed by `predict()`.

    # SLOW: ok — one tiny MLP forward per latent symbol; correctness over
    # speed per CLAUDE.md (research codebase).
    """

    def __init__(self, context_model: SpatialContextModel, channels: int, latent_h: int, latent_w: int) -> None:
        self.context_model = context_model
        self.channels = channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.spatial_size = latent_h * latent_w
        self.buffer = np.zeros((channels, latent_h, latent_w), dtype=np.float64)
        self.pos = 0

    def _position(self) -> tuple[int, int, int]:
        channel = self.pos // self.spatial_size
        rem = self.pos % self.spatial_size
        h, w = divmod(rem, self.latent_w)
        return channel, h, w

    def predict(self) -> np.ndarray:
        channel, h, w = self._position()
        left = self.buffer[channel, h, w - 1] if w > 0 else 0.0
        top = self.buffer[channel, h - 1, w] if h > 0 else 0.0

        with torch.no_grad():
            left_t = torch.tensor([left], dtype=torch.float32)
            top_t = torch.tensor([top], dtype=torch.float32)
            channel_t = torch.tensor([channel], dtype=torch.float32)
            mean, log_scale = self.context_model(left_t, top_t, channel_t)

        scale = max(float(torch.exp(log_scale[0]).item()), 1e-6)
        return gaussian_prob_table(float(mean[0].item()), scale)

    def update(self, symbol: int) -> None:
        channel, h, w = self._position()
        self.buffer[channel, h, w] = symbol - 128
        self.pos += 1


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def _make_predictor_factory(model: ConvAutoencoder, latent_h: int, latent_w: int):
    context_model = model.context_model
    channels = model.latent_channels
    return lambda: ContextLatentPredictor(context_model, channels, latent_h, latent_w)


def compress_image(image: np.ndarray, model: ConvAutoencoder) -> bytes:
    """`image`: (H, W, 3) uint8 array. Returns header + arithmetic-coded latent."""
    height, width = image.shape[:2]

    x = torch.from_numpy(image.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = _pad_to_multiple(x, _DOWNSAMPLE)

    with torch.no_grad():
        z_hat = model.encode(x)

    _, channels, latent_h, latent_w = z_hat.shape
    symbols = (z_hat[0] + 128).round().to(torch.uint8).numpy().reshape(-1)

    make_predictor = _make_predictor_factory(model, latent_h, latent_w)
    payload = coder_compress(symbols.tobytes(), make_predictor)

    return _HEADER.pack(height, width, channels) + payload


def decompress_image(data: bytes, model: ConvAutoencoder) -> np.ndarray:
    """Returns the reconstructed (H, W, 3) uint8 image."""
    height, width, channels = _HEADER.unpack_from(data, 0)
    payload = data[_HEADER.size :]

    pad_h = height + (-height) % _DOWNSAMPLE
    pad_w = width + (-width) % _DOWNSAMPLE
    latent_h, latent_w = pad_h // _DOWNSAMPLE, pad_w // _DOWNSAMPLE

    make_predictor = _make_predictor_factory(model, latent_h, latent_w)
    raw = coder_decompress(payload, make_predictor)

    symbols = np.frombuffer(raw, dtype=np.uint8).reshape(channels, latent_h, latent_w)
    z_hat = torch.from_numpy(symbols.astype(np.float32) - 128.0).unsqueeze(0)

    with torch.no_grad():
        x_hat = model.decode(z_hat)

    x_hat = x_hat[0, :, :height, :width].clamp(0.0, 1.0)
    return (x_hat.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
