"""Phase 4 generative-extreme codec: TAESD latent + Order-0 arithmetic coding.

The "pre-staged matter" here is a frozen, pretrained tiny VAE (TAESD,
`madebyollin/taesd`, ~2.4M params, 4 latent channels, /8 spatial downsample).
We transmit a heavily downsampled, quantized version of its latent -- on the
order of a few hundred bytes for a Kodak-sized image -- and let the VAE's
decoder regenerate a full image from it.

Wire format: an 8-byte header (orig height, width) followed by the
arithmetic-coded quantized small-latent symbols (channel-major, raster order
within each channel).

Pipeline:
  encode: image -> TAESD encoder -> latent (4, H/8, W/8) -> scale to [0, 1]
          -> average-pool by _DOWNSAMPLE -> quantize to uint8
          -> Order0Predictor + Phase 1 arithmetic coder
  decode: bytes -> Order0Predictor + Phase 1 arithmetic decoder -> uint8
          symbols -> [0, 1] -> nearest-upsample by _DOWNSAMPLE
          -> unscale -> TAESD decoder -> image

This is lossy and extremely lossy in rate terms (an additional _DOWNSAMPLE x
beyond TAESD's own /8). What must stay exact is the entropy coding of the
quantized small-latent symbols -- tests/test_generative.py checks that round
trip.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from teleport.coder import compress as coder_compress
from teleport.coder import decompress as coder_decompress
from teleport.predictors.order0 import Order0Predictor

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

_HEADER = struct.Struct(">II")  # height, width
_TAESD_DOWNSAMPLE = 8  # TAESD's own /8 spatial downsample
_DOWNSAMPLE = 8  # additional downsample of the TAESD latent
_TOTAL_DOWNSAMPLE = _TAESD_DOWNSAMPLE * _DOWNSAMPLE
_LATENT_CHANNELS = 4


@lru_cache(maxsize=1)
def _load_taesd():
    from diffusers import AutoencoderTiny

    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    vae.eval()
    return vae


def _make_predictor():
    return Order0Predictor()


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def compress_image(image: np.ndarray) -> bytes:
    """`image`: (H, W, 3) uint8 array. Returns header + arithmetic-coded
    small-latent symbols."""
    vae = _load_taesd()
    height, width = image.shape[:2]

    x = torch.from_numpy(image.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x * 2.0 - 1.0  # [-1, 1], TAESD's expected input range
    x = _pad_to_multiple(x, _TOTAL_DOWNSAMPLE)

    with torch.no_grad():
        latents = vae.encode(x).latents
        scaled = vae.scale_latents(latents)  # [0, 1]
        small = F.avg_pool2d(scaled, kernel_size=_DOWNSAMPLE)

    symbols = (small[0] * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(-1)
    payload = coder_compress(symbols.tobytes(), _make_predictor)

    return _HEADER.pack(height, width) + payload


def decompress_image(data: bytes) -> np.ndarray:
    """Returns the reconstructed (H, W, 3) uint8 image."""
    vae = _load_taesd()
    height, width = _HEADER.unpack_from(data, 0)
    payload = data[_HEADER.size :]

    pad_h = height + (-height) % _TOTAL_DOWNSAMPLE
    pad_w = width + (-width) % _TOTAL_DOWNSAMPLE
    latent_h, latent_w = pad_h // _TAESD_DOWNSAMPLE, pad_w // _TAESD_DOWNSAMPLE
    small_h, small_w = latent_h // _DOWNSAMPLE, latent_w // _DOWNSAMPLE

    raw = coder_decompress(payload, _make_predictor)
    symbols = np.frombuffer(raw, dtype=np.uint8).reshape(_LATENT_CHANNELS, small_h, small_w)

    small = torch.from_numpy(symbols.astype(np.float32) / 255.0).unsqueeze(0)
    scaled = F.interpolate(small, size=(latent_h, latent_w), mode="nearest")

    with torch.no_grad():
        unscaled = vae.unscale_latents(scaled)
        dec = vae.decode(unscaled).sample

    dec = dec[0, :, :height, :width].clamp(-1.0, 1.0)
    img = (dec + 1.0) / 2.0 * 255.0
    return img.permute(1, 2, 0).round().numpy().astype(np.uint8)
