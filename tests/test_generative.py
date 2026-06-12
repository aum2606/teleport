"""Phase 4 acceptance tests for the generative-extreme codec.

Uses TAESD (`madebyollin/taesd`), a frozen pretrained tiny VAE -- no training
happens here. These tests check the *pipeline* (lossless entropy coding of
the quantized small latent, end-to-end shapes, determinism, and that the
payload really is tiny). Quality numbers vs JPEG/WebP come from
`scripts/eval_generative.py` (results.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diffusers")

from teleport.coder import compress as coder_compress
from teleport.coder import decompress as coder_decompress
from teleport.generative.codec import _make_predictor

_TAESD_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--madebyollin--taesd"

pytestmark_taesd = pytest.mark.skipif(
    not _TAESD_CACHE.exists(), reason="TAESD weights not cached (run a script that loads madebyollin/taesd)"
)


def test_small_latent_entropy_coding_roundtrip():
    """The arithmetic coding of quantized small-latent symbols must be lossless."""
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 256, size=4 * 2 * 2, dtype=np.uint8)

    payload = coder_compress(symbols.tobytes(), _make_predictor)
    restored = np.frombuffer(coder_decompress(payload, _make_predictor), dtype=np.uint8)

    np.testing.assert_array_equal(restored, symbols)


@pytestmark_taesd
def test_compress_decompress_roundtrip_shape():
    from teleport.generative import compress_image, decompress_image

    img = (np.random.default_rng(1).random((128, 128, 3)) * 255).astype(np.uint8)

    compressed = compress_image(img)
    restored = decompress_image(compressed)

    assert restored.shape == img.shape
    assert restored.dtype == np.uint8


@pytestmark_taesd
def test_compress_decompress_non_multiple_of_64():
    from teleport.generative import compress_image, decompress_image

    img = (np.random.default_rng(2).random((100, 150, 3)) * 255).astype(np.uint8)

    compressed = compress_image(img)
    restored = decompress_image(compressed)

    assert restored.shape == img.shape


@pytestmark_taesd
def test_compress_is_deterministic():
    from teleport.generative import compress_image

    img = (np.random.default_rng(3).random((128, 128, 3)) * 255).astype(np.uint8)

    a = compress_image(img)
    b = compress_image(img)
    assert a == b


@pytestmark_taesd
def test_payload_is_hundreds_of_bytes():
    """A 512x768 image's payload should be a few hundred bytes -- the
    "hundreds of bytes -> full image" extreme of Phase 4."""
    from teleport.generative import compress_image

    img = (np.random.default_rng(4).random((512, 768, 3)) * 255).astype(np.uint8)

    compressed = compress_image(img)

    assert 100 < len(compressed) < 1000
