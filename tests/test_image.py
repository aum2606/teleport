"""Phase 3 acceptance tests for the image codec.

Uses a tiny, randomly-initialized (untrained) model — these tests check the
*pipeline* (entropy model validity, lossless latent coding, end-to-end
shapes), not reconstruction quality. Quality numbers come from
`scripts/train_image.py` + `scripts/eval_image.py` (results.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from teleport.coder import compress as coder_compress
from teleport.coder import decompress as coder_decompress
from teleport.image import ConvAutoencoder, compress_image, decompress_image, gaussian_prob_table, ms_ssim
from teleport.image.codec import ContextLatentPredictor, _make_predictor_factory


def _tiny_model() -> ConvAutoencoder:
    torch.manual_seed(0)
    model = ConvAutoencoder(latent_channels=4, hidden=8, context_hidden=8)
    model.eval()
    return model


def test_gaussian_prob_table_is_valid_distribution():
    for mean, scale in [(0.0, 1.0), (5.0, 3.0), (-10.0, 0.5), (0.0, 1e-8)]:
        table = gaussian_prob_table(mean, scale)
        assert table.shape == (256,)
        assert table.dtype == np.float64
        assert np.all(table > 0.0)
        assert table.sum() == pytest.approx(1.0, rel=1e-9)


def test_context_predictor_uses_left_and_top_context():
    """Feeding different already-decoded neighbor values must change the
    predicted distribution (the model is conditioned on them)."""
    model = _tiny_model()
    make_predictor = _make_predictor_factory(model, latent_h=4, latent_w=4)

    # Predictor A: leave the left neighbor at its initial (zero) value.
    predictor_a: ContextLatentPredictor = make_predictor()
    predictor_a.update(128)  # position (0, 0) -> centered value 0
    table_a = predictor_a.predict()  # context for position (0, 1)

    # Predictor B: same position (0, 0) decoded to a very different value.
    predictor_b: ContextLatentPredictor = make_predictor()
    predictor_b.update(255)  # position (0, 0) -> centered value 127
    table_b = predictor_b.predict()  # context for position (0, 1)

    assert not np.allclose(table_a, table_b)


def test_context_predictor_visits_channel_major_raster_order():
    model = _tiny_model()
    make_predictor = _make_predictor_factory(model, latent_h=2, latent_w=2)
    predictor: ContextLatentPredictor = make_predictor()

    expected_positions = [
        (c, h, w) for c in range(model.latent_channels) for h in range(2) for w in range(2)
    ]
    for expected in expected_positions:
        assert predictor._position() == expected
        predictor.update(128)


def test_latent_entropy_coding_roundtrip():
    """The arithmetic coding of quantized latent symbols must be lossless."""
    model = _tiny_model()
    latent_h, latent_w = 4, 4
    make_predictor = _make_predictor_factory(model, latent_h, latent_w)

    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 256, size=model.latent_channels * latent_h * latent_w, dtype=np.uint8)

    payload = coder_compress(symbols.tobytes(), make_predictor)
    restored = np.frombuffer(coder_decompress(payload, make_predictor), dtype=np.uint8)

    np.testing.assert_array_equal(restored, symbols)


def test_compress_decompress_roundtrip_shape():
    model = _tiny_model()
    img = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)

    compressed = compress_image(img, model)
    restored = decompress_image(compressed, model)

    assert restored.shape == img.shape
    assert restored.dtype == np.uint8


def test_compress_decompress_non_multiple_of_8():
    """Images whose dims aren't multiples of 8 must round trip via padding/cropping."""
    model = _tiny_model()
    img = (np.random.default_rng(2).random((20, 30, 3)) * 255).astype(np.uint8)

    compressed = compress_image(img, model)
    restored = decompress_image(compressed, model)

    assert restored.shape == img.shape


def test_ms_ssim_identical_is_one_and_noisy_is_lower():
    rng = np.random.default_rng(4)
    img = (rng.random((192, 256, 3)) * 255).astype(np.float64)
    a = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    assert ms_ssim(a, a.clone()) == pytest.approx(1.0, abs=1e-6)

    noisy = np.clip(img + rng.normal(0, 30, img.shape), 0, 255)
    b = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0)
    assert ms_ssim(a, b) < 1.0


def test_compress_is_deterministic():
    model = _tiny_model()
    img = (np.random.default_rng(3).random((16, 16, 3)) * 255).astype(np.uint8)

    a = compress_image(img, model)
    b = compress_image(img, model)
    assert a == b
