"""Phase 2 acceptance tests for the GRU predictors.

Kept separate from test_predictors.py because each predict()/update() pair
of the online predictor does a real forward+backward pass (# SLOW: ok) —
these tests use very small inputs to stay fast.
"""

from __future__ import annotations

import pytest

from teleport.coder import compress, decompress
from teleport.predictors.rnn import DEFAULT_WEIGHTS, PretrainedRNNPredictor, RNNPredictor


def test_roundtrip_tiny():
    data = b"the quick brown fox jumps over the lazy dog"
    compressed = compress(data, RNNPredictor)
    assert decompress(compressed, RNNPredictor) == data


def test_roundtrip_repetitive_data_compresses():
    """A highly repetitive stream should compress well as the model adapts."""
    data = b"ab" * 100
    compressed = compress(data, RNNPredictor)
    assert decompress(compressed, RNNPredictor) == data
    assert len(compressed) < len(data)


def test_roundtrip_empty():
    data = b""
    compressed = compress(data, RNNPredictor)
    assert decompress(compressed, RNNPredictor) == data


def test_construction_is_deterministic():
    """Two fresh predictors must start with identical weights/predictions."""
    import numpy as np

    p1 = RNNPredictor()
    p2 = RNNPredictor()
    np.testing.assert_array_equal(p1.predict(), p2.predict())


pytestmark_pretrained = pytest.mark.skipif(
    not DEFAULT_WEIGHTS.exists(), reason="models/rnn_shared.pt not built (run scripts/pretrain_rnn.py)"
)


@pytestmark_pretrained
def test_pretrained_roundtrip_tiny():
    data = b"the quick brown fox jumps over the lazy dog"
    compressed = compress(data, PretrainedRNNPredictor)
    assert decompress(compressed, PretrainedRNNPredictor) == data


@pytestmark_pretrained
def test_pretrained_roundtrip_empty():
    data = b""
    compressed = compress(data, PretrainedRNNPredictor)
    assert decompress(compressed, PretrainedRNNPredictor) == data


@pytestmark_pretrained
def test_pretrained_is_frozen_no_training():
    """update() must not change model weights — only the hidden state."""
    import copy

    p = PretrainedRNNPredictor()
    before = copy.deepcopy(p.model.state_dict())

    for byte in b"the quick brown fox":
        p.predict()
        p.update(byte)

    after = p.model.state_dict()
    for key in before:
        assert (before[key] == after[key]).all(), f"{key} changed during inference"


@pytestmark_pretrained
def test_pretrained_construction_is_deterministic():
    import numpy as np

    p1 = PretrainedRNNPredictor()
    p2 = PretrainedRNNPredictor()
    np.testing.assert_array_equal(p1.predict(), p2.predict())
