import numpy as np
import pytest

from teleport.predictors import PREDICTORS
from teleport.predictors.rnn import DEFAULT_WEIGHTS


@pytest.mark.parametrize("name", list(PREDICTORS.keys()))
def test_predict_is_valid_distribution(name):
    if name == "rnn_pretrained" and not DEFAULT_WEIGHTS.exists():
        pytest.skip("models/rnn_shared.pt not built (run scripts/pretrain_rnn.py)")
    predictor = PREDICTORS[name]()
    data = b"abracadabra mississippi banana " * 5

    for byte in data:
        probs = predictor.predict()
        assert probs.shape == (256,)
        assert probs.dtype == np.float64
        assert np.all(probs > 0.0)
        assert probs.sum() == pytest.approx(1.0, rel=1e-9)
        predictor.update(byte)


@pytest.mark.parametrize("name", ["order0", "ppm2", "ppm3"])
def test_predictor_is_causal(name):
    """predict() before observing the next byte must not depend on it:
    two predictors fed the same prefix give the same prediction regardless
    of what comes after."""
    prefix = b"the quick brown fox"
    p1 = PREDICTORS[name]()
    p2 = PREDICTORS[name]()

    for byte in prefix:
        p1.predict()
        p1.update(byte)
        p2.predict()
        p2.update(byte)

    np.testing.assert_array_equal(p1.predict(), p2.predict())


@pytest.mark.parametrize("name", ["order0"])
def test_predictor_learns_skewed_distribution(name):
    """A predictor should assign higher probability to a byte that
    dominates the stream than to one that never appears."""
    predictor = PREDICTORS[name]()
    data = b"a" * 500 + b"b" * 10
    for byte in data:
        predictor.update(byte)

    probs = predictor.predict()
    assert probs[ord("a")] > probs[ord("b")]
    assert probs[ord("a")] > probs[ord("z")]
