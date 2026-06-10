import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from teleport.coder import compress, cross_entropy_bits, decompress
from teleport.predictors import PREDICTORS
from teleport.predictors.order0 import Order0Predictor
from teleport.predictors.uniform import UniformPredictor


@pytest.mark.parametrize("name", list(PREDICTORS.keys()))
@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\xff",
        b"\x00" * 1000,
        b"\xff" * 1000,
        b"the quick brown fox jumps over the lazy dog " * 50,
        bytes(range(256)),
        bytes(range(256)) * 4,
    ],
)
def test_roundtrip_adversarial(name, data):
    make_predictor = PREDICTORS[name]
    encoded = compress(data, make_predictor)
    decoded = decompress(encoded, make_predictor)
    assert decoded == data


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.binary(min_size=0, max_size=200))
def test_roundtrip_property_order0(data):
    encoded = compress(data, Order0Predictor)
    decoded = decompress(encoded, Order0Predictor)
    assert decoded == data


def test_uniform_predictor_gives_8_bpc():
    data = bytes((i * 37) % 256 for i in range(2000))
    bits = cross_entropy_bits(data, UniformPredictor)
    bpc = bits / len(data)
    assert bpc == pytest.approx(8.0, abs=1e-9)


@pytest.mark.parametrize("name", list(PREDICTORS.keys()))
def test_coder_near_cross_entropy(name):
    make_predictor = PREDICTORS[name]
    data = b"the quick brown fox jumps over the lazy dog. " * 200

    encoded = compress(data, make_predictor)
    compressed_bits = (len(encoded) - 8) * 8  # minus 8-byte length header

    expected_bits = cross_entropy_bits(data, make_predictor)

    # Coder overhead has two parts: ~2 bits of flush, plus a per-symbol cap
    # from the 1/2^16 probability floor (any predicted probability above
    # (TOTAL-255)/TOTAL gets capped, costing ~0.0056 bits/symbol in the worst
    # case of a maximally confident predictor). Allow generous headroom.
    quantization_allowance = 0.01 * len(data) * 8
    assert compressed_bits <= expected_bits + quantization_allowance + 8
    # And it shouldn't be wildly under either (sanity bound) - quantization
    # can also round in the coder's favor for individual symbols.
    assert compressed_bits >= expected_bits - quantization_allowance - 8


def test_decoder_does_not_see_future_symbols():
    """A predictor that peeked ahead would desync; verify two different
    suffixes sharing a prefix decode that prefix identically."""
    make_predictor = lambda: PREDICTORS["ppm3"]()

    a = b"abcdefgh" + b"X" * 10
    b = b"abcdefgh" + b"Y" * 10

    enc_a = compress(a, make_predictor)
    enc_b = compress(b, make_predictor)

    dec_a = decompress(enc_a, make_predictor)
    dec_b = decompress(enc_b, make_predictor)

    assert dec_a == a
    assert dec_b == b
    assert dec_a[:8] == dec_b[:8] == b"abcdefgh"


def test_empty_and_single_byte_all_predictors():
    for make_predictor in PREDICTORS.values():
        for data in (b"", b"\x42"):
            assert decompress(compress(data, make_predictor), make_predictor) == data
