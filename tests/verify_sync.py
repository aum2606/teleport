"""Encoder/decoder lockstep check for the online neural predictor (Phase 2).

CLAUDE.md rule 3 (determinism): two independently constructed `RNNPredictor`
instances, fed the exact same byte sequence and the same predict/update
calls, must produce bit-identical probability distributions and weight
updates at every step (CPU-only, single-threaded, fixed seed). If they
diverge, the encoder and decoder would desync and corrupt the stream.

Run after any change to `predictors/rnn*` or its training loop:

    python tests/verify_sync.py
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np

from teleport.coder import compress, decompress
from teleport.predictors.base import Predictor
from teleport.predictors.rnn import DEFAULT_WEIGHTS, PretrainedRNNPredictor, RNNPredictor

SAMPLE = (b"the quick brown fox jumps over the lazy dog. " * 4)[:100]


def check_lockstep(make_predictor: Callable[[], Predictor], data: bytes, label: str) -> None:
    """Two fresh predictors fed the same bytes must match at every step."""
    encoder_side = make_predictor()
    decoder_side = make_predictor()

    for i, byte in enumerate(data):
        p_enc = encoder_side.predict()
        p_dec = decoder_side.predict()
        if not np.array_equal(p_enc, p_dec):
            raise AssertionError(f"{label}: predict() diverged at step {i}")
        encoder_side.update(byte)
        decoder_side.update(byte)

    print(f"{label} lockstep: {len(data)} bytes, predictors stayed bit-identical")


def check_roundtrip(make_predictor: Callable[[], Predictor], data: bytes, label: str) -> None:
    """Full compress/decompress round trip must be byte-exact."""
    compressed = compress(data, make_predictor)
    restored = decompress(compressed, make_predictor)
    if restored != data:
        raise AssertionError(f"{label}: round trip mismatch")
    print(f"{label} round trip: {len(data)} bytes -> {len(compressed)} bytes, exact")


def main() -> None:
    check_lockstep(RNNPredictor, SAMPLE, "rnn (online)")
    check_roundtrip(RNNPredictor, SAMPLE, "rnn (online)")

    if DEFAULT_WEIGHTS.exists():
        check_lockstep(PretrainedRNNPredictor, SAMPLE, "rnn_pretrained")
        check_roundtrip(PretrainedRNNPredictor, SAMPLE, "rnn_pretrained")
    else:
        print(f"rnn_pretrained: skipped ({DEFAULT_WEIGHTS} not built)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"verify_sync: FAIL ({exc})", file=sys.stderr)
        sys.exit(1)
