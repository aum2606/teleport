from __future__ import annotations

import math
from typing import Callable

from teleport.coder.codec import compress, decompress
from teleport.predictors.base import Predictor

__all__ = ["compress", "decompress", "cross_entropy_bits", "PredictorCompressor"]


def cross_entropy_bits(data: bytes, make_predictor: Callable[[], Predictor]) -> float:
    """Total -log2 P(symbol) over `data`, in bits, under a fresh predictor.

    This is the information-theoretic lower bound the arithmetic coder should
    match to within ~1% (CLAUDE.md Phase 1 acceptance criterion).
    """
    predictor = make_predictor()
    total_bits = 0.0
    for byte in data:
        probs = predictor.predict()
        total_bits -= math.log2(float(probs[byte]))
        predictor.update(byte)
    return total_bits


class PredictorCompressor:
    """Adapts a (name, predictor-factory) pair to bench.py's Compressor interface."""

    def __init__(self, name: str, make_predictor: Callable[[], Predictor]) -> None:
        self.name = name
        self._make_predictor = make_predictor

    def compress(self, data: bytes) -> bytes:
        return compress(data, self._make_predictor)

    def decompress(self, data: bytes) -> bytes:
        return decompress(data, self._make_predictor)
