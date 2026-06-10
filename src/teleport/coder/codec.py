"""High-level lossless codec: Predictor + arithmetic coder -> bytes.

Wire format: 8-byte big-endian length prefix (number of original bytes),
followed by the arithmetic-coded payload.
"""

from __future__ import annotations

import struct
from typing import Callable

import numpy as np

from teleport.coder.arithmetic import ArithmeticDecoder, ArithmeticEncoder
from teleport.coder.freqs import TOTAL, probs_to_cumfreqs
from teleport.predictors.base import Predictor

_HEADER = struct.Struct(">Q")


def compress(data: bytes, make_predictor: Callable[[], Predictor]) -> bytes:
    predictor = make_predictor()
    encoder = ArithmeticEncoder()

    for byte in data:
        cumfreqs = probs_to_cumfreqs(predictor.predict())
        encoder.encode(int(cumfreqs[byte]), int(cumfreqs[byte + 1]), TOTAL)
        predictor.update(byte)

    return _HEADER.pack(len(data)) + encoder.finish()


def decompress(data: bytes, make_predictor: Callable[[], Predictor]) -> bytes:
    (length,) = _HEADER.unpack_from(data, 0)
    predictor = make_predictor()
    decoder = ArithmeticDecoder(data[_HEADER.size :])

    out = bytearray(length)
    for i in range(length):
        cumfreqs = probs_to_cumfreqs(predictor.predict())
        target = decoder.get_cum_freq(TOTAL)
        symbol = int(np.searchsorted(cumfreqs, target, side="right") - 1)
        decoder.decode(int(cumfreqs[symbol]), int(cumfreqs[symbol + 1]), TOTAL)
        out[i] = symbol
        predictor.update(symbol)

    return bytes(out)
