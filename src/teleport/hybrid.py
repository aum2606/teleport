"""Hybrid codec: try multiple methods, keep the smallest output.

Wire format: `[1 byte: method id][payload bytes]`.

| id | method | payload |
|---|---|---|
| 0x00 | stored | raw bytes, uncompressed (incompressible-input guard) |
| 0x01 | classical | stdlib `lzma` (preset 9) output |
| 0x02 | classical-alt | stdlib `bz2` (level 9) output |
| 0x03 | neural | Phase 1 arithmetic coder + `PretrainedRNNPredictor` output |

This neutralizes the Phase 2.2 out-of-domain failure mode (5.21 bpc on
Python source, see results.md) at the cost of exactly one header byte, while
keeping the in-domain win (2.92 bpc): worst case is best-classical + 1 byte.

# SLOW: ok — compression runs every candidate (dominated by the neural
path, ~2.4 KB/s); decompression only runs the selected method
(CLAUDE.md rule 5).
"""

from __future__ import annotations

import bz2
import lzma
from pathlib import Path
from typing import Callable

from teleport.coder import compress as coder_compress
from teleport.coder import cross_entropy_bits, decompress as coder_decompress
from teleport.predictors.base import Predictor
from teleport.predictors.rnn import DEFAULT_WEIGHTS, PretrainedRNNPredictor

METHOD_STORED = 0x00
METHOD_LZMA = 0x01
METHOD_BZ2 = 0x02
METHOD_NEURAL = 0x03

_LZMA_PRESET = 9
_BZ2_LEVEL = 9

# Encoder-side heuristic only (decoder is unaffected): if the neural
# predictor's cross-entropy over a small probe is already this bad, the full
# encode is hopeless (it will lose to lzma/bz2 anyway) and can be skipped.
_PROBE_BYTES = 2048
_PROBE_THRESHOLD_BPC = 4.5


def _make_neural_predictor(weights_path: Path | str) -> Callable[[], Predictor]:
    return lambda: PretrainedRNNPredictor(weights_path)


def _neural_probe_bpc(data: bytes, weights_path: Path | str) -> float:
    sample = data[:_PROBE_BYTES]
    if not sample:
        return 0.0
    return cross_entropy_bits(sample, _make_neural_predictor(weights_path)) / len(sample)


def compress_hybrid(data: bytes, model_path: Path | str = DEFAULT_WEIGHTS, probe: bool = True) -> bytes:
    candidates: dict[int, bytes] = {
        METHOD_STORED: data,
        METHOD_LZMA: lzma.compress(data, preset=_LZMA_PRESET),
        METHOD_BZ2: bz2.compress(data, compresslevel=_BZ2_LEVEL),
    }

    weights_path = Path(model_path)
    if weights_path.exists():
        if not probe or _neural_probe_bpc(data, weights_path) <= _PROBE_THRESHOLD_BPC:
            candidates[METHOD_NEURAL] = coder_compress(data, _make_neural_predictor(weights_path))

    method, payload = min(candidates.items(), key=lambda kv: len(kv[1]))
    return bytes([method]) + payload


def decompress_hybrid(data: bytes, model_path: Path | str = DEFAULT_WEIGHTS) -> bytes:
    method, payload = data[0], data[1:]

    if method == METHOD_STORED:
        return payload
    if method == METHOD_LZMA:
        return lzma.decompress(payload)
    if method == METHOD_BZ2:
        return bz2.decompress(payload)
    if method == METHOD_NEURAL:
        return coder_decompress(payload, _make_neural_predictor(model_path))

    raise ValueError(f"unknown hybrid method id: {method:#x}")


class HybridCompressor:
    """Adapts compress_hybrid/decompress_hybrid to bench.py's Compressor interface."""

    name = "hybrid"

    def __init__(self, model_path: Path | str = DEFAULT_WEIGHTS) -> None:
        self.model_path = model_path

    def compress(self, data: bytes) -> bytes:
        return compress_hybrid(data, self.model_path)

    def decompress(self, data: bytes) -> bytes:
        return decompress_hybrid(data, self.model_path)
