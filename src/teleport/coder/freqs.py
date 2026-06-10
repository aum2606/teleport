"""Convert a float probability distribution into integer cumulative
frequencies the arithmetic coder can use.

Per CLAUDE.md: probabilities passed to the coder must be > 0 for every
symbol. We floor at 1/TOTAL and renormalize so the integer frequencies sum to
exactly TOTAL (a fixed power of two), deterministically given the same input
floats on both encoder and decoder.
"""

from __future__ import annotations

import numpy as np

TOTAL_BITS = 16
TOTAL = 1 << TOTAL_BITS


def probs_to_cumfreqs(probs: np.ndarray) -> np.ndarray:
    """Return a length-257 int64 array of cumulative frequencies in [0, TOTAL]."""
    freqs = np.maximum(1, np.floor(probs * TOTAL)).astype(np.int64)

    diff = TOTAL - int(freqs.sum())
    if diff > 0:
        freqs[int(np.argmax(freqs))] += diff
    elif diff < 0:
        need = -diff
        for idx in np.argsort(-freqs):
            available = int(freqs[idx]) - 1
            take = min(need, available)
            freqs[idx] -= take
            need -= take
            if need == 0:
                break
        if need > 0:
            raise ValueError("cannot renormalize frequencies to TOTAL")

    cumfreqs = np.zeros(257, dtype=np.int64)
    np.cumsum(freqs, out=cumfreqs[1:])
    return cumfreqs
