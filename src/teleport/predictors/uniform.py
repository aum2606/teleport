"""Sanity-check predictor: every byte is equally likely.

Should produce exactly 8.00 bpc through the coder (log2(256) = 8).
"""

from __future__ import annotations

import numpy as np

UNIFORM = np.full(256, 1.0 / 256, dtype=np.float64)


class UniformPredictor:
    def predict(self) -> np.ndarray:
        return UNIFORM

    def update(self, symbol: int) -> None:
        pass
