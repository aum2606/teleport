"""Order-0 adaptive predictor: global byte-frequency counts with Laplace
(+1) smoothing so every symbol always has nonzero probability.
"""

from __future__ import annotations

import numpy as np


class Order0Predictor:
    def __init__(self) -> None:
        self.counts = np.ones(256, dtype=np.float64)

    def predict(self) -> np.ndarray:
        return self.counts / self.counts.sum()

    def update(self, symbol: int) -> None:
        self.counts[symbol] += 1
