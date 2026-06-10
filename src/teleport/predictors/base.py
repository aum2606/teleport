"""The Predictor contract (CLAUDE.md). Every predictor implements this.

`predict()` must be strictly causal: it may depend only on symbols already
passed to `update()`, never on the symbol currently being coded.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Predictor(Protocol):
    def predict(self) -> np.ndarray:
        """Return a (256,) float64 array of probabilities, summing to 1, all > 0."""
        ...

    def update(self, symbol: int) -> None:
        """Observe the symbol that actually occurred at the current position."""
        ...
