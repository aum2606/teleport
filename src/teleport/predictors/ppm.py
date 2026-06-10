"""PPM-style context model with order blending (escape/backoff).

For each step, we walk down from the highest order context to order 0 and
finally to a uniform order(-1) base. At each order, the symbols seen in that
context get a share `(1 - escape) * count / total` of the remaining
probability mass; the rest (`escape`) is passed down to the next-lower order.
The order(-1) uniform layer absorbs whatever mass is left, guaranteeing every
symbol gets nonzero probability without needing a separate renormalization
step.

Escape mass uses PPM method C: escape weight = number of distinct symbols
seen in the context, so `escape = distinct / (total + distinct)`.

This is "PPM-style" rather than textbook PPM-C: it does not implement
exclusion (higher-order symbols are not excluded from lower-order counts).
Exclusion would tighten the estimate further but adds significant complexity;
blending without it is still a strict causal predictor and still benefits
from longer contexts.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class PPMPredictor:
    def __init__(self, max_order: int = 3) -> None:
        self.max_order = max_order
        # contexts[k]: maps a k-byte context tuple -> Counter(symbol -> count)
        self.contexts: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(max_order + 1)
        ]
        self.history: list[int] = []

    def predict(self) -> np.ndarray:
        dist = np.zeros(256, dtype=np.float64)
        remaining = 1.0

        for order in range(self.max_order, -1, -1):
            if remaining <= 0.0:
                break
            ctx = tuple(self.history[-order:]) if order > 0 else ()
            counts = self.contexts[order].get(ctx)
            if not counts:
                continue

            total = sum(counts.values())
            distinct = len(counts)
            denom = total + distinct
            for sym, c in counts.items():
                dist[sym] += remaining * (c / denom)
            remaining *= distinct / denom

        if remaining > 0.0:
            dist += remaining / 256.0

        dist /= dist.sum()
        return dist

    def update(self, symbol: int) -> None:
        for order in range(self.max_order + 1):
            ctx = tuple(self.history[-order:]) if order > 0 else ()
            self.contexts[order][ctx][symbol] += 1
        self.history.append(symbol)
        if len(self.history) > self.max_order:
            del self.history[: -self.max_order]
