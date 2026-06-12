"""Online adaptive neural predictor — Phase 2.1 (IMPLEMENTATION_PLAN.md 2.1).

A tiny char/byte-level GRU that predicts the next byte, then immediately
trains on the byte that actually occurred (teacher forcing, one SGD step per
byte). No model is ever transmitted: encoder and decoder each start from the
same fixed random seed and run the identical, deterministic sequence of
predict()/update() calls, so their weights evolve in lockstep. The wire only
carries the arithmetic-coded "surprise".

Determinism (CLAUDE.md rule 3): CPU-only, single-threaded, fixed seed,
`torch.use_deterministic_algorithms(True)`. Run `tests/verify_sync.py` after
any change here.

# SLOW: ok — one forward + backward pass per byte; Phase 2 research code,
not optimized for speed (CLAUDE.md rule 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

_SEED = 1234
_BOS = 256  # beginning-of-stream token, fed as the "previous symbol" at t=0
_VOCAB = 257  # 256 byte values + BOS

# Shared frozen weights for Phase 2.2 (scripts/pretrain_rnn.py writes this).
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "rnn_shared.pt"


class _CharGRU(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(_VOCAB, embedding_dim)
        self.cell = nn.GRUCell(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, 256)

    def forward(
        self, input_idx: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_idx)
        new_hidden = self.cell(emb, hidden)
        logits = self.fc(new_hidden)
        return logits, new_hidden


class RNNPredictor:
    """Online (adaptive) GRU byte predictor.

    `predict()` runs one GRU step from the cached hidden state and the last
    observed byte (or BOS at the start), and caches the resulting logits and
    hidden state. `update(symbol)` then does one teacher-forced
    cross-entropy training step against `symbol` and commits the new hidden
    state for the next call — predict() is always followed by exactly one
    update() in this codec, so the cache is safe.
    """

    def __init__(
        self, embedding_dim: int = 16, hidden_size: int = 64, lr: float = 0.01
    ) -> None:
        torch.manual_seed(_SEED)
        self.model = _CharGRU(embedding_dim, hidden_size).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.hidden = torch.zeros(1, hidden_size, dtype=torch.float64)
        self.last_input = torch.tensor([_BOS], dtype=torch.long)
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def predict(self) -> np.ndarray:
        logits, new_hidden = self.model(self.last_input, self.hidden)
        probs = torch.softmax(logits, dim=-1)[0]
        self._cache = (logits, new_hidden)

        arr = probs.detach().numpy().astype(np.float64)
        arr = np.maximum(arr, 1e-9)
        return arr / arr.sum()

    def update(self, symbol: int) -> None:
        if self._cache is None:
            raise RuntimeError("update() called before predict()")
        logits, new_hidden = self._cache
        self._cache = None

        target = torch.tensor([symbol], dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.hidden = new_hidden.detach()
        self.last_input = torch.tensor([symbol], dtype=torch.long)


class PretrainedRNNPredictor:
    """Frozen shared-model predictor — Phase 2.2 ("teleport" configuration,
    IMPLEMENTATION_PLAN.md 2.2).

    Loads weights pretrained offline on a held-out corpus
    (`scripts/pretrain_rnn.py`). Encoder and decoder both load the identical
    frozen checkpoint — the shared model is the "pre-staged matter" at both
    ends; the wire carries only the arithmetic-coded residual surprise. No
    training happens during compression: `predict()`/`update()` are pure
    forward passes (`torch.no_grad()`), only the GRU hidden state advances.
    """

    def __init__(self, weights_path: Path | str = DEFAULT_WEIGHTS) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found — run `python scripts/pretrain_rnn.py` first"
            )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        self.model = _CharGRU(checkpoint["embedding_dim"], checkpoint["hidden_size"]).double()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.hidden = torch.zeros(1, checkpoint["hidden_size"], dtype=torch.float64)
        self.last_input = torch.tensor([_BOS], dtype=torch.long)
        self._cached_hidden: torch.Tensor | None = None

    def predict(self) -> np.ndarray:
        with torch.no_grad():
            logits, new_hidden = self.model(self.last_input, self.hidden)
            probs = torch.softmax(logits, dim=-1)[0]
        self._cached_hidden = new_hidden

        arr = probs.numpy().astype(np.float64)
        arr = np.maximum(arr, 1e-9)
        return arr / arr.sum()

    def update(self, symbol: int) -> None:
        if self._cached_hidden is None:
            raise RuntimeError("update() called before predict()")
        self.hidden = self._cached_hidden
        self._cached_hidden = None
        self.last_input = torch.tensor([symbol], dtype=torch.long)
