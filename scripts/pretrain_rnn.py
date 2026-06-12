"""Pretrain the shared GRU model for Phase 2.2 ("teleport" configuration,
IMPLEMENTATION_PLAN.md section 2.2).

Trains on a held-out chunk of `data/enwik6` (bytes 200,000-400,000),
disjoint from every evaluation slice used in results.md, with truncated
BPTT over `nn.GRU` (vectorized, fast — only the *pretraining* is batched;
inference in `predictors/rnn.PretrainedRNNPredictor` remains strictly
per-byte/causal via `nn.GRUCell`). Saves frozen weights, in the GRUCell
parameter layout, to `models/rnn_shared.pt`.

Determinism: CPU-only, single-threaded, fixed seed (CLAUDE.md rule 3) — the
checkpoint is reproducible from this script alone.

Usage:
    python scripts/pretrain_rnn.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "enwik6"
OUT = ROOT / "models" / "rnn_shared.pt"

EMBEDDING_DIM = 16
HIDDEN_SIZE = 128
SEQ_LEN = 128
EPOCHS = 8
LR = 0.002
SEED = 1234

# Disjoint from the eval slices used in results.md (enwik6_20k = [0:20000],
# in-domain eval = [600000:620000]).
TRAIN_START = 200_000
TRAIN_END = 400_000

VOCAB = 257
BOS = 256


class _SeqGRU(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 256)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        out, h = self.gru(emb, h0)
        return self.fc(out), h


def main() -> None:
    torch.manual_seed(SEED)
    data = DATA.read_bytes()[TRAIN_START:TRAIN_END]

    model = _SeqGRU(EMBEDDING_DIM, HIDDEN_SIZE).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    targets_all = list(data)
    inputs_all = [BOS] + targets_all[:-1]
    n_chunks = len(targets_all) // SEQ_LEN

    for epoch in range(EPOCHS):
        hidden = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float64)
        total_loss = 0.0
        for c in range(n_chunks):
            lo, hi = c * SEQ_LEN, (c + 1) * SEQ_LEN
            x = torch.tensor([inputs_all[lo:hi]], dtype=torch.long)
            y = torch.tensor([targets_all[lo:hi]], dtype=torch.long)

            logits, hidden = model(x, hidden.detach())
            loss = nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_nats = total_loss / n_chunks
        print(f"epoch {epoch + 1}/{EPOCHS}: loss {avg_nats:.4f} nats (~{avg_nats / math.log(2):.4f} bpc)")

    state = model.state_dict()
    cell_state = {
        "embedding.weight": state["embedding.weight"],
        "cell.weight_ih": state["gru.weight_ih_l0"],
        "cell.weight_hh": state["gru.weight_hh_l0"],
        "cell.bias_ih": state["gru.bias_ih_l0"],
        "cell.bias_hh": state["gru.bias_hh_l0"],
        "fc.weight": state["fc.weight"],
        "fc.bias": state["fc.bias"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "state_dict": cell_state,
            "train_range": [TRAIN_START, TRAIN_END],
            "seed": SEED,
        },
        OUT,
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
