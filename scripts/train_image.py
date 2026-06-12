"""Train the Phase 3 image autoencoder (IMPLEMENTATION_PLAN.md section 3.1).

Trains on random 64x64 patches cropped from `kodim01..kodim18` (held out:
`kodim19..kodim24`, used for evaluation in results.md). Loss is
`rate (bits/pixel) + lambda * 255^2 * MSE` — additive uniform-noise
quantization proxy during training (Balle et al. style), hard rounding at
test time (`ConvAutoencoder.encode`).

Determinism: CPU-only, single-threaded, fixed seed (CLAUDE.md rule 3) — the
checkpoint is reproducible from this script alone.

Usage:
    python scripts/train_image.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from teleport.image.model import ConvAutoencoder

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parent.parent
KODAK_DIR = ROOT / "data" / "images" / "kodak"
OUT = ROOT / "models" / "image_codec.pt"

TRAIN_IDS = range(1, 19)  # kodim01..kodim18
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 32
LATENT_CHANNELS = 4
HIDDEN = 32
CONTEXT_HIDDEN = 16
LAMBDA = 50.0  # rate + LAMBDA * 255^2 * MSE
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
SEED = 1234


def _load_patches() -> torch.Tensor:
    rng = np.random.default_rng(SEED)
    patches = []
    for idx in TRAIN_IDS:
        path = KODAK_DIR / f"kodim{idx:02d}.png"
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        for _ in range(PATCHES_PER_IMAGE):
            top = rng.integers(0, h - PATCH_SIZE + 1)
            left = rng.integers(0, w - PATCH_SIZE + 1)
            patch = img[top : top + PATCH_SIZE, left : left + PATCH_SIZE, :]
            patches.append(patch)
    arr = np.stack(patches).transpose(0, 3, 1, 2)  # (N, 3, H, W)
    return torch.from_numpy(arr)


def main() -> None:
    torch.manual_seed(SEED)
    patches = _load_patches()
    n = patches.shape[0]
    print(f"loaded {n} patches of {PATCH_SIZE}x{PATCH_SIZE}")

    model = ConvAutoencoder(latent_channels=LATENT_CHANNELS, hidden=HIDDEN, context_hidden=CONTEXT_HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_pixels = PATCH_SIZE * PATCH_SIZE
    rng = np.random.default_rng(SEED + 1)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        total_rate, total_mse, total_loss, n_batches = 0.0, 0.0, 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            batch = patches[idx]

            x_hat, _, z_tilde = model(batch)
            mse = torch.nn.functional.mse_loss(x_hat, batch)
            rate_bpp = model.rate_bits(z_tilde) / (batch.shape[0] * num_pixels)
            loss = rate_bpp + LAMBDA * (255.0**2) * mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_rate += rate_bpp.item()
            total_mse += mse.item()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            psnr = 10 * np.log10(1.0 / max(total_mse / n_batches, 1e-12))
            print(
                f"epoch {epoch + 1}/{EPOCHS}: loss {total_loss / n_batches:.4f} "
                f"rate {total_rate / n_batches:.4f} bpp  psnr {psnr:.2f} dB"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_channels": LATENT_CHANNELS,
            "hidden": HIDDEN,
            "context_hidden": CONTEXT_HIDDEN,
            "state_dict": model.state_dict(),
            "lambda": LAMBDA,
            "seed": SEED,
        },
        OUT,
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
