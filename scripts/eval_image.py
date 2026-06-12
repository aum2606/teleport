"""Phase 3 evaluation: trained codec vs JPEG/WebP on held-out Kodak images.

Held-out set: kodim19..kodim24 (kodim01..kodim18 were used for training,
see scripts/train_image.py). For each image:

  - run our codec (compress_image/decompress_image), measure bpp and PSNR
  - find a JPEG quality and a WebP quality whose file size is closest to
    ours, measure their PSNR at that matched bpp

Results are printed as a markdown table (paste into results.md).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from teleport.image import ConvAutoencoder, compress_image, decompress_image, ms_ssim

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parent.parent
KODAK_DIR = ROOT / "data" / "images" / "kodak"
CHECKPOINT = ROOT / "models" / "image_codec.pt"

EVAL_IDS = range(19, 25)  # kodim19..kodim24


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def _to_chw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float64)).permute(2, 0, 1).unsqueeze(0)


def _msssim(a: np.ndarray, b: np.ndarray) -> float:
    return ms_ssim(_to_chw(a), _to_chw(b))


def _load_model() -> ConvAutoencoder:
    checkpoint = torch.load(CHECKPOINT, weights_only=True)
    model = ConvAutoencoder(
        latent_channels=checkpoint["latent_channels"],
        hidden=checkpoint["hidden"],
        context_hidden=checkpoint["context_hidden"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _codec_at_bytes(img: Image.Image, fmt: str, target_bytes: int) -> tuple[bytes, np.ndarray]:
    """Search quality 1..95 for the encoding closest to (but not exceeding,
    if possible) target_bytes; return (encoded_bytes, decoded_array)."""
    best = None
    for quality in range(1, 96):
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        data = buf.getvalue()
        if best is None or abs(len(data) - target_bytes) < abs(len(best) - target_bytes):
            best = data
        if len(data) >= target_bytes:
            break
    decoded = np.asarray(Image.open(io.BytesIO(best)).convert("RGB"), dtype=np.uint8)
    return best, decoded


def main() -> None:
    model = _load_model()

    rows = []
    for idx in EVAL_IDS:
        path = KODAK_DIR / f"kodim{idx:02d}.png"
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape[:2]
        num_pixels = h * w

        compressed = compress_image(arr, model)
        restored = decompress_image(compressed, model)
        ours_bpp = len(compressed) * 8 / num_pixels
        ours_psnr = _psnr(arr, restored)
        ours_msssim = _msssim(arr, restored)

        jpeg_bytes, jpeg_decoded = _codec_at_bytes(img, "JPEG", len(compressed))
        jpeg_bpp = len(jpeg_bytes) * 8 / num_pixels
        jpeg_psnr = _psnr(arr, jpeg_decoded)
        jpeg_msssim = _msssim(arr, jpeg_decoded)

        webp_bytes, webp_decoded = _codec_at_bytes(img, "WEBP", len(compressed))
        webp_bpp = len(webp_bytes) * 8 / num_pixels
        webp_psnr = _psnr(arr, webp_decoded)
        webp_msssim = _msssim(arr, webp_decoded)

        rows.append(
            (
                idx, h, w,
                ours_bpp, ours_psnr, ours_msssim,
                jpeg_bpp, jpeg_psnr, jpeg_msssim,
                webp_bpp, webp_psnr, webp_msssim,
            )
        )
        print(
            f"kodim{idx:02d} ({h}x{w}): "
            f"ours {ours_bpp:.4f} bpp / {ours_psnr:.2f} dB / {ours_msssim:.4f} MS-SSIM  |  "
            f"jpeg {jpeg_bpp:.4f} bpp / {jpeg_psnr:.2f} dB / {jpeg_msssim:.4f} MS-SSIM  |  "
            f"webp {webp_bpp:.4f} bpp / {webp_psnr:.2f} dB / {webp_msssim:.4f} MS-SSIM"
        )

    print()
    print(
        "| image | size | ours bpp | ours PSNR | ours MS-SSIM | "
        "JPEG bpp | JPEG PSNR | JPEG MS-SSIM | WebP bpp | WebP PSNR | WebP MS-SSIM |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for idx, h, w, ob, op, om, jb, jp, jm, wb, wp, wm in rows:
        print(
            f"| kodim{idx:02d} | {h}x{w} | {ob:.4f} | {op:.2f} | {om:.4f} | "
            f"{jb:.4f} | {jp:.2f} | {jm:.4f} | {wb:.4f} | {wp:.2f} | {wm:.4f} |"
        )

    avg_ours_bpp = np.mean([r[3] for r in rows])
    avg_ours_psnr = np.mean([r[4] for r in rows])
    avg_ours_msssim = np.mean([r[5] for r in rows])
    avg_jpeg_bpp = np.mean([r[6] for r in rows])
    avg_jpeg_psnr = np.mean([r[7] for r in rows])
    avg_jpeg_msssim = np.mean([r[8] for r in rows])
    avg_webp_bpp = np.mean([r[9] for r in rows])
    avg_webp_psnr = np.mean([r[10] for r in rows])
    avg_webp_msssim = np.mean([r[11] for r in rows])
    print(
        f"| **avg** | | {avg_ours_bpp:.4f} | {avg_ours_psnr:.2f} | {avg_ours_msssim:.4f} | "
        f"{avg_jpeg_bpp:.4f} | {avg_jpeg_psnr:.2f} | {avg_jpeg_msssim:.4f} | "
        f"{avg_webp_bpp:.4f} | {avg_webp_psnr:.2f} | {avg_webp_msssim:.4f} |"
    )


if __name__ == "__main__":
    main()
