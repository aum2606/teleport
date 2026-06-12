"""Phase 4 evaluation: generative-extreme codec vs JPEG/WebP on held-out
Kodak images, at matched (tiny) byte budgets.

Held-out set: kodim19..kodim24 (same as Phase 3, see scripts/eval_image.py).
For each image:

  - run our codec (teleport.generative.compress_image/decompress_image),
    measure total bytes, bpp, PSNR, MS-SSIM
  - find a JPEG quality and a WebP quality whose file size is closest to
    ours (typically quality 1, since our payload is only hundreds of bytes),
    measure their PSNR/MS-SSIM at that matched byte budget

This is the "hundreds of bytes -> full image" extreme: our codec transmits a
tiny quantized VAE latent and regenerates plausible detail via TAESD's
decoder, vs. classical codecs starved down to the same byte budget (which
produces severe blocking/banding artifacts).

Note: IMPLEMENTATION_PLAN.md section 4 suggests LPIPS for perceptual
similarity. We reuse the hand-rolled MS-SSIM from Phase 3 instead, to avoid
adding a large pretrained AlexNet/VGG dependency -- documented here and in
results.md as an honest substitution (same precedent as the Kodak-vs-CIFAR10
substitution in Phase 3).

Results are printed as a markdown table (paste into results.md).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from teleport.generative import compress_image, decompress_image
from teleport.image import ms_ssim
from scripts.eval_image import _codec_at_bytes, _msssim, _psnr

ROOT = Path(__file__).resolve().parent.parent
KODAK_DIR = ROOT / "data" / "images" / "kodak"

EVAL_IDS = range(19, 25)  # kodim19..kodim24


def main() -> None:
    rows = []
    for idx in EVAL_IDS:
        path = KODAK_DIR / f"kodim{idx:02d}.png"
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape[:2]
        num_pixels = h * w

        compressed = compress_image(arr)
        restored = decompress_image(compressed)
        ours_bytes = len(compressed)
        ours_bpp = ours_bytes * 8 / num_pixels
        ours_psnr = _psnr(arr, restored)
        ours_msssim = _msssim(arr, restored)

        jpeg_bytes, jpeg_decoded = _codec_at_bytes(img, "JPEG", ours_bytes)
        jpeg_bpp = len(jpeg_bytes) * 8 / num_pixels
        jpeg_psnr = _psnr(arr, jpeg_decoded)
        jpeg_msssim = _msssim(arr, jpeg_decoded)

        webp_bytes, webp_decoded = _codec_at_bytes(img, "WEBP", ours_bytes)
        webp_bpp = len(webp_bytes) * 8 / num_pixels
        webp_psnr = _psnr(arr, webp_decoded)
        webp_msssim = _msssim(arr, webp_decoded)

        rows.append(
            (
                idx, h, w, ours_bytes,
                ours_bpp, ours_psnr, ours_msssim,
                len(jpeg_bytes), jpeg_bpp, jpeg_psnr, jpeg_msssim,
                len(webp_bytes), webp_bpp, webp_psnr, webp_msssim,
            )
        )
        print(
            f"kodim{idx:02d} ({h}x{w}): "
            f"ours {ours_bytes}B / {ours_bpp:.5f} bpp / {ours_psnr:.2f} dB / {ours_msssim:.4f} MS-SSIM  |  "
            f"jpeg {len(jpeg_bytes)}B / {jpeg_bpp:.5f} bpp / {jpeg_psnr:.2f} dB / {jpeg_msssim:.4f} MS-SSIM  |  "
            f"webp {len(webp_bytes)}B / {webp_bpp:.5f} bpp / {webp_psnr:.2f} dB / {webp_msssim:.4f} MS-SSIM"
        )

    print()
    print(
        "| image | size | ours bytes | ours bpp | ours PSNR | ours MS-SSIM | "
        "JPEG bytes | JPEG bpp | JPEG PSNR | JPEG MS-SSIM | WebP bytes | WebP bpp | WebP PSNR | WebP MS-SSIM |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for idx, h, w, ob, obpp, op, om, jb, jbpp, jp, jm, wb, wbpp, wp, wm in rows:
        print(
            f"| kodim{idx:02d} | {h}x{w} | {ob} | {obpp:.5f} | {op:.2f} | {om:.4f} | "
            f"{jb} | {jbpp:.5f} | {jp:.2f} | {jm:.4f} | "
            f"{wb} | {wbpp:.5f} | {wp:.2f} | {wm:.4f} |"
        )

    avg_ours_bytes = np.mean([r[3] for r in rows])
    avg_ours_bpp = np.mean([r[4] for r in rows])
    avg_ours_psnr = np.mean([r[5] for r in rows])
    avg_ours_msssim = np.mean([r[6] for r in rows])
    avg_jpeg_bytes = np.mean([r[7] for r in rows])
    avg_jpeg_bpp = np.mean([r[8] for r in rows])
    avg_jpeg_psnr = np.mean([r[9] for r in rows])
    avg_jpeg_msssim = np.mean([r[10] for r in rows])
    avg_webp_bytes = np.mean([r[11] for r in rows])
    avg_webp_bpp = np.mean([r[12] for r in rows])
    avg_webp_psnr = np.mean([r[13] for r in rows])
    avg_webp_msssim = np.mean([r[14] for r in rows])
    print(
        f"| **avg** | | {avg_ours_bytes:.0f} | {avg_ours_bpp:.5f} | {avg_ours_psnr:.2f} | {avg_ours_msssim:.4f} | "
        f"{avg_jpeg_bytes:.0f} | {avg_jpeg_bpp:.5f} | {avg_jpeg_psnr:.2f} | {avg_jpeg_msssim:.4f} | "
        f"{avg_webp_bytes:.0f} | {avg_webp_bpp:.5f} | {avg_webp_psnr:.2f} | {avg_webp_msssim:.4f} |"
    )


if __name__ == "__main__":
    main()
