"""Fetch benchmark corpora into data/ (gitignored).

Usage:
    python scripts/get_data.py            # enwik6 only
    python scripts/get_data.py --images    # also fetch the Kodak image set
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK6_BYTES = 1_000_000

KODAK_URL_TEMPLATE = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
KODAK_COUNT = 24


def fetch_enwik6() -> Path:
    out_path = DATA_DIR / "enwik6"
    if out_path.exists():
        print(f"{out_path} already exists, skipping download")
        return out_path

    print(f"Downloading {ENWIK8_URL} ...")
    resp = requests.get(ENWIK8_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("enwik8") as f:
            data = f.read(ENWIK6_BYTES)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {out_path}")
    return out_path


def fetch_kodak_images() -> Path:
    out_dir = DATA_DIR / "images" / "kodak"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, KODAK_COUNT + 1):
        out_path = out_dir / f"kodim{i:02d}.png"
        if out_path.exists():
            continue
        url = KODAK_URL_TEMPLATE.format(i)
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

    print(f"Kodak images in {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch teleport benchmark corpora.")
    parser.add_argument(
        "--images", action="store_true", help="Also fetch the Kodak image set (Phase 3)"
    )
    args = parser.parse_args()

    fetch_enwik6()
    if args.images:
        fetch_kodak_images()


if __name__ == "__main__":
    main()
