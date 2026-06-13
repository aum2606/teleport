"""Reference compressors used as baselines in bench.py.

Every compressor here implements the same minimal interface:
    compress(data: bytes) -> bytes
    decompress(data: bytes) -> bytes
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import zlib
from dataclasses import dataclass
from typing import Protocol

import zstandard


class Compressor(Protocol):
    name: str

    def compress(self, data: bytes) -> bytes: ...

    def decompress(self, data: bytes) -> bytes: ...


@dataclass
class GzipCompressor:
    level: int = 9
    name: str = "gzip-9"

    def __post_init__(self) -> None:
        self.name = f"gzip-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


@dataclass
class Bzip2Compressor:
    level: int = 9
    name: str = "bzip2-9"

    def __post_init__(self) -> None:
        self.name = f"bzip2-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return bz2.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)


@dataclass
class LzmaCompressor:
    preset: int = 9
    name: str = "lzma-9"

    def __post_init__(self) -> None:
        self.name = f"lzma-{self.preset}"

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.preset)

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)


@dataclass
class ZstdCompressor:
    level: int = 19
    name: str = "zstd-19"

    def __post_init__(self) -> None:
        self.name = f"zstd-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return zstandard.ZstdCompressor(level=self.level).compress(data)

    def decompress(self, data: bytes) -> bytes:
        return zstandard.ZstdDecompressor().decompress(data)


def default_baselines() -> list[Compressor]:
    """Baselines compared against in every bench run (per CLAUDE.md)."""
    return [GzipCompressor(9), Bzip2Compressor(9), ZstdCompressor(19)]


# Re-export raw zlib level for completeness / debugging.
__all__ = [
    "Compressor",
    "GzipCompressor",
    "Bzip2Compressor",
    "LzmaCompressor",
    "ZstdCompressor",
    "default_baselines",
    "zlib",
]
