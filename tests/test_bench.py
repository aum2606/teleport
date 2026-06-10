from pathlib import Path

import pytest

from teleport.baselines import Bzip2Compressor, GzipCompressor, ZstdCompressor
from teleport.bench import bench_compressor, format_table


@pytest.mark.parametrize(
    "compressor", [GzipCompressor(9), Bzip2Compressor(9), ZstdCompressor(19)]
)
def test_baseline_roundtrip(compressor):
    data = b"the quick brown fox jumps over the lazy dog " * 100
    result = bench_compressor(compressor, "synthetic", data)
    assert result.roundtrip_ok
    assert result.raw_bytes == len(data)
    assert result.compressed_bytes > 0
    assert result.bpc > 0


@pytest.mark.parametrize(
    "compressor", [GzipCompressor(9), Bzip2Compressor(9), ZstdCompressor(19)]
)
def test_baseline_roundtrip_empty(compressor):
    result = bench_compressor(compressor, "empty", b"")
    assert result.roundtrip_ok
    assert result.raw_bytes == 0
    assert result.bpc == 0.0


def test_format_table_smoke():
    data = b"abc" * 50
    results = [bench_compressor(GzipCompressor(9), "synthetic", data)]
    table = format_table(results)
    assert "gzip-9" in table
    assert "synthetic" in table


def test_enwik6_roundtrip_if_present():
    enwik6 = Path(__file__).resolve().parent.parent / "data" / "enwik6"
    if not enwik6.exists():
        pytest.skip("data/enwik6 not fetched (run scripts/get_data.py)")
    data = enwik6.read_bytes()
    assert len(data) == 1_000_000
    result = bench_compressor(GzipCompressor(9), "enwik6", data)
    assert result.roundtrip_ok
