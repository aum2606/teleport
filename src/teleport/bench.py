"""Measurement harness.

Every compression number quoted anywhere in this project (results.md, commit
messages, plots) must come from this module. It reports, for a given
compressor and corpus:

  - compressed size in bytes
  - bits-per-character (bpc) = compressed_bytes * 8 / raw_bytes
  - compression and decompression wall time
  - round-trip verification (decompress(compress(x)) == x)

Usage:
    python -m teleport.bench --corpus data/enwik6
    python -m teleport.bench --corpus data/enwik6 --append results.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from teleport.baselines import Compressor, LzmaCompressor, default_baselines
from teleport.coder import PredictorCompressor
from teleport.hybrid import HybridCompressor
from teleport.predictors import PREDICTORS


@dataclass
class BenchResult:
    name: str
    corpus: str
    raw_bytes: int
    compressed_bytes: int
    bpc: float
    compress_time: float
    decompress_time: float
    roundtrip_ok: bool


def bench_compressor(compressor: Compressor, corpus_name: str, data: bytes) -> BenchResult:
    t0 = time.perf_counter()
    compressed = compressor.compress(data)
    compress_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decompressed = compressor.decompress(compressed)
    decompress_time = time.perf_counter() - t0

    raw_bytes = len(data)
    compressed_bytes = len(compressed)
    bpc = (compressed_bytes * 8) / raw_bytes if raw_bytes else 0.0

    return BenchResult(
        name=compressor.name,
        corpus=corpus_name,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        bpc=bpc,
        compress_time=compress_time,
        decompress_time=decompress_time,
        roundtrip_ok=decompressed == data,
    )


def bench_corpus(
    corpus_path: Path, compressors: list[Compressor] | None = None
) -> list[BenchResult]:
    compressors = compressors if compressors is not None else default_baselines()
    data = corpus_path.read_bytes()
    return [bench_compressor(c, corpus_path.name, data) for c in compressors]


def format_table(results: list[BenchResult]) -> str:
    header = "| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r.corpus} | {r.name} | {r.raw_bytes} | {r.compressed_bytes} "
            f"| {r.bpc:.4f} | {r.compress_time:.4f} | {r.decompress_time:.4f} "
            f"| {'pass' if r.roundtrip_ok else 'FAIL'} |"
        )
    return "\n".join(rows)


def append_to_results_md(results_path: Path, results: list[BenchResult]) -> None:
    table = format_table(results)
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n" + table + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the teleport benchmark harness.")
    parser.add_argument("--corpus", required=True, type=Path, help="Path to corpus file")
    parser.add_argument(
        "--append", type=Path, default=None, help="results.md path to append the table to"
    )
    parser.add_argument(
        "--predictor",
        choices=sorted(PREDICTORS),
        default=None,
        help="Also benchmark this predictor (in addition to the baseline compressors)",
    )
    parser.add_argument(
        "--predictor-only",
        action="store_true",
        help="With --predictor, skip the gzip/bzip2/zstd baseline rows",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Also benchmark the hybrid codec (best-of stored/lzma/bz2/neural, see hybrid.py)",
    )
    parser.add_argument(
        "--lzma",
        action="store_true",
        help="Also benchmark stdlib lzma-9 (one of the hybrid codec's classical candidates)",
    )
    args = parser.parse_args()

    compressors: list[Compressor] = [] if args.predictor_only else default_baselines()
    if args.predictor:
        compressors.append(PredictorCompressor(args.predictor, PREDICTORS[args.predictor]))
    if args.lzma:
        compressors.append(LzmaCompressor())
    if args.hybrid:
        compressors.append(HybridCompressor())

    results = bench_corpus(args.corpus, compressors)
    table = format_table(results)
    print(table)

    for r in results:
        if not r.roundtrip_ok:
            raise SystemExit(f"ROUND-TRIP FAILURE for {r.name} on {r.corpus}")

    if args.append:
        append_to_results_md(args.append, results)


if __name__ == "__main__":
    main()
