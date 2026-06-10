# Results

All numbers are produced by `python -m teleport.bench` (see `src/teleport/bench.py`).
Each row is appended automatically — no numbers are quoted from memory.

## Phase 0 — baseline reference compressors

Corpus: `data/enwik6` (first 1,000,000 bytes of enwik8, the Hutter Prize English
Wikipedia dump). bpc = compressed_bytes * 8 / raw_bytes.

| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |
|---|---|---|---|---|---|---|---|
| enwik6 | gzip-9 | 1000000 | 355791 | 2.8463 | 0.1039 | 0.0073 | pass |
| enwik6 | bzip2-9 | 1000000 | 281323 | 2.2506 | 0.1239 | 0.0439 | pass |
| enwik6 | zstd-19 | 1000000 | 300075 | 2.4006 | 0.5212 | 0.0029 | pass |

These are the numbers every later predictor must beat (or honestly fail to beat).
Phase 1's uniform predictor should land at exactly 8.00 bpc as a sanity check;
order-2/3 PPM should approach or beat the bzip2 row above.
