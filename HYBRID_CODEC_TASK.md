# Task Brief — Hybrid Codec (one session)

**Goal:** a codec that is *never* meaningfully worse than a classical compressor and
sometimes much better, by trying multiple methods and keeping the smallest output.
This neutralizes the Phase 2.2 out-of-domain failure mode (5.21 bpc on Python source)
at the cost of exactly one header byte, while keeping the in-domain win (2.92 bpc).

Read CLAUDE.md first. All its rules apply, especially: round-trip is sacred,
no new dependencies, all numbers via bench.py.

---

## Design

### Container format
```
[1 byte: method id][payload bytes]
```

| id | method | payload |
|---|---|---|
| 0x00 | stored | raw bytes, uncompressed (incompressible-input guard) |
| 0x01 | classical | stdlib `lzma` (preset 9) output |
| 0x02 | classical-alt | stdlib `bz2` (level 9) output |
| 0x03 | neural | Phase 1 arithmetic coder + `PretrainedRNNPredictor` output |

Notes:
- **No new dependencies** (CLAUDE.md): the classical fallbacks are Python stdlib
  `lzma` and `bz2`, not zstd. lzma typically beats our bzip2-9 baseline anyway;
  keep bz2 as a second cheap candidate since it won Phase 1's table on enwik6.
- Method 0x00 guarantees worst case = raw + 1 byte even on random/encrypted input,
  where every compressor expands.
- Reserve ids 0x04+ for future predictors (ppm3 could be added later as a
  mid-speed candidate; out of scope today).

### Encoder
```python
def compress_hybrid(data: bytes, model_path: str = "models/rnn_shared.pt") -> bytes:
    candidates = {
        0x00: data,
        0x01: lzma.compress(data, preset=9),
        0x02: bz2.compress(data, 9),
        0x03: compress_neural(data, model_path),   # existing Phase 2.2 path
    }
    method, payload = min(candidates.items(), key=lambda kv: len(kv[1]))
    return bytes([method]) + payload
```
Ties: `min` keeps the lowest id — fine (prefers simpler method on a tie).

### Decoder
Dispatch on the header byte to `identity` / `lzma.decompress` / `bz2.decompress` /
`decompress_neural`. Unknown id → raise `ValueError` (never guess).

### File placement
- `src/teleport/hybrid.py` — `compress_hybrid`, `decompress_hybrid`, the method table
- `tests/test_hybrid.py`
- bench.py: register compressor name `hybrid` so it appears as a normal row

---

## Tests (write these first)

1. **Round-trip exact** on: `data/enwik6_indomain_20k`, `data/code_ood_20k`,
   1-byte input, empty input, 1KB of `os.urandom` (must select 0x00),
   1KB of `b"\x00"*1024` (any method, must round-trip).
2. **Never-worse property:** for each corpus,
   `len(hybrid) <= min(len(raw), len(lzma), len(bz2), len(neural)) + 1`.
   This is the entire point of the codec — assert it literally.
3. **Method selection sanity:** in-domain slice selects 0x03 (neural);
   code_ood slice selects 0x01 or 0x02 (classical); urandom selects 0x00.
   If in-domain does NOT select neural, that's a finding, not a test bug —
   investigate before "fixing" the assertion (CLAUDE.md: don't loosen tests).
4. **Unknown header id** raises.

---

## Bench rows to produce (append to results.md)

Run on the same two Phase 2.2 slices so the comparison is direct:

| corpus | compressor | expected outcome |
|---|---|---|
| enwik6_indomain_20k | hybrid | ~2.92 bpc + 0.0004 (header) — matches rnn_pretrained, beats all classical |
| code_ood_20k | hybrid | ~lzma/bz2-level (~2.3 bpc) — the 5.21 bpc OOD failure is gone |

Also add `lzma-9` and `bz2-9` standalone rows for these slices if not present,
so the "hybrid = best-of + 1 byte" claim is verifiable from the table alone.

Write a short results.md narrative paragraph: worst case is now
best-classical + 1 byte; in-domain advantage retained; this is the standard
production pattern (codecs with format negotiation / dictionary fallback).

---

## Honest costs to state in results.md

- **Compression time = sum of all candidates**, dominated by the neural path
  (~2.4 KB/s ⇒ ~8s per 20KB slice). Decompression pays only the chosen method.
  `# SLOW: ok` per CLAUDE.md rule 5 — but say it in the table.
- The 1-byte header is ~0.0004 bpc on 20KB inputs; negligible but nonzero — include
  it in the reported bpc (bench.py measures total container size, so this is
  automatic; just don't special-case it out).

## Stretch (only if the session has time left)

A cheap **probe** to skip the neural candidate when it's hopeless: run the
pretrained predictor over the first 2KB only, measure cross-entropy; if it's
worse than, say, 4.5 bpc, skip the full neural encode. Changes nothing about
the format (encoder-side heuristic only — decoder is untouched). Must add a
bench row showing identical output sizes with and without the probe on both
slices, and the compress-time saving on code_ood.

## Acceptance (session is done when)

1. All tests in `tests/test_hybrid.py` pass, plus the full existing suite.
2. results.md has the two hybrid rows + lzma/bz2 reference rows + narrative.
3. The never-worse property holds on both slices by direct measurement.
