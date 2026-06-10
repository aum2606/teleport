# CLAUDE.md

## What this project is

`teleport` — a learned-compression research project proving one idea end to end:
**a model that predicts data well IS a compressor; better prediction = fewer bits.**
We build a swappable-predictor arithmetic-coding engine, climb from n-gram → neural →
lossy image → generative compression, and measure honestly at every step.

The guiding metaphor: don't move the data, move the instruction to reconstruct it.
The shared model at both ends is "pre-staged matter"; the wire carries only selection.

## Repository layout

```
teleport/
├── CLAUDE.md
├── IMPLEMENTATION_PLAN.md   # phased plan — read before starting any new phase
├── results.md               # benchmark table; append a row after every milestone
├── data/                    # benchmark corpora (gitignored; fetch via scripts/get_data.py)
├── src/teleport/
│   ├── coder/               # arithmetic coder (Phase 1) — FROZEN after Phase 1 passes
│   ├── predictors/          # uniform, order0, ppm, rnn, transformer — all implement Predictor
│   ├── image/               # Phase 3 autoencoder codec
│   └── bench.py             # measurement harness; ALL numbers come from here
├── tests/
└── scripts/
```

## Hard rules (do not violate)

1. **Round-trip is sacred.** Every lossless change must pass
   `decompress(compress(x)) == x` byte-exact on the full test corpora before merge.
   A compressor that corrupts data silently is worse than no compressor.
2. **The Predictor interface is the contract.**
   `predict() -> distribution over 256 symbols`, then `update(symbol)`.
   Predictors must be strictly causal — `predict()` may never see the current symbol.
   Never modify the coder to accommodate a predictor; fix the predictor.
3. **Determinism in neural mode.** Encoder and decoder must produce bit-identical
   model states. CPU-only for adaptive neural compression; fix all RNG seeds;
   `torch.use_deterministic_algorithms(True)`; single-threaded
   (`torch.set_num_threads(1)`). Run `tests/verify_sync.py` after touching anything
   in `predictors/rnn*` or the training loop.
4. **All benchmark numbers go through `bench.py`** and into `results.md` with:
   corpus, predictor/config, bpc or bpp, wall time, baseline comparison
   (gzip -9, zstd -19; JPEG/WebP for images). No numbers quoted from memory.
5. **Don't optimize the coder for speed during Phases 1–2.** Correctness and
   clarity first; this is a research codebase. Mark slow paths with `# SLOW: ok`.

## Conventions

- Python 3.11+, type hints everywhere, `numpy` for distributions
  (float64 in the coder path — float32 caused cumulative-probability ties).
- Symbols are bytes (alphabet size 256) everywhere unless a module says otherwise.
- Probabilities passed to the coder must be > 0 for every symbol (floor at 1/2^16
  and renormalize). The coder must never receive a zero-probability true symbol.
- Tests: `pytest -q`. Coder tests live in `tests/test_coder.py` and include
  property-based round trips (hypothesis) — run them before and after any coder edit.
- Commits: one logical change; message says what metric moved if any
  (e.g., "ppm order-3: 2.71 → 2.58 bpc on enwik6").

## Common commands

```bash
pytest -q                          # full test suite
python scripts/get_data.py         # fetch benchmark corpora into data/
python -m teleport.bench --predictor ppm --corpus data/enwik6
python tests/verify_sync.py        # encoder/decoder lockstep check (neural mode)
```

## Current phase & state

> Update this section as work progresses — it is the single source of truth
> for "where are we".

- **Phase:** 0 complete — bench harness, baselines, and enwik6 corpus are in place.
- **Next milestone:** Phase 1 — integer arithmetic coder + Predictor protocol +
  uniform/order-0/PPM predictors (see IMPLEMENTATION_PLAN.md section 1).
- **Frozen components:** none yet (coder freezes at end of Phase 1)

## Things that will bite you (learned the hard way / known pitfalls)

- Arithmetic-coder carry propagation and the E3 underflow case are where 90% of
  bugs live. Test all-0xFF and all-0x00 inputs and 1-byte files explicitly.
- Off-by-one between encoder's `update()` timing and decoder's: both must update
  with the *decoded/encoded* symbol at the same step index. Desync shows up as
  garbage output starting mid-file, not at byte 0.
- If compressed size is *worse* than the predictor's measured cross-entropy by
  more than ~1%, the coder is leaking bits — investigate before blaming the model.
- In Phase 3, evaluate at LOW bitrates first (< 0.3 bpp); learned codecs lose to
  JPEG at high bitrates and that's expected, not a bug.

## What NOT to do

- Don't add dependencies beyond numpy/torch/pillow/pytest/matplotlib without
  noting why here.
- Don't "fix" failing round-trip tests by loosening the assertion to fuzzy
  equality. Lossless means lossless.
- Don't train large models; this project's models are deliberately small
  (≤ ~10M params through Phase 2). The thesis is about the *principle*, not SOTA.
- Don't start Phase N+1 while Phase N acceptance criteria are unmet
  (see IMPLEMENTATION_PLAN.md).
