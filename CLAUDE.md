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

- **Phase:** 4 (Option A, generative/semantic extreme) complete.
  - 2.1 (online neural predictor): `predictors/rnn.RNNPredictor` is a small
    online GRU (embedding 16, hidden 64, double precision, single-threaded,
    fixed seed, Adam) trained as it compresses; 5.0412 bpc on a 20KB enwik6
    slice (worse than gzip/bzip2/zstd/order0 — honest negative result for
    online-from-scratch at this scale, see results.md).
  - 2.2 (pretrained shared-model "teleport" mode):
    `predictors/rnn.PretrainedRNNPredictor` loads a frozen GRU (embedding
    16, hidden 128) pretrained by `scripts/pretrain_rnn.py` on
    `enwik6[200000:400000]`. In-domain: 2.9192 bpc — beats gzip/bzip2/zstd.
    Out-of-domain (Python source): 5.2100 bpc — worse than all classical
    baselines. The in-domain/out-of-domain gap is the project thesis made
    concrete (see results.md).
  - `tests/verify_sync.py` checks encoder/decoder lockstep determinism for
    both predictors and passes.
  - 3 (lossy image compression, small end-to-end slice): a tiny conv
    autoencoder (`src/teleport/image/model.py`, 4 latent channels, /8
    downsample) with a `SpatialContextModel` entropy model — a 2-layer
    MLP "PixelCNN-lite" predicting each latent symbol's Gaussian(mean,
    scale) from its left/top causal neighbors and channel index — trained
    on 64x64 patches from `kodim01..kodim18` (`scripts/train_image.py`,
    `models/image_codec.pt`). Latent symbols are entropy-coded losslessly
    via the frozen Phase 1 coder, with `ContextLatentPredictor`
    (`src/teleport/image/codec.py`) supplying a fresh per-position table
    from the context model. Evaluated on held-out `kodim19..kodim24`
    (`scripts/eval_image.py`): **0.1785 bpp avg, 23.52 dB PSNR / 0.8900
    MS-SSIM — beats JPEG (0.1774 bpp, 22.01 dB / 0.7460 MS-SSIM) on every
    one of the 6 held-out images on both metrics**, by a wide MS-SSIM
    margin (0.8900 vs 0.7460). WebP still leads (0.1830 bpp, 28.38 dB /
    0.9219 MS-SSIM), but the MS-SSIM gap (0.032) is much smaller than the
    PSNR gap (4.86 dB). MS-SSIM implemented by hand, torch only
    (`src/teleport/image/metrics.py`, Wang/Simoncelli/Bovik 2003). An
    earlier version with a per-channel global Gaussian (no spatial
    context) scored 0.36 bpp / 23.22 dB — same encoder/decoder/data/
    lambda, only the entropy model changed, and bpp dropped 0.34 -> 0.21
    at equal PSNR on the training set. IMPLEMENTATION_PLAN.md section 3's
    literal acceptance criterion ("beat JPEG on MS-SSIM at < 0.3 bpp") is
    met (see results.md). The latent-symbol round trip through the
    unmodified arithmetic coder is exact and tested
    (`tests/test_image.py`).
  - 4 (generative/semantic extreme, Option A — latent-diffusion
    "teleport"): TAESD (`madebyollin/taesd`, frozen, ~2.4M params /
    9.79MB, 4 latent channels, /8 downsample) is the shared "pre-staged
    matter". `src/teleport/generative/codec.py` encodes via TAESD,
    average-pools the scaled latent by an extra 8x (64x total per side),
    quantizes to uint8, and entropy-codes the resulting tiny grid (4 x
    H/64 x W/64) with `Order0Predictor` through the unmodified Phase 1
    coder; the decoder upsamples (nearest) and runs TAESD's decoder.
    Lossless round trip of the quantized symbols is tested in
    `tests/test_generative.py`. Evaluated on held-out `kodim19..kodim24`
    (`scripts/eval_generative.py`): **avg 354 bytes (0.0072 bpp), 17.25 dB
    PSNR / 0.4913 MS-SSIM**. JPEG/WebP at quality 1 (their smallest
    possible size) are 7.8-11.0 KB — ~20-25x larger than our payload —
    and score 21.68 dB/0.7283 and 27.46 dB/0.9072 respectively; **classical
    codecs cannot reach this byte budget at all**. LPIPS (suggested by
    IMPLEMENTATION_PLAN.md section 4) was substituted with the Phase 3
    hand-rolled MS-SSIM, documented in `scripts/eval_generative.py` and
    results.md. Writeup of "when is this real compression vs a parlor
    trick?" is in results.md (Phase 4 section).
- **Next milestone:** none — the user chose to stop after Phase 4. All
  4 planned phases are complete with documented, honest results
  (results.md). If resumed later, IMPLEMENTATION_PLAN.md's "Stretch /
  Research directions (after Phase 4)" section has candidate next steps
  (bit-identical int8 neural inference, delta/content-addressed transport,
  federated dictionary research).
- **Frozen components:** `src/teleport/coder/` (arithmetic.py, bitio.py,
  freqs.py, codec.py) — round-trip-exact on enwik6 for all four text
  predictors and on the Phase 3 image latent symbols. Do not modify; fix
  predictors instead.

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
  - Phase 4 added `diffusers` + `safetensors` (optional `generative` extra)
    to load TAESD (`madebyollin/taesd`, ~2.4M params, 4 latent channels, /8
    spatial downsample), a tiny pretrained VAE used as the "pre-staged
    matter" for the generative-extreme demo. No training of this model is
    done — it's loaded frozen.
- Don't "fix" failing round-trip tests by loosening the assertion to fuzzy
  equality. Lossless means lossless.
- Don't train large models; this project's models are deliberately small
  (≤ ~10M params through Phase 2). The thesis is about the *principle*, not SOTA.
- Don't start Phase N+1 while Phase N acceptance criteria are unmet
  (see IMPLEMENTATION_PLAN.md).
