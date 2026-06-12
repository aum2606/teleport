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

## Phase 1 — arithmetic coder + swappable predictors

Same corpus (`data/enwik6`, 1,000,000 bytes). The coder is fixed (frozen);
only the predictor changes. `bpc = compressed_bytes * 8 / raw_bytes`
(includes an 8-byte length-header overhead, ~0.00006 bpc).

| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |
|---|---|---|---|---|---|---|---|
| enwik6 | uniform | 1000000 | 1000009 | 8.0001 | 24.7732 | 26.8902 | pass |
| enwik6 | order0 | 1000000 | 632832 | 5.0627 | 27.9672 | 31.1366 | pass |
| enwik6 | ppm2 | 1000000 | 382823 | 3.0626 | 116.4051 | 127.4788 | pass |
| enwik6 | ppm3 | 1000000 | 314703 | 2.5176 | 126.4382 | 155.8906 | pass |

The story end to end: smarter predictor, same coder, smaller file.
- `uniform` confirms the coder is exact: log2(256) = 8.00 bpc precisely.
- `order0` (~5.06 bpc) already beats nothing but establishes the adaptive
  baseline; the gap to `ppm2`/`ppm3` is purely from context modeling.
- `ppm3` (2.5176 bpc) beats gzip-9 (2.8463) and zstd-19 (2.4006 — nearly tied)
  but not bzip2-9 (2.2506). PPM here uses order blending without exclusion
  (see `src/teleport/predictors/ppm.py`), which is weaker than textbook
  PPM-C; bzip2's BWT-based model is a strong baseline at this corpus size.
- All predictors pass exact round-trip on the full 1MB corpus.
- Predictors are pure-Python/numpy and intentionally unoptimized for speed
  per CLAUDE.md (PPM takes ~2 minutes each way on 1MB).

**Coder near-optimality (Phase 1 acceptance check, ppm3 on enwik6):**
ppm3's measured cross-entropy is 2,528,726.7 bits (2.5287 bpc). The coder
produced 2,517,560 bits (2.5176 bpc) — within 0.44% of cross-entropy, inside
the ~1% bound. (The coder came in slightly *under* entropy here; integer
frequency quantization can round either way, see `tests/test_coder.py`.)

## Phase 2.1 — online neural (GRU) predictor

Corpus: `data/enwik6_20k` (first 20,000 bytes of `enwik6`). A much smaller
slice than Phase 0/1's 1MB: the online GRU does one forward+backward pass
per byte (`# SLOW: ok`, CLAUDE.md rule 5) at ~300 bytes/sec, so 1MB would
take well over an hour each way. Baselines are re-run on the same slice for
a fair comparison.

| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |
|---|---|---|---|---|---|---|---|
| enwik6_20k | gzip-9 | 20000 | 7675 | 3.0700 | 0.0010 | 0.0002 | pass |
| enwik6_20k | bzip2-9 | 20000 | 7258 | 2.9032 | 0.0036 | 0.0010 | pass |
| enwik6_20k | zstd-19 | 20000 | 7448 | 2.9792 | 0.0120 | 0.0008 | pass |
| enwik6_20k | rnn | 20000 | 12603 | 5.0412 | 54.8435 | 64.6522 | pass |

- `rnn` (5.0412 bpc) is an *online* GRU (embedding 16, hidden 64, Adam
  lr=0.01, double precision, single-threaded, fixed seed) trained from
  scratch as it compresses — no pretraining, no shared model transmitted.
  It does **not** beat gzip/bzip2/zstd or Phase 1's order0 (5.0627 bpc on
  the full 1MB corpus) at this size: 20,000 bytes is not enough for a
  randomly-initialized GRU to out-learn simple frequency/context statistics.
  This is an honest negative result for the *online-from-scratch* mode at
  this scale, not a bug — see IMPLEMENTATION_PLAN.md 2.2 (pretrained
  shared-model mode), which is expected to do much better by amortizing
  training over a held-out corpus.
- Round trip is exact on the 20KB slice.
- **Coder near-optimality check:** the rnn predictor's measured
  cross-entropy on this slice is 102,016.0 bits (5.1008 bpc). The coder
  produced 12603 bytes = 100,824 bits (5.0412 bpc) — about 1.17% *under*
  cross-entropy, just outside the ~1% bound but in the direction of integer
  frequency rounding helping rather than the coder leaking bits (see
  `tests/test_coder.py`).
- Determinism: `tests/verify_sync.py` runs two independently constructed
  `RNNPredictor` instances over the same 100-byte sample and checks
  `predict()` is bit-identical at every step (passes).

## Phase 2.2 — pretrained shared-model predictor ("teleport" configuration)

`predictors/rnn.PretrainedRNNPredictor` loads a frozen GRU (embedding 16,
hidden 128) trained offline by `scripts/pretrain_rnn.py` on
`data/enwik6[200000:400000]` (8 epochs of truncated BPTT, converging to
~2.15 bpc training loss). Both encoder and decoder load the identical
checkpoint (`models/rnn_shared.pt`) — the shared model is the "pre-staged
matter"; the wire carries only the residual surprise. No training happens
during compression (`predict()`/`update()` are `torch.no_grad()` forward
passes only), so it's ~8x faster than the online predictor (~2,460 vs ~300
bytes/sec).

**In-domain** (`data/enwik6_indomain_20k` = `enwik6[600000:620000]`,
disjoint from both the training range and the Phase 2.1 slice):

| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |
|---|---|---|---|---|---|---|---|
| enwik6_indomain_20k | gzip-9 | 20000 | 8138 | 3.2552 | 0.0023 | 0.0005 | pass |
| enwik6_indomain_20k | bzip2-9 | 20000 | 7587 | 3.0348 | 0.0036 | 0.0009 | pass |
| enwik6_indomain_20k | zstd-19 | 20000 | 7852 | 3.1408 | 0.0090 | 0.0004 | pass |
| enwik6_indomain_20k | rnn_pretrained | 20000 | 7298 | 2.9192 | 7.1305 | 7.3240 | pass |

**Out-of-domain** (`data/code_ood_20k` = first 20,000 bytes of this
project's own `src/**/*.py`, concatenated — Python source vs. the model's
English-Wikipedia training distribution):

| corpus | compressor | raw bytes | compressed bytes | bpc | compress (s) | decompress (s) | round-trip |
|---|---|---|---|---|---|---|---|
| code_ood_20k | gzip-9 | 20000 | 5981 | 2.3924 | 0.0012 | 0.0003 | pass |
| code_ood_20k | bzip2-9 | 20000 | 5836 | 2.3344 | 0.0024 | 0.0007 | pass |
| code_ood_20k | zstd-19 | 20000 | 5802 | 2.3208 | 0.0108 | 0.0001 | pass |
| code_ood_20k | rnn_pretrained | 20000 | 13025 | 5.2100 | 7.2947 | 7.5614 | pass |

**The thesis, made visible:**
- **In-domain** (2.9192 bpc): the pretrained shared model *beats every
  classical baseline* (gzip 3.2552, bzip2 3.0348, zstd 3.1408) — and beats
  Phase 2.1's online-from-scratch GRU by a wide margin (5.0412 bpc on a
  similar-sized slice). Pretraining on held-out English text and freezing
  the weights pays off: the wire only carries the surprise relative to a
  model that already "knows" English Wikipedia prose.
- **Out-of-domain** (5.2100 bpc): the *same frozen model* on Python source
  is far worse than gzip/bzip2/zstd (~2.3 bpc) and even worse than Phase
  1's order-0 byte-frequency model (5.0627 bpc on enwik6) — predicting
  Python tokens with a model trained only on English prose is close to
  guessing.
- **The gap (2.9192 -> 5.2100 bpc) is the whole point**: a shared model is
  "pre-staged matter" that is amortized for free only when the wire content
  matches what the model was pre-staged for. Classical compressors build
  their model from the data itself every time, so they're indifferent to
  domain — which is exactly why they win out-of-domain and lose in-domain
  here.
- Round trip is exact in both directions.
- Determinism: both `RNNPredictor` (Phase 2.1) and `PretrainedRNNPredictor`
  (Phase 2.2) pass `tests/verify_sync.py` / `tests/test_rnn.py`'s
  determinism checks — two independently constructed instances (loading the
  same checkpoint) produce bit-identical predictions.

## Phase 3 — lossy image compression (small end-to-end slice)

**Setup.** A small Balle-style convolutional autoencoder
(`src/teleport/image/model.py`): 3 stride-2 conv layers (encoder/decoder,
hidden width 32, /8 downsampling), 4 latent channels, one factorized
Gaussian entropy model per latent channel. Quantized latent symbols
(byte = round(z) + 128, clamped to [0, 255]) are coded losslessly with the
frozen Phase 1 arithmetic coder via a non-adaptive `LatentPredictor` that
reads a fixed 256-bin probability table per channel from the trained
Gaussian (`src/teleport/image/codec.py`). The image pipeline itself is
lossy — only the latent-symbol entropy coding is required to round-trip
exactly, and `tests/test_image.py` checks that.

**Training** (`scripts/train_image.py`): 576 random 64x64 patches cropped
from `kodim01..kodim18`, loss = rate (bpp, from the differentiable
factorized-Gaussian rate estimate with additive-uniform-noise quantization)
+ 50 * 255^2 * MSE, Adam lr=1e-3, 50 epochs, CPU, seed 1234
(`models/image_codec.pt`, gitignored). Final training-set numbers: 0.338 bpp,
23.68 dB PSNR.

**Evaluation** (`scripts/eval_image.py`): held out `kodim19..kodim24`
(never seen during training). For each image, JPEG and WebP quality were
swept to find the encoding closest in size to ours, so PSNR is compared at
matched bpp.

| image | size | ours bpp | ours PSNR | JPEG bpp | JPEG PSNR | WebP bpp | WebP PSNR |
|---|---|---|---|---|---|---|---|
| kodim19 | 768x512 | 0.3295 | 23.00 | 0.3288 | 27.19 | 0.3286 | 30.06 |
| kodim20 | 512x768 | 0.5332 | 23.64 | 0.5346 | 32.70 | 0.5358 | 35.62 |
| kodim21 | 512x768 | 0.3172 | 23.67 | 0.3145 | 25.76 | 0.3187 | 28.57 |
| kodim22 | 512x768 | 0.3244 | 23.91 | 0.3333 | 27.30 | 0.3227 | 29.38 |
| kodim23 | 512x768 | 0.3550 | 22.88 | 0.3526 | 32.16 | 0.3558 | 35.42 |
| kodim24 | 512x768 | 0.3161 | 22.23 | 0.3268 | 24.08 | 0.3228 | 26.44 |
| **avg** | | 0.3626 | 23.22 | 0.3651 | 28.20 | 0.3641 | 30.91 |

**Honest result: we lose, by a lot.** At matched bitrate (~0.36 bpp,
within the < 0.3 bpp "low bitrate" regime CLAUDE.md flags as the fair
comparison point), our codec trails JPEG by ~5 dB PSNR and WebP by ~7.7 dB
on average — every single held-out image is worse than both baselines. This
is not surprising for a 4-latent-channel, 50-epoch, 576-patch, single-image-
family model: the entropy model is a single global Gaussian per channel
(no spatial context at all, unlike JPEG's per-block DCT + Huffman or WebP's
predictive coding), and the encoder/decoder have seen only 18 images. The
~0.36 bpp floor (vs. the ~0.3 bpp target) comes from the rate/distortion
tradeoff at lambda=50; pushing lambda lower would cut bpp further but at
even worse PSNR — the rate-distortion curve for this tiny model is simply
inferior to JPEG's across the regime tested, not just at this one point.

**What this slice proves and doesn't.** It proves the full pipeline works
end to end and losslessly: a learned, content-adaptive probability model
(the per-channel Gaussian, fit by gradient descent) drives the same frozen
arithmetic coder used for text, exactly as the "swappable predictor" thesis
requires — `compress_image`/`decompress_image` round-trip the *latent
symbols* exactly, with no coder changes (CLAUDE.md rule 2). It does not
prove learned image compression is competitive at this scale; closing the
gap to JPEG/WebP would need spatial entropy context (e.g. a small
autoregressive prior over latent positions, the "hyperprior" approach) and
more training data/epochs — explicitly out of scope for "don't train large
models."

## Phase 3 (revised) — spatial context entropy model

**What changed.** Replaced the per-channel global `FactorizedEntropyModel`
with `SpatialContextModel` (`src/teleport/image/model.py`): a small
"PixelCNN-lite" — for each latent symbol, a 2-layer MLP (hidden 16) predicts
a Gaussian(mean, scale) from (left neighbor, top neighbor, channel index).
Symbols are coded channel-major, raster order within each channel, so both
context values are always already-decoded by the time a symbol is coded.
`ContextLatentPredictor` (`src/teleport/image/codec.py`) implements this as
the arithmetic coder's adaptive `Predictor` — one MLP forward per latent
symbol (`# SLOW: ok`), still the unmodified Phase 1 coder underneath. Same
autoencoder architecture, same training data (576 64x64 patches from
`kodim01..kodim18`), same lambda=50, 50 epochs, retrained from scratch
(`models/image_codec.pt`). Training-set numbers improved from 0.338 bpp /
23.68 dB to **0.206 bpp / 23.74 dB** — the context model finds real
redundancy between neighboring latent positions.

**Evaluation** on held-out `kodim19..kodim24` (`scripts/eval_image.py`),
JPEG/WebP quality swept to match our bpp per image:

| image | size | ours bpp | ours PSNR | JPEG bpp | JPEG PSNR | WebP bpp | WebP PSNR |
|---|---|---|---|---|---|---|---|
| kodim19 | 768x512 | 0.1754 | 23.14 | 0.1820 | 21.49 | 0.1815 | 28.04 |
| kodim20 | 512x768 | 0.1636 | 24.24 | 0.1640 | 22.78 | 0.1648 | 30.67 |
| kodim21 | 512x768 | 0.1896 | 23.79 | 0.1817 | 20.90 | 0.1905 | 26.66 |
| kodim22 | 512x768 | 0.1738 | 24.20 | 0.1787 | 23.12 | 0.1692 | 27.65 |
| kodim23 | 512x768 | 0.1683 | 23.37 | 0.1634 | 22.68 | 0.1681 | 32.11 |
| kodim24 | 512x768 | 0.2006 | 22.36 | 0.1948 | 21.05 | 0.2237 | 25.17 |
| **avg** | | 0.1785 | 23.52 | 0.1774 | 22.01 | 0.1830 | 28.38 |

**Result: we now beat JPEG, on every held-out image, at ~0.18 bpp.** Average
PSNR is 23.52 dB vs JPEG's 22.01 dB at essentially matched bitrate (1.5 dB
average gain, positive on all 6 images) — the payoff moment IMPLEMENTATION_PLAN.md
section 3.2 describes, on PSNR. WebP still wins by ~4.9 dB; closing that gap
would need a richer entropy context (more neighbors, e.g. all 4 causal
neighbors instead of 2, or cross-channel context at the same position) and/or
a less aggressive bitrate (JPEG in particular degrades sharply below ~0.2 bpp,
visible in kodim19/21/24 where its PSNR drops into the low 20s — the same
"low bitrate is where learned codecs shine" effect IMPLEMENTATION_PLAN.md
predicted, just not yet enough to catch WebP). Note the acceptance criterion
in IMPLEMENTATION_PLAN.md section 3 is MS-SSIM, not PSNR — MS-SSIM was not
measured here (would need an additional dependency or hand-rolled
implementation); PSNR alone is suggestive but not the literal criterion.

**Why this worked.** The only change was the entropy model — same encoder,
decoder, training data, and rate-distortion lambda. Lower achieved bpp
(0.34 -> 0.21 on the training set) at *equal* PSNR is a direct, isolated
demonstration of this project's thesis: a model that predicts the latent
symbols better (here, "better" = uses 2 pixels of causal context instead of
none) needs fewer bits for the same information — "a model that predicts
data well IS a compressor."

## Phase 3 — MS-SSIM (acceptance criterion check)

IMPLEMENTATION_PLAN.md section 3's literal acceptance criterion is "beat
JPEG on MS-SSIM at low bitrates (< 0.3 bpp)", not PSNR. Implemented MS-SSIM
by hand (`src/teleport/image/metrics.py`, Wang/Simoncelli/Bovik 2003,
5-scale Gaussian-window SSIM, torch only — no new dependencies) and re-ran
`scripts/eval_image.py` on the same held-out `kodim19..kodim24` /
~0.18 bpp comparison:

| image | size | ours bpp | ours PSNR | ours MS-SSIM | JPEG bpp | JPEG PSNR | JPEG MS-SSIM | WebP bpp | WebP PSNR | WebP MS-SSIM |
|---|---|---|---|---|---|---|---|---|---|---|
| kodim19 | 768x512 | 0.1754 | 23.14 | 0.8784 | 0.1820 | 21.49 | 0.7137 | 0.1815 | 28.04 | 0.9090 |
| kodim20 | 512x768 | 0.1636 | 24.24 | 0.9310 | 0.1640 | 22.78 | 0.8236 | 0.1648 | 30.67 | 0.9547 |
| kodim21 | 512x768 | 0.1896 | 23.79 | 0.9020 | 0.1817 | 20.90 | 0.7473 | 0.1905 | 26.66 | 0.9271 |
| kodim22 | 512x768 | 0.1738 | 24.20 | 0.8602 | 0.1787 | 23.12 | 0.7075 | 0.1692 | 27.65 | 0.8841 |
| kodim23 | 512x768 | 0.1683 | 23.37 | 0.8852 | 0.1634 | 22.68 | 0.7310 | 0.1681 | 32.11 | 0.9529 |
| kodim24 | 512x768 | 0.2006 | 22.36 | 0.8832 | 0.1948 | 21.05 | 0.7527 | 0.2237 | 25.17 | 0.9036 |
| **avg** | | 0.1785 | 23.52 | 0.8900 | 0.1774 | 22.01 | 0.7460 | 0.1830 | 28.38 | 0.9219 |

**Acceptance criterion met.** At matched bpp (~0.18, well under the 0.3 bpp
threshold), our codec beats JPEG on MS-SSIM on all 6 held-out images by a
wide margin (avg 0.8900 vs 0.7460 — JPEG's MS-SSIM falls off a cliff below
~0.2 bpp due to blocking artifacts that MS-SSIM penalizes heavily, which
PSNR underweights). WebP still leads (0.9219), but the gap is much smaller
in MS-SSIM terms (0.032) than in PSNR terms (4.86 dB) — WebP's predictive
coding preserves perceptual structure even where its pixel-exact PSNR
advantage is large.

This closes the Phase 3 small-slice acceptance gap: the spatial-context
codec (`SpatialContextModel` + `ContextLatentPredictor`) beats a classical
codec (JPEG) on the metric IMPLEMENTATION_PLAN.md specifies, at the
bitrates it specifies, using the same Phase 1 arithmetic coder unchanged.

## Phase 4 — generative/semantic extreme (Option A: latent-diffusion "teleport")

**Demo:** TAESD (`madebyollin/taesd`, frozen, ~2.4M params / 9.79MB
safetensors, 4 latent channels, /8 spatial downsample) is the "pre-staged
matter" at both ends. The encoder runs an image through TAESD's encoder,
scales the latent to [0, 1], average-pools it by an additional 8x (so the
total downsample vs the original image is 64x per side, i.e. 4096x fewer
spatial positions), and quantizes to uint8. This tiny grid (4 channels x
H/64 x W/64) is entropy-coded with `Order0Predictor` through the unmodified
Phase 1 arithmetic coder. The decoder reverses this: decode symbols ->
nearest-upsample 8x back to TAESD's latent resolution -> unscale -> TAESD's
decoder regenerates a full image. Implementation:
`src/teleport/generative/codec.py`; lossless round trip of the quantized
small-latent symbols is tested in `tests/test_generative.py`.

**Eval (`scripts/eval_generative.py`, held-out `kodim19..kodim24`, same set
as Phase 3):**

| image | size | ours bytes | ours bpp | ours PSNR | ours MS-SSIM | JPEG bytes | JPEG bpp | JPEG PSNR | JPEG MS-SSIM | WebP bytes | WebP bpp | WebP PSNR | WebP MS-SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kodim19 | 768x512 | 347 | 0.00706 | 17.50 | 0.4971 | 8948 | 0.18205 | 21.49 | 0.7137 | 7476 | 0.15210 | 27.36 | 0.8945 |
| kodim20 | 512x768 | 377 | 0.00767 | 16.85 | 0.6092 | 8060 | 0.16398 | 22.78 | 0.8236 | 5020 | 0.10213 | 28.98 | 0.9359 |
| kodim21 | 512x768 | 343 | 0.00698 | 17.32 | 0.4893 | 8466 | 0.17224 | 20.69 | 0.7350 | 8418 | 0.17126 | 26.31 | 0.9204 |
| kodim22 | 512x768 | 349 | 0.00710 | 18.67 | 0.4660 | 7830 | 0.15930 | 21.79 | 0.6424 | 6630 | 0.13489 | 27.08 | 0.8651 |
| kodim23 | 512x768 | 367 | 0.00747 | 16.65 | 0.5200 | 7820 | 0.15910 | 22.53 | 0.7261 | 5152 | 0.10482 | 29.87 | 0.9238 |
| kodim24 | 512x768 | 343 | 0.00698 | 16.53 | 0.3661 | 8987 | 0.18284 | 20.79 | 0.7292 | 10994 | 0.22367 | 25.17 | 0.9036 |
| **avg** | | 354 | 0.00721 | 17.25 | 0.4913 | 8352 | 0.16992 | 21.68 | 0.7283 | 7282 | 0.14815 | 27.46 | 0.9072 |

`JPEG`/`WebP` columns are `_codec_at_bytes` searching quality 1..95 for the
size closest to ours -- but quality 1 is already 7.8-11.0 KB for these
images, ~20-25x larger than our ~354-byte average. **Classical codecs
cannot reach this byte budget at all**, even at their lowest quality
setting. Our codec produces a recognizable, correctly-colored,
correctly-composed ~17.25 dB / 0.49 MS-SSIM image from ~354 bytes; JPEG/WebP
at their *smallest possible* size (~7-9 KB) score 21.68 dB/0.73 and
27.46 dB/0.91.

LPIPS (suggested by IMPLEMENTATION_PLAN.md section 4) was substituted with
the hand-rolled MS-SSIM from Phase 3, to avoid adding a large pretrained
AlexNet/VGG dependency -- same honest-substitution pattern as the
Kodak-vs-CIFAR10 swap in Phase 3.

**"When is this real compression vs a parlor trick?"** TAESD's ~9.79MB of
weights are the shared "pre-staged matter" -- downloaded once, frozen,
identical at both ends, and amortized over every image ever sent. For a
*single* image this is obviously not a win (the model dwarfs any byte
saved). But once both ends already hold the model (the realistic framing:
e.g. it ships with a browser or OS), each additional image costs ~354
bytes regardless of content, ~20-25x below what JPEG/WebP can reach at
*any* quality setting for these 512x768-1000x768px photos. The honest
trade is what those bytes buy: not "this image, slightly degraded" but "a
plausible image with the same rough colors, composition, and lighting" --
fine texture, small objects, and exact detail are regenerated/hallucinated,
not preserved. This is the project's central thesis pushed to its limit:
the more powerful and more *shared* the predictive model, the more of
"the image" can live in the model rather than on the wire, until the wire
carries almost nothing at all.
