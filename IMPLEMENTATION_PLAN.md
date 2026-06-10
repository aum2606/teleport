# Learned Compression — Implementation Plan

**Project codename:** `teleport` — "don't move the data, move the instruction to reconstruct it"

**Core thesis being proven:** Any model that predicts data well can be turned into a compressor. Better prediction = fewer bits on the wire. We build this in four phases, each one a working, measurable artifact.

---

## Phase 0 — Project Setup (half a day)

**Goal:** Reproducible environment, benchmark data, and a measurement harness so every later phase reports honest numbers.

**Deliverables:**
- Python 3.11+ project with `uv` or `venv`, pinned dependencies
- `data/` folder with benchmark corpora:
  - `enwik6` (first 1 MB of English Wikipedia dump — small cousin of the famous enwik8/enwik9 benchmark)
  - A folder of 50–100 small images (e.g., a CIFAR-10 subset or Kodak images for Phase 3)
- `bench.py`: a harness that, given a compressor, reports:
  - compressed size (bytes), bits-per-character (bpc) or bits-per-pixel (bpp)
  - compression + decompression wall time
  - **round-trip verification** (decompress(compress(x)) == x for lossless phases)
  - comparison row against `gzip -9`, `bzip2`, `zstd -19`
- A `results.md` that gets a new table row after every milestone

**Why this first:** the whole project is an empirical argument. If measurement is sloppy, nothing downstream means anything.

---

## Phase 1 — Arithmetic Coder + N-gram Predictor (Week 1)

**Goal:** A working lossless compressor where the predictor is swappable. This is the engine for everything else.

### 1.1 Arithmetic coder (the hard, fiddly part)
- Implement integer-based arithmetic coding (32-bit registers, carry propagation, E1/E2/E3 renormalization)
- Interface contract:
  - Encoder consumes `(symbol, cumulative_distribution)` pairs, emits bits
  - Decoder consumes bits + the same distributions, emits symbols
  - **The predictor must be causal**: distribution for symbol *t* may only depend on symbols < *t*, so encoder and decoder stay in lockstep
- Unit tests: known-distribution round trips, adversarial inputs (all-same-byte, random bytes, empty file)
- Acceptance: round-trip exact on all of `enwik6`; compressed size within ~1% of the predictor's measured cross-entropy (this proves the coder is near-optimal — any gap is coder overhead)

### 1.2 Predictor interface
```python
class Predictor(Protocol):
    def predict(self) -> np.ndarray      # prob distribution over next symbol (256 bytes)
    def update(self, symbol: int) -> None # observe the symbol that actually occurred
```
Everything later (n-gram, neural, anything) implements this. The coder never changes again.

### 1.3 Predictors, in order
1. **Uniform** (sanity check — should give exactly 8.00 bpc)
2. **Order-0 adaptive** (byte frequency counts) — expect ~5 bpc on English text
3. **Order-2 / order-3 context model with escape/backoff (PPM-style)** — expect ~2.5–3 bpc, competitive with gzip or better

### Milestone metric
A table in `results.md`: predictor vs bpc vs gzip/zstd. **The story of the whole project is visible in this one table: smarter model → smaller file, same coder.**

---

## Phase 2 — Neural Predictor (Weeks 2–3)

**Goal:** Replace the n-gram with a small neural model and watch compression beat classical tools. This is the prediction↔compression equivalence made flesh.

### 2.1 Online (adaptive) neural compressor
- Small char/byte-level model: start with a GRU/LSTM (~1–5M params), or a tiny transformer
- **Online mode** (the conceptually clean one): model starts from a fixed random seed, trains *as it compresses*, and the decoder runs the identical training loop to stay in sync. No model needs to be transmitted — both sides derive it from the data stream itself.
- Critical engineering constraint: **determinism**. Fix all seeds, single-threaded, fixed op order, no nondeterministic CUDA kernels (or stay on CPU). One floating-point divergence between encoder and decoder corrupts everything from that point on. Build a `verify_sync.py` test that runs encode and decode and diffs the internal model states every N steps.
- Expect ~1.8–2.2 bpc on enwik-style text — typically beating zstd, approaching strong PPM/CM compressors

### 2.2 Pretrained (shared-model) mode — the "teleport" configuration
- Pretrain the same small model on a held-out corpus; freeze it; both encoder and decoder load identical weights
- Now the wire carries only the arithmetic-coded residual surprise — this is the project thesis in pure form: **the shared model is the "pre-staged matter" at the destination; the wire carries only the selection**
- Measure: bpc on in-domain text vs out-of-domain text (the gap teaches you when shared-model compression wins and when it loses)

### Milestone metric
`results.md` gains: online-neural bpc, pretrained-shared bpc (in/out of domain), with wall-clock honesty (neural compression is *slow* — say so in the table).

---

## Phase 3 — Lossy Image Compression (Weeks 4–6)

**Goal:** Move from lossless text to the rate–distortion world: autoencoder → quantize → entropy-code the latent.

### 3.1 Baseline learned image codec
- Convolutional autoencoder (encoder → latent ~192 channels @ /16 resolution → decoder), trained on CIFAR-10 or a small image set
- Quantization: additive uniform noise during training, hard rounding at test (the standard Ballé-style trick)
- Entropy model over the latent: start with a factorized learned prior; arithmetic-code the quantized latent with it — **reusing the Phase 1 coder**
- Loss: `rate + λ · distortion` (MSE first, then try MS-SSIM); train 3–4 models across λ values to trace an R-D curve

### 3.2 Evaluation
- Plot rate–distortion curve (bpp vs PSNR / MS-SSIM) against JPEG and WebP at matched bitrates
- Side-by-side image grid at equal bpp — *seeing* learned compression beat JPEG at low bitrates is the payoff moment

### Acceptance
- Beat JPEG on MS-SSIM at low bitrates (< 0.3 bpp) on the test set
- Honest writeup of where it loses (high bitrates, out-of-domain images, decode speed)

---

## Phase 4 — Generative / Semantic Extreme (Weeks 7–8, exploratory)

**Goal:** Demonstrate the far end: hundreds of bytes → full image, because the shared model supplies everything generic.

Pick ONE demo (scope control):
- **Option A — Latent diffusion "teleport":** encode an image to a compact latent + seed using a pretrained open model (e.g., a small Stable-Diffusion-class VAE); transmit latent (heavily quantized) + seed; regenerate at the receiver. Measure bytes sent vs perceptual similarity (LPIPS), vs JPEG at the same byte budget.
- **Option B — Semantic text compression:** compress an article to a structured summary + entity list; regenerate with a small LLM at the receiver; measure bytes vs human-judged meaning preservation. (This is lossy-for-text — wide-open research territory.)

Deliverable: a demo script + a short writeup answering *"when is this real compression vs a parlor trick?"* (key question: the shared model is gigabytes — when is it fairly amortized?)

---

## Stretch / Research directions (after Phase 4)
- Bit-identical neural inference across machines (integer-only inference, e.g., quantized int8) — solves the determinism problem properly
- Delta/content-addressed transport layer around the codec (the "move semantics" thread)
- Federated dictionary: which shared contexts (model? zstd dictionary? prior versions?) pay off at which scale

---

## What you need to build this

**Hardware**
- Phases 0–2: any laptop. CPU is fine (and *required* for determinism in 2.1 unless you do careful CUDA work).
- Phase 3: a GPU helps a lot (training autoencoders). A free Colab/Kaggle GPU or a small cloud instance is enough; CIFAR-scale models train in hours.
- Phase 4: pretrained models run on a mid-range GPU (8 GB VRAM) or slowly on CPU.

**Software stack**
- Python 3.11+, NumPy, PyTorch
- `pytest` for the coder tests; `matplotlib` for R-D curves; `Pillow` for images
- Reference baselines installed: gzip, bzip2, zstd, plus JPEG/WebP via Pillow

**Knowledge prerequisites (learn just-in-time, in this order)**
1. Entropy, cross-entropy, and why `bits = -log2 P(symbol)` (Phase 1)
2. Arithmetic coding mechanics — the one genuinely fiddly algorithm in the project (Phase 1)
3. Basic PyTorch + RNN/transformer training loop (Phase 2)
4. Rate–distortion theory at intuition level; the Ballé et al. learned-compression recipe (Phase 3)
5. VAEs / diffusion at user level, LPIPS metric (Phase 4)

**Time:** ~8 weeks part-time. Phase 1 is the foundation — do not rush it; every later phase reuses its coder and its measurement harness.

**Risks and mitigations**
| Risk | Mitigation |
|---|---|
| Arithmetic coder bugs (silent corruption) | Exhaustive round-trip tests before any predictor work; property-based tests |
| Encoder/decoder desync in neural mode | CPU-only, fixed seeds, `verify_sync.py` state-diff harness |
| Neural compression is impractically slow | Report speed honestly; it's a research demo, not a product — say so |
| Phase 3 model won't beat JPEG | Compare at *low* bitrates first (where learned codecs shine); use MS-SSIM not just PSNR |
| Scope creep in Phase 4 | One demo option only; timebox to 2 weeks |
