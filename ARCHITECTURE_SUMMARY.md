# Echo TTS API - Architecture & Implementation Summary

## 1. Model Architecture

### Core Model: EchoDiT (Diffusion Transformer)

**EchoDiT** is a **latent diffusion model** that generates audio by denoising a sequence of latent tokens over multiple diffusion steps.

#### Architecture Components:

1. **Text Encoder** (`TextEncoder`)
   - Encodes text tokens → text embeddings
   - 14-layer bidirectional transformer
   - Output: `(batch, seq_len, 1280)` text states
   - Used for text conditioning via KV cache

2. **Speaker Encoder** (`SpeakerEncoder`)
   - Encodes speaker audio → speaker embeddings
   - 14-layer causal transformer
   - Processes speaker latents in patches (patch_size=4)
   - Output: `(batch, speaker_len/4, 1280)` speaker states
   - Used for voice cloning/conditioning

3. **Latent Encoder** (`LatentEncoder`)
   - Encodes already-generated prefix latents → memory embeddings
   - Same architecture as SpeakerEncoder
   - Provides "memory" of previously generated tokens for autoregressive-style context
   - Output: `(batch, prefix_len/4, 1280)` latent states

4. **Main Transformer Blocks** (24 layers)
   - Each block uses **JointAttention** that attends over:
     - **Self-attention**: current latent tokens being generated
     - **Latent KV cache**: prefix memory (already generated tokens)
     - **Text KV cache**: text conditioning
     - **Speaker KV cache**: speaker conditioning
   - **AdaLN conditioning**: timestep embedding modulates attention/MLP via LowRankAdaLN
   - **MLP**: standard feedforward network

5. **Output Projection**
   - Projects transformer output back to latent space: `(batch, seq_len, 80)`
   - 80-dimensional latent space (matches Fish DAC autoencoder)

#### Key Design Choices:

- **Multi-modal conditioning**: Text + Speaker + Latent memory all attend jointly
- **Causal + non-causal mixing**: Self-attention is causal, but can attend to non-causal text/speaker caches
- **Patch-based processing**: Speaker/latent sequences processed in patches (4 tokens → 1 patch) for efficiency
- **KV caching**: Text/speaker/latent embeddings pre-computed and cached to avoid re-encoding

### Autoencoder: Fish DAC

- **Encoder**: Audio → latent codes (80-dim, downsampled by 2048x)
- **Decoder**: Latent codes → audio (44.1kHz PCM16)
- **PCA projection**: Applied to latents for dimensionality reduction
- **Streaming decoder**: Maintains bounded context window for block-wise decoding

---

## 2. Inference Flow for a Single Request

### Phase 1: Initialization (Pre-compute Conditioning)

1. **Text Processing**
   - Preprocess text (normalize, add speaker tags)
   - Tokenize to byte-level tokens
   - Pad/truncate to fixed 768 tokens
   - Encode via `TextEncoder` → text states
   - Compute text KV cache: `get_kv_cache_text()` → `List[(K, V)]` per layer

2. **Speaker Processing**
   - Load speaker audio file (or use base64)
   - Encode via Fish DAC encoder → speaker latent `(1, L, 80)`
   - Pad to patch_size multiple (4)
   - Encode via `SpeakerEncoder` → speaker states
   - Compute speaker KV cache: `get_kv_cache_speaker()` → `List[(K, V)]` per layer

3. **Unconditional Variants** (for CFG)
   - Text uncond: BOS token only
   - Speaker uncond: zeros
   - Pre-compute uncond KV caches

### Phase 2: Block-wise Diffusion Generation

The model generates audio in **blocks** (e.g., `[32, 128, 480]` tokens) with different step counts per block (e.g., `[8, 15, 20]` steps).

For each block:

1. **Initialize Latent KV Cache**
   - Encode current `prefix_latent` (all previously generated tokens) via `LatentEncoder`
   - Compute latent KV cache: `get_kv_cache_latent()` → `List[(K, V)]` per layer

2. **Diffusion Loop** (Euler-style denoising)
   - Initialize: `x_t = randn(block_size, 80) * truncation_factor`
   - For each step `i` in `num_steps`:
     - Current timestep: `t = t_schedule[i]`
     - **CFG Decision**: If `cfg_min_t <= t <= cfg_max_t`:
       - Run model **3x** (cond, uncond-text, uncond-speaker)
       - Combine: `v_pred = v_cond + cfg_text*(v_cond - v_uncond_text) + cfg_speaker*(v_cond - v_uncond_speaker)`
     - Else: Run model **1x** (conditioned only)
     - Update: `x_t = x_t + v_pred * (t_next - t)`

3. **Block Completion**
   - Commit `x_t` to `prefix_latent[pos_id:pos_id+block_size]`
   - **Early Stop Check**: If `early_stop_on_zero=True`:
     - Check if tail of `x_t` is near-zero (fraction of values < `zero_eps`)
     - If yes, mark `early_stop = True`
   - **Flattening Point Detection** (if finishing):
     - Scan `prefix_latent` backward to find start of stable near-zero tail
     - Uses sliding window heuristic: `std < 0.02`, `mean_abs < 0.02`, `max_abs < 0.05`
   - **Streaming Decode**:
     - Decode `x_t` via `StreamingAEDecoder.decode_next()`
     - If `flatten_point` set, trim audio to that point
     - Yield PCM16 bytes

4. **Advance to Next Block**
   - `pos_id += block_size`
   - Repeat until all blocks done OR early stop triggered

### Phase 3: Output

- PCM16 audio chunks streamed to client
- Final audio trimmed at flattening point to remove trailing noise

---

## 3. Changes Made to Support Batching

### New Component: `BatchScheduler` (`new/batching.py`)

**Goal**: Allow multiple concurrent websocket requests to share GPU work by batching diffusion steps.

#### Key Design Decisions:

1. **Micro-batching at Step Level**
   - Groups requests by identical `(block_idx, step_idx, block_size, num_steps, cfg_active)`
   - Runs **one model forward** for the entire batch
   - Scatters results back to individual requests

2. **Speaker Latent Bucketing**
   - Pads speaker latents to nearest bucket from `ECHO_SPEAKER_LATENT_BUCKETS`
   - Ensures stable shapes for `torch.compile` and efficient batching
   - Example: 164-length latent → padded to 256 bucket

3. **Per-Request State Management**
   - Each request has `_ReqState` tracking:
     - Diffusion state: `block_idx`, `step_idx`, `x_t`, `prefix_latent`, `pos_id`
     - Conditioning: pre-computed KV caches (text, speaker, latent)
     - Output: `asyncio.Queue[bytes]` for PCM chunks
   - State initialized once per request, then updated per step

4. **Scheduler Loop** (`_run_loop()`)
   - Continuously:
     - Pulls new requests from `_in_q`
     - Groups active requests by batch key
     - Runs batched diffusion step (`_run_one_step()`)
     - Finishes blocks (`_finish_block()`) when `step_idx >= block_steps`
     - Enqueues PCM chunks to per-request queues

5. **Early Stop & Flattening Point** (now in batching path)
   - **Early stop detection**: Same logic as engine (check tail for near-zero)
   - **Flattening point**: Same heuristic as engine (backward scan for stable tail)
   - Applied per-request when finishing a block

6. **Torch Compile Warmup for Batched Shapes**
   - Added `BatchScheduler.warmup()` that compiles common batched shapes (B=2,4,8)
   - Called during server startup to avoid first-request compilation stall
   - Uses same speaker buckets as runtime

#### Integration Points:

- **WebSocket endpoint** (`server.py`):
  - WS handler submits request to scheduler via `scheduler.submit()`
  - Reads PCM chunks from returned queue
  - No longer blocks event loop (GPU work happens in scheduler background task)

- **Server initialization** (`server.py`):
  - Creates `BatchScheduler` instance
  - Starts scheduler task in lifespan startup
  - Runs batch warmup after model load

---

## 4. Torch Compile Usage

### Compilation Strategy:

1. **Model Compilation**
   - Main model: `torch.compile(EchoDiT)`
   - KV cache methods: `torch.compile(get_kv_cache_text)`, etc.
   - Fish AE: selective compilation of quantizer ops

2. **Compile Cache Persistence**
   - **Save**: After first generation with new `block_sizes`, save artifacts to disk
   - **Load**: On startup, load cached artifacts for known `block_sizes`
   - Cache files: `/tmp/echo_tts_compile_cache_{block_sizes}_{version}.gz`
   - Saves ~70-90MB per config (speeds up subsequent startups)

3. **Shape Stability**
   - **Text**: Fixed to 768 tokens (pad/truncate)
   - **Speaker**: Bucketed to stable lengths (128, 256, 512, 1024, 2048)
   - **Block sizes**: Fixed per config (e.g., `[32, 128, 480]`)
   - Critical for compile cache hits

4. **Warmup Strategy**
   - **Single-request warmup**: Compiles B=1 paths for all speaker buckets
   - **Batch warmup**: Compiles B=2,4,8 paths for common buckets
   - Runs during server startup (before accepting requests)

5. **Compile Disable Fallback**
   - If compilation fails, `disable_compile()` called
   - Falls back to eager execution (slower but functional)

### Performance Impact:

- **First request**: ~200-300ms TTFB (with warmup)
- **Subsequent requests**: ~200-250ms TTFB (cache hits)
- **Compile time**: ~30-60s per new shape (one-time cost, cached)

---

## 5. Scope for Improvement

### A. Batching & Concurrency

1. **Mixed Config Batching**
   - **Current**: Only batches requests with identical sampler configs
   - **Improvement**: Support batching across different `block_sizes`/`num_steps` (requires padding/grouping logic)

2. **Dynamic Batch Sizing**
   - **Current**: Fixed `max_batch_size=8`
   - **Improvement**: Adaptive sizing based on GPU memory/utilization

3. **Request Cancellation**
   - **Current**: Best-effort (queue drops when full)
   - **Improvement**: Proper cancellation propagation to scheduler, cleanup of in-flight state

4. **Multi-GPU Support**
   - **Current**: Single GPU per process
   - **Improvement**: Model parallelism or multi-instance routing

### B. Latency Optimization

1. **KV Cache Pre-computation**
   - **Current**: Text/speaker KV caches computed on first use
   - **Improvement**: Pre-compute common voices at startup

2. **Streaming Decode Optimization**
   - **Current**: Decodes full block each time
   - **Improvement**: Incremental decode (only new tokens)

3. **CFG Optimization**
   - **Current**: Always runs 3x forward when CFG active
   - **Improvement**: Skip CFG for low-timestep steps (already done for `t < cfg_min_t`)

### C. Quality & Robustness

1. **Flattening Point Tuning**
   - **Current**: Fixed heuristic (`window_size=32`, `std_threshold=0.02`)
   - **Improvement**: Adaptive thresholds, or ML-based detection

2. **Early Stop Sensitivity**
   - **Current**: Fixed `zero_eps=0.02`, `zero_tail_min_frac=0.95`
   - **Improvement**: Per-voice or per-text adaptive thresholds

3. **Audio Quality Validation**
   - **Current**: No automatic quality checks
   - **Improvement**: Add metrics (SNR, silence detection) to catch degradation

### D. Infrastructure

1. **Metrics & Observability**
   - **Current**: Basic debug logs
   - **Improvement**: Prometheus metrics (TTFB, throughput, batch utilization, GPU memory)

2. **Error Handling**
   - **Current**: Exceptions bubble up to WS handler
   - **Improvement**: Graceful degradation, retry logic, error recovery

3. **Configuration Management**
   - **Current**: Environment variables
   - **Improvement**: Config file support, hot-reload for non-model settings

### E. Model Optimizations

1. **Attention Optimization**
   - **Current**: Standard scaled dot-product attention
   - **Improvement**: Flash Attention 2, or custom kernels for joint attention

2. **Quantization**
   - **Current**: bfloat16 for model, float32 for AE
   - **Improvement**: INT8 quantization for model (with calibration)

3. **Speculative Decoding**
   - **Current**: Sequential block generation
   - **Improvement**: Parallel generation of multiple blocks (with validation)

### F. Production Readiness

1. **Load Balancing**
   - **Current**: Single instance
   - **Improvement**: Multi-instance with sticky routing (by voice) for cache efficiency

2. **Graceful Shutdown**
   - **Current**: Basic cleanup
   - **Improvement**: Drain in-flight requests, save compile caches

3. **Health Checks**
   - **Current**: Basic `/health` endpoint
   - **Improvement**: Detailed health (GPU memory, queue depth, model loaded)

4. **Rate Limiting**
   - **Current**: None
   - **Improvement**: Per-client rate limits, queue depth limits

---

## Summary

**Current State**: MVP batching implementation that successfully batches concurrent websocket requests, achieving ~488ms TTFB for 4 concurrent requests (vs ~220ms for single request). Quality matches single-request path after adding early-stop and flattening point logic.

**Key Achievements**:
- ✅ Micro-batching at diffusion step level
- ✅ Speaker latent bucketing for shape stability
- ✅ Torch compile cache persistence
- ✅ Early-stop and flattening point in batching path
- ✅ Non-blocking websocket handlers

**Next Steps**: Focus on production hardening (metrics, error handling, cancellation) and latency optimization (KV cache pre-computation, streaming decode improvements).

