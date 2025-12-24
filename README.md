# Echo-TTS

> Fork: Echo TTS Streaming API — adds a FastAPI server in `api_server.py` that serves `/v1/audio/speech` with streaming PCM output. It keeps upstream behavior but layers chunked text handling, configurable sampling defaults, and runtime switches via `ECHO_*` env vars.

## Echo TTS Streaming API
- Run: `python api_server.py` (uses `PORT` env var, default `8000`; FastAPI lifespan loads models and optional compile caches).
- Optional dependency: install `ffmpeg` (e.g., `apt-get install ffmpeg` or the official builds on Windows/macOS) if you want non-stream outputs encoded to MP3 by default.
- Endpoint: `POST /v1/audio/speech` with body fields:
  - `input` (text), `voice` (name of a prompt file or folder under `audio_prompts/`; accepts explicit filenames/extensions, base64-encoded audio, or directory names when folder support is on; folders are concatenated—per file with 1s gaps—before encoding), `response_format` (`pcm`, `wav`, or `mp3`), `stream` (bool, default true), `seed`, `extra_body` (sampler overrides such as `block_sizes`, `num_steps`, `chunking_enabled`, `chunk_target_seconds`, etc.).
- Text normalization: prompts are normalized for Echo TTS; `[S1]` is automatically prefixed when missing. Exclamation runs are normalized by default (single `!` -> `.`, multiple `!` -> `!`); toggle via `ECHO_NORMALIZE_EXCLAMATION`.
- Streaming behavior depends on `response_format`:
  - `pcm` (default for streaming): emits raw PCM bytes live as they are generated, with header `X-Audio-Sample-Rate: 44100`. Ideal for real-time playback.
  - `wav` or `mp3`: buffers all audio, then returns the complete file at the end of generation. Still benefits from early stop (reduced compute when the model finishes early), but audio is not delivered until generation completes. Useful when you need a standard audio format but want early stop savings.
- Non-stream returns a single response; default format is MP3 when ffmpeg is available (falls back to WAV otherwise).
- Chunking: enabled by default; long text is split into timed chunks (target 30s, min 20s, max 40s) based on chars/word per second heuristics. Each chunk is synthesized separately and streamed in order; secondary chunks default to the non-stream block shape unless overridden.
- Streaming defaults: `DEFAULT_BLOCK_SIZES = [32, 128, 480]` and `DEFAULT_NUM_STEPS = [8, 15, 20]` are tuned for real-time streaming with low TTFB (~200–300ms on a 3090 when compiled).
- Voices: accepts single prompt files or whole folders (if enabled) under `audio_prompts/`; base64 voices are also supported.
- Voices listing: `GET /v1/voices` returns an OpenAI-style list of voices (`object: voice`, `id`, `name`) sourced from `audio_prompts/` (files and folders with audio when folder support is on).
- Debug: when enabled, last generation is saved to `api_generation.wav`.
- Seed: pass `"seed": <int>` in `extra_body` for reproducible generations; defaults to random if omitted or `-1`.
- Example (streaming PCM, voice `expresso_02_ex03-ex01_calm_005`):
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output out.pcm \
  -d '{
    "input": "[S1] Hello, this is Echo TTS streaming.",
    "voice": "expresso_02_ex03-ex01_calm_005",
    "stream": true,
    "extra_body": {"seed": 42}
  }'
```
- Example (streaming PCM with stronger speaker forcing via `speaker_kv_scale`):
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output out.pcm \
  -d '{
    "input": "[S1] Please stick to the reference speaker.",
    "voice": "expresso_02_ex03-ex01_calm_005",
    "stream": true,
    "extra_body": {
      "speaker_kv_scale": 1.25,
      "speaker_kv_min_t": 0.9,
      "speaker_kv_max_layers": 24
    }
  }'
```
- Example (non-stream, WAV response):
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output out.wav \
  -d '{
    "input": "[S1] Hello, this is a WAV response example.",
    "voice": "expresso_02_ex03-ex01_calm_005",
    "stream": false,
    "response_format": "wav"
  }'
```

### Server Environment Flags
- `ECHO_MODEL_REPO` (default `jordand/echo-tts-base`) selects the main model; `ECHO_FISH_REPO` (default `jordand/fish-s1-dac-min`) selects the decoder.
- `ECHO_DEVICE` / `ECHO_FISH_DEVICE` (default `cuda`) pick devices; set to `cpu` to avoid GPU requirements. `ECHO_MODEL_DTYPE` (default `bfloat16`) and `ECHO_FISH_DTYPE` (default `float32`) control dtypes.
- `ECHO_COMPILE` (default `0`) toggles `torch.compile` for the main model; `ECHO_COMPILE_AE` (default `1`) separately compiles the decoder; `ECHO_COMPILE_LORA_ONLY` is ignored when LoRA is unused.
- Cache/logging: `ECHO_CACHE_DIR` (default `/tmp`) and `ECHO_CACHE_VERSION` label saved compile artifacts; `ECHO_CACHE_SPEAKER_ON_GPU` (default `0`) caches speaker latents per device; `ECHO_DEBUG_LOGS` (default `0`) enables verbose timing/debug prints.
- Chunking/text defaults: `ECHO_CHUNKING` (default `1`), `ECHO_CHUNK_CHARS_PER_SECOND` (default `14`), `ECHO_CHUNK_WORDS_PER_SECOND` (default `2.7`), `ECHO_NORMALIZE_EXCLAMATION` (default `1`) normalizes `!` (single -> `.`, multiple -> `!`).
- Reference audio handling: `ECHO_MAX_SPEAKER_LATENT_LENGTH` (default `6400`), `ECHO_FOLDER_SUPPORT` (default `1` to allow folder prompts), `ECHO_WARMUP_VOICE` and `ECHO_WARMUP_TEXT` seed optional compile warmup.
- Optional dependency: ffmpeg (on PATH) is required for `response_format='mp3'`; when present, non-stream defaults to MP3, otherwise WAV.
- Performance presets (streaming only): `ECHO_PERFORMANCE_PRESET` (default `default`) sets streaming sampler defaults: `default` uses `block_sizes=[32, 128, 480]` / `num_steps=[8, 15, 20]`; `low_mid` keeps those blocks with `num_steps=[8, 10, 15]`; `low` uses `block_sizes=[32, 64, 272, 272]` and `num_steps=[8, 10, 15, 15]`; `equal` uses `block_sizes=[213, 213, 214]` and `num_steps=[15, 15, 15]` for three ~10s blocks with uniform quality. Unknown values fall back to default with a warning; non-streaming uses its own steps.
- Non-streaming steps: `ECHO_NUM_STEPS_NONSTREAM` (default `20`) controls the fixed non-stream sampler steps (recommended range 10–40); block size stays `640` by default unless overridden via request.
- VAD-based reroll: `ECHO_VAD_REROLL_ENABLED` (default `0`) enables silence detection using Silero VAD; when a generated block contains silence >= `ECHO_VAD_SILENCE_THRESHOLD_MS` (default `1000`ms), it is regenerated with a new seed up to `ECHO_VAD_MAX_REROLLS` (default `3`) times. This helps mitigate the model's tendency to produce unnaturally long pauses (see Model Quirks below).
- Note: enabling `torch.compile` (model and/or decoder) can increase peak VRAM; disable `ECHO_COMPILE`/`ECHO_COMPILE_AE` if memory is tight.

### Performance / VRAM notes
- Quick presets (streaming): set `ECHO_PERFORMANCE_PRESET=low_mid` to reduce steps or `ECHO_PERFORMANCE_PRESET=low` to also shrink blocks; both lower compute/VRAM at some quality cost. Non-streaming always defaults to 20 steps unless you set `ECHO_NUM_STEPS_NONSTREAM` (10–40 recommended).
- Lower-end GPUs: prefer `ECHO_PERFORMANCE_PRESET=low_mid` (fewer streaming steps) or `ECHO_PERFORMANCE_PRESET=low` (smaller blocks + fewer steps) instead of manual step tweaks.
- Uniform streaming quality: `ECHO_PERFORMANCE_PRESET=equal` splits generation into three ~10s blocks with 15 steps each, trading faster TTFB for consistent quality across all chunks (no small fast first block).
- Compile vs presets: with `ECHO_COMPILE=1` you may be able to keep the higher (default) preset while staying real-time, but it raises peak VRAM; if memory is tight, turn compile off before lowering presets.
- VRAM reduction: set `ECHO_FISH_DTYPE=bfloat16` (or `bf16`) to run the decoder in bf16 at a small quality cost.
- Disable compile to save memory: set `ECHO_COMPILE=0` (model) and `ECHO_COMPILE_AE=0` (Fish AE, which defaults to compiled) if VRAM is constrained; expect slower generations.

### Model Quirks
- **Unnaturally long pauses**: The model sometimes inserts long silences (1+ seconds) in the middle of speech, especially with certain text patterns or speaker references. The VAD-based reroll feature (`ECHO_VAD_REROLL_ENABLED=1`) can automatically detect and regenerate blocks with excessive silence.
- **Tail artifacts on short generations**: Very short text inputs (e.g., "Hello") may produce audio with artifacts or noise at the end. This is more pronounced with streaming mode's smaller initial blocks. Future updates may address this with improved tail detection and trimming.

### Skyrim/Fallout Voice AI Integration
Recommended settings for game voice AI applications (e.g., Mantella, xVASynth replacement):
- **Server environment**:
  ```bash
  ECHO_PERFORMANCE_PRESET=equal ECHO_VAD_REROLL_ENABLED=1 ECHO_COMPILE_AE=1 python api_server.py
  ```
- **Request settings**:
  ```json
  {
    "input": "[S1] Your dialogue text here.",
    "voice": "your_character_voice",
    "stream": true,
    "response_format": "wav"
  }
  ```
- **Why these settings**:
  - `ECHO_PERFORMANCE_PRESET=equal`: Uses three uniform ~10s blocks with 15 steps each. This enables early stop to save compute efficiently—short text stops after 1 block (1/3 compute), medium text after 2 blocks (2/3 compute), long text uses all 3. Also ensures consistent quality across the generation.
  - `ECHO_VAD_REROLL_ENABLED=1`: Automatically regenerates blocks with excessive silence, reducing the chance of awkward pauses in dialogue.
  - `stream: true` + `response_format: wav`: Benefits from early stop (model stops generating when speech ends, saving compute) while returning a standard WAV file compatible with game engines.

#### SkyrimNet Integration
For [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin) users, Echo-TTS provides Inworld TTS API compatibility:

1. **Start the server with voice cloning enabled**:
   ```bash
   ECHO_PERFORMANCE_PRESET=equal ECHO_VAD_REROLL_ENABLED=1 ECHO_COMPILE_AE=1 ECHO_INWORLD_CLONE_ENABLED=1 python api_server.py
   ```

2. **Configure SkyrimNet**:
   - In SkyrimNet settings, set the TTS API URL to your Echo-TTS server: `http://<your-ip>:8000`
   - Select "Inworld TTS" as the TTS provider

3. **Voice cloning**: SkyrimNet can clone character voices via the `/voices/v1/workspaces/{workspace}/voices:clone` endpoint. Cloned voices are saved with the format `{workspace}__{voice}` (e.g., `default__Lydia.wav`).

### Inworld TTS API Compatibility

Echo-TTS includes Inworld TTS API-compatible endpoints for drop-in replacement with tools like Mantella. These endpoints are **enabled by default**.

#### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /tts/v1/voices` | List all available voices |
| `POST /tts/v1/voice` | Synthesize speech (non-streaming, returns complete audio) |
| `POST /tts/v1/voice:stream` | Synthesize speech (streaming JSON chunks with base64 audio) |
| `GET /voices/v1/workspaces/{workspace}/voices/{voice}` | Get voice metadata |
| `POST /voices/v1/workspaces/{workspace}/voices:clone` | Clone a voice from audio sample (disabled by default) |
| `DELETE /voices/v1/workspaces/{workspace}/voices/{voice}` | Delete a cloned voice (disabled by default) |

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ECHO_INWORLD_COMPAT` | `1` | Enable Inworld-compatible endpoints |
| `ECHO_INWORLD_CLONE_ENABLED` | `0` | Enable voice cloning/deletion (security-sensitive) |
| `ECHO_INWORLD_MAX_SAMPLE_SIZE` | `104857600` | Max voice sample size in bytes (100 MB) |

#### Example: List Voices
```bash
curl http://localhost:8000/tts/v1/voices
```

Response:
```json
{
  "voices": [
    {"languages": ["en"], "voiceId": "expresso_02_ex03-ex01_calm_005", "displayName": "expresso_02_ex03-ex01_calm_005", "description": "Built-in voice", "tags": ["built-in"]},
    {"languages": ["en"], "voiceId": "default__John", "displayName": "John", "description": "Cloned voice", "tags": ["cloned"]}
  ]
}
```

#### Example: Synthesize Speech
```bash
curl -X POST http://localhost:8000/tts/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voiceId": "expresso_02_ex03-ex01_calm_005",
    "modelId": "inworld-tts-1",
    "audioConfig": {"audioEncoding": "LINEAR16"}
  }'
```

Response:
```json
{
  "audioContent": "<base64-encoded WAV>",
  "timestampInfo": {"wordAlignment": {...}, "characterAlignment": {...}}
}
```

#### Example: Get Voice Metadata
```bash
curl http://localhost:8000/voices/v1/workspaces/default/voices/expresso_02_ex03-ex01_calm_005
```

Response:
```json
{
  "name": "workspaces/default/voices/expresso_02_ex03-ex01_calm_005",
  "voiceId": "expresso_02_ex03-ex01_calm_005",
  "displayName": "expresso_02_ex03-ex01_calm_005",
  "langCode": "EN_US",
  "description": "Built-in voice: expresso_02_ex03-ex01_calm_005",
  "tags": ["built-in"]
}
```

#### Audio Format Support
- `LINEAR16` → WAV (16-bit PCM with header)
- `MP3` → MP3 (requires ffmpeg)
- Other formats return 400 error

#### Voice Cloning Security
Voice cloning is **disabled by default** (`ECHO_INWORLD_CLONE_ENABLED=0`). When enabled:
- Voice names are sanitized (alphanumeric, underscore, hyphen, space only)
- Only WAV and MP3 uploads are accepted (validated by file header)
- Maximum upload size: 100 MB
- Cloned voices use Inworld format: `{workspace}__{voice}` (e.g., `default__John`)
- Only voices with this format can be deleted through the API
- Original/manually-added voices cannot be deleted through the API

# Original README

A multi-speaker text-to-speech model with speaker reference conditioning. See the [blog post](https://jordandarefsky.com/blog/2025/echo/) for technical details.

**Model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) | **Demo:** [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)

## Responsible Use

Don't use this model to:
- Impersonate real people without their consent
- Generate deceptive audio (e.g., fraud, misinformation, deepfakes)

You are responsible for complying with local laws regarding biometric data and voice cloning.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU with at least 8GB VRAM.

## Quick Start

### Gradio UI

```bash
python gradio_app.py
```

### Python API

```python
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)
from functools import partial
import torchaudio

# Load models (downloads from HuggingFace on first run)
model = load_model_from_hf(delete_blockwise_modules=True)
fish_ae = load_fish_ae_from_hf()
pca_state = load_pca_state_from_hf()

# Load speaker reference (or set to None for no reference)
speaker_audio = load_audio("speaker.wav").cuda()

# Configure sampler
sample_fn = partial(
    sample_euler_cfg_independent_guidances,
    num_steps=40,
    cfg_scale_text=3.0,
    cfg_scale_speaker=8.0,
    cfg_min_t=0.5,
    cfg_max_t=1.0,
    truncation_factor=None,
    rescale_k=None,
    rescale_sigma=None,
    speaker_kv_scale=None,
    speaker_kv_max_layers=None,
    speaker_kv_min_t=None,
    sequence_length=640, # (~30 seconds)
)

# Generate
text = "[S1] Hello, this is a test of the Echo TTS model."
audio_out, _ = sample_pipeline(
    model=model,
    fish_ae=fish_ae,
    pca_state=pca_state,
    sample_fn=sample_fn,
    text_prompt=text,
    speaker_audio=speaker_audio,
    rng_seed=0,
)

torchaudio.save("output.wav", audio_out[0].cpu(), 44100)
```

See also:
- `inference.py` -- lower-level usage example at the bottom of the file
- `inference_blockwise.py` -- examples of blockwise/continuation generation

## Low VRAM (8GB)

In `gradio_app.py`, adjust:

```python
FISH_AE_DTYPE = torch.bfloat16  # instead of float32
DEFAULT_SAMPLE_LATENT_LENGTH = 576  # (< 640 depending on what fits) instead of 640
```

## Tips

### Generation Length

Echo is trained to generate up to 30 seconds of audio (640 latents) given text and reference audio. Since the supplied text always corresponded to ≤30 seconds of audio during training, the model will attempt to fit any text prompt at inference into the 30 seconds of generated audio (and thus, e.g., long text prompts may result in faster speaking rates). On the other hand, shorter text prompts will work and will produce shorter outputs (as the model generates latent padding automatically).

If "Sample Latent Length" (in Custom Shapes in gradio)/sequence_length is set to less than 640, the model will attempt to generate the prefix corresponding to that length. I.e., if you set this to 320, and supply ~30 seconds worth of text, the model will likely generate the first half of the text (rather than try to fit the entirety of the text into the first 15 seconds).

### Reference Audio

You can condition on up to 5 minutes of reference audio, but shorter clips (e.g., 10 seconds or shorter) work well too.

### Force Speaker (KV Scaling)

Sometimes out-of-distribution text for a given reference speaker will cause the model to generate a different speaker entirely. Enabling "Force Speaker" (which scales speaker KV for a portion of timesteps, default scale 1.5) generally fixes this. However, high values may introduce artifacts or "overconditioning." Aim for the lowest scale that produces the correct speaker: 1.0 is baseline, 1.5 is the default when enabled and will usually force the speaker, but lower values (e.g., 1.3, 1.1) may suffice.

### Text Prompt Format

Text prompts use the format from [WhisperD](https://huggingface.co/jordand/whisper-d-v1a). Colons, semicolons, and emdashes are normalized to commas (see inference.py tokenizer_encode) by default, and "[S1] " will be added to the beginning of the prompt if not already present. Commas generally function as pauses. Exclamation points (and other non-bland punctuation) may lead to increased expressiveness but also potentially lower quality on occasion; improving controllability is an important direction for future work.

The included text presets are stylistically in-distribution with the WhisperD transcription style.

### Blockwise Generation

`inference_blockwise.py` includes blockwise sampling, which allows generating audio in smaller blocks as well as producing continuations of existing audio (where the prefix and continuation are up to 30 seconds combined). The model released on HF is a fully fine-tuned model (not the LoRA as described in the blog). Blockwise generation enables audio streaming (not included in current code) since the S1-DAC decoder is causal. Blockwise functionality hasn't been thoroughly tested and may benefit from different (e.g., smaller) CFG scales.

## License

Code in this repo is MIT‑licensed except where file headers specify otherwise (e.g., autoencoder.py is Apache‑2.0).

Regardless of our model license, audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the Fish Speech S1-DAC autoencoder, which is CC-BY-NC-SA-4.0.

We have chosen to release the Echo-TTS weights under CC-BY-NC-SA-4.0.

For included audio prompts, see `audio_prompts/LICENSE`.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```
