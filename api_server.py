import base64
import gzip
import math
import os
import tempfile
import io
import wave
import shutil
import subprocess
from dataclasses import dataclass, replace
from contextlib import asynccontextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import time
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.concurrency import iterate_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import torchaudio
from utils import preprocess_text, chunk_text_by_time
from inference import (
    PCAState,
    ae_decode as _base_ae_decode,
    find_flattening_point,
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    load_audio,
    load_fish_ae_from_hf,
    load_model_from_hf,
    load_pca_state_from_hf,
)
from autoencoder import DAC
from samplers import (
    GuidanceMode,
    sample_euler_cfg_any,
    _get_first_n_kv_cache,
    _get_uncond_text_input_ids_and_mask,
    _multiply_speaker_kv_cache,
    _temporal_score_rescale,
)

SAMPLE_RATE = 44_100

DEFAULT_BLOCK_SIZES = [32, 128, 480]
DEFAULT_NUM_STEPS = [8, 15, 20]
DEFAULT_CFG_TEXT = 3.0
DEFAULT_CFG_SPEAKER = 8.0
DEFAULT_CFG_MIN_T = 0.5
DEFAULT_CFG_MAX_T = 1.0
DEFAULT_EARLY_STOP = True
DEFAULT_ZERO_EPS = 2.0e-2  # even looser to trigger early_stop on low-energy tails
DEFAULT_ZERO_TAIL_FRAMES = 16
DEFAULT_ZERO_TAIL_MIN_FRAC = 0.95
DEFAULT_BLOCK_SIZE_NONSTREAM = 640
DEFAULT_NUM_STEPS_NONSTREAM = int(os.getenv("ECHO_NUM_STEPS_NONSTREAM", "20"))
DEBUG_LOGS_ENABLED = os.getenv("ECHO_DEBUG_LOGS", "0") == "1"
FFMPEG_PATH = shutil.which("ffmpeg")

MODEL_REPO = os.getenv("ECHO_MODEL_REPO", "jordand/echo-tts-base") # Model repo overriden when using LoRA
PCA_REPO = os.getenv("ECHO_PCA_REPO", MODEL_REPO)
FISH_REPO = os.getenv("ECHO_FISH_REPO", "jordand/fish-s1-dac-min")
DEVICE = os.getenv("ECHO_DEVICE", "cuda")
FISH_DEVICE = os.getenv("ECHO_FISH_DEVICE", DEVICE)
MODEL_DTYPE = os.getenv("ECHO_MODEL_DTYPE", "bfloat16")  # keep half-precision by default to avoid doubling VRAM
FISH_DTYPE = os.getenv("ECHO_FISH_DTYPE", "float32")     # keep decoder in fp32 by default for quality
USE_COMPILE = os.getenv("ECHO_COMPILE", "1") == "1" # Takes several minutes to compile but cuts TTFB by 100~200ms
COMPILE_AE = os.getenv("ECHO_COMPILE_AE", "1") == "1"
CACHE_SPEAKER_ON_GPU = os.getenv("ECHO_CACHE_SPEAKER_ON_GPU", "0") == "1" # Provides speed-up by 20ms~60ms TTFB per request at cost of VRAM usage
CACHE_VERSION = os.getenv("ECHO_CACHE_VERSION", "v1_0")
CACHE_DIR = Path(os.getenv("ECHO_CACHE_DIR", "/tmp"))
WARMUP_VOICE = os.getenv("ECHO_WARMUP_VOICE")
WARMUP_TEXT = os.getenv("ECHO_WARMUP_TEXT", "[S1] Warmup compile run.")
# Chunking config (wiring TBD; previewed in scripts/chunk_preview.py)
CHUNKING_ENABLED = os.getenv("ECHO_CHUNKING", "1") == "1"
CHUNK_CHARS_PER_SECOND = float(os.getenv("ECHO_CHUNK_CHARS_PER_SECOND", "14"))
CHUNK_WORDS_PER_SECOND = float(os.getenv("ECHO_CHUNK_WORDS_PER_SECOND", "2.7"))
NORMALIZE_EXCLAMATION = os.getenv("ECHO_NORMALIZE_EXCLAMATION", "1") == "1"
MAX_SPEAKER_LATENT_LENGTH = int(os.getenv("ECHO_MAX_SPEAKER_LATENT_LENGTH", "6400"))
FOLDER_SUPPORT = os.getenv("ECHO_FOLDER_SUPPORT", "1") == "1"
# Performance presets
_PERFORMANCE_PRESET_RAW = os.getenv("ECHO_PERFORMANCE_PRESET", "default")
PERFORMANCE_PRESET = _PERFORMANCE_PRESET_RAW.strip().lower().replace("-", "_")
_PERFORMANCE_PRESETS = {
    "default": {"block_sizes": [32, 128, 480], "num_steps": [8, 15, 20]},
    "low_mid": {"block_sizes": [32, 128, 480], "num_steps": [8, 10, 15]},
    "low": {"block_sizes": [32, 64, 272, 272], "num_steps": [8, 10, 15, 15]},
}
if PERFORMANCE_PRESET in _PERFORMANCE_PRESETS:
    preset = _PERFORMANCE_PRESETS[PERFORMANCE_PRESET]
    DEFAULT_BLOCK_SIZES = list(preset["block_sizes"])
    DEFAULT_NUM_STEPS = list(preset["num_steps"])
elif PERFORMANCE_PRESET not in {"", "default"}:
    print(
        f"⚠️ Unknown ECHO_PERFORMANCE_PRESET '{_PERFORMANCE_PRESET_RAW}'; using default sampler settings."
    )
# LoRA settings
LORA_FIRST_BLOCK = os.getenv("ECHO_LORA_FIRST_BLOCK", "0") == "1"
LORA_REPO = os.getenv("ECHO_LORA_REPO", "")
LORA_HF_NAME = os.getenv("ECHO_LORA_HF_NAME", "lora_lr1e-5_skip02_noema_huber0005_cfgdecay_90ksteps.safetensors")
LORA_SCALE = float(os.getenv("ECHO_LORA_SCALE", "1.0"))
LORA_ALPHA = float(os.getenv("ECHO_LORA_ALPHA", "32.0"))
COMPILE_LORA_ONLY = os.getenv("ECHO_COMPILE_LORA_ONLY", "0") == "1"
if LORA_FIRST_BLOCK:
    DEFAULT_NUM_STEPS[0] = 5 # 5 steps for first block when using LoRA
    MODEL_REPO = LORA_REPO

# Keep torch.compile caches on restarts similar to test_ttfb_optimization.py
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _run_startup_tasks()
    yield


VOICE_DIRS = [
    Path(__file__).resolve().parent / "audio_prompts",
    Path(__file__).resolve().parent / "prompt_audio",
    Path(__file__).resolve().parent / "extra_prompt_audio",
]
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

app = FastAPI(title="Echo-TTS API (streaming)", lifespan=lifespan)

_MODEL = None
_MODEL_LORA = None
_FISH_AE = None
_PCA_STATE = None
_SPEAKER_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
_SPEAKER_CACHE_GPU: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
_LOADED_CACHE_PATHS: set[Path] = set()
_SAVED_CACHE_PATHS: set[Path] = set()
_WARMUP_RAN = False
_COMPILE_DISABLED = False


def _log_debug(msg: str) -> None:
    if DEBUG_LOGS_ENABLED:
        print(msg, flush=True)


def _device_key(device: torch.device) -> str:
    """Stable per-device key (cuda->cuda:0) to avoid mismatch between 'cuda' and 'cuda:0'."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        return f"cuda:{idx}"
    return str(device)


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    name = (name or "").lower()
    mapping: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name in {"none", ""}:
        return None
    if name not in mapping:
        raise ValueError(
            f"Invalid dtype '{name}'. Choose from: {', '.join(sorted(mapping))} or 'none'."
        )
    return mapping[name]


def _cache_file_path(block_sizes: List[int]) -> Path:
    block_sizes_str = "_".join(map(str, block_sizes))
    return CACHE_DIR / f"echo_tts_compile_cache_{block_sizes_str}_{CACHE_VERSION}.gz"


def _load_compile_cache(block_sizes: List[int]) -> bool:
    if not USE_COMPILE or _COMPILE_DISABLED:
        return False
    path = _cache_file_path(block_sizes)
    if path in _LOADED_CACHE_PATHS or not path.exists():
        return False
    try:
        with gzip.open(path, "rb") as f:
            artifact_bytes = f.read()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        _LOADED_CACHE_PATHS.add(path)
        print(f"✅ Loaded torch.compile cache from {path}")
        return True
    except Exception as exc:
        print(f"⚠️ Could not load compile cache {path}: {exc}")
        return False


def _save_compile_cache(block_sizes: List[int]) -> bool:
    if not USE_COMPILE or _COMPILE_DISABLED:
        return False
    path = _cache_file_path(block_sizes)
    if path in _SAVED_CACHE_PATHS:
        return False
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            print("⚠️ No compile artifacts to save yet")
            return False
        artifact_bytes, _ = artifacts
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            f.write(artifact_bytes)
        _SAVED_CACHE_PATHS.add(path)
        _LOADED_CACHE_PATHS.add(path)
        size_mb = len(artifact_bytes) / 1024 / 1024
        print(f"✅ Saved torch.compile cache to {path} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"⚠️ Could not save compile cache {path}: {exc}")
        return False


def _disable_compile(reason: str) -> None:
    """Disable torch.compile use for the remainder of the process after a hard failure."""
    global _COMPILE_DISABLED
    if _COMPILE_DISABLED:
        return
    _COMPILE_DISABLED = True
    print(f"⚠️ Disabling torch.compile for this process due to error: {reason}")


def _ensure_cache_aliases(model: torch.nn.Module) -> None:
    """
    New model.py renamed kv-cache helpers (get_kv_cache_*). Ensure legacy names exist
    so samplers / older call sites keep working.
    """
    if hasattr(model, "get_kv_cache_text") and not hasattr(model, "get_text_kv_cache"):
        model.get_text_kv_cache = model.get_kv_cache_text  # type: ignore[attr-defined]
    if hasattr(model, "get_kv_cache_speaker") and not hasattr(model, "get_speaker_kv_cache"):
        model.get_speaker_kv_cache = model.get_kv_cache_speaker  # type: ignore[attr-defined]
    if hasattr(model, "get_kv_cache_latent") and not hasattr(model, "get_latent_kv_cache"):
        model.get_latent_kv_cache = model.get_kv_cache_latent  # type: ignore[attr-defined]


def _load_components(
    *,
    force_reinit: bool = False,
    force_compile: Optional[bool] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, Any]:
    """
    Load or reload components. force_compile overrides env-driven compile flag.
    """
    global _MODEL, _MODEL_LORA, _FISH_AE, _PCA_STATE, _COMPILE_DISABLED

    if force_reinit:
        _MODEL = None
        _MODEL_LORA = None
        _FISH_AE = None
        _PCA_STATE = None

    lora_requested = LORA_FIRST_BLOCK and bool(LORA_HF_NAME)
    compile_flag = USE_COMPILE and not _COMPILE_DISABLED if force_compile is None else force_compile
    ae_compile_flag = COMPILE_AE and not _COMPILE_DISABLED if force_compile is None else force_compile
    base_compile_flag = compile_flag and not (COMPILE_LORA_ONLY and lora_requested)
    lora_compile_flag = compile_flag

    if _MODEL is None:
        _MODEL = load_model_from_hf(
            repo_id=MODEL_REPO,
            device=DEVICE,
            dtype=_dtype_from_name(MODEL_DTYPE),
            compile=base_compile_flag,
        )
        _ensure_cache_aliases(_MODEL)
        print(f"[load] model repo={MODEL_REPO} device={DEVICE} dtype={MODEL_DTYPE} compile={base_compile_flag}")

    if lora_requested and _MODEL_LORA is None:
        try:
            _MODEL_LORA = load_model_from_hf(
                repo_id=LORA_REPO,
                device=DEVICE,
                dtype=_dtype_from_name(MODEL_DTYPE),
                compile=lora_compile_flag,
                lora_hf_hub_name=LORA_HF_NAME,
                lora_scale=LORA_SCALE,
                lora_alpha=LORA_ALPHA,
            )
            _ensure_cache_aliases(_MODEL_LORA)
            print(
                f"[load] lora model repo={LORA_REPO} weight={LORA_HF_NAME} "
                f"scale={LORA_SCALE} alpha={LORA_ALPHA} device={DEVICE} dtype={MODEL_DTYPE} compile={lora_compile_flag}"
            )
        except Exception as exc:
            _MODEL_LORA = None
            print(f"⚠️ Failed to load LoRA model ({LORA_HF_NAME}): {exc}")
            print("⚠️ First-block LoRA disabled; continuing with base model.")

    if _FISH_AE is None:
        _FISH_AE = load_fish_ae_from_hf(
            repo_id=FISH_REPO,
            device=FISH_DEVICE,
            dtype=_dtype_from_name(FISH_DTYPE),
            compile=ae_compile_flag,
        )
        print(f"[load] fish_ae repo={FISH_REPO} device={FISH_DEVICE} dtype={FISH_DTYPE} compile={ae_compile_flag}")

    if _PCA_STATE is None:
        _PCA_STATE = load_pca_state_from_hf(repo_id=PCA_REPO, device=DEVICE)
        print(f"[load] pca repo={PCA_REPO} device={DEVICE}")

    return _MODEL, _FISH_AE, _PCA_STATE


def _voice_roots() -> List[Path]:
    """Resolved voice roots (skip paths that cannot be resolved)."""
    roots: List[Path] = []
    for directory in VOICE_DIRS:
        try:
            roots.append(directory.resolve())
        except (OSError, RuntimeError):
            continue
    return roots


def _is_path_within_voice_dirs(path: Path, roots: Iterable[Path]) -> bool:
    """Ensure the real path stays under one of the allowed voice directories."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False

    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _find_voice_file(name: str) -> Optional[Path]:
    """Find a voice file or directory by name in VOICE_DIRS (no traversal)."""
    sanitized = name.strip()
    if not sanitized:
        return None

    roots = _voice_roots()
    name_path = Path(sanitized)

    for directory in VOICE_DIRS:
        if not directory.exists():
            continue
        # First check for exact directory match if folder support is enabled
        if FOLDER_SUPPORT:
            dir_path = directory / sanitized
            if _is_path_within_voice_dirs(dir_path, roots) and dir_path.is_dir():
                return dir_path

        # Support callers providing the full filename with extension.
        direct_path = directory / sanitized
        if (
            name_path.suffix
            and name_path.suffix.lower() in _AUDIO_EXTS
            and _is_path_within_voice_dirs(direct_path, roots)
            and direct_path.is_file()
        ):
            return direct_path

        # Otherwise look for a matching stem with an allowed extension (case-insensitive).
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in _AUDIO_EXTS:
                continue
            if path.stem != sanitized:
                continue
            if not _is_path_within_voice_dirs(path, roots):
                continue
            return path
    return None


def _list_voice_options() -> List[Dict[str, Any]]:
    """
    Enumerate available voice ids (file stems and folder names) across VOICE_DIRS.
    Deduplicate by name; skip hidden entries; require at least one valid audio file for folders.
    """
    voices: List[Dict[str, Any]] = []
    seen: set[str] = set()
    roots = _voice_roots()

    for directory in VOICE_DIRS:
        if not directory.exists():
            continue
        try:
            entries = list(directory.iterdir())
        except (OSError, RuntimeError):
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                if entry.suffix.lower() not in _AUDIO_EXTS:
                    continue
                voice_name = entry.stem
                if voice_name in seen:
                    continue
                if not _is_path_within_voice_dirs(entry, roots):
                    continue
                voices.append(
                    {
                        "object": "voice",
                        "id": voice_name,
                        "name": voice_name,
                        "metadata": {"source": directory.name, "type": "file"},
                    }
                )
                seen.add(voice_name)
            elif entry.is_dir() and FOLDER_SUPPORT:
                voice_name = entry.name
                if voice_name in seen:
                    continue
                if not _is_path_within_voice_dirs(entry, roots):
                    continue
                has_audio = False
                try:
                    for child in entry.iterdir():
                        if child.is_file() and child.suffix.lower() in _AUDIO_EXTS:
                            has_audio = True
                            break
                except (OSError, RuntimeError):
                    continue
                if not has_audio:
                    continue
                voices.append(
                    {
                        "object": "voice",
                        "id": voice_name,
                        "name": voice_name,
                        "metadata": {"source": directory.name, "type": "folder"},
                    }
                )
                seen.add(voice_name)
    return voices


def _decode_base64_audio(encoded: str) -> Tuple[torch.Tensor, str]:
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 voice: {exc}") from exc

    cache_key = f"base64:{sha256(raw).hexdigest()}"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return audio, cache_key


def _load_speaker_latent_from_directory(directory_path: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Load all audio files from a directory, concatenate with 1s gaps,
    then encode once (trimmed to 5 minutes max).
    
    Returns: (speaker_latent, speaker_mask, cache_key)
    """
    voice_roots = _voice_roots()
    if not _is_path_within_voice_dirs(directory_path, voice_roots):
        raise HTTPException(status_code=400, detail="Invalid voice directory")

    if not directory_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Voice directory not found: {directory_path}")

    # Find all audio files in the directory
    audio_files: List[Path] = []
    for candidate in directory_path.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _AUDIO_EXTS:
            continue
        if not _is_path_within_voice_dirs(candidate, voice_roots):
            continue
        audio_files.append(candidate)

    # Sort for consistent ordering
    audio_files = sorted(audio_files)

    if not audio_files:
        raise HTTPException(
            status_code=400,
            detail=f"No audio files found in directory: {directory_path}",
        )

    print(f"[folder] Found {len(audio_files)} audio files in {directory_path.name}")

    # Load components for encoding
    model, fish_ae, pca_state = _load_components()

    # Maximum duration: 5 minutes = 300 seconds
    MAX_DURATION_SECONDS = 300
    MAX_SAMPLES = int(MAX_DURATION_SECONDS * SAMPLE_RATE)
    SILENCE = torch.zeros((1, SAMPLE_RATE), dtype=torch.float32)

    # Load each clip, trim to the remaining budget, and concatenate with 1s gaps.
    segments: List[torch.Tensor] = []
    total_samples = 0
    for idx, audio_file in enumerate(audio_files):
        audio = load_audio(str(audio_file))
        audio_samples = audio.shape[-1]

        if total_samples >= MAX_SAMPLES:
            print(f"[folder] Reached max duration (5 min), stopping at {audio_file.name}")
            break

        remaining_samples = MAX_SAMPLES - total_samples
        if audio_samples > remaining_samples:
            audio = audio[..., :remaining_samples]
            audio_samples = remaining_samples
            print(f"[folder] Trimming {audio_file.name} to fit 5 min limit")

        print(f"[folder] Processing {audio_file.name} ({audio_samples / SAMPLE_RATE:.2f}s)")
        segments.append(audio)
        total_samples += audio_samples

        # Add 1s of silence between clips if there is room and this is not the last file.
        if idx < len(audio_files) - 1 and total_samples < MAX_SAMPLES:
            silence_len = min(SAMPLE_RATE, MAX_SAMPLES - total_samples)
            if silence_len > 0:
                segments.append(SILENCE[:, :silence_len])
                total_samples += silence_len

    if not segments:
        raise HTTPException(
            status_code=400,
            detail=f"No valid audio could be loaded from directory: {directory_path}",
        )

    combined_audio = torch.cat(segments, dim=1)

    with torch.inference_mode():
        fish_device = next(fish_ae.parameters()).device
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            fish_ae,
            pca_state,
            combined_audio.to(device=fish_device, dtype=fish_ae.dtype),
            max_speaker_latent_length=MAX_SPEAKER_LATENT_LENGTH,
        )

    # Trim to MAX_SPEAKER_LATENT_LENGTH if needed
    if speaker_latent.shape[1] > MAX_SPEAKER_LATENT_LENGTH:
        print(
            f"[folder] Trimming concatenated latent from {speaker_latent.shape[1]} to {MAX_SPEAKER_LATENT_LENGTH}"
        )
        speaker_latent = speaker_latent[:, :MAX_SPEAKER_LATENT_LENGTH]
        speaker_mask = speaker_mask[:, :MAX_SPEAKER_LATENT_LENGTH]

    # Pad to patch size if needed (will be done in _get_speaker_latent, but calculate expected size)
    cache_key = f"dir:{directory_path.resolve()}:{len(audio_files)}"

    print(
        f"[folder] Combined {len(audio_files)} files into latent of length {speaker_latent.shape[1]} "
        f"({combined_audio.shape[-1] / SAMPLE_RATE:.2f}s total with gaps)"
    )

    return speaker_latent, speaker_mask, cache_key


def _resolve_voice(voice: str) -> Tuple[torch.Tensor, str]:

    local_path = _find_voice_file(voice)
    if local_path is not None:
        return load_audio(str(local_path)), f"file:{local_path.resolve()}"
    return _decode_base64_audio(voice)


def _get_speaker_latent(voice: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model, fish_ae, pca_state = _load_components()
    t_start = time.time()
    target_device = model.device
    target_device_key = _device_key(target_device)
    
    # Check if voice is a directory
    local_path = _find_voice_file(voice)
    is_directory = local_path is not None and local_path.is_dir()
    _log_debug(f"[voice] lookup voice={voice} path={local_path} is_dir={is_directory}")
    if not is_directory and local_path is not None:
        cache_key = f"file:{local_path.resolve()}"
        if cache_key in _SPEAKER_CACHE:
            _log_debug(f"[voice] cache hit {cache_key} (pre-resolve file) { (time.time() - t_start)*1000:.2f} ms")
            cached_latent, cached_mask = _SPEAKER_CACHE[cache_key]
            return cached_latent.to(model.device), cached_mask.to(model.device)
    
    # Handle directory case
    if is_directory and FOLDER_SUPPORT:
        # Generate stable cache key based on directory path
        # We'll use path + file count to ensure cache invalidation if files change
        audio_files = []
        for ext in _AUDIO_EXTS:
            audio_files.extend(local_path.glob(f"*{ext}"))
        cache_key = f"dir:{local_path.resolve()}:{len(sorted(audio_files))}"
        
        if cache_key in _SPEAKER_CACHE:
            _log_debug(f"[voice] cache hit {cache_key} (dir) { (time.time() - t_start)*1000:.2f} ms")
            _log_debug(f"[cache] Using cached speaker latent for directory: {local_path.name}")
            cached_latent, cached_mask = _SPEAKER_CACHE[cache_key]
            gpu_hit = (
                _SPEAKER_CACHE_GPU.get(cache_key, {}).get(target_device_key)
                if CACHE_SPEAKER_ON_GPU
                else None
            )
            if gpu_hit:
                _log_debug(f"[voice] gpu cache hit {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")
                return gpu_hit[0], gpu_hit[1]
            result = cached_latent.to(target_device), cached_mask.to(target_device)
            if CACHE_SPEAKER_ON_GPU:
                _SPEAKER_CACHE_GPU.setdefault(cache_key, {})[target_device_key] = result
                _log_debug(f"[voice] gpu cache store {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")
            return result
        
        # Load and encode all files in directory
        speaker_latent, speaker_mask, _ = _load_speaker_latent_from_directory(local_path)
        
        # Apply patch size padding
        patch_size = getattr(model, "speaker_patch_size", 4)
        target_len = int(math.ceil(speaker_latent.shape[1] / patch_size) * patch_size)
        if target_len != speaker_latent.shape[1]:
            pad_amt = target_len - speaker_latent.shape[1]
            speaker_latent = torch.nn.functional.pad(speaker_latent, (0, 0, 0, pad_amt))
            speaker_mask = torch.nn.functional.pad(speaker_mask, (0, pad_amt))
        
        # Cache on CPU to avoid holding GPU memory across requests
        _SPEAKER_CACHE[cache_key] = (
            speaker_latent.cpu(),
            speaker_mask.cpu(),
        )
        _log_debug(f"[cache] Cached speaker latent for directory: {local_path.name}")
        _log_debug(f"[voice] cache store {cache_key} (dir) { (time.time() - t_start)*1000:.2f} ms")
        if CACHE_SPEAKER_ON_GPU:
            _SPEAKER_CACHE_GPU.setdefault(cache_key, {})[target_device_key] = (
                speaker_latent.to(target_device),
                speaker_mask.to(target_device),
            )
            _log_debug(f"[voice] gpu cache store {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")
        
        return _SPEAKER_CACHE[cache_key][0].to(model.device), _SPEAKER_CACHE[cache_key][1].to(model.device)
    
    # Handle single file case (existing logic)
    load_start = time.time()
    audio, cache_key = _resolve_voice(voice)
    _log_debug(f"[voice] load_audio {cache_key} {(time.time() - load_start)*1000:.2f} ms")

    if cache_key in _SPEAKER_CACHE:
        _log_debug(f"[voice] cache hit {cache_key} (file) { (time.time() - t_start)*1000:.2f} ms")
        cached_latent, cached_mask = _SPEAKER_CACHE[cache_key]
        gpu_hit = (
            _SPEAKER_CACHE_GPU.get(cache_key, {}).get(target_device_key)
            if CACHE_SPEAKER_ON_GPU
            else None
        )
        if gpu_hit:
            _log_debug(f"[voice] gpu cache hit {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")
            return gpu_hit[0], gpu_hit[1]
        result = cached_latent.to(target_device), cached_mask.to(target_device)
        if CACHE_SPEAKER_ON_GPU:
            _SPEAKER_CACHE_GPU.setdefault(cache_key, {})[target_device_key] = result
            _log_debug(f"[voice] gpu cache store {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")
        return result

    with torch.inference_mode():
        fish_device = next(fish_ae.parameters()).device
        to_device_start = time.time()
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            fish_ae,
            pca_state,
            audio.to(device=fish_device, dtype=fish_ae.dtype),
            max_speaker_latent_length=MAX_SPEAKER_LATENT_LENGTH,
        )
        _log_debug(f"[voice] encode speaker {cache_key} {(time.time() - to_device_start)*1000:.2f} ms")

        patch_size = getattr(model, "speaker_patch_size", 4)
        target_len = int(math.ceil(speaker_latent.shape[1] / patch_size) * patch_size)
        if target_len != speaker_latent.shape[1]:
            pad_amt = target_len - speaker_latent.shape[1]
            speaker_latent = torch.nn.functional.pad(speaker_latent, (0, 0, 0, pad_amt))
            speaker_mask = torch.nn.functional.pad(speaker_mask, (0, pad_amt))

        # Cache on CPU to avoid holding GPU memory across requests; move to device on return.
        _SPEAKER_CACHE[cache_key] = (
            speaker_latent.cpu(),
            speaker_mask.cpu(),
        )
        _log_debug(f"[voice] cache store {cache_key} (file) { (time.time() - t_start)*1000:.2f} ms")
        if CACHE_SPEAKER_ON_GPU:
            _SPEAKER_CACHE_GPU.setdefault(cache_key, {})[target_device_key] = (
                speaker_latent.to(target_device),
                speaker_mask.to(target_device),
            )
            _log_debug(f"[voice] gpu cache store {cache_key} dev={target_device_key} {(time.time() - t_start)*1000:.2f} ms")

    return _SPEAKER_CACHE[cache_key][0].to(target_device), _SPEAKER_CACHE[cache_key][1].to(target_device)


def _pick_warmup_voice() -> Optional[str]:
    if WARMUP_VOICE:
        return WARMUP_VOICE
    for directory in VOICE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix and path.suffix.lower() not in _AUDIO_EXTS:
                continue
            if not path.suffix:
                continue
            return path.stem
    return None


def _to_list(value: Any, length: int, field: str) -> List[Any]:
    if isinstance(value, list):
        if len(value) != length:
            raise HTTPException(
                status_code=400,
                detail=f"{field} length ({len(value)}) must match block_sizes length ({length})",
            )
        return value
    return [value for _ in range(length)]


@dataclass
class SamplerConfig:
    block_sizes: List[int]
    num_steps: List[int]
    cfg_scale_text: float
    cfg_scale_speaker: float
    cfg_min_t: float
    cfg_max_t: float
    truncation_factor: Optional[List[float]]
    init_scale: Optional[List[float]]
    rescale_k: Optional[float]
    rescale_sigma: Optional[float]
    speaker_kv_scale: Optional[float]
    speaker_kv_min_t: Optional[float]
    speaker_kv_max_layers: Optional[int]
    early_stop_on_zero: bool
    zero_eps: float
    zero_tail_min_frac: float
    zero_tail_frames: int
    guidance_mode: GuidanceMode
    max_text_length: int


def _parse_sampler_config(extra_body: Dict[str, Any]) -> SamplerConfig:
    block_sizes_raw = extra_body.get("block_sizes", DEFAULT_BLOCK_SIZES)
    if isinstance(block_sizes_raw, int):
        block_sizes = [int(block_sizes_raw)]
    else:
        block_sizes = [int(x) for x in block_sizes_raw]

    num_steps_raw = extra_body.get("num_steps", DEFAULT_NUM_STEPS)
    if isinstance(num_steps_raw, int):
        num_steps = [int(num_steps_raw) for _ in block_sizes]
    else:
        num_steps = [int(x) for x in num_steps_raw]
        if len(num_steps) != len(block_sizes):
            raise HTTPException(
                status_code=400,
                detail="num_steps list must match block_sizes length",
            )

    cfg_scale_text = float(extra_body.get("cfg_scale_text", DEFAULT_CFG_TEXT))
    cfg_scale_speaker = float(extra_body.get("cfg_scale_speaker", DEFAULT_CFG_SPEAKER))
    cfg_min_t = float(extra_body.get("cfg_min_t", DEFAULT_CFG_MIN_T))
    cfg_max_t = float(extra_body.get("cfg_max_t", DEFAULT_CFG_MAX_T))

    truncation_raw = extra_body.get("truncation_factor", None)
    truncation = (
        None
        if truncation_raw is None
        else _to_list(truncation_raw, len(block_sizes), "truncation_factor")
    )

    init_scale_raw = extra_body.get("init_scale", None)
    init_scale = (
        None
        if init_scale_raw is None
        else _to_list(init_scale_raw, len(block_sizes), "init_scale")
    )

    rescale_k = extra_body.get("rescale_k", None)
    rescale_sigma = extra_body.get("rescale_sigma", None)
    speaker_kv_scale = extra_body.get("speaker_kv_scale", None)
    speaker_kv_min_t = extra_body.get("speaker_kv_min_t", None)
    speaker_kv_max_layers = extra_body.get("speaker_kv_max_layers", None)
    _log_debug(
        f"[parse] speaker_kv_scale={speaker_kv_scale} speaker_kv_min_t={speaker_kv_min_t} speaker_kv_max_layers={speaker_kv_max_layers}"
    )

    zero_eps = float(extra_body.get("zero_eps", DEFAULT_ZERO_EPS))
    zero_tail_min_frac = float(
        extra_body.get("zero_tail_min_frac", DEFAULT_ZERO_TAIL_MIN_FRAC)
    )
    zero_tail_frames = int(extra_body.get("zero_tail_frames", DEFAULT_ZERO_TAIL_FRAMES))
    early_stop_on_zero = bool(extra_body.get("early_stop_on_zero", DEFAULT_EARLY_STOP))
    max_text_length = int(extra_body.get("max_text_length", 768))

    guidance_mode_raw = str(
        extra_body.get("guidance_mode", GuidanceMode.INDEPENDENT.value)
    )
    try:
        guidance_mode = GuidanceMode(guidance_mode_raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid guidance_mode '{guidance_mode_raw}'",
        ) from exc

    return SamplerConfig(
        block_sizes=block_sizes,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        truncation_factor=truncation,
        init_scale=init_scale,
        rescale_k=float(rescale_k) if rescale_k is not None else None,
        rescale_sigma=float(rescale_sigma) if rescale_sigma is not None else None,
        speaker_kv_scale=float(speaker_kv_scale)
        if speaker_kv_scale is not None
        else None,
        speaker_kv_min_t=float(speaker_kv_min_t)
        if speaker_kv_min_t is not None
        else None,
        speaker_kv_max_layers=int(speaker_kv_max_layers)
        if speaker_kv_max_layers is not None
        else None,
        early_stop_on_zero=early_stop_on_zero,
        zero_eps=zero_eps,
        zero_tail_min_frac=zero_tail_min_frac,
        zero_tail_frames=zero_tail_frames,
        guidance_mode=guidance_mode,
        max_text_length=max_text_length,
    )


def _audio_to_pcm(audio: torch.Tensor) -> bytes:
    audio = audio.detach().float().cpu().squeeze()
    audio = torch.clamp(audio, -1.0, 1.0)
    return (audio * 32767.0).to(torch.int16).numpy().tobytes()


def _pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap little-endian 16-bit PCM into a WAV container."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buffer.getvalue()


def _encode_mp3_from_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """
    Encode raw PCM to MP3 using ffmpeg via stdin/stdout to avoid temp files.
    Requires ffmpeg to be present on PATH; otherwise raises HTTPException.
    """
    if not FFMPEG_PATH:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires ffmpeg in PATH",
        )
    try:
        result = subprocess.run(
            [
                FFMPEG_PATH,
                "-v",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "-",
                "-f",
                "mp3",
                "-",
            ],
            input=pcm,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to encode mp3") from exc
    if result.returncode != 0 or not result.stdout:
        err = (result.stderr or b"").decode(errors="ignore").strip()
        _log_debug(f"[mp3] ffmpeg encode error rc={result.returncode} stderr={err}")
        raise HTTPException(status_code=500, detail="Failed to encode mp3")
    return bytes(result.stdout)


def _save_debug_wav(tensor: torch.Tensor, path: str) -> None:
    try:
        torchaudio.save(path, tensor.detach().cpu().unsqueeze(0).float(), SAMPLE_RATE)
        _log_debug(f"[debug] saved {path}")
    except Exception as exc:
        _log_debug(f"[debug] failed to save {path}: {exc}")


def _ae_decode_with_flatten(
    fish_ae: DAC,
    pca_state: PCAState,
    latent: torch.Tensor,
    flattening_point: int | None = None,
) -> torch.Tensor:
    """
    Mirror prior behavior: decode, then truncate to flattening point (latent units),
    This keeps padding tails out when downstream decode slices are used.
    """
    if flattening_point is None:
        flattening_point = find_flattening_point(latent[0])
    if flattening_point <= 0:
        return latent.new_zeros((latent.shape[0], 1, 0))
    latent = latent[:, :flattening_point, ...]
    audio = _base_ae_decode(fish_ae, pca_state, latent)
    return audio


class StreamingAEDecoder:
    """
    Stateful decoder that keeps a bounded latent tail so each block is decoded with context,
    while emitting only the newly added samples in global time.
    """

    def __init__(
        self,
        fish_ae: DAC,
        pca_state: PCAState,
        samples_per_latent: int | None = None,
        max_latent_ctx: int | None = None,
    ) -> None:
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.samples_per_latent = samples_per_latent or getattr(fish_ae, "frame_length", 2048)

        # Default to post-transformer window (128) + small margin.
        default_ctx = 160
        # Include decoder delay as extra latent context.
        delay_latents = math.ceil(getattr(fish_ae, "delay", 0) / self.samples_per_latent)
        self.max_latent_ctx = max_latent_ctx or (default_ctx + delay_latents)

        self._tail_latents: Optional[torch.Tensor] = None
        self._emitted_samples: int = 0
        self._total_latents_seen: int = 0

    def decode_next(self, latent_block: torch.Tensor, flattening_point: int | None = None) -> torch.Tensor:
        """Decode next block with context; return only new audio in global timeline."""
        block_len = latent_block.shape[1]
        self._total_latents_seen += block_len

        max_len = self.max_latent_ctx + block_len  # always keep full new block plus context

        if self._tail_latents is None:
            self._tail_latents = latent_block
        else:
            self._tail_latents = torch.cat([self._tail_latents, latent_block], dim=1)
            if self._tail_latents.shape[1] > max_len:
                self._tail_latents = self._tail_latents[:, -max_len:]

        tail_len = self._tail_latents.shape[1]
        global_start_latent = self._total_latents_seen - tail_len
        global_start_samples = global_start_latent * self.samples_per_latent

        local_flatten = None
        if flattening_point is not None:
            local_flatten = max(0, min(flattening_point - global_start_latent, tail_len))

        if local_flatten is None:
            audio = _base_ae_decode(self.fish_ae, self.pca_state, self._tail_latents)
        else:
            audio = _ae_decode_with_flatten(self.fish_ae, self.pca_state, self._tail_latents, local_flatten)

        audio_len = audio.shape[-1]
        total_samples_global = global_start_samples + audio_len

        # Figure out slice of this decode that hasn't been emitted yet.
        new_start = max(0, self._emitted_samples - global_start_samples)
        if new_start >= audio.shape[-1]:
            return audio[..., 0:0]

        new_audio = audio[..., new_start:]
        if flattening_point is not None:
            target_samples = flattening_point * self.samples_per_latent
            remaining = target_samples - self._emitted_samples
            if remaining <= 0:
                return new_audio[..., 0:0]
            if new_audio.shape[-1] > remaining:
                new_audio = new_audio[..., :remaining]
            self._emitted_samples = min(target_samples, self._emitted_samples + new_audio.shape[-1])
        else:
            self._emitted_samples = total_samples_global
        return new_audio

    def is_finished(self, flattening_point: int | None) -> bool:
        if flattening_point is None:
            return False
        return self._emitted_samples >= flattening_point * self.samples_per_latent


@torch.inference_mode()
def _generate_full_audio_bytes(
    text: str,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    cfg: SamplerConfig,
    rng_seed: int,
) -> bytes:
    text = preprocess_text(text, normalize_exclamation=NORMALIZE_EXCLAMATION)

    model, fish_ae, pca_state = _load_components()
    device = model.device

    text_input_ids, text_mask = get_text_input_ids_and_mask(
        [text], min(cfg.max_text_length, 768), device=device
    )

    sample_fn = partial(
        sample_euler_cfg_any,
        block_sizes=cfg.block_sizes,
        guidance_mode=cfg.guidance_mode,
        num_steps=cfg.num_steps,
        cfg_scale_text=cfg.cfg_scale_text,
        cfg_scale_speaker=cfg.cfg_scale_speaker,
        cfg_min_t=cfg.cfg_min_t,
        cfg_max_t=cfg.cfg_max_t,
        truncation_factor=cfg.truncation_factor,
        init_scale=cfg.init_scale,
        rescale_k=cfg.rescale_k,
        rescale_sigma=cfg.rescale_sigma,
        speaker_kv_scale=cfg.speaker_kv_scale,
        speaker_kv_min_t=cfg.speaker_kv_min_t,
        speaker_kv_max_layers=cfg.speaker_kv_max_layers,
        early_stop_on_zero=cfg.early_stop_on_zero,
        zero_eps=cfg.zero_eps,
        zero_tail_min_frac=cfg.zero_tail_min_frac,
        zero_tail_frames=cfg.zero_tail_frames,
    )

    latent_out = sample_fn(
        model,
        speaker_latent,
        speaker_mask,
        text_input_ids,
        text_mask,
        rng_seed,
    )

    audio_out = _ae_decode_with_flatten(fish_ae, pca_state, latent_out)
    pcm = _audio_to_pcm(audio_out)
    torch.cuda.empty_cache()
    return pcm


def _warmup_compile(block_sizes: List[int], num_steps: List[int]) -> None:
    global _WARMUP_RAN
    if _WARMUP_RAN or not USE_COMPILE or _COMPILE_DISABLED:
        return

    voice = _pick_warmup_voice()
    if voice is None:
        print("⚠️ Warmup skipped (no voice file found in audio_prompts/prompt_audio/extra_prompt_audio).")
        return

    warm_sets: List[Tuple[List[int], List[int]]] = [
        (block_sizes, num_steps),
    ]
    nonstream_blocks = [DEFAULT_BLOCK_SIZE_NONSTREAM]
    nonstream_steps = [DEFAULT_NUM_STEPS_NONSTREAM]
    if (block_sizes, num_steps) != (nonstream_blocks, nonstream_steps):
        warm_sets.append((nonstream_blocks, nonstream_steps))

    try:
        speaker_latent, speaker_mask = _get_speaker_latent(voice)
        for idx, (blocks, steps) in enumerate(warm_sets, start=1):
            print(f"Warmup compile run {idx}/{len(warm_sets)} with voice '{voice}' block sizes {blocks}...")
            base_cfg = SamplerConfig(
                block_sizes=blocks,
                num_steps=steps,
                cfg_scale_text=DEFAULT_CFG_TEXT,
                cfg_scale_speaker=DEFAULT_CFG_SPEAKER,
                cfg_min_t=DEFAULT_CFG_MIN_T,
                cfg_max_t=DEFAULT_CFG_MAX_T,
                truncation_factor=None,
                init_scale=None,
                rescale_k=None,
                rescale_sigma=None,
                speaker_kv_scale=None,
                speaker_kv_min_t=None,
                speaker_kv_max_layers=None,
                early_stop_on_zero=False,
                zero_eps=DEFAULT_ZERO_EPS,
                zero_tail_min_frac=DEFAULT_ZERO_TAIL_MIN_FRAC,
                zero_tail_frames=DEFAULT_ZERO_TAIL_FRAMES,
                guidance_mode=GuidanceMode.INDEPENDENT,
                max_text_length=768,
            )

            # Warm model path (LoRA is used automatically on first block if enabled)
            for _ in _stream_blocks(
                text=WARMUP_TEXT,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                cfg=base_cfg,
                rng_seed=0,
            ):
                pass

        _WARMUP_RAN = True
        print("✅ Warmup compile finished; cache should be saved.")
    except Exception as exc:
        print(f"⚠️ Warmup compile failed: {exc}")


def _kv_cache_text(model: torch.nn.Module, text_input_ids: torch.Tensor, text_mask: torch.Tensor):
    if hasattr(model, "get_kv_cache_text"):
        return model.get_kv_cache_text(text_input_ids, text_mask)  # type: ignore[attr-defined]
    if hasattr(model, "get_text_kv_cache"):
        return model.get_text_kv_cache(text_input_ids, text_mask)  # type: ignore[attr-defined]
    raise RuntimeError("Model is missing text kv-cache helpers")


def _kv_cache_speaker(model: torch.nn.Module, speaker_latent: torch.Tensor):
    if hasattr(model, "get_kv_cache_speaker"):
        return model.get_kv_cache_speaker(speaker_latent)  # type: ignore[attr-defined]
    if hasattr(model, "get_speaker_kv_cache"):
        return model.get_speaker_kv_cache(speaker_latent)  # type: ignore[attr-defined]
    raise RuntimeError("Model is missing speaker kv-cache helpers")


def _kv_cache_latent(model: torch.nn.Module, prefix_latent: torch.Tensor):
    if hasattr(model, "get_kv_cache_latent"):
        return model.get_kv_cache_latent(prefix_latent)  # type: ignore[attr-defined]
    if hasattr(model, "get_latent_kv_cache"):
        return model.get_latent_kv_cache(prefix_latent)  # type: ignore[attr-defined]
    raise RuntimeError("Model is missing latent kv-cache helpers")


def _is_dynamo_fx_error(exc: Exception) -> bool:
    msg = str(exc)
    return "symbolically trace a dynamo-optimized function" in msg or "dynamo-optimized function" in msg


@torch.inference_mode()
def _stream_blocks(
    text: str,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    cfg: SamplerConfig,
    rng_seed: int,
) -> Iterator[bytes]:
    text = preprocess_text(text, normalize_exclamation=NORMALIZE_EXCLAMATION)
    if cfg.guidance_mode != GuidanceMode.INDEPENDENT:
        raise HTTPException(
            status_code=400,
            detail="Streaming path currently supports guidance_mode='independent' only",
        )

    base_model, fish_ae, pca_state = _load_components()
    lora_model = _MODEL_LORA if LORA_FIRST_BLOCK and _MODEL_LORA is not None else None

    base_device = base_model.device
    if lora_model is not None and lora_model.device != base_device:
        try:
            lora_model.to(base_device)
            torch.cuda.empty_cache()
            print(f"[lora] Moved LoRA model to {base_device} to match base model.")
        except RuntimeError as exc:
            print(
                f"⚠️ LoRA model device {lora_model.device} differs from base {base_device} "
                f"and move failed ({exc}); using base model for all blocks."
            )
            lora_model = None

    use_lora_first_block = lora_model is not None
    if LORA_FIRST_BLOCK and lora_model is None and LORA_HF_NAME:
        print("⚠️ LoRA first-block requested but unavailable; falling back to base model.")
    if use_lora_first_block:
        print("[lora] Using LoRA-merged model for first block; base model for subsequent blocks.")

    start_time_wall = time.time()

    model_first = lora_model if use_lora_first_block else base_model
    device, dtype = model_first.device, model_first.dtype
    batch_size = 1

    text_input_ids, text_mask = get_text_input_ids_and_mask(
        [text], min(cfg.max_text_length, 768), device=device
    )

    torch.manual_seed(rng_seed)

    if isinstance(cfg.num_steps, int):
        step_counts = [cfg.num_steps] * len(cfg.block_sizes)
    else:
        step_counts = cfg.num_steps

    if cfg.truncation_factor is None or isinstance(cfg.truncation_factor, float):
        truncation_factors = [cfg.truncation_factor] * len(cfg.block_sizes)
    else:
        truncation_factors = cfg.truncation_factor

    default_init_scale = 0.999
    if cfg.init_scale is None or isinstance(cfg.init_scale, float):
        init_scales = [
            default_init_scale if cfg.init_scale is None else cfg.init_scale
        ] * len(cfg.block_sizes)
    else:
        init_scales = cfg.init_scale

    text_input_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(
        text_input_ids.shape[0], text_input_ids.shape[1], device=device
    )

    speaker_latent_uncond, speaker_mask_uncond = (
        torch.zeros_like(speaker_latent),
        torch.zeros_like(speaker_mask),
    )

    full_text_input_ids = torch.cat(
        [text_input_ids, text_input_ids_uncond, text_input_ids], dim=0
    )
    full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)

    full_speaker_latent = torch.cat(
        [speaker_latent, speaker_latent, speaker_latent_uncond], dim=0
    )
    full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

    # Build condition caches lazily per model (base vs LoRA) so we don't duplicate work on TTFB.
    condition_caches: Dict[str, Tuple[Any, Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = {  # type: ignore[type-arg]
        "base": None,
        "lora": None,
    }

    lora_cond_cache: Tuple[Any, Any, torch.Tensor, torch.Tensor] | None = None

    def _get_condition_caches(model_for_caches: torch.nn.Module, key: str):
        caches = condition_caches[key]
        if caches is None:
            text_ids_local = full_text_input_ids.to(device=model_for_caches.device)
            text_mask_local = text_mask.to(device=model_for_caches.device)
            full_text_mask_local = full_text_mask.to(device=model_for_caches.device)
            speaker_mask_local = speaker_mask.to(device=model_for_caches.device)
            full_speaker_mask_local = full_speaker_mask.to(device=model_for_caches.device)

            kv_cache_text_full = _kv_cache_text(model_for_caches, text_ids_local, full_text_mask_local)
            kv_cache_text = _get_first_n_kv_cache(kv_cache_text_full, batch_size)

            kv_cache_speaker_full = _kv_cache_speaker(
                model_for_caches,
                full_speaker_latent.to(device=model_for_caches.device, dtype=model_for_caches.dtype),
            )
            kv_cache_speaker = _get_first_n_kv_cache(kv_cache_speaker_full, batch_size)

            caches = (
                kv_cache_text_full,
                kv_cache_text,
                kv_cache_speaker_full,
                kv_cache_speaker,
                text_mask_local,
                full_text_mask_local,
                speaker_mask_local,
                full_speaker_mask_local,
            )
            condition_caches[key] = caches
        return caches

    def _get_lora_cond_cache(model_for_caches: torch.nn.Module):
        nonlocal lora_cond_cache
        if lora_cond_cache is None:
            text_ids_local = text_input_ids.to(device=model_for_caches.device)
            text_mask_local = text_mask.to(device=model_for_caches.device)
            speaker_mask_local = speaker_mask.to(device=model_for_caches.device)

            kv_cache_text = _kv_cache_text(model_for_caches, text_ids_local, text_mask_local)
            kv_cache_speaker = _kv_cache_speaker(
                model_for_caches,
                speaker_latent.to(device=model_for_caches.device, dtype=model_for_caches.dtype),
            )
            lora_cond_cache = (kv_cache_text, kv_cache_speaker, text_mask_local, speaker_mask_local)
        return lora_cond_cache

    streaming_decoder = StreamingAEDecoder(fish_ae, pca_state)

    prefix_total = sum(cfg.block_sizes)
    prefix_latent = torch.zeros(
        (batch_size, prefix_total, 80), device=device, dtype=torch.float32
    )

    pos_id = 0
    first_block_started_at = None
    ttfb_reported = False
    final_block_start_samples = 0

    for block_idx, (block_size, block_steps) in enumerate(
        zip(cfg.block_sizes, step_counts)
    ):
        block_start_time = time.time()  # Track total generation time for this chunk
        block_trunc = truncation_factors[block_idx]
        block_init_scale = init_scales[block_idx]

        use_lora_block = use_lora_first_block and block_idx == 0
        model_for_block = lora_model if use_lora_block else base_model
        block_dtype = model_for_block.dtype
        block_device = model_for_block.device
        cache_key = "lora" if use_lora_block else "base"

        kv_cache_text_full = kv_cache_text = kv_cache_speaker_full = kv_cache_speaker = None
        block_text_mask = block_full_text_mask = block_speaker_mask = block_full_speaker_mask = None
        kv_cache_text_full_lora = kv_cache_speaker_full_lora = block_text_mask_lora = block_speaker_mask_lora = None  # type: ignore[assignment]

        if use_lora_block:
            kv_cache_text_full_lora, kv_cache_speaker_full_lora, block_text_mask_lora, block_speaker_mask_lora = _get_lora_cond_cache(model_for_block)
        else:
            (
                kv_cache_text_full,
                kv_cache_text,
                kv_cache_speaker_full,
                kv_cache_speaker,
                block_text_mask,
                block_full_text_mask,
                block_speaker_mask,
                block_full_speaker_mask,
            ) = _get_condition_caches(model_for_block, cache_key)

        t_schedule = (
            torch.linspace(1.0, 0.0, block_steps + 1, device=block_device) * block_init_scale
        )

        if cfg.speaker_kv_scale is not None:
            target_kv = kv_cache_speaker_full_lora if use_lora_block else kv_cache_speaker_full
            _multiply_speaker_kv_cache(
                target_kv,
                cfg.speaker_kv_scale,
                text_input_ids.shape[-1],
                cfg.speaker_kv_max_layers,
            )

        # Time KV cache latent computation
        kv_start = time.time()
        if use_lora_block:
            full_prefix_latent = prefix_latent
            kv_cache_latent_full = _kv_cache_latent(model_for_block, full_prefix_latent.to(block_dtype))
            kv_cache_latent = kv_cache_latent_full
        else:
            full_prefix_latent = torch.cat(
                [prefix_latent, prefix_latent, prefix_latent], dim=0
            )
            kv_cache_latent_full = _kv_cache_latent(model_for_block, full_prefix_latent.to(block_dtype))
            kv_cache_latent = _get_first_n_kv_cache(kv_cache_latent_full, batch_size)
        kv_time = time.time() - kv_start

        # Time diffusion steps
        diffusion_start = time.time()
        x_t = torch.randn((batch_size, block_size, 80), device=block_device, dtype=torch.float32)
        if block_trunc is not None:
            x_t = x_t * block_trunc

        for i in range(block_steps):
            t, t_next = t_schedule[i], t_schedule[i + 1]
            has_cfg = ((t >= cfg.cfg_min_t) * (t <= cfg.cfg_max_t)).item()

            if use_lora_block:
                v_pred = model_for_block(
                    x=x_t.to(block_dtype),
                    t=(torch.ones((batch_size,), device=block_device) * t).to(block_dtype),
                    text_mask=block_text_mask_lora,
                    speaker_mask=block_speaker_mask_lora,
                    start_pos=pos_id,
                    kv_cache_text=kv_cache_text_full_lora,
                    kv_cache_speaker=kv_cache_speaker_full_lora,
                    kv_cache_latent=kv_cache_latent_full,
                ).float()
            elif has_cfg:
                v_cond, v_uncond_text, v_uncond_speaker = model_for_block(
                    x=torch.cat([x_t, x_t, x_t], dim=0).to(block_dtype),
                    t=(torch.ones((batch_size * 3,), device=block_device) * t).to(block_dtype),
                    text_mask=block_full_text_mask,
                    speaker_mask=block_full_speaker_mask,
                    start_pos=pos_id,
                    kv_cache_text=kv_cache_text_full,
                    kv_cache_speaker=kv_cache_speaker_full,
                    kv_cache_latent=kv_cache_latent_full,
                ).float().chunk(3, dim=0)

                v_pred = (
                    v_cond
                    + cfg.cfg_scale_text * (v_cond - v_uncond_text)
                    + cfg.cfg_scale_speaker * (v_cond - v_uncond_speaker)
                )
            else:
                v_pred = model_for_block(
                    x=x_t.to(block_dtype),
                    t=(torch.ones((batch_size,), device=block_device) * t).to(block_dtype),
                    text_mask=block_text_mask,
                    speaker_mask=block_speaker_mask,
                    start_pos=pos_id,
                    kv_cache_text=kv_cache_text,
                    kv_cache_speaker=kv_cache_speaker,
                    kv_cache_latent=kv_cache_latent,
                ).float()

            if cfg.rescale_k is not None and cfg.rescale_sigma is not None:
                v_pred = _temporal_score_rescale(
                    v_pred, x_t, float(t), cfg.rescale_k, cfg.rescale_sigma
                )

            if (
                cfg.speaker_kv_scale is not None
                and cfg.speaker_kv_min_t is not None
                and t_next < cfg.speaker_kv_min_t
                and t >= cfg.speaker_kv_min_t
            ):
                target_kv = kv_cache_speaker_full_lora if use_lora_block else kv_cache_speaker_full
                _multiply_speaker_kv_cache(
                    target_kv,
                    1.0 / cfg.speaker_kv_scale,
                    text_input_ids.shape[-1],
                    cfg.speaker_kv_max_layers,
                )

            x_t = x_t + v_pred * (t_next - t)

        diffusion_time = time.time() - diffusion_start
        prefix_latent[:, pos_id : pos_id + block_size] = x_t
        pos_id += block_size

        # Early stop detection per block
        early_stop = False
        if cfg.early_stop_on_zero:
            tail_len = min(cfg.zero_tail_frames, x_t.shape[1])
            tail = x_t[:, -tail_len:]
            tail_abs = torch.abs(tail)
            zero_frac = float((tail_abs <= cfg.zero_eps).float().mean().item())
            tail_absmax = float(tail_abs.max().item())
            zero_ok = zero_frac >= cfg.zero_tail_min_frac and tail_absmax <= cfg.zero_eps
            if zero_ok:
                early_stop = True
                print(
                    f"[early_stop] block {block_idx+1}/{len(cfg.block_sizes)} "
                    f"tail_len={tail_len} zero_frac={zero_frac:.3f} "
                    f"absmax={tail_absmax:.3e} eps={cfg.zero_eps} min_frac={cfg.zero_tail_min_frac}"
                )
            else:
                _log_debug(
                    f"[zero_tail_check] block {block_idx+1}/{len(cfg.block_sizes)} "
                    f"tail_len={tail_len} zero_frac={zero_frac:.3f} "
                    f"absmax={tail_absmax:.3e} eps={cfg.zero_eps} min_frac={cfg.zero_tail_min_frac}"
                )

        is_last_planned = (block_idx == len(cfg.block_sizes) - 1)
        will_finish = early_stop or is_last_planned

        flatten_point = None
        if will_finish:
            prefix_latent_trim = prefix_latent[:, :pos_id]
            flatten_point = find_flattening_point(prefix_latent_trim[0], window_size=32, std_threshold=0.02)
            already_emitted_latents = streaming_decoder._emitted_samples // streaming_decoder.samples_per_latent
            flatten_point = max(flatten_point, already_emitted_latents)

        # Decode block audio with streaming context
        decode_start = time.time()
        new_audio = streaming_decoder.decode_next(
            x_t.to(next(fish_ae.parameters()).device),
            flattening_point=flatten_point,
        )
        decode_time = time.time() - decode_start

        # Calculate audio duration and RTFx
        audio_samples = new_audio.numel() if new_audio.numel() > 0 else 0
        audio_duration_ms = (audio_samples / SAMPLE_RATE) * 1000.0 if audio_samples > 0 else 0.0
        
        # Total generation time for this chunk (diffusion + decode)
        chunk_gen_time = time.time() - block_start_time
        chunk_gen_time_ms = chunk_gen_time * 1000.0
        
        # RTFx: generation_time / audio_duration (lower is better, <1.0 means faster than realtime)
        rtfx = chunk_gen_time / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0
        
        print(
            f"[chunk {block_idx+1}/{len(cfg.block_sizes)}] "
            f"RTFx={rtfx:.3f} | "
            f"gen_time={chunk_gen_time_ms:.2f}ms | "
            f"audio_len={audio_duration_ms:.2f}ms | "
            f"kv_cache={kv_time*1000:.2f}ms | "
            f"diffusion={diffusion_time*1000:.2f}ms | "
            f"decode={decode_time*1000:.2f}ms"
        )

        if not ttfb_reported:
            # TTFB ~= time from function entry to first audio block ready
            ttfb_sec = time.time() - start_time_wall
            print(f"[stream] first block ready, TTFB {ttfb_sec*1000:.2f} ms (decode {decode_time*1000:.2f} ms)")
            ttfb_reported = True

        if new_audio.numel() > 0:
            yield _audio_to_pcm(new_audio)

        if will_finish:
            break

    _save_compile_cache(cfg.block_sizes)
    torch.cuda.empty_cache()


class SpeechRequest(BaseModel):
    model: str = Field(default="echo-tts")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice name (audio_prompts/prompt_audio/extra_prompt_audio) or base64 audio")
    response_format: Optional[str] = Field(
        default=None,
        description="pcm only for streaming; non-stream supports pcm/wav/mp3 (defaults to mp3 if ffmpeg is available, else wav).",
    )
    stream: bool = Field(default=True)
    seed: int = Field(default=0)
    extra_body: Dict[str, Any] = Field(default_factory=dict, description="Optional sampler overrides")


def _run_startup_tasks() -> None:
    _load_components()
    _load_compile_cache(DEFAULT_BLOCK_SIZES)
    if DEFAULT_BLOCK_SIZES != [DEFAULT_BLOCK_SIZE_NONSTREAM]:
        _load_compile_cache([DEFAULT_BLOCK_SIZE_NONSTREAM])
    if USE_COMPILE:
        _warmup_compile(DEFAULT_BLOCK_SIZES, DEFAULT_NUM_STEPS)


def _resolve_response_format(requested: Optional[str], stream: bool) -> str:
    """Return a validated response format, defaulting by stream/non-stream."""
    if requested is None or str(requested).strip() == "":
        return "pcm" if stream else ("mp3" if FFMPEG_PATH else "wav")
    fmt = str(requested).strip().lower()
    allowed = {"pcm"} if stream else {"pcm", "wav", "mp3"}
    if fmt not in allowed:
        raise HTTPException(status_code=400, detail="response_format must be one of 'pcm', 'wav', 'mp3'")
    if stream and fmt != "pcm":
        raise HTTPException(status_code=400, detail="Streaming currently supports response_format='pcm' only")
    if (not stream) and fmt == "mp3" and not FFMPEG_PATH:
        raise HTTPException(status_code=400, detail="response_format='mp3' requires ffmpeg in PATH")
    return fmt


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/voices")
def list_voices() -> Dict[str, Any]:
    """
    OpenAI-compatible voices listing.
    Returns audio file stems and folder names (if folder support is enabled) across VOICE_DIRS.
    """
    voices = _list_voice_options()
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
def create_speech(request: Request, payload: SpeechRequest = Body(...)) -> StreamingResponse:
    route_start = time.time()
    _load_components()
    _log_debug(f"[route] after load_components: {(time.time() - route_start)*1000:.2f} ms")
    response_format = _resolve_response_format(payload.response_format, payload.stream)
    sampler_cfg = _parse_sampler_config(payload.extra_body)
    _log_debug(f"[route] after parse_sampler_config: {(time.time() - route_start)*1000:.2f} ms")
    speaker_latent, speaker_mask = _get_speaker_latent(payload.voice)
    _log_debug(f"[route] after get_speaker_latent: {(time.time() - route_start)*1000:.2f} ms")
    chunking_raw = payload.extra_body.get("chunking_enabled", CHUNKING_ENABLED)
    if isinstance(chunking_raw, str):
        chunking_enabled = chunking_raw.strip().lower() not in {"0", "false", "no", "off", ""}
    else:
        chunking_enabled = bool(chunking_raw)
    chunk_target_seconds = float(payload.extra_body.get("chunk_target_seconds", 30.0))
    chunk_min_seconds = float(payload.extra_body.get("chunk_min_seconds", 20.0))
    chunk_max_seconds = float(payload.extra_body.get("chunk_max_seconds", 40.0))
    chunk_chars_per_sec = float(
        payload.extra_body.get("chunk_chars_per_second", CHUNK_CHARS_PER_SECOND)
    )
    chunk_words_per_sec = float(
        payload.extra_body.get("chunk_words_per_second", CHUNK_WORDS_PER_SECOND)
    )
    chunks: List[str]
    if chunking_enabled:
        chunks = chunk_text_by_time(
            payload.input,
            target_seconds=chunk_target_seconds,
            min_seconds=chunk_min_seconds,
            max_seconds=chunk_max_seconds,
            chars_per_second=chunk_chars_per_sec,
            words_per_second=chunk_words_per_sec,
            normalize_exclamation=NORMALIZE_EXCLAMATION,
        )
        if not chunks:
            chunks = [payload.input]
    else:
        chunks = [payload.input]
    _log_debug(f"[route] chunking_enabled={chunking_enabled} chunks={len(chunks)}")

    if not payload.stream:
        if "block_sizes" not in payload.extra_body:
            sampler_cfg.block_sizes = [DEFAULT_BLOCK_SIZE_NONSTREAM]
        if "num_steps" not in payload.extra_body:
            sampler_cfg.num_steps = [DEFAULT_NUM_STEPS_NONSTREAM for _ in sampler_cfg.block_sizes]

    chunk_cfgs: List[SamplerConfig]
    override_secondary = (
        payload.stream
        and chunking_enabled
        and len(chunks) > 1
        and "block_sizes" not in payload.extra_body
        and "num_steps" not in payload.extra_body
    )
    if override_secondary:
        secondary_cfg = replace(
            sampler_cfg,
            block_sizes=[DEFAULT_BLOCK_SIZE_NONSTREAM],
            num_steps=[DEFAULT_NUM_STEPS_NONSTREAM],
        )
        chunk_cfgs = [sampler_cfg] + [secondary_cfg for _ in chunks[1:]]
    else:
        chunk_cfgs = [sampler_cfg for _ in chunks]

    if payload.stream:
        collected: Optional[bytearray] = bytearray() if DEBUG_LOGS_ENABLED else None
        disconnect_exception: type[Exception] = type("ClientDisconnected", (Exception,), {})

        def _run_stream() -> Iterable[bytes]:
            for idx, chunk in enumerate(chunks):
                chunk_seed = payload.seed + idx
                cfg_for_chunk = chunk_cfgs[idx] if idx < len(chunk_cfgs) else sampler_cfg
                _log_debug(f"[route] starting chunk {idx+1}/{len(chunks)} seed={chunk_seed}")
                for block in _stream_blocks(
                    text=chunk,
                    speaker_latent=speaker_latent,
                    speaker_mask=speaker_mask,
                    cfg=cfg_for_chunk,
                    rng_seed=chunk_seed,
                ):
                    yield block

        async def _drain_stream(stream_iter: Iterator[bytes]) -> AsyncIterator[bytes]:
            async for chunk in iterate_in_threadpool(stream_iter):
                if await request.is_disconnected():
                    _log_debug("[route] client disconnected; stopping stream generation")
                    raise disconnect_exception()
                yield chunk

        def _close_iter(stream_iter: Optional[Iterator[bytes]]) -> None:
            if stream_iter is None:
                return
            close_fn = getattr(stream_iter, "close", None)
            if close_fn is None:
                return
            try:
                close_fn()
            except Exception:
                pass

        async def _generator() -> AsyncIterator[bytes]:
            emitted = False
            _log_debug(f"[route] generator entered: {(time.time() - route_start)*1000:.2f} ms")

            stream_iter: Optional[Iterator[bytes]] = None
            try:
                stream_iter = _run_stream()
                try:
                    async for chunk in _drain_stream(stream_iter):
                        emitted = True
                        if collected is not None:
                            collected.extend(chunk)
                        yield chunk
                except disconnect_exception:
                    _log_debug("[route] disconnect propagated; aborting remaining chunks")
                except RuntimeError as exc:
                    if emitted or not USE_COMPILE:
                        raise
                    if _is_dynamo_fx_error(exc):
                        _disable_compile(str(exc))
                        _load_components(force_reinit=True, force_compile=False)
                        _close_iter(stream_iter)
                        stream_iter = _run_stream()
                        async for chunk in _drain_stream(stream_iter):
                            if collected is not None:
                                collected.extend(chunk)
                            yield chunk
                    else:
                        raise
            finally:
                _close_iter(stream_iter)

        response = StreamingResponse(
            _generator(),
            media_type="application/octet-stream",
            headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE)},
        )
        _log_debug(
            f"[route] response constructed (headers about to send): {(time.time() - route_start)*1000:.2f} ms"
        )

        async def _save_after():
            if collected:
                try:
                    int16_audio = torch.frombuffer(memoryview(collected), dtype=torch.int16)
                    audio = int16_audio.float() / 32767.0
                    _save_debug_wav(audio, "api_generation.wav")
                except Exception as exc:
                    _log_debug(f"[debug] failed to save api_generation.wav: {exc}")

        if collected is not None:
            response.background = _save_after  # type: ignore
        return response

    # For stream=false, generate full audio, save for inspection, then return
    try:
        collected = bytearray()
        for idx, chunk in enumerate(chunks):
            chunk_seed = payload.seed + idx
            cfg_for_chunk = chunk_cfgs[idx] if idx < len(chunk_cfgs) else sampler_cfg
            _log_debug(f"[route] (non-stream) starting chunk {idx+1}/{len(chunks)} seed={chunk_seed}")
            audio_tensor = _generate_full_audio_bytes(
                text=chunk,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                cfg=cfg_for_chunk,
                rng_seed=chunk_seed,
            )
            collected.extend(audio_tensor)
            audio_tensor = bytes(collected)
    except RuntimeError as exc:
        if USE_COMPILE and _is_dynamo_fx_error(exc):
            _disable_compile(str(exc))
            _load_components(force_reinit=True, force_compile=False)
            collected = bytearray()
            for idx, chunk in enumerate(chunks):
                chunk_seed = payload.seed + idx
                cfg_for_chunk = chunk_cfgs[idx] if idx < len(chunk_cfgs) else sampler_cfg
                _log_debug(f"[route] (non-stream) retry chunk {idx+1}/{len(chunks)} seed={chunk_seed}")
                audio_tensor_part = _generate_full_audio_bytes(
                    text=chunk,
                    speaker_latent=speaker_latent,
                    speaker_mask=speaker_mask,
                    cfg=cfg_for_chunk,
                    rng_seed=chunk_seed,
                )
                collected.extend(audio_tensor_part)
            audio_tensor = bytes(collected)
        else:
            raise
    if DEBUG_LOGS_ENABLED:
        try:
            int16_audio = torch.frombuffer(memoryview(audio_tensor), dtype=torch.int16)
            audio = int16_audio.float() / 32767.0
            _save_debug_wav(audio, "api_generation.wav")
        except Exception as exc:
            _log_debug(f"[debug] failed to save api_generation.wav: {exc}")

    response_bytes = audio_tensor
    media_type = "application/octet-stream"
    if response_format == "wav":
        response_bytes = _pcm16_to_wav_bytes(audio_tensor, SAMPLE_RATE)
        media_type = "audio/wav"
    elif response_format == "mp3":
        response_bytes = _encode_mp3_from_pcm(audio_tensor, SAMPLE_RATE)
        media_type = "audio/mpeg"

    return StreamingResponse(
        content=iter([response_bytes]),
        media_type=media_type,
        headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
