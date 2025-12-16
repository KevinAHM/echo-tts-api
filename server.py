"""
Echo TTS API Server.
Clean FastAPI application for text-to-speech with voice cloning.
"""

import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional

import torch
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.concurrency import iterate_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from new.config import config
from new.inference_engine import engine, GenerationConfig
from new.speaker_latents import SpeakerLatentManager, create_speaker_latent_manager
from samplers import GuidanceMode
from utils import chunk_text_by_time


# Global state
_speaker_manager: Optional[SpeakerLatentManager] = None
FFMPEG_PATH = shutil.which("ffmpeg")


def get_speaker_manager() -> SpeakerLatentManager:
    """Get the global speaker latent manager."""
    global _speaker_manager
    if _speaker_manager is None:
        raise RuntimeError("Speaker manager not initialized. Server not started properly.")
    return _speaker_manager


def _encode_mp3_from_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """Encode raw PCM to MP3 using ffmpeg."""
    if not FFMPEG_PATH:
        raise HTTPException(status_code=400, detail="response_format='mp3' requires ffmpeg in PATH")
    
    try:
        result = subprocess.run(
            [
                FFMPEG_PATH, "-v", "error",
                "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
                "-f", "mp3", "-",
            ],
            input=pcm,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to encode mp3") from exc
    
    if result.returncode != 0 or not result.stdout:
        raise HTTPException(status_code=500, detail="Failed to encode mp3")
    return bytes(result.stdout)


def _resolve_response_format(requested: Optional[str], stream: bool) -> str:
    """Return a validated response format, defaulting by stream/non-stream."""
    if requested is None or str(requested).strip() == "":
        return "pcm" if stream else ("mp3" if FFMPEG_PATH else "wav")
    
    fmt = str(requested).strip().lower()
    allowed = {"pcm"} if stream else {"pcm", "wav", "mp3"}
    
    if fmt not in allowed:
        raise HTTPException(status_code=400, detail="response_format must be one of 'pcm', 'wav', 'mp3'")
    if stream and fmt != "pcm":
        raise HTTPException(status_code=400, detail="Streaming only supports response_format='pcm'")
    if (not stream) and fmt == "mp3" and not FFMPEG_PATH:
        raise HTTPException(status_code=400, detail="response_format='mp3' requires ffmpeg in PATH")
    
    return fmt


def _parse_generation_config(extra_body: Dict[str, Any], streaming: bool = True) -> GenerationConfig:
    """Parse generation config from request extra_body."""
    
    # Start with defaults based on streaming mode
    if streaming:
        gen_config = GenerationConfig.from_defaults()
    else:
        gen_config = GenerationConfig.for_non_streaming()
    
    # Override block sizes if provided
    if "block_sizes" in extra_body:
        block_sizes_raw = extra_body["block_sizes"]
        if isinstance(block_sizes_raw, int):
            gen_config.block_sizes = [int(block_sizes_raw)]
        else:
            gen_config.block_sizes = [int(x) for x in block_sizes_raw]
    
    # Override num_steps if provided
    if "num_steps" in extra_body:
        num_steps_raw = extra_body["num_steps"]
        if isinstance(num_steps_raw, int):
            gen_config.num_steps = [int(num_steps_raw) for _ in gen_config.block_sizes]
        else:
            gen_config.num_steps = [int(x) for x in num_steps_raw]
            if len(gen_config.num_steps) != len(gen_config.block_sizes):
                raise HTTPException(status_code=400, detail="num_steps list must match block_sizes length")
    
    # Override other parameters
    if "cfg_scale_text" in extra_body:
        gen_config.cfg_scale_text = float(extra_body["cfg_scale_text"])
    if "cfg_scale_speaker" in extra_body:
        gen_config.cfg_scale_speaker = float(extra_body["cfg_scale_speaker"])
    if "cfg_min_t" in extra_body:
        gen_config.cfg_min_t = float(extra_body["cfg_min_t"])
    if "cfg_max_t" in extra_body:
        gen_config.cfg_max_t = float(extra_body["cfg_max_t"])
    
    # Truncation and init scale
    if "truncation_factor" in extra_body:
        trunc_raw = extra_body["truncation_factor"]
        if isinstance(trunc_raw, (int, float)):
            gen_config.truncation_factor = [float(trunc_raw)] * len(gen_config.block_sizes)
        else:
            gen_config.truncation_factor = [float(x) for x in trunc_raw]
    
    if "init_scale" in extra_body:
        init_raw = extra_body["init_scale"]
        if isinstance(init_raw, (int, float)):
            gen_config.init_scale = [float(init_raw)] * len(gen_config.block_sizes)
        else:
            gen_config.init_scale = [float(x) for x in init_raw]
    
    # Rescale parameters
    if "rescale_k" in extra_body:
        gen_config.rescale_k = float(extra_body["rescale_k"])
    if "rescale_sigma" in extra_body:
        gen_config.rescale_sigma = float(extra_body["rescale_sigma"])
    
    # Speaker KV scaling
    if "speaker_kv_scale" in extra_body:
        gen_config.speaker_kv_scale = float(extra_body["speaker_kv_scale"])
    if "speaker_kv_min_t" in extra_body:
        gen_config.speaker_kv_min_t = float(extra_body["speaker_kv_min_t"])
    if "speaker_kv_max_layers" in extra_body:
        gen_config.speaker_kv_max_layers = int(extra_body["speaker_kv_max_layers"])
    
    # Early stop parameters
    if "early_stop_on_zero" in extra_body:
        gen_config.early_stop_on_zero = bool(extra_body["early_stop_on_zero"])
    if "zero_eps" in extra_body:
        gen_config.zero_eps = float(extra_body["zero_eps"])
    if "zero_tail_min_frac" in extra_body:
        gen_config.zero_tail_min_frac = float(extra_body["zero_tail_min_frac"])
    if "zero_tail_frames" in extra_body:
        gen_config.zero_tail_frames = int(extra_body["zero_tail_frames"])
    
    # Max text length
    if "max_text_length" in extra_body:
        gen_config.max_text_length = int(extra_body["max_text_length"])
    
    # Guidance mode
    if "guidance_mode" in extra_body:
        guidance_mode_raw = str(extra_body["guidance_mode"])
        try:
            gen_config.guidance_mode = GuidanceMode(guidance_mode_raw)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid guidance_mode '{guidance_mode_raw}'")
    
    return gen_config


def _initialize_server() -> None:
    """Initialize all server components."""
    global _speaker_manager
    
    print("🚀 Initializing Echo TTS Server...")
    
    # Load inference engine components
    print("  Loading model components...")
    engine.load_components()
    
    # Load compile caches
    print("  Loading compile caches...")
    engine.load_compile_cache(config.sampler.block_sizes)
    if config.sampler.block_sizes != [config.sampler.block_size_nonstream]:
        engine.load_compile_cache([config.sampler.block_size_nonstream])
    
    # Create speaker latent manager
    print("  Creating speaker latent manager...")
    _speaker_manager = create_speaker_latent_manager()
    _speaker_manager.set_patch_size(getattr(engine.model, "speaker_patch_size", 4))
    
    # Pre-warm specified voices
    if config.voice.prewarm_voices:
        print(f"  Pre-warming {len(config.voice.prewarm_voices)} voices...")
        _speaker_manager.prewarm_voices(config.voice.prewarm_voices, engine.device)
    
    # Run warmup generation if compile is enabled
    if config.model.use_compile:
        print("  Running warmup compilation...")
        _run_warmup()
    
    print("✅ Server initialization complete!")


def _run_warmup() -> None:
    """Run warmup generation to trigger torch.compile."""
    # Find a voice for warmup
    warmup_voice = config.warmup.warmup_voice
    if warmup_voice is None:
        available = get_speaker_manager().list_available_voices()
        if available:
            warmup_voice = available[0]["id"]
    
    if warmup_voice is None:
        print("  ⚠️ No voice available for warmup, skipping")
        return
    
    try:
        speaker_latent = get_speaker_manager().get_latent(warmup_voice, engine.device)
        gen_config = GenerationConfig.from_defaults()
        gen_config.early_stop_on_zero = False  # Don't early stop during warmup
        
        print(f"  Running warmup with voice '{warmup_voice}'...")
        for _ in engine.generate_streaming(
            text=config.warmup.warmup_text,
            speaker_latent=speaker_latent.latent,
            speaker_mask=speaker_latent.mask,
            gen_config=gen_config,
            rng_seed=0,
        ):
            pass
        
        # Also warmup non-streaming config
        gen_config_nonstream = GenerationConfig.for_non_streaming()
        gen_config_nonstream.early_stop_on_zero = False
        for _ in engine.generate_streaming(
            text=config.warmup.warmup_text,
            speaker_latent=speaker_latent.latent,
            speaker_mask=speaker_latent.mask,
            gen_config=gen_config_nonstream,
            rng_seed=0,
        ):
            pass
        
        print("  ✅ Warmup complete")
    except Exception as exc:
        print(f"  ⚠️ Warmup failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    _initialize_server()
    yield


# Create FastAPI app
app = FastAPI(
    title="Echo TTS API",
    description="High-quality text-to-speech with voice cloning",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class SpeechRequest(BaseModel):
    """Request model for speech synthesis."""
    model: str = Field(default="echo-tts", description="Model name (ignored)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice name or base64 audio")
    response_format: Optional[str] = Field(
        default=None,
        description="Output format: 'pcm' (streaming), 'wav', 'mp3' (non-streaming)"
    )
    stream: bool = Field(default=True, description="Enable streaming output")
    seed: int = Field(default=0, description="Random seed for reproducibility")
    extra_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sampler parameters"
    )


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/voices")
def list_voices() -> Dict[str, Any]:
    """
    List available voices.
    Returns voice IDs from audio_prompts, prompt_audio, and extra_prompt_audio directories.
    """
    voices = get_speaker_manager().list_available_voices()
    return {"object": "list", "data": voices}


@app.get("/v1/cache/stats")
def cache_stats() -> Dict[str, Any]:
    """Get speaker latent cache statistics."""
    return get_speaker_manager().get_cache_stats()


@app.post("/v1/audio/speech")
def create_speech(request: Request, payload: SpeechRequest = Body(...)) -> StreamingResponse:
    """
    Generate speech from text.
    
    Supports both streaming (PCM chunks) and non-streaming (WAV/MP3) output.
    """
    route_start = time.time()
    
    # Validate response format
    response_format = _resolve_response_format(payload.response_format, payload.stream)
    
    # Parse generation config
    gen_config = _parse_generation_config(payload.extra_body, streaming=payload.stream)
    
    # Get speaker latent
    speaker_latent = get_speaker_manager().get_latent(payload.voice, engine.device)
    
    if config.server.debug_logs:
        print(f"[route] Speaker latent loaded in {(time.time() - route_start)*1000:.2f}ms")
    
    # Handle text chunking
    chunking_raw = payload.extra_body.get("chunking_enabled", config.text.chunking_enabled)
    if isinstance(chunking_raw, str):
        chunking_enabled = chunking_raw.strip().lower() not in {"0", "false", "no", "off", ""}
    else:
        chunking_enabled = bool(chunking_raw)
    
    chunk_target_seconds = float(payload.extra_body.get("chunk_target_seconds", 30.0))
    chunk_min_seconds = float(payload.extra_body.get("chunk_min_seconds", 20.0))
    chunk_max_seconds = float(payload.extra_body.get("chunk_max_seconds", 40.0))
    
    # Split text into chunks
    if chunking_enabled:
        chunks = chunk_text_by_time(
            payload.input,
            target_seconds=chunk_target_seconds,
            min_seconds=chunk_min_seconds,
            max_seconds=chunk_max_seconds,
            chars_per_second=config.text.chunk_chars_per_second,
            words_per_second=config.text.chunk_words_per_second,
            normalize_exclamation=config.text.normalize_exclamation,
        )
        if not chunks:
            chunks = [payload.input]
    else:
        chunks = [payload.input]
    
    if config.server.debug_logs:
        print(f"[route] Text split into {len(chunks)} chunks")
    
    # Prepare configs for each chunk
    # For streaming with multiple chunks, use non-streaming config for chunks after the first
    override_secondary = (
        payload.stream
        and chunking_enabled
        and len(chunks) > 1
        and "block_sizes" not in payload.extra_body
        and "num_steps" not in payload.extra_body
    )
    
    if override_secondary:
        secondary_config = GenerationConfig.for_non_streaming()
        chunk_configs = [gen_config] + [secondary_config for _ in chunks[1:]]
    else:
        chunk_configs = [gen_config for _ in chunks]
    
    if payload.stream:
        # Streaming response
        disconnect_exception = type("ClientDisconnected", (Exception,), {})
        
        def _run_stream() -> Iterable[bytes]:
            for idx, chunk in enumerate(chunks):
                chunk_seed = payload.seed + idx
                chunk_config = chunk_configs[idx] if idx < len(chunk_configs) else gen_config
                
                if config.server.debug_logs:
                    print(f"[route] Generating chunk {idx+1}/{len(chunks)}")
                
                for block in engine.generate_streaming(
                    text=chunk,
                    speaker_latent=speaker_latent.latent,
                    speaker_mask=speaker_latent.mask,
                    gen_config=chunk_config,
                    rng_seed=chunk_seed,
                ):
                    yield block
        
        async def _drain_stream(stream_iter: Iterator[bytes]) -> AsyncIterator[bytes]:
            async for chunk in iterate_in_threadpool(stream_iter):
                if await request.is_disconnected():
                    if config.server.debug_logs:
                        print("[route] Client disconnected")
                    raise disconnect_exception()
                yield chunk
        
        async def _generator() -> AsyncIterator[bytes]:
            stream_iter = None
            try:
                stream_iter = _run_stream()
                async for chunk in _drain_stream(stream_iter):
                    yield chunk
            except disconnect_exception:
                pass
            finally:
                if stream_iter is not None:
                    close_fn = getattr(stream_iter, "close", None)
                    if close_fn:
                        try:
                            close_fn()
                        except Exception:
                            pass
        
        return StreamingResponse(
            _generator(),
            media_type="application/octet-stream",
            headers={"X-Audio-Sample-Rate": str(config.sample_rate)},
        )
    
    else:
        # Non-streaming response
        collected = bytearray()
        
        for idx, chunk in enumerate(chunks):
            chunk_seed = payload.seed + idx
            chunk_config = chunk_configs[idx] if idx < len(chunk_configs) else gen_config
            
            if config.server.debug_logs:
                print(f"[route] Generating chunk {idx+1}/{len(chunks)} (non-stream)")
            
            audio_bytes = engine.generate_full(
                text=chunk,
                speaker_latent=speaker_latent.latent,
                speaker_mask=speaker_latent.mask,
                gen_config=chunk_config,
                rng_seed=chunk_seed,
            )
            collected.extend(audio_bytes)
        
        audio_data = bytes(collected)
        
        # Convert to requested format
        if response_format == "wav":
            response_bytes = engine.pcm_to_wav(audio_data, config.sample_rate)
            media_type = "audio/wav"
        elif response_format == "mp3":
            response_bytes = _encode_mp3_from_pcm(audio_data, config.sample_rate)
            media_type = "audio/mpeg"
        else:
            response_bytes = audio_data
            media_type = "application/octet-stream"
        
        return StreamingResponse(
            content=iter([response_bytes]),
            media_type=media_type,
            headers={"X-Audio-Sample-Rate": str(config.sample_rate)},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )

