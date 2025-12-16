"""
Inference engine for Echo TTS.
Handles model loading, compilation, and audio generation.
"""

import gzip
import io
import math
import time
import wave
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from .config import config
from autoencoder import DAC
from inference import (
    PCAState,
    ae_decode as _base_ae_decode,
    find_flattening_point,
    get_text_input_ids_and_mask,
    load_fish_ae_from_hf,
    load_model_from_hf,
    load_pca_state_from_hf,
)
from model import EchoDiT
from samplers import (
    GuidanceMode,
    _get_first_n_kv_cache,
    _get_uncond_text_input_ids_and_mask,
    _multiply_speaker_kv_cache,
    _temporal_score_rescale,
)
from utils import preprocess_text


@dataclass
class GenerationConfig:
    """Configuration for a single generation request."""
    
    block_sizes: List[int]
    num_steps: List[int]
    cfg_scale_text: float
    cfg_scale_speaker: float
    cfg_min_t: float
    cfg_max_t: float
    truncation_factor: Optional[List[float]] = None
    init_scale: Optional[List[float]] = None
    rescale_k: Optional[float] = None
    rescale_sigma: Optional[float] = None
    speaker_kv_scale: Optional[float] = None
    speaker_kv_min_t: Optional[float] = None
    speaker_kv_max_layers: Optional[int] = None
    early_stop_on_zero: bool = True
    zero_eps: float = 2.0e-2
    zero_tail_min_frac: float = 0.95
    zero_tail_frames: int = 16
    guidance_mode: GuidanceMode = GuidanceMode.INDEPENDENT
    max_text_length: int = 768
    
    @classmethod
    def from_defaults(cls) -> "GenerationConfig":
        """Create config with default values from global config."""
        return cls(
            block_sizes=list(config.sampler.block_sizes),
            num_steps=list(config.sampler.num_steps),
            cfg_scale_text=config.sampler.cfg_text,
            cfg_scale_speaker=config.sampler.cfg_speaker,
            cfg_min_t=config.sampler.cfg_min_t,
            cfg_max_t=config.sampler.cfg_max_t,
            early_stop_on_zero=config.sampler.early_stop,
            zero_eps=config.sampler.zero_eps,
            zero_tail_min_frac=config.sampler.zero_tail_min_frac,
            zero_tail_frames=config.sampler.zero_tail_frames,
        )
    
    @classmethod
    def for_non_streaming(cls) -> "GenerationConfig":
        """Create config optimized for non-streaming generation."""
        cfg = cls.from_defaults()
        cfg.block_sizes = [config.sampler.block_size_nonstream]
        cfg.num_steps = [config.sampler.num_steps_nonstream]
        return cfg


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
        
        default_ctx = 160
        delay_latents = math.ceil(getattr(fish_ae, "delay", 0) / self.samples_per_latent)
        self.max_latent_ctx = max_latent_ctx or (default_ctx + delay_latents)
        
        self._tail_latents: Optional[torch.Tensor] = None
        self._emitted_samples: int = 0
        self._total_latents_seen: int = 0
    
    def decode_next(self, latent_block: torch.Tensor, flattening_point: int | None = None) -> torch.Tensor:
        """Decode next block with context; return only new audio in global timeline."""
        block_len = latent_block.shape[1]
        self._total_latents_seen += block_len
        
        max_len = self.max_latent_ctx + block_len
        
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
            audio = self._ae_decode_with_flatten(self._tail_latents, local_flatten)
        
        audio_len = audio.shape[-1]
        total_samples_global = global_start_samples + audio_len
        
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
    
    def _ae_decode_with_flatten(self, latent: torch.Tensor, flattening_point: int) -> torch.Tensor:
        """Decode with truncation to flattening point."""
        if flattening_point <= 0:
            return latent.new_zeros((latent.shape[0], 1, 0))
        latent = latent[:, :flattening_point, ...]
        return _base_ae_decode(self.fish_ae, self.pca_state, latent)
    
    def is_finished(self, flattening_point: int | None) -> bool:
        if flattening_point is None:
            return False
        return self._emitted_samples >= flattening_point * self.samples_per_latent


class InferenceEngine:
    """
    Main inference engine for Echo TTS.
    
    Handles:
    - Model loading and compilation
    - Streaming and non-streaming generation
    - KV cache management
    - Compile cache persistence
    """
    
    def __init__(self):
        self._model: Optional[EchoDiT] = None
        self._model_lora: Optional[EchoDiT] = None
        self._fish_ae: Optional[DAC] = None
        self._pca_state: Optional[PCAState] = None
        
        self._compile_disabled: bool = False
        self._loaded_cache_paths: set[Path] = set()
        self._saved_cache_paths: set[Path] = set()
        self._warmup_ran: bool = False
    
    @property
    def model(self) -> EchoDiT:
        """Get the main model, loading if necessary."""
        if self._model is None:
            self.load_components()
        return self._model
    
    @property
    def fish_ae(self) -> DAC:
        """Get the Fish AE decoder, loading if necessary."""
        if self._fish_ae is None:
            self.load_components()
        return self._fish_ae
    
    @property
    def pca_state(self) -> PCAState:
        """Get the PCA state, loading if necessary."""
        if self._pca_state is None:
            self.load_components()
        return self._pca_state
    
    @property
    def device(self) -> torch.device:
        """Get the model device."""
        return self.model.device
    
    def _ensure_cache_aliases(self, model: torch.nn.Module) -> None:
        """Ensure legacy KV cache method names exist."""
        if hasattr(model, "get_kv_cache_text") and not hasattr(model, "get_text_kv_cache"):
            model.get_text_kv_cache = model.get_kv_cache_text
        if hasattr(model, "get_kv_cache_speaker") and not hasattr(model, "get_speaker_kv_cache"):
            model.get_speaker_kv_cache = model.get_kv_cache_speaker
        if hasattr(model, "get_kv_cache_latent") and not hasattr(model, "get_latent_kv_cache"):
            model.get_latent_kv_cache = model.get_kv_cache_latent
    
    def load_components(self, force_reinit: bool = False, force_compile: Optional[bool] = None) -> None:
        """Load or reload all model components."""
        if force_reinit:
            self._model = None
            self._model_lora = None
            self._fish_ae = None
            self._pca_state = None
        
        lora_requested = config.lora.lora_first_block and bool(config.lora.lora_hf_name)
        compile_flag = config.model.use_compile and not self._compile_disabled if force_compile is None else force_compile
        ae_compile_flag = config.model.compile_ae and not self._compile_disabled if force_compile is None else force_compile
        base_compile_flag = compile_flag and not (config.lora.compile_lora_only and lora_requested)
        lora_compile_flag = compile_flag
        
        if self._model is None:
            self._model = load_model_from_hf(
                repo_id=config.model.model_repo,
                device=config.model.device,
                dtype=config.model.model_dtype,
                compile=base_compile_flag,
            )
            self._ensure_cache_aliases(self._model)
            print(f"[engine] Loaded model: repo={config.model.model_repo} device={config.model.device} dtype={config.model.model_dtype_str} compile={base_compile_flag}")
        
        if lora_requested and self._model_lora is None:
            try:
                self._model_lora = load_model_from_hf(
                    repo_id=config.lora.lora_repo,
                    device=config.model.device,
                    dtype=config.model.model_dtype,
                    compile=lora_compile_flag,
                    lora_hf_hub_name=config.lora.lora_hf_name,
                    lora_scale=config.lora.lora_scale,
                    lora_alpha=config.lora.lora_alpha,
                )
                self._ensure_cache_aliases(self._model_lora)
                print(f"[engine] Loaded LoRA model: repo={config.lora.lora_repo} weight={config.lora.lora_hf_name}")
            except Exception as exc:
                self._model_lora = None
                print(f"⚠️ Failed to load LoRA model: {exc}")
        
        if self._fish_ae is None:
            self._fish_ae = load_fish_ae_from_hf(
                repo_id=config.model.fish_repo,
                device=config.model.fish_device,
                dtype=config.model.fish_dtype,
                compile=ae_compile_flag,
            )
            print(f"[engine] Loaded Fish AE: repo={config.model.fish_repo} device={config.model.fish_device}")
        
        if self._pca_state is None:
            self._pca_state = load_pca_state_from_hf(
                repo_id=config.model.pca_repo,
                device=config.model.device,
            )
            print(f"[engine] Loaded PCA state: repo={config.model.pca_repo}")
    
    def _cache_file_path(self, block_sizes: List[int]) -> Path:
        """Get path for compile cache file."""
        block_sizes_str = "_".join(map(str, block_sizes))
        return config.model.cache_dir / f"echo_tts_compile_cache_{block_sizes_str}_{config.model.cache_version}.gz"
    
    def load_compile_cache(self, block_sizes: List[int]) -> bool:
        """Load torch.compile cache from disk."""
        if not config.model.use_compile or self._compile_disabled:
            return False
        
        path = self._cache_file_path(block_sizes)
        if path in self._loaded_cache_paths or not path.exists():
            return False
        
        try:
            with gzip.open(path, "rb") as f:
                artifact_bytes = f.read()
            torch.compiler.load_cache_artifacts(artifact_bytes)
            self._loaded_cache_paths.add(path)
            print(f"✅ Loaded torch.compile cache from {path}")
            return True
        except Exception as exc:
            print(f"⚠️ Could not load compile cache {path}: {exc}")
            return False
    
    def save_compile_cache(self, block_sizes: List[int]) -> bool:
        """Save torch.compile cache to disk."""
        if not config.model.use_compile or self._compile_disabled:
            return False
        
        path = self._cache_file_path(block_sizes)
        if path in self._saved_cache_paths:
            return False
        
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is None:
                return False
            artifact_bytes, _ = artifacts
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wb") as f:
                f.write(artifact_bytes)
            self._saved_cache_paths.add(path)
            self._loaded_cache_paths.add(path)
            size_mb = len(artifact_bytes) / 1024 / 1024
            print(f"✅ Saved torch.compile cache to {path} ({size_mb:.1f} MB)")
            return True
        except Exception as exc:
            print(f"⚠️ Could not save compile cache {path}: {exc}")
            return False
    
    def disable_compile(self, reason: str) -> None:
        """Disable torch.compile for the remainder of the process."""
        if self._compile_disabled:
            return
        self._compile_disabled = True
        print(f"⚠️ Disabling torch.compile: {reason}")
    
    def _kv_cache_text(self, model: torch.nn.Module, text_input_ids: torch.Tensor, text_mask: torch.Tensor):
        if hasattr(model, "get_kv_cache_text"):
            return model.get_kv_cache_text(text_input_ids, text_mask)
        if hasattr(model, "get_text_kv_cache"):
            return model.get_text_kv_cache(text_input_ids, text_mask)
        raise RuntimeError("Model is missing text kv-cache helpers")
    
    def _kv_cache_speaker(self, model: torch.nn.Module, speaker_latent: torch.Tensor):
        if hasattr(model, "get_kv_cache_speaker"):
            return model.get_kv_cache_speaker(speaker_latent)
        if hasattr(model, "get_speaker_kv_cache"):
            return model.get_speaker_kv_cache(speaker_latent)
        raise RuntimeError("Model is missing speaker kv-cache helpers")
    
    def _kv_cache_latent(self, model: torch.nn.Module, prefix_latent: torch.Tensor):
        if hasattr(model, "get_kv_cache_latent"):
            return model.get_kv_cache_latent(prefix_latent)
        if hasattr(model, "get_latent_kv_cache"):
            return model.get_latent_kv_cache(prefix_latent)
        raise RuntimeError("Model is missing latent kv-cache helpers")
    
    @staticmethod
    def audio_to_pcm(audio: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM16 bytes."""
        audio = audio.detach().float().cpu().squeeze()
        audio = torch.clamp(audio, -1.0, 1.0)
        return (audio * 32767.0).to(torch.int16).numpy().tobytes()
    
    @staticmethod
    def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
        """Wrap PCM16 bytes in a WAV container."""
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm)
            return buffer.getvalue()
    
    @torch.inference_mode()
    def generate_streaming(
        self,
        text: str,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        gen_config: GenerationConfig,
        rng_seed: int = 0,
    ) -> Iterator[bytes]:
        """
        Generate audio in streaming mode, yielding PCM chunks.
        
        Args:
            text: Text to synthesize
            speaker_latent: Speaker latent tensor
            speaker_mask: Speaker mask tensor
            gen_config: Generation configuration
            rng_seed: Random seed
            
        Yields:
            PCM16 audio bytes for each block
        """
        text = preprocess_text(text, normalize_exclamation=config.text.normalize_exclamation)
        
        if gen_config.guidance_mode != GuidanceMode.INDEPENDENT:
            raise ValueError("Streaming only supports guidance_mode='independent'")
        
        base_model = self.model
        lora_model = self._model_lora if config.lora.lora_first_block and self._model_lora is not None else None
        
        base_device = base_model.device
        if lora_model is not None and lora_model.device != base_device:
            try:
                lora_model.to(base_device)
                torch.cuda.empty_cache()
            except RuntimeError:
                lora_model = None
        
        use_lora_first_block = lora_model is not None
        start_time_wall = time.time()
        
        model_first = lora_model if use_lora_first_block else base_model
        device, dtype = model_first.device, model_first.dtype
        batch_size = 1
        
        text_input_ids, text_mask = get_text_input_ids_and_mask(
            [text], min(gen_config.max_text_length, 768), device=device
        )
        
        torch.manual_seed(rng_seed)
        
        step_counts = gen_config.num_steps if isinstance(gen_config.num_steps, list) else [gen_config.num_steps] * len(gen_config.block_sizes)
        
        if gen_config.truncation_factor is None or isinstance(gen_config.truncation_factor, float):
            truncation_factors = [gen_config.truncation_factor] * len(gen_config.block_sizes)
        else:
            truncation_factors = gen_config.truncation_factor
        
        default_init_scale = 0.999
        if gen_config.init_scale is None or isinstance(gen_config.init_scale, float):
            init_scales = [default_init_scale if gen_config.init_scale is None else gen_config.init_scale] * len(gen_config.block_sizes)
        else:
            init_scales = gen_config.init_scale
        
        # Prepare unconditional inputs
        text_input_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(
            text_input_ids.shape[0], text_input_ids.shape[1], device=device
        )
        speaker_latent_uncond = torch.zeros_like(speaker_latent)
        speaker_mask_uncond = torch.zeros_like(speaker_mask)
        
        full_text_input_ids = torch.cat([text_input_ids, text_input_ids_uncond, text_input_ids], dim=0)
        full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
        full_speaker_latent = torch.cat([speaker_latent, speaker_latent, speaker_latent_uncond], dim=0)
        full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)
        
        # Lazy condition cache
        condition_caches: Dict[str, Any] = {"base": None, "lora": None}
        lora_cond_cache = None
        
        def _get_condition_caches(model_for_caches: torch.nn.Module, key: str):
            caches = condition_caches[key]
            if caches is None:
                text_ids_local = full_text_input_ids.to(device=model_for_caches.device)
                text_mask_local = text_mask.to(device=model_for_caches.device)
                full_text_mask_local = full_text_mask.to(device=model_for_caches.device)
                speaker_mask_local = speaker_mask.to(device=model_for_caches.device)
                full_speaker_mask_local = full_speaker_mask.to(device=model_for_caches.device)
                
                kv_cache_text_full = self._kv_cache_text(model_for_caches, text_ids_local, full_text_mask_local)
                kv_cache_text = _get_first_n_kv_cache(kv_cache_text_full, batch_size)
                
                kv_cache_speaker_full = self._kv_cache_speaker(
                    model_for_caches,
                    full_speaker_latent.to(device=model_for_caches.device, dtype=model_for_caches.dtype),
                )
                kv_cache_speaker = _get_first_n_kv_cache(kv_cache_speaker_full, batch_size)
                
                caches = (
                    kv_cache_text_full, kv_cache_text,
                    kv_cache_speaker_full, kv_cache_speaker,
                    text_mask_local, full_text_mask_local,
                    speaker_mask_local, full_speaker_mask_local,
                )
                condition_caches[key] = caches
            return caches
        
        def _get_lora_cond_cache(model_for_caches: torch.nn.Module):
            nonlocal lora_cond_cache
            if lora_cond_cache is None:
                text_ids_local = text_input_ids.to(device=model_for_caches.device)
                text_mask_local = text_mask.to(device=model_for_caches.device)
                speaker_mask_local = speaker_mask.to(device=model_for_caches.device)
                
                kv_cache_text = self._kv_cache_text(model_for_caches, text_ids_local, text_mask_local)
                kv_cache_speaker = self._kv_cache_speaker(
                    model_for_caches,
                    speaker_latent.to(device=model_for_caches.device, dtype=model_for_caches.dtype),
                )
                lora_cond_cache = (kv_cache_text, kv_cache_speaker, text_mask_local, speaker_mask_local)
            return lora_cond_cache
        
        streaming_decoder = StreamingAEDecoder(self.fish_ae, self.pca_state)
        
        prefix_total = sum(gen_config.block_sizes)
        prefix_latent = torch.zeros((batch_size, prefix_total, 80), device=device, dtype=torch.float32)
        
        pos_id = 0
        ttfb_reported = False
        
        for block_idx, (block_size, block_steps) in enumerate(zip(gen_config.block_sizes, step_counts)):
            block_start_time = time.time()
            block_trunc = truncation_factors[block_idx]
            block_init_scale = init_scales[block_idx]
            
            use_lora_block = use_lora_first_block and block_idx == 0
            model_for_block = lora_model if use_lora_block else base_model
            block_dtype = model_for_block.dtype
            block_device = model_for_block.device
            cache_key = "lora" if use_lora_block else "base"
            
            # Get condition caches
            if use_lora_block:
                kv_cache_text_full_lora, kv_cache_speaker_full_lora, block_text_mask_lora, block_speaker_mask_lora = _get_lora_cond_cache(model_for_block)
            else:
                (
                    kv_cache_text_full, kv_cache_text,
                    kv_cache_speaker_full, kv_cache_speaker,
                    block_text_mask, block_full_text_mask,
                    block_speaker_mask, block_full_speaker_mask,
                ) = _get_condition_caches(model_for_block, cache_key)
            
            t_schedule = torch.linspace(1.0, 0.0, block_steps + 1, device=block_device) * block_init_scale
            
            if gen_config.speaker_kv_scale is not None:
                target_kv = kv_cache_speaker_full_lora if use_lora_block else kv_cache_speaker_full
                _multiply_speaker_kv_cache(
                    target_kv,
                    gen_config.speaker_kv_scale,
                    text_input_ids.shape[-1],
                    gen_config.speaker_kv_max_layers,
                )
            
            # Compute latent KV cache
            if use_lora_block:
                full_prefix_latent = prefix_latent
                kv_cache_latent_full = self._kv_cache_latent(model_for_block, full_prefix_latent.to(block_dtype))
                kv_cache_latent = kv_cache_latent_full
            else:
                full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)
                kv_cache_latent_full = self._kv_cache_latent(model_for_block, full_prefix_latent.to(block_dtype))
                kv_cache_latent = _get_first_n_kv_cache(kv_cache_latent_full, batch_size)
            
            # Diffusion loop
            x_t = torch.randn((batch_size, block_size, 80), device=block_device, dtype=torch.float32)
            if block_trunc is not None:
                x_t = x_t * block_trunc
            
            for i in range(block_steps):
                t, t_next = t_schedule[i], t_schedule[i + 1]
                has_cfg = ((t >= gen_config.cfg_min_t) * (t <= gen_config.cfg_max_t)).item()
                
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
                        + gen_config.cfg_scale_text * (v_cond - v_uncond_text)
                        + gen_config.cfg_scale_speaker * (v_cond - v_uncond_speaker)
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
                
                if gen_config.rescale_k is not None and gen_config.rescale_sigma is not None:
                    v_pred = _temporal_score_rescale(v_pred, x_t, float(t), gen_config.rescale_k, gen_config.rescale_sigma)
                
                if (
                    gen_config.speaker_kv_scale is not None
                    and gen_config.speaker_kv_min_t is not None
                    and t_next < gen_config.speaker_kv_min_t
                    and t >= gen_config.speaker_kv_min_t
                ):
                    target_kv = kv_cache_speaker_full_lora if use_lora_block else kv_cache_speaker_full
                    _multiply_speaker_kv_cache(
                        target_kv,
                        1.0 / gen_config.speaker_kv_scale,
                        text_input_ids.shape[-1],
                        gen_config.speaker_kv_max_layers,
                    )
                
                x_t = x_t + v_pred * (t_next - t)
            
            prefix_latent[:, pos_id:pos_id + block_size] = x_t
            pos_id += block_size
            
            # Early stop detection
            early_stop = False
            if gen_config.early_stop_on_zero:
                tail_len = min(gen_config.zero_tail_frames, x_t.shape[1])
                tail = x_t[:, -tail_len:]
                tail_abs = torch.abs(tail)
                zero_frac = float((tail_abs <= gen_config.zero_eps).float().mean().item())
                tail_absmax = float(tail_abs.max().item())
                zero_ok = zero_frac >= gen_config.zero_tail_min_frac and tail_absmax <= gen_config.zero_eps
                if zero_ok:
                    early_stop = True
                    print(f"[early_stop] block {block_idx+1}/{len(gen_config.block_sizes)}")
            
            is_last_planned = (block_idx == len(gen_config.block_sizes) - 1)
            will_finish = early_stop or is_last_planned
            
            flatten_point = None
            if will_finish:
                prefix_latent_trim = prefix_latent[:, :pos_id]
                flatten_point = find_flattening_point(prefix_latent_trim[0], window_size=32, std_threshold=0.02)
                already_emitted_latents = streaming_decoder._emitted_samples // streaming_decoder.samples_per_latent
                flatten_point = max(flatten_point, already_emitted_latents)
            
            # Decode block audio
            new_audio = streaming_decoder.decode_next(
                x_t.to(next(self.fish_ae.parameters()).device),
                flattening_point=flatten_point,
            )
            
            # Log timing
            audio_samples = new_audio.numel() if new_audio.numel() > 0 else 0
            audio_duration_ms = (audio_samples / config.sample_rate) * 1000.0 if audio_samples > 0 else 0.0
            chunk_gen_time = time.time() - block_start_time
            chunk_gen_time_ms = chunk_gen_time * 1000.0
            rtfx = chunk_gen_time / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0
            
            if config.server.debug_logs:
                print(f"[chunk {block_idx+1}/{len(gen_config.block_sizes)}] RTFx={rtfx:.3f} gen={chunk_gen_time_ms:.2f}ms audio={audio_duration_ms:.2f}ms")
            
            if not ttfb_reported:
                ttfb_sec = time.time() - start_time_wall
                print(f"[stream] TTFB {ttfb_sec*1000:.2f}ms")
                ttfb_reported = True
            
            if new_audio.numel() > 0:
                yield self.audio_to_pcm(new_audio)
            
            if will_finish:
                break
        
        self.save_compile_cache(gen_config.block_sizes)
        torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def generate_full(
        self,
        text: str,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        gen_config: GenerationConfig,
        rng_seed: int = 0,
    ) -> bytes:
        """
        Generate complete audio (non-streaming).
        
        Args:
            text: Text to synthesize
            speaker_latent: Speaker latent tensor
            speaker_mask: Speaker mask tensor
            gen_config: Generation configuration
            rng_seed: Random seed
            
        Returns:
            Complete PCM16 audio bytes
        """
        # Use streaming internally and collect all chunks
        chunks = []
        for chunk in self.generate_streaming(text, speaker_latent, speaker_mask, gen_config, rng_seed):
            chunks.append(chunk)
        return b"".join(chunks)
    
    def warmup(self, voice_name: Optional[str] = None, warmup_text: Optional[str] = None) -> bool:
        """
        Run warmup generation to trigger torch.compile.
        
        Args:
            voice_name: Voice to use for warmup (uses default if None)
            warmup_text: Text to synthesize (uses default if None)
            
        Returns:
            True if warmup succeeded
        """
        if self._warmup_ran or not config.model.use_compile or self._compile_disabled:
            return True
        
        warmup_text = warmup_text or config.warmup.warmup_text
        print(f"🔥 Running warmup compile with text: {warmup_text[:50]}...")
        
        # This requires speaker latent to be passed in
        # The actual warmup will be done in the server which has access to SpeakerLatentManager
        self._warmup_ran = True
        return True


# Global engine instance
engine = InferenceEngine()

