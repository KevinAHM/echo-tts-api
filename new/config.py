"""
Configuration module for Echo TTS API.
All environment variables and settings are centralized here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    """Convert string dtype name to torch.dtype."""
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


@dataclass
class ModelConfig:
    """Model and inference configuration."""
    
    # Model repositories
    model_repo: str = os.getenv("ECHO_MODEL_REPO", "jordand/echo-tts-base")
    pca_repo: str = os.getenv("ECHO_PCA_REPO", os.getenv("ECHO_MODEL_REPO", "jordand/echo-tts-base"))
    fish_repo: str = os.getenv("ECHO_FISH_REPO", "jordand/fish-s1-dac-min")
    
    # Device configuration
    device: str = os.getenv("ECHO_DEVICE", "cuda")
    fish_device: str = os.getenv("ECHO_FISH_DEVICE", os.getenv("ECHO_DEVICE", "cuda"))
    
    # Dtype configuration
    model_dtype_str: str = os.getenv("ECHO_MODEL_DTYPE", "bfloat16")
    fish_dtype_str: str = os.getenv("ECHO_FISH_DTYPE", "float32")
    
    # Compile settings
    use_compile: bool = os.getenv("ECHO_COMPILE", "1") == "1"
    compile_ae: bool = os.getenv("ECHO_COMPILE_AE", "1") == "1"
    
    # Cache settings
    cache_speaker_on_gpu: bool = os.getenv("ECHO_CACHE_SPEAKER_ON_GPU", "0") == "1"
    cache_version: str = os.getenv("ECHO_CACHE_VERSION", "v1_0")
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("ECHO_CACHE_DIR", "/tmp")))
    
    # Speaker latent settings
    max_speaker_latent_length: int = int(os.getenv("ECHO_MAX_SPEAKER_LATENT_LENGTH", "6400"))
    speaker_latent_buckets: str = os.getenv("ECHO_SPEAKER_LATENT_BUCKETS", "128,256,512,1024,2048")
    warmup_speaker_buckets: str = os.getenv("ECHO_WARMUP_SPEAKER_BUCKETS", "128,256,512,1024,2048")
    
    @property
    def model_dtype(self) -> torch.dtype | None:
        return _dtype_from_name(self.model_dtype_str)
    
    @property
    def fish_dtype(self) -> torch.dtype | None:
        return _dtype_from_name(self.fish_dtype_str)


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    
    lora_first_block: bool = os.getenv("ECHO_LORA_FIRST_BLOCK", "0") == "1"
    lora_repo: str = os.getenv("ECHO_LORA_REPO", "")
    lora_hf_name: str = os.getenv("ECHO_LORA_HF_NAME", "lora_lr1e-5_skip02_noema_huber0005_cfgdecay_90ksteps.safetensors")
    lora_scale: float = float(os.getenv("ECHO_LORA_SCALE", "1.0"))
    lora_alpha: float = float(os.getenv("ECHO_LORA_ALPHA", "32.0"))
    compile_lora_only: bool = os.getenv("ECHO_COMPILE_LORA_ONLY", "0") == "1"


@dataclass
class SamplerConfig:
    """Sampler configuration with performance presets."""
    
    # Performance preset
    performance_preset: str = os.getenv("ECHO_PERFORMANCE_PRESET", "default").strip().lower().replace("-", "_")
    
    # Default sampler settings (may be overridden by preset)
    cfg_text: float = 3.0
    cfg_speaker: float = 8.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    
    # Early stop settings
    early_stop: bool = True
    zero_eps: float = 2.0e-2
    zero_tail_frames: int = 16
    zero_tail_min_frac: float = 0.95
    
    # Non-streaming defaults
    block_size_nonstream: int = 640
    num_steps_nonstream: int = int(os.getenv("ECHO_NUM_STEPS_NONSTREAM", "20"))
    
    def __post_init__(self):
        """Apply performance preset to block sizes and steps."""
        presets = {
            "default": {"block_sizes": [32, 128, 480], "num_steps": [8, 15, 20]},
            "low_mid": {"block_sizes": [32, 128, 480], "num_steps": [8, 10, 15]},
            "low": {"block_sizes": [32, 64, 272, 272], "num_steps": [8, 10, 15, 15]},
        }
        
        if self.performance_preset in presets:
            preset = presets[self.performance_preset]
            self.block_sizes = list(preset["block_sizes"])
            self.num_steps = list(preset["num_steps"])
        else:
            self.block_sizes = [32, 128, 480]
            self.num_steps = [8, 15, 20]
            if self.performance_preset not in {"", "default"}:
                print(f"⚠️ Unknown ECHO_PERFORMANCE_PRESET '{self.performance_preset}'; using default.")


@dataclass 
class TextConfig:
    """Text processing configuration."""
    
    chunking_enabled: bool = os.getenv("ECHO_CHUNKING", "1") == "1"
    chunk_chars_per_second: float = float(os.getenv("ECHO_CHUNK_CHARS_PER_SECOND", "14"))
    chunk_words_per_second: float = float(os.getenv("ECHO_CHUNK_WORDS_PER_SECOND", "2.7"))
    normalize_exclamation: bool = os.getenv("ECHO_NORMALIZE_EXCLAMATION", "1") == "1"


@dataclass
class VoiceConfig:
    """Voice directory configuration."""
    
    voice_dirs: List[Path] = field(default_factory=lambda: [
        Path(__file__).resolve().parent.parent / "audio_prompts",
        Path(__file__).resolve().parent.parent / "prompt_audio",
        Path(__file__).resolve().parent.parent / "extra_prompt_audio",
    ])
    folder_support: bool = os.getenv("ECHO_FOLDER_SUPPORT", "1") == "1"
    audio_extensions: set = field(default_factory=lambda: {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"})
    
    # Voices to pre-warm at startup (comma-separated)
    prewarm_voices: List[str] = field(default_factory=lambda: [
        v.strip() for v in os.getenv("ECHO_PREWARM_VOICES", "").split(",") if v.strip()
    ])


@dataclass
class WarmupConfig:
    """Warmup configuration."""
    
    warmup_voice: str = os.getenv("ECHO_WARMUP_VOICE", "maya_ref")
    warmup_text: str = os.getenv("ECHO_WARMUP_TEXT", "[S1] Warmup compile run.")


@dataclass
class ServerConfig:
    """Server configuration."""
    
    host: str = os.getenv("ECHO_HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug_logs: bool = os.getenv("ECHO_DEBUG_LOGS", "1") == "1"


@dataclass
class Config:
    """Main configuration container."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    text: TextConfig = field(default_factory=TextConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # Audio constants
    sample_rate: int = 44_100
    
    def __post_init__(self):
        """Apply LoRA overrides if enabled."""
        if self.lora.lora_first_block:
            self.sampler.num_steps[0] = 5  # 5 steps for first block when using LoRA
            self.model.model_repo = self.lora.lora_repo


# Global config instance
config = Config()


# Set torch compile cache directory
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

