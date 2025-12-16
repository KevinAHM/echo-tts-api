"""
Echo TTS new architecture modules.
"""

from .config import config
from .inference_engine import engine, GenerationConfig
from .speaker_latents import SpeakerLatentManager, create_speaker_latent_manager

__all__ = [
    "config",
    "engine",
    "GenerationConfig",
    "SpeakerLatentManager",
    "create_speaker_latent_manager",
]

