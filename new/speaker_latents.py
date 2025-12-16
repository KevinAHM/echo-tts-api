"""
Speaker latent generation and caching module.
Handles loading voice audio files, encoding to latents, and caching.
"""

import base64
import math
import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import config
from inference import (
    PCAState,
    get_speaker_latent_and_mask,
    load_audio,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
)
from autoencoder import DAC


@dataclass
class SpeakerLatent:
    """Container for speaker latent and mask tensors."""
    latent: torch.Tensor
    mask: torch.Tensor
    cache_key: str
    
    def to(self, device: torch.device) -> "SpeakerLatent":
        """Move tensors to specified device."""
        return SpeakerLatent(
            latent=self.latent.to(device),
            mask=self.mask.to(device),
            cache_key=self.cache_key,
        )


class SpeakerLatentManager:
    """
    Manages speaker latent generation and caching.
    
    Features:
    - CPU cache for all computed latents
    - Optional GPU cache for frequently used voices
    - Support for single files and directories of audio
    - Pre-warming of specified voices at startup
    """
    
    def __init__(
        self,
        fish_ae: DAC,
        pca_state: PCAState,
        voice_dirs: List[Path],
        audio_extensions: set,
        folder_support: bool = True,
        cache_on_gpu: bool = False,
        max_latent_length: int = 6400,
    ):
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.voice_dirs = voice_dirs
        self.audio_extensions = audio_extensions
        self.folder_support = folder_support
        self.cache_on_gpu = cache_on_gpu
        self.max_latent_length = max_latent_length
        
        # Caches
        self._cpu_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._gpu_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # Model patch size (will be set when used with model)
        self._patch_size: int = 4
    
    def set_patch_size(self, patch_size: int) -> None:
        """Set the model's speaker patch size for padding calculations."""
        self._patch_size = patch_size
    
    def _device_key(self, device: torch.device) -> str:
        """Stable per-device key to avoid mismatch between 'cuda' and 'cuda:0'."""
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            return f"cuda:{idx}"
        return str(device)
    
    def _voice_roots(self) -> List[Path]:
        """Resolved voice roots (skip paths that cannot be resolved)."""
        roots: List[Path] = []
        for directory in self.voice_dirs:
            try:
                roots.append(directory.resolve())
            except (OSError, RuntimeError):
                continue
        return roots
    
    def _is_path_within_voice_dirs(self, path: Path, roots: Iterable[Path]) -> bool:
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
    
    def find_voice_file(self, name: str) -> Optional[Path]:
        """Find a voice file or directory by name in voice directories."""
        sanitized = name.strip()
        if not sanitized:
            return None
        
        roots = self._voice_roots()
        name_path = Path(sanitized)
        
        for directory in self.voice_dirs:
            if not directory.exists():
                continue
            
            # Check for exact directory match if folder support is enabled
            if self.folder_support:
                dir_path = directory / sanitized
                if self._is_path_within_voice_dirs(dir_path, roots) and dir_path.is_dir():
                    return dir_path
            
            # Support callers providing the full filename with extension
            direct_path = directory / sanitized
            if (
                name_path.suffix
                and name_path.suffix.lower() in self.audio_extensions
                and self._is_path_within_voice_dirs(direct_path, roots)
                and direct_path.is_file()
            ):
                return direct_path
            
            # Look for a matching stem with an allowed extension
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self.audio_extensions:
                    continue
                if path.stem != sanitized:
                    continue
                if not self._is_path_within_voice_dirs(path, roots):
                    continue
                return path
        
        return None
    
    def list_available_voices(self) -> List[Dict[str, Any]]:
        """
        Enumerate available voice ids (file stems and folder names).
        Returns list of voice metadata dictionaries.
        """
        voices: List[Dict[str, Any]] = []
        seen: set[str] = set()
        roots = self._voice_roots()
        
        for directory in self.voice_dirs:
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
                    if entry.suffix.lower() not in self.audio_extensions:
                        continue
                    voice_name = entry.stem
                    if voice_name in seen:
                        continue
                    if not self._is_path_within_voice_dirs(entry, roots):
                        continue
                    voices.append({
                        "object": "voice",
                        "id": voice_name,
                        "name": voice_name,
                        "metadata": {"source": directory.name, "type": "file"},
                    })
                    seen.add(voice_name)
                    
                elif entry.is_dir() and self.folder_support:
                    voice_name = entry.name
                    if voice_name in seen:
                        continue
                    if not self._is_path_within_voice_dirs(entry, roots):
                        continue
                    
                    # Check if directory has audio files
                    has_audio = False
                    try:
                        for child in entry.iterdir():
                            if child.is_file() and child.suffix.lower() in self.audio_extensions:
                                has_audio = True
                                break
                    except (OSError, RuntimeError):
                        continue
                    
                    if not has_audio:
                        continue
                    
                    voices.append({
                        "object": "voice",
                        "id": voice_name,
                        "name": voice_name,
                        "metadata": {"source": directory.name, "type": "folder"},
                    })
                    seen.add(voice_name)
        
        return voices
    
    def _decode_base64_audio(self, encoded: str) -> Tuple[torch.Tensor, str]:
        """Decode base64 encoded audio to tensor."""
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 voice: {exc}") from exc
        
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
    
    def _encode_audio_to_latent(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio tensor to speaker latent."""
        with torch.inference_mode():
            fish_device = next(self.fish_ae.parameters()).device
            speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                self.fish_ae,
                self.pca_state,
                audio.to(device=fish_device, dtype=self.fish_ae.dtype),
                max_speaker_latent_length=self.max_latent_length,
            )
        return speaker_latent, speaker_mask
    
    def _pad_to_patch_size(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad latent and mask to be divisible by patch size."""
        target_len = int(math.ceil(latent.shape[1] / self._patch_size) * self._patch_size)
        if target_len != latent.shape[1]:
            pad_amt = target_len - latent.shape[1]
            latent = torch.nn.functional.pad(latent, (0, 0, 0, pad_amt))
            mask = torch.nn.functional.pad(mask, (0, pad_amt))
        return latent, mask
    
    def _load_directory_latent(self, directory_path: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load all audio files from a directory, concatenate with 1s gaps,
        then encode once (trimmed to 5 minutes max).
        """
        voice_roots = self._voice_roots()
        if not self._is_path_within_voice_dirs(directory_path, voice_roots):
            raise ValueError("Invalid voice directory")
        
        if not directory_path.is_dir():
            raise ValueError(f"Voice directory not found: {directory_path}")
        
        # Find all audio files in the directory
        audio_files: List[Path] = []
        for candidate in directory_path.iterdir():
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self.audio_extensions:
                continue
            if not self._is_path_within_voice_dirs(candidate, voice_roots):
                continue
            audio_files.append(candidate)
        
        audio_files = sorted(audio_files)
        
        if not audio_files:
            raise ValueError(f"No audio files found in directory: {directory_path}")
        
        print(f"[speaker_latents] Found {len(audio_files)} audio files in {directory_path.name}")
        
        # Maximum duration: 5 minutes = 300 seconds
        MAX_DURATION_SECONDS = 300
        MAX_SAMPLES = int(MAX_DURATION_SECONDS * config.sample_rate)
        SILENCE = torch.zeros((1, config.sample_rate), dtype=torch.float32)
        
        segments: List[torch.Tensor] = []
        total_samples = 0
        
        for idx, audio_file in enumerate(audio_files):
            audio = load_audio(str(audio_file))
            audio_samples = audio.shape[-1]
            
            if total_samples >= MAX_SAMPLES:
                print(f"[speaker_latents] Reached max duration (5 min), stopping at {audio_file.name}")
                break
            
            remaining_samples = MAX_SAMPLES - total_samples
            if audio_samples > remaining_samples:
                audio = audio[..., :remaining_samples]
                audio_samples = remaining_samples
                print(f"[speaker_latents] Trimming {audio_file.name} to fit 5 min limit")
            
            print(f"[speaker_latents] Processing {audio_file.name} ({audio_samples / config.sample_rate:.2f}s)")
            segments.append(audio)
            total_samples += audio_samples
            
            # Add 1s of silence between clips
            if idx < len(audio_files) - 1 and total_samples < MAX_SAMPLES:
                silence_len = min(config.sample_rate, MAX_SAMPLES - total_samples)
                if silence_len > 0:
                    segments.append(SILENCE[:, :silence_len])
                    total_samples += silence_len
        
        if not segments:
            raise ValueError(f"No valid audio could be loaded from directory: {directory_path}")
        
        combined_audio = torch.cat(segments, dim=1)
        speaker_latent, speaker_mask = self._encode_audio_to_latent(combined_audio)
        
        # Trim to max length if needed
        if speaker_latent.shape[1] > self.max_latent_length:
            print(f"[speaker_latents] Trimming latent from {speaker_latent.shape[1]} to {self.max_latent_length}")
            speaker_latent = speaker_latent[:, :self.max_latent_length]
            speaker_mask = speaker_mask[:, :self.max_latent_length]
        
        cache_key = f"dir:{directory_path.resolve()}:{len(audio_files)}"
        
        print(
            f"[speaker_latents] Combined {len(audio_files)} files into latent of length {speaker_latent.shape[1]} "
            f"({combined_audio.shape[-1] / config.sample_rate:.2f}s total)"
        )
        
        return speaker_latent, speaker_mask, cache_key
    
    def get_latent(
        self,
        voice: str,
        target_device: torch.device,
    ) -> SpeakerLatent:
        """
        Get speaker latent for a voice.
        
        Args:
            voice: Voice name, path, or base64 encoded audio
            target_device: Device to place the returned tensors on
            
        Returns:
            SpeakerLatent containing latent tensor, mask tensor, and cache key
        """
        target_device_key = self._device_key(target_device)
        
        # Check if voice is a directory or file
        local_path = self.find_voice_file(voice)
        is_directory = local_path is not None and local_path.is_dir()
        
        # Handle directory case
        if is_directory and self.folder_support:
            audio_files = []
            for ext in self.audio_extensions:
                audio_files.extend(local_path.glob(f"*{ext}"))
            cache_key = f"dir:{local_path.resolve()}:{len(sorted(audio_files))}"
            
            # Check CPU cache
            if cache_key in self._cpu_cache:
                cached_latent, cached_mask = self._cpu_cache[cache_key]
                
                # Check GPU cache
                gpu_hit = self._gpu_cache.get(cache_key, {}).get(target_device_key) if self.cache_on_gpu else None
                if gpu_hit:
                    return SpeakerLatent(latent=gpu_hit[0], mask=gpu_hit[1], cache_key=cache_key)
                
                result = (cached_latent.to(target_device), cached_mask.to(target_device))
                if self.cache_on_gpu:
                    self._gpu_cache.setdefault(cache_key, {})[target_device_key] = result
                return SpeakerLatent(latent=result[0], mask=result[1], cache_key=cache_key)
            
            # Load and encode directory
            speaker_latent, speaker_mask, cache_key = self._load_directory_latent(local_path)
            speaker_latent, speaker_mask = self._pad_to_patch_size(speaker_latent, speaker_mask)
            
            # Cache on CPU
            self._cpu_cache[cache_key] = (speaker_latent.cpu(), speaker_mask.cpu())
            
            if self.cache_on_gpu:
                self._gpu_cache.setdefault(cache_key, {})[target_device_key] = (
                    speaker_latent.to(target_device),
                    speaker_mask.to(target_device),
                )
            
            return SpeakerLatent(
                latent=self._cpu_cache[cache_key][0].to(target_device),
                mask=self._cpu_cache[cache_key][1].to(target_device),
                cache_key=cache_key,
            )
        
        # Handle single file case
        if local_path is not None:
            cache_key = f"file:{local_path.resolve()}"
            audio = load_audio(str(local_path))
        else:
            # Try base64 decode
            audio, cache_key = self._decode_base64_audio(voice)
        
        # Check CPU cache
        if cache_key in self._cpu_cache:
            cached_latent, cached_mask = self._cpu_cache[cache_key]
            
            # Check GPU cache
            gpu_hit = self._gpu_cache.get(cache_key, {}).get(target_device_key) if self.cache_on_gpu else None
            if gpu_hit:
                return SpeakerLatent(latent=gpu_hit[0], mask=gpu_hit[1], cache_key=cache_key)
            
            result = (cached_latent.to(target_device), cached_mask.to(target_device))
            if self.cache_on_gpu:
                self._gpu_cache.setdefault(cache_key, {})[target_device_key] = result
            return SpeakerLatent(latent=result[0], mask=result[1], cache_key=cache_key)
        
        # Encode audio
        speaker_latent, speaker_mask = self._encode_audio_to_latent(audio)
        speaker_latent, speaker_mask = self._pad_to_patch_size(speaker_latent, speaker_mask)
        
        # Cache on CPU
        self._cpu_cache[cache_key] = (speaker_latent.cpu(), speaker_mask.cpu())
        
        if self.cache_on_gpu:
            self._gpu_cache.setdefault(cache_key, {})[target_device_key] = (
                speaker_latent.to(target_device),
                speaker_mask.to(target_device),
            )
        
        return SpeakerLatent(
            latent=self._cpu_cache[cache_key][0].to(target_device),
            mask=self._cpu_cache[cache_key][1].to(target_device),
            cache_key=cache_key,
        )
    
    def prewarm_voices(self, voices: List[str], target_device: torch.device) -> None:
        """Pre-compute and cache latents for a list of voices."""
        if not voices:
            print("⚠️ No voices to prewarm")
            return
        
        print(f"🔥 Pre-warming {len(voices)} voices...")
        import time
        
        for voice in voices:
            try:
                t0 = time.time()
                speaker_latent = self.get_latent(voice, target_device)
                elapsed = (time.time() - t0) * 1000
                print(f"  ✅ Warmed '{voice}' in {elapsed:.1f}ms, latent shape: {speaker_latent.latent.shape}")
            except Exception as exc:
                print(f"  ❌ Failed to warm '{voice}': {exc}")
        
        print("✅ Voice prewarming complete")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "cpu_cache_size": len(self._cpu_cache),
            "gpu_cache_size": sum(len(v) for v in self._gpu_cache.values()),
            "cached_voices": list(self._cpu_cache.keys()),
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cpu_cache.clear()
        self._gpu_cache.clear()


def create_speaker_latent_manager() -> SpeakerLatentManager:
    """Factory function to create a SpeakerLatentManager with default config."""
    fish_ae = load_fish_ae_from_hf(
        repo_id=config.model.fish_repo,
        device=config.model.fish_device,
        dtype=config.model.fish_dtype,
        compile=config.model.compile_ae,
    )
    
    pca_state = load_pca_state_from_hf(
        repo_id=config.model.pca_repo,
        device=config.model.device,
    )
    
    return SpeakerLatentManager(
        fish_ae=fish_ae,
        pca_state=pca_state,
        voice_dirs=config.voice.voice_dirs,
        audio_extensions=config.voice.audio_extensions,
        folder_support=config.voice.folder_support,
        cache_on_gpu=config.model.cache_speaker_on_gpu,
        max_latent_length=config.model.max_speaker_latent_length,
    )

