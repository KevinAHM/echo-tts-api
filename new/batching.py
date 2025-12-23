"""
MVP micro-batching scheduler for Echo TTS websocket streaming.

Goal: allow multiple concurrent websocket requests to share GPU work by batching
diffusion steps across requests that are at the same (block_idx, step_idx).

Scope (MVP):
- No LoRA path.
- Assumes GuidanceMode.INDEPENDENT (streaming path already enforces this).
- Uses per-request speaker latents, padded to a bucket length for stable shapes.
- Produces PCM16 chunks at block boundaries (same as current engine streaming).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import config
from .inference_engine import GenerationConfig, InferenceEngine, StreamingAEDecoder
from inference import find_flattening_point, get_text_input_ids_and_mask
from samplers import _get_uncond_text_input_ids_and_mask, _get_first_n_kv_cache
from utils import preprocess_text


def _parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in (csv or "").split(",")]
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def _pick_bucket(target_len: int, buckets: List[int], hard_max: int) -> int:
    if target_len <= 0:
        return min((buckets[0] if buckets else 4), hard_max)
    for b in buckets:
        if b >= target_len:
            return min(b, hard_max)
    return hard_max


def _pad_to_len(latent: torch.Tensor, mask: torch.Tensor, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # latent: (1, T, 80), mask: (1, T)
    cur = int(latent.shape[1])
    if cur == length:
        return latent, mask
    if cur > length:
        return latent[:, :length], mask[:, :length]
    pad = length - cur
    latent = torch.nn.functional.pad(latent, (0, 0, 0, pad))
    mask = torch.nn.functional.pad(mask, (0, pad))
    return latent, mask


@dataclass
class _ReqState:
    request_id: str
    text: str
    voice: str
    seed: int
    gen_config: GenerationConfig
    speaker_latent: torch.Tensor  # (1, L, 80) on model device
    speaker_mask: torch.Tensor  # (1, L) on model device

    # Output
    out_q: asyncio.Queue[Optional[bytes]] = field(default_factory=lambda: asyncio.Queue(maxsize=32))

    # Timing
    created_t: float = field(default_factory=time.perf_counter)
    ttfb_t: Optional[float] = None

    # Per-request diffusion state (initialized by scheduler)
    block_idx: int = 0
    step_idx: int = 0
    pos_id: int = 0
    prefix_latent: Optional[torch.Tensor] = None  # (1, prefix_total, 80) fp32
    x_t: Optional[torch.Tensor] = None  # (1, block_size, 80) fp32
    t_schedule: Optional[torch.Tensor] = None  # (steps+1,)

    # Conditioning caches (initialized once)
    text_input_ids: Optional[torch.Tensor] = None  # (1, 768)
    text_mask: Optional[torch.Tensor] = None  # (1, 768)
    full_text_input_ids: Optional[torch.Tensor] = None  # (3, 768)
    full_text_mask: Optional[torch.Tensor] = None  # (3, 768)
    full_speaker_latent: Optional[torch.Tensor] = None  # (3, L, 80)
    full_speaker_mask: Optional[torch.Tensor] = None  # (3, L)

    kv_cache_text_full: Any = None
    kv_cache_text: Any = None
    kv_cache_speaker_full: Any = None
    kv_cache_speaker: Any = None

    # Block-local latent cache (changes per block)
    kv_cache_latent_full: Any = None
    kv_cache_latent: Any = None

    # Streaming decode state
    decoder: Optional[StreamingAEDecoder] = None

    # Completion
    done: bool = False


class BatchScheduler:
    """
    Single-GPU scheduler that micro-batches diffusion steps across active requests.

    MVP constraints:
    - All requests are run on the single global model in `InferenceEngine`.
    - Best-effort grouping by identical sampler config; mismatched configs fall back to single-item groups.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        max_batch_size: int = 8,
        batch_wait_ms: float = 4.0,
    ) -> None:
        self._engine = engine
        self._max_batch_size = int(max(1, max_batch_size))
        self._batch_wait_ms = float(max(0.0, batch_wait_ms))
        self._in_q: asyncio.Queue[_ReqState] = asyncio.Queue()
        self._active: List[_ReqState] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

        self._speaker_buckets = _parse_int_list(config.model.speaker_latent_buckets)
        self._batch_warmup_buckets = _parse_int_list(
            (config.model.warmup_speaker_buckets or "").strip()
        ) or self._speaker_buckets

    @torch.inference_mode()
    def warmup(self, *, batch_size: Optional[int] = None, buckets: Optional[List[int]] = None) -> None:
        """
        Warm up torch.compile for common *batched* shapes used by the scheduler.

        Why: the regular server warmup compiles B=1 paths; the scheduler introduces new
        shapes (e.g. B>1, and thus CFG path uses 3B) which otherwise compile on first
        real request causing multi-minute TTFB spikes.
        """
        if not config.model.use_compile:
            return

        model = self._engine.model
        device, dtype = model.device, model.dtype

        B = int(batch_size or self._max_batch_size)
        B = max(1, B)

        # Use a couple representative speaker buckets (default: config warmup buckets).
        warm_buckets = list(buckets) if buckets is not None else list(self._batch_warmup_buckets)
        if not warm_buckets:
            warm_buckets = [256]

        # Fixed shapes used by API path
        text_len = min(768, 768)
        prefix_total = sum(config.sampler.block_sizes)
        block_size = int(config.sampler.block_sizes[0])

        # Build text inputs
        text_ids = torch.zeros((B, text_len), dtype=torch.int32, device=device)
        text_mask = torch.zeros((B, text_len), dtype=torch.bool, device=device)
        text_mask[:, 0] = True
        text_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(B, text_len, device=device)
        full_text_ids = torch.cat([text_ids, text_ids_uncond, text_ids], dim=0)
        full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)

        # Prefix latents are zeros during first block
        prefix_latent = torch.zeros((B, prefix_total, 80), device=device, dtype=torch.float32).to(dtype)
        full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)

        # One CFG step forward; compile the CFG path (3B) which dominates TTFB.
        t = (torch.ones((B * 3,), device=device) * 0.999).to(dtype)

        x_in = torch.randn((B * 3, block_size, 80), device=device, dtype=torch.float32).to(dtype)

        for spk_len in warm_buckets[:3]:  # cap to avoid excessive startup time
            spk_len = int(spk_len)
            if spk_len <= 0:
                continue

            speaker_latent = torch.zeros((B, spk_len, 80), device=device, dtype=dtype)
            speaker_mask = torch.zeros((B, spk_len), device=device, dtype=torch.bool)
            speaker_mask[:, : min(4, spk_len)] = True
            speaker_latent_uncond = torch.zeros_like(speaker_latent)
            speaker_mask_uncond = torch.zeros_like(speaker_mask)
            full_speaker_latent = torch.cat([speaker_latent, speaker_latent, speaker_latent_uncond], dim=0)
            full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

            # KV caches
            kv_text_full = self._engine._kv_cache_text(model, full_text_ids, full_text_mask)  # noqa: SLF001
            kv_speaker_full = self._engine._kv_cache_speaker(model, full_speaker_latent)  # noqa: SLF001
            kv_latent_full = self._engine._kv_cache_latent(model, full_prefix_latent)  # noqa: SLF001

            _ = model(
                x=x_in,
                t=t,
                text_mask=full_text_mask,
                speaker_mask=full_speaker_mask,
                start_pos=0,
                kv_cache_text=kv_text_full,
                kv_cache_speaker=kv_speaker_full,
                kv_cache_latent=kv_latent_full,
            )

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="echo-tts-batch-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._task = None

    async def submit(
        self,
        *,
        request_id: str,
        text: str,
        voice: str,
        seed: int,
        gen_config: GenerationConfig,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
    ) -> asyncio.Queue[Optional[bytes]]:
        # Normalize & validate here; heavy GPU work happens in scheduler loop.
        text = preprocess_text(text, normalize_exclamation=config.text.normalize_exclamation)

        # Bucket speaker length for stable shapes / batching.
        # NOTE: speaker_latent is already padded to patch size (multiple of 4).
        target_len = int(speaker_latent.shape[1])
        bucket = _pick_bucket(target_len, self._speaker_buckets, config.model.max_speaker_latent_length)
        speaker_latent, speaker_mask = _pad_to_len(speaker_latent, speaker_mask, bucket)

        st = _ReqState(
            request_id=request_id,
            text=text,
            voice=voice,
            seed=int(seed),
            gen_config=gen_config,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
        )
        await self._in_q.put(st)
        return st.out_q

    def _init_request_state(self, reqs: List[_ReqState]) -> None:
        """Initialize conditioning caches and per-request tensors in a batch."""
        model = self._engine.model
        device, dtype = model.device, model.dtype

        # Text ids/masks (pad to fixed 768 for stable shapes)
        max_len = min(768, min(r.gen_config.max_text_length for r in reqs))
        text_ids, text_mask = get_text_input_ids_and_mask([r.text for r in reqs], max_len, device=device)

        # Unconditional (BOS only)
        text_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(text_ids.shape[0], text_ids.shape[1], device=device)

        # Speaker uncond
        speaker_latent = torch.cat([r.speaker_latent.to(device=device, dtype=dtype) for r in reqs], dim=0)
        speaker_mask = torch.cat([r.speaker_mask.to(device=device) for r in reqs], dim=0)
        speaker_latent_uncond = torch.zeros_like(speaker_latent)
        speaker_mask_uncond = torch.zeros_like(speaker_mask)

        # Full (cond, uncond-text, uncond-speaker)
        full_text_ids = torch.cat([text_ids, text_ids_uncond, text_ids], dim=0)
        full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
        full_speaker_latent = torch.cat([speaker_latent, speaker_latent, speaker_latent_uncond], dim=0)
        full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

        # KV caches (compute once per request init)
        kv_cache_text_full = self._engine._kv_cache_text(model, full_text_ids, full_text_mask)  # noqa: SLF001
        kv_cache_speaker_full = self._engine._kv_cache_speaker(model, full_speaker_latent)  # noqa: SLF001
        kv_cache_text = _get_first_n_kv_cache(kv_cache_text_full, text_ids.shape[0])
        kv_cache_speaker = _get_first_n_kv_cache(kv_cache_speaker_full, text_ids.shape[0])

        # Per-request init
        prefix_total = sum(reqs[0].gen_config.block_sizes)
        for i, r in enumerate(reqs):
            r.text_input_ids = text_ids[i : i + 1]
            r.text_mask = text_mask[i : i + 1]
            r.full_text_input_ids = full_text_ids[i : i + 1].repeat(3, 1)  # not used directly
            r.full_text_mask = full_text_mask[i : i + 1].repeat(3, 1)      # not used directly
            r.full_speaker_latent = full_speaker_latent[i : i + 1].repeat(3, 1, 1)  # not used directly
            r.full_speaker_mask = full_speaker_mask[i : i + 1].repeat(3, 1)         # not used directly
            r.kv_cache_text_full = kv_cache_text_full
            r.kv_cache_text = kv_cache_text
            r.kv_cache_speaker_full = kv_cache_speaker_full
            r.kv_cache_speaker = kv_cache_speaker

            r.prefix_latent = torch.zeros((1, prefix_total, 80), device=device, dtype=torch.float32)
            r.pos_id = 0
            r.block_idx = 0
            r.step_idx = 0
            r.decoder = StreamingAEDecoder(self._engine.fish_ae, self._engine.pca_state)

        # Block init (shared across requests if same config)
        self._prepare_block(reqs)

    def _prepare_block(self, reqs: List[_ReqState]) -> None:
        """Prepare x_t, t_schedule, latent kv caches for the current block for a group (same block_idx)."""
        model = self._engine.model
        device, dtype = model.device, model.dtype

        # Assumption (MVP): same sampler config across grouped requests.
        r0 = reqs[0]
        block_idx = r0.block_idx
        block_sizes = r0.gen_config.block_sizes
        step_counts = r0.gen_config.num_steps
        block_size = int(block_sizes[block_idx])
        block_steps = int(step_counts[block_idx])

        # Determine init scale (per-config list supported)
        init_scales = r0.gen_config.init_scale
        default_init_scale = 0.999
        if init_scales is None:
            init_scale = default_init_scale
        elif isinstance(init_scales, (int, float)):
            init_scale = float(init_scales)
        else:
            init_scale = float(init_scales[block_idx])

        t_schedule = torch.linspace(1.0, 0.0, block_steps + 1, device=device) * init_scale

        # Create per-request x_t with per-request RNG generator.
        # Using independent generators avoids global RNG races.
        for r in reqs:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(r.seed) + int(block_idx))
            x_t = torch.randn((1, block_size, 80), device=device, dtype=torch.float32, generator=gen)

            # Optional truncation
            trunc = r.gen_config.truncation_factor
            if trunc is not None:
                if isinstance(trunc, (int, float)):
                    trunc_val = float(trunc)
                else:
                    trunc_val = float(trunc[block_idx])
                x_t = x_t * trunc_val

            r.x_t = x_t
            r.t_schedule = t_schedule

        # Latent KV cache depends on prefix_latent (per request) but can be batched.
        prefix_latent = torch.cat([r.prefix_latent for r in reqs], dim=0).to(dtype)
        full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)

        kv_cache_latent_full = self._engine._kv_cache_latent(model, full_prefix_latent)  # noqa: SLF001
        kv_cache_latent = _get_first_n_kv_cache(kv_cache_latent_full, prefix_latent.shape[0])

        for r in reqs:
            r.kv_cache_latent_full = kv_cache_latent_full
            r.kv_cache_latent = kv_cache_latent

    def _group_key(self, r: _ReqState) -> Tuple[Any, ...]:
        # MVP grouping by (block_idx, step_idx, block_size, block_steps) and cfg-active (depends on t)
        block_idx = r.block_idx
        block_size = int(r.gen_config.block_sizes[block_idx])
        block_steps = int(r.gen_config.num_steps[block_idx])
        return (block_idx, r.step_idx, block_size, block_steps)

    @torch.inference_mode()
    def _run_one_step(self, reqs: List[_ReqState]) -> None:
        """Run a single diffusion step for a grouped batch."""
        model = self._engine.model
        device, dtype = model.device, model.dtype

        r0 = reqs[0]
        block_idx = r0.block_idx
        step_idx = r0.step_idx
        block_steps = int(r0.gen_config.num_steps[block_idx])
        t_schedule = r0.t_schedule
        assert t_schedule is not None

        t = t_schedule[step_idx]
        t_next = t_schedule[step_idx + 1]
        has_cfg = bool(((t >= r0.gen_config.cfg_min_t) * (t <= r0.gen_config.cfg_max_t)).item())

        # Batch x_t
        x_t = torch.cat([r.x_t for r in reqs], dim=0)  # (B, S, 80)
        B = int(x_t.shape[0])

        # Masks
        text_mask = torch.cat([r.text_mask for r in reqs], dim=0)
        speaker_mask = torch.cat([r.speaker_mask for r in reqs], dim=0)

        if has_cfg:
            x_in = torch.cat([x_t, x_t, x_t], dim=0).to(dtype)
            t_in = (torch.ones((B * 3,), device=device) * t).to(dtype)

            # Full masks for (cond, uncond-text, uncond-speaker)
            text_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(B, text_mask.shape[1], device=device)
            full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
            speaker_mask_uncond = torch.zeros_like(speaker_mask)
            full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

            v_cond, v_uncond_text, v_uncond_speaker = model(
                x=x_in,
                t=t_in,
                text_mask=full_text_mask,
                speaker_mask=full_speaker_mask,
                start_pos=r0.pos_id,
                kv_cache_text=r0.kv_cache_text_full,
                kv_cache_speaker=r0.kv_cache_speaker_full,
                kv_cache_latent=r0.kv_cache_latent_full,
            ).float().chunk(3, dim=0)

            v_pred = (
                v_cond
                + r0.gen_config.cfg_scale_text * (v_cond - v_uncond_text)
                + r0.gen_config.cfg_scale_speaker * (v_cond - v_uncond_speaker)
            )
        else:
            t_in = (torch.ones((B,), device=device) * t).to(dtype)
            v_pred = model(
                x=x_t.to(dtype),
                t=t_in,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                start_pos=r0.pos_id,
                kv_cache_text=r0.kv_cache_text,
                kv_cache_speaker=r0.kv_cache_speaker,
                kv_cache_latent=r0.kv_cache_latent,
            ).float()

        x_t = x_t + v_pred * (t_next - t)

        # Scatter back
        for i, r in enumerate(reqs):
            r.x_t = x_t[i : i + 1]
            r.step_idx += 1
            # If we finished the block, commit block + emit audio.
            if r.step_idx >= block_steps:
                self._finish_block(r)

    def _finish_block(self, r: _ReqState) -> None:
        """Commit current block to prefix_latent, decode and enqueue PCM chunk, and advance to next block."""
        assert r.x_t is not None and r.prefix_latent is not None and r.decoder is not None

        block_size = int(r.x_t.shape[1])
        r.prefix_latent[:, r.pos_id : r.pos_id + block_size] = r.x_t
        r.pos_id += block_size

        # --- Match engine.generate_streaming() tail handling ---
        # 1) Optional early-stop based on "near-zero" tail of the just-finished block.
        early_stop = False
        if r.gen_config.early_stop_on_zero:
            tail_len = min(int(r.gen_config.zero_tail_frames), int(r.x_t.shape[1]))
            if tail_len > 0:
                tail = r.x_t[:, -tail_len:]
                tail_abs = torch.abs(tail)
                zero_frac = float((tail_abs <= float(r.gen_config.zero_eps)).float().mean().item())
                tail_absmax = float(tail_abs.max().item())
                zero_ok = zero_frac >= float(r.gen_config.zero_tail_min_frac) and tail_absmax <= float(r.gen_config.zero_eps)
                if zero_ok:
                    early_stop = True

        # 2) If finishing (early stop OR last planned block), find flattening point to trim trailing junk.
        is_last_planned = (r.block_idx == len(r.gen_config.block_sizes) - 1)
        will_finish = early_stop or is_last_planned

        flatten_point: int | None = None
        if will_finish:
            prefix_latent_trim = r.prefix_latent[:, : r.pos_id]
            # Heuristic: find the start of a stable near-zero latent tail.
            flatten_point = int(
                find_flattening_point(prefix_latent_trim[0], window_size=32, std_threshold=0.02)
            )
            # Never trim before already-emitted audio.
            already_emitted_latents = int(r.decoder._emitted_samples // r.decoder.samples_per_latent)  # noqa: SLF001
            flatten_point = max(flatten_point, already_emitted_latents)

        # Decode and enqueue PCM (trimmed if finishing)
        new_audio = r.decoder.decode_next(
            r.x_t.to(next(self._engine.fish_ae.parameters()).device),
            flattening_point=flatten_point,
        )
        if new_audio.numel() > 0:
            if r.ttfb_t is None:
                r.ttfb_t = time.perf_counter()
            pcm = self._engine.audio_to_pcm(new_audio)
            # Best-effort: if client is slow, drop rather than block GPU loop forever.
            try:
                r.out_q.put_nowait(pcm)
            except asyncio.QueueFull:
                pass

        if will_finish:
            r.done = True
            # Signal completion
            try:
                r.out_q.put_nowait(None)
            except asyncio.QueueFull:
                # If full, ensure eventual completion signal by draining one.
                try:
                    _ = r.out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    r.out_q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            return

        # Advance block (continue generation)
        r.block_idx += 1
        r.step_idx = 0

        # Next block will be prepared by scheduler (batched) on the next tick.

    async def _drain_new(self) -> None:
        # Pull in newly submitted requests, optionally waiting a short window for batching.
        if self._batch_wait_ms > 0:
            try:
                st = await asyncio.wait_for(self._in_q.get(), timeout=self._batch_wait_ms / 1000.0)
                self._active.append(st)
            except asyncio.TimeoutError:
                return

            # Grab a few more without blocking
            t_end = time.perf_counter() + (self._batch_wait_ms / 1000.0)
            while len(self._active) < self._max_batch_size and time.perf_counter() < t_end:
                try:
                    st = self._in_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._active.append(st)
        else:
            while True:
                try:
                    st = self._in_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._active.append(st)

    async def _run_loop(self) -> None:
        # Main loop: initialize new requests, then repeatedly batch one step at a time.
        while not self._stop.is_set():
            await self._drain_new()

            # Initialize any requests that haven't been initialized yet.
            new_reqs = [r for r in self._active if r.prefix_latent is None]
            if new_reqs:
                # MVP: initialize in one batch, but cap to max_batch_size.
                init_batch = new_reqs[: self._max_batch_size]
                self._init_request_state(init_batch)

            # Remove completed
            self._active = [r for r in self._active if not r.done]
            if not self._active:
                await asyncio.sleep(0.001)
                continue

            # Group by step key and run one step for each group.
            groups: Dict[Tuple[Any, ...], List[_ReqState]] = {}
            for r in self._active:
                groups.setdefault(self._group_key(r), []).append(r)

            # Run largest groups first for better batching efficiency
            for _, reqs in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
                if not reqs:
                    continue
                # Cap batch size; spill remainder to next tick
                batch = reqs[: self._max_batch_size]
                self._run_one_step(batch)

                # If we just finished a block for this batch, prepare next block KV caches (batched)
                # for the subset that advanced to the next block and are not done.
                advanced = [r for r in batch if (not r.done) and r.step_idx == 0]
                if advanced:
                    # Recompute latent kv cache and init x_t/t_schedule for next block in a batch.
                    self._prepare_block(advanced)

            # Yield control to event loop; keep loop tight for low latency
            await asyncio.sleep(0)


