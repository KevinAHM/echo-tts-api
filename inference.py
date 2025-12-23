import gc
from dataclasses import dataclass
from typing import Callable, List, Tuple

from huggingface_hub import hf_hub_download
import safetensors.torch as st
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder

from autoencoder import DAC, build_ae
from model import EchoDiT


LORA_ALPHA = 32.0


def merge_lora_into_state_dict(
    base_state: dict[str, torch.Tensor],
    lora_state: dict[str, torch.Tensor],
    lora_scale: float = 1.0,   # extra user scale on top of training scale
    alpha: float = LORA_ALPHA, # should match training (32.0 in your JAX LoRA)
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights (produced by flax_lora_to_torch) into a base state_dict.

    Expects LoRA keys like:
      "<prefix>.lora_w1.weight"  (shape: [r, in])
      "<prefix>.lora_w2.weight"  (shape: [out, r])

    and base weights like:
      "<prefix>.weight"          (shape: [out, in])

    The merged update is:
      W_new = W + (alpha / r) * lora_scale * (lora_w2 @ lora_w1)
    """
    base_state = dict(base_state)  # work on a copy so we don't mutate the original

    for k in list(lora_state.keys()):
        if not k.endswith(".lora_w1.weight"):
            continue

        prefix = k[: -len(".lora_w1.weight")]
        w1_key = prefix + ".lora_w1.weight"
        w2_key = prefix + ".lora_w2.weight"
        base_key = prefix + ".weight"

        if w2_key not in lora_state:
            print(f"[LoRA] Missing {w2_key}, skipping {prefix}")
            continue

        if base_key not in base_state:
            continue

        w1 = lora_state[w1_key]  # (r, in)
        w2 = lora_state[w2_key]  # (out, r)
        W = base_state[base_key]  # (out, in)

        if w1.ndim != 2 or w2.ndim != 2 or W.ndim != 2:
            print(f"[LoRA] Unexpected tensor rank for {prefix}, skipping")
            continue

        r, in_dim = w1.shape
        out_dim, r2 = w2.shape

        if r2 != r:
            print(f"[LoRA] Rank mismatch for {prefix}: w1 {w1.shape}, w2 {w2.shape}, skipping")
            continue

        if W.shape != (out_dim, in_dim):
            print(f"[LoRA] Shape mismatch for {prefix}: base {W.shape}, expected {(out_dim, in_dim)}, skipping")
            continue

        scale = (alpha / float(r)) * float(lora_scale)
        delta = (w2.float() @ w1.float()) * scale  # (out, in)

        base_state[base_key] = (W.float() + delta).to(W.dtype)

    return base_state


def load_model_from_hf(
    repo_id: str = "jordand/echo-tts-base",
    device: str = "cuda",
    dtype: torch.dtype | None = torch.bfloat16,
    compile: bool = False,
    token: str | None = None,
    delete_blockwise_modules: bool = False,
    lora_hf_hub_name: str | None = None,
    lora_scale: float = 1.0,
    lora_alpha: float = LORA_ALPHA,
    base_state_dict: dict[str, torch.Tensor] | None = None,
) -> EchoDiT:
    if lora_hf_hub_name is not None and lora_hf_hub_name.lower() == "none":
        lora_hf_hub_name = None

    model = EchoDiT(
        latent_size=80, model_size=2048, num_layers=24, num_heads=16,
        intermediate_size=5888, norm_eps=1e-5,
        text_vocab_size=256, text_model_size=1280, text_num_layers=14,
        text_num_heads=10, text_intermediate_size=3328,
        speaker_patch_size=4, speaker_model_size=1280, speaker_num_layers=14,
        speaker_num_heads=10, speaker_intermediate_size=3328,
        timestep_embed_size=512, adaln_rank=256,
    )
    if base_state_dict is None:
        w_path = hf_hub_download(repo_id, "pytorch_model.safetensors", token=token)
        # Load straight to target device to avoid large CPU copies.
        state = st.load_file(w_path, device=device)
        if dtype is not None:
            state = {k: v.to(dtype=dtype, device=device) for k, v in state.items()}
    else:
        # Reuse an already-loaded base state; move directly to target device/dtype.
        state = {
            k: v.to(device=device, dtype=(dtype or v.dtype))
            for k, v in base_state_dict.items()
        }

    if delete_blockwise_modules:
        state = {k: v for k, v in state.items() if not (
            k.startswith("latent_encoder.") or 
            k.startswith("latent_norm") or 
            ".wk_latent" in k or 
            ".wv_latent" in k
        )}

    if lora_hf_hub_name is not None:
        lora_path = hf_hub_download(repo_id, lora_hf_hub_name, token=token)
        # Load LoRA weights directly on device to avoid host copies.
        lora_state = st.load_file(lora_path, device=device)
        state = merge_lora_into_state_dict(
            base_state=state,
            lora_state=lora_state,
            lora_scale=lora_scale,
            alpha=lora_alpha,
        )
        del lora_state
        gc.collect()

    model.load_state_dict(state, strict=False, assign=True)
    model = model.to(device).eval()
    # Free CPU copies promptly; state dicts can be large.
    del state
    gc.collect()

    if compile:
        model = compile_model(model)

    return model

def compile_model(model: EchoDiT) -> EchoDiT:
    model = torch.compile(model)
    model.get_kv_cache_text = torch.compile(model.get_kv_cache_text)
    model.get_kv_cache_speaker = torch.compile(model.get_kv_cache_speaker)
    if hasattr(model, "get_kv_cache_latent"):
        model.get_kv_cache_latent = torch.compile(model.get_kv_cache_latent)
    return model

def load_fish_ae_from_hf(repo_id: str = "jordand/fish-s1-dac-min", device: str = "cuda", dtype: torch.dtype | None = torch.float32, compile: bool = False, token: str | None = None) -> DAC:
   
    with torch.device("meta"):
        fish_ae = build_ae()

    w_path = hf_hub_download(repo_id, "pytorch_model.safetensors", token=token)
    state = st.load_file(w_path, device=device)
    if dtype is not None and dtype != torch.float32:
        state = {k: v.to(dtype=dtype, device=device) for k, v in state.items()}
    fish_ae.load_state_dict(state, strict=False, assign=True)
    del state

    fish_ae = fish_ae.eval().to(device)
    gc.collect()

    if compile:
        fish_ae = compile_fish_ae(fish_ae)

    return fish_ae

def compile_fish_ae(fish_ae: DAC) -> DAC:
    fish_ae.quantizer.upsample = torch.compile(fish_ae.quantizer.upsample)
    fish_ae.quantizer.downsample = torch.compile(fish_ae.quantizer.downsample)
    fish_ae.quantizer.pre_module = torch.compile(fish_ae.quantizer.pre_module)
    fish_ae.quantizer.post_module = torch.compile(fish_ae.quantizer.post_module)
    return fish_ae


@dataclass
class PCAState:
    pca_components: torch.Tensor
    pca_mean: torch.Tensor
    latent_scale: float

def load_pca_state_from_hf(repo_id: str = "jordand/echo-tts-base", device: str = "cuda", filename: str = "pca_state.safetensors", token: str | None = None) -> PCAState:
    p_path = hf_hub_download(repo_id, filename, token=token)
    t = st.load_file(p_path, device=device)
    return PCAState(
        pca_components=t["pca_components"],
        pca_mean=t["pca_mean"],
        latent_scale=float(t["latent_scale"].item()),
    )


# ________

def load_audio(path: str, max_duration: int = 300) -> torch.Tensor:

    decoder = AudioDecoder(path)
    sr = decoder.metadata.sample_rate
    audio = decoder.get_samples_played_in_range(0, max_duration)
    audio = audio.data.mean(dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, 44_100)
    audio = audio / torch.maximum(audio.abs().max(), torch.tensor(1.))
    # is this better than clipping? should we target a specific energy level?
    return audio

def tokenizer_encode(text: str, append_bos: bool = True, normalize: bool = True, return_normalized_text: bool = False) -> torch.Tensor | Tuple[torch.Tensor, str]:

    if normalize:
        text = text.replace("…", "...")        
        text = text.replace('’', "'")
        text = text.replace('”', '"')
        text = text.replace('”', '"')
        text = text.replace("\n", " ")
        text = text.replace(":", ",")
        text = text.replace(";", ",")
        if not text.startswith("[") and not text.startswith("(") and 'S1' not in text and 'S2' not in text:
            text = "[S1] " + text

    b = list(text.encode("utf-8"))
    if append_bos:
        b.insert(0, 0)

    if return_normalized_text:
        return torch.tensor(b), text

    return torch.tensor(b)

def get_text_input_ids_and_mask(text_arr: List[str], max_length: int | None, device: str | None = None, normalize: bool = True, return_normalized_text: bool = False, pad_to_max: bool = True) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, List[str]]:
    encoded_texts = [tokenizer_encode(text, normalize=normalize, return_normalized_text=True) for text in text_arr]
    
    if max_length is None:
        max_length = max(len(enc) for enc, _ in encoded_texts)
    
    tokens = torch.zeros((len(text_arr), max_length), dtype=torch.int32)
    mask = torch.zeros((len(text_arr), max_length), dtype=torch.bool)
    
    for i, (encoded, _) in enumerate(encoded_texts):
        length = min(len(encoded), max_length)
        tokens[i, :length] = encoded[:length]
        mask[i, :length] = 1

    if not pad_to_max and max_length is not None:
        tokens, mask = tokens[:, :max_length], mask[:, :max_length]

    if device is not None:
        tokens, mask = tokens.to(device), mask.to(device)

    if return_normalized_text:
        return tokens, mask, [text for _, text in encoded_texts]
    return tokens, mask

# ________

@torch.inference_mode()
def ae_encode(fish_ae: DAC, pca_state: PCAState, audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 3 and audio.shape[1] == 1 # (b, 1, length)
    z_q = fish_ae.encode_zq(audio).float()
    z_q = (z_q.transpose(1, 2) - pca_state.pca_mean) @ pca_state.pca_components.T
    z_q = z_q * pca_state.latent_scale
    return z_q

@torch.inference_mode()
def ae_decode(fish_ae: DAC, pca_state: PCAState, z_q: torch.Tensor) -> torch.Tensor:
    z_q = (z_q / pca_state.latent_scale) @ pca_state.pca_components + pca_state.pca_mean
    return fish_ae.decode_zq(z_q.transpose(1, 2).to(fish_ae.dtype)).float()

@torch.inference_mode()
def ae_reconstruct(fish_ae: DAC, pca_state: PCAState, audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 3 and audio.shape[1] == 1 # (b, 1, length)
    z_q = ae_encode(fish_ae, pca_state, audio.to(fish_ae.dtype))
    return ae_decode(fish_ae, pca_state, z_q)

# ________

@torch.inference_mode()
def get_speaker_latent_and_mask(
    fish_ae: DAC,
    pca_state: PCAState,
    audio: torch.Tensor, # (1, length)
    max_speaker_latent_length: int = 6400, # pretrained max length
    audio_chunk_size: int = 640 * 2048, # (~30 seconds, 1/10 max speaker condition size; max chunk seen in training)
    pad_to_max: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # gets speaker latent and mask from audio, computes in chunks and concatenates (similar to training setup)
    
    AE_DOWNSAMPLE_FACTOR = 2048
    max_audio_len_length = max_speaker_latent_length * AE_DOWNSAMPLE_FACTOR
    
    assert audio.ndim == 2 and audio.shape[0] == 1  # (1, length)
    audio = audio[:, :max_audio_len_length]

    latent_arr = []
    
    for i in range(0, audio.shape[1], audio_chunk_size):
        audio_chunk = audio[:, i:i + audio_chunk_size]
        if audio_chunk.shape[1] < audio_chunk_size:
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, audio_chunk_size - audio_chunk.shape[1]))

        latent_chunk = ae_encode(fish_ae, pca_state, audio_chunk.unsqueeze(0))
        latent_arr.append(latent_chunk)
    
    speaker_latent = torch.cat(latent_arr, dim=1)
    
    actual_latent_length = audio.shape[1] // AE_DOWNSAMPLE_FACTOR
    speaker_mask = (torch.arange(speaker_latent.shape[1], device=speaker_latent.device) < actual_latent_length).unsqueeze(0)
    
    if pad_to_max and speaker_latent.shape[1] < max_speaker_latent_length:
        speaker_latent = torch.nn.functional.pad(speaker_latent, (0, 0, 0, max_speaker_latent_length - speaker_latent.shape[1]))
        speaker_mask = torch.nn.functional.pad(speaker_mask, (0, max_speaker_latent_length - speaker_mask.shape[1]))
    elif not pad_to_max:
        speaker_latent = speaker_latent[:, :actual_latent_length]
        speaker_mask = speaker_mask[:, :actual_latent_length]
    
    return speaker_latent, speaker_mask


# ________

def find_flattening_point(
    data,
    target_value=0.0,
    window_size=20,
    std_threshold=0.1,
    mean_abs_threshold=0.02,
    max_abs_threshold=0.05,
    min_runs=2,
):
    """
    Heuristic to find the start of the flat (near-zero) tail of the latent sequence.
    Scans backward from the end, requiring consecutive flat windows.
    """
    if data.numel() == 0:
        return 0

    data = torch.cat([data, data.new_zeros((window_size,) + data.shape[1:])], dim=0)

    flat_start = len(data)
    runs = 0

    for i in range(len(data) - window_size, -1, -1):
        window = data[i : i + window_size]
        std_ok = window.std() < std_threshold
        mean_ok = (window - target_value).abs().mean() < mean_abs_threshold
        max_ok = window.abs().max() < max_abs_threshold
        if std_ok and mean_ok and max_ok:
            runs += 1
            if runs >= min_runs:
                flat_start = i
        else:
            if runs > 0:
                break
    return flat_start

SampleFn = Callable[
    [EchoDiT, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    torch.Tensor
]

@torch.inference_mode()
def sample_pipeline(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    sample_fn: SampleFn,
    text_prompt: str,
    speaker_audio: torch.Tensor | None,
    rng_seed: int,
    pad_to_max_speaker_latent_length: int | None = None,
    pad_to_max_text_length: int | None = None,
    normalize_text: bool = True,
) -> Tuple[torch.Tensor, str]:

    MAX_SPEAKER_LATENT_LENGTH = 6400 # max seen during training, though maybe can go higher?
    MAX_TEXT_LENGTH = 768

    device, dtype = model.device, model.dtype

    text_input_ids, text_mask, normalized_text = get_text_input_ids_and_mask([text_prompt], max_length=min(pad_to_max_text_length or MAX_TEXT_LENGTH, MAX_TEXT_LENGTH), device=device, normalize=normalize_text, return_normalized_text=True, pad_to_max=(pad_to_max_text_length is not None))

    if speaker_audio is None:
        speaker_latent = torch.zeros((1, pad_to_max_speaker_latent_length or 4, 80), device=device, dtype=dtype)
        speaker_mask = torch.zeros((1, pad_to_max_speaker_latent_length or 4), device=device, dtype=torch.bool)
    else:
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            fish_ae, 
            pca_state, 
            speaker_audio.to(fish_ae.dtype).to(device), 
            max_speaker_latent_length=pad_to_max_speaker_latent_length or MAX_SPEAKER_LATENT_LENGTH,
            pad_to_max=(pad_to_max_speaker_latent_length is not None)
        )
        speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // 4 * 4] # have to be divis by patch size
        speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // 4 * 4]

        
    latent_out = sample_fn(model, speaker_latent, speaker_mask, text_input_ids, text_mask, rng_seed)

    audio_out = ae_decode(fish_ae, pca_state, latent_out)

    flattening_point = find_flattening_point(latent_out[0])
    audio_out = audio_out[..., :flattening_point * 2048]

    return audio_out, normalized_text[0]




# ________


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]

def _concat_kv_caches(*caches: KVCache) -> KVCache: 
    # helper that concatenates multiple KV caches along the batch dimension
    num_layers = len(caches[0])
    result = []
    for i in range(num_layers):
        k = torch.cat([c[i][0] for c in caches], dim=0)
        v = torch.cat([c[i][1] for c in caches], dim=0)
        result.append((k, v))
    return result

def _multiply_kv_cache(cache: KVCache, scale: float, max_layers: int | None = None) -> None:
    # helper that multiplies KV cache values in-place, for kv speaker scaling
    num_layers = len(cache) if max_layers is None else min(max_layers, len(cache))
    for i in range(num_layers):
        k, v = cache[i]
        k.mul_(scale)
        v.mul_(scale)

def _temporal_score_rescale(
    v_pred: torch.Tensor, x_t: torch.Tensor, t: float, rescale_k: float, rescale_sigma: float
) -> torch.Tensor:
    # for https://arxiv.org/pdf/2510.01184
    if t < 1:
        snr = (1 - t) ** 2 / (t ** 2)
        ratio = (snr * rescale_sigma ** 2 + 1) / (snr * rescale_sigma ** 2 / rescale_k + 1)
        return 1 / (1 - t) * (ratio * ((1 - t) * v_pred + x_t) - x_t)
    return v_pred


@torch.inference_mode()
def sample_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    rng_seed: int,
    num_steps: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float | None,
    rescale_k: float | None,
    rescale_sigma: float | None,
    speaker_kv_scale: float | None,
    speaker_kv_max_layers: int | None,
    speaker_kv_min_t: float | None,
    sequence_length: int | None = None,
) -> torch.Tensor:

    if sequence_length is None:
        sequence_length = 640 # max sequence length during training

    torch.manual_seed(rng_seed)

    INIT_SCALE = 0.999 # so that we can apply rescale to first step

    device, dtype = model.device, model.dtype
    batch_size = text_input_ids.shape[0]

    t_schedule = torch.linspace(1., 0., num_steps + 1, device=device) * INIT_SCALE

    text_mask_uncond = torch.zeros_like(text_mask)
    speaker_mask_uncond = torch.zeros_like(speaker_mask)

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent.to(dtype))

    if speaker_kv_scale is not None:
        _multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers)

    # masks prevent decoder from attending to unconds:
    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

    full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
    full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

    x_t = torch.randn((batch_size, sequence_length, 80), device=device, dtype=torch.float32)
    if truncation_factor is not None:
        x_t = x_t * truncation_factor

    for i in range(num_steps):
        t, t_next = t_schedule[i], t_schedule[i + 1]

        has_cfg = ((t >= cfg_min_t) * (t <= cfg_max_t)).item()

        if has_cfg:
            v_cond, v_uncond_text, v_uncond_speaker = model(
                x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                t=(torch.ones((batch_size * 3,), device=device) * t).to(dtype),
                text_mask=full_text_mask,
                speaker_mask=full_speaker_mask,
                kv_cache_text=kv_text_full,
                kv_cache_speaker=kv_speaker_full,
            ).float().chunk(3, dim=0)
            v_pred = v_cond + cfg_scale_text * (v_cond - v_uncond_text) + cfg_scale_speaker * (v_cond - v_uncond_speaker) # can also use a single, joint unconditional for fewer NFE
        else:
            v_pred = model(
                x=x_t.to(dtype),
                t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                kv_cache_text=kv_text_cond,
                kv_cache_speaker=kv_speaker_cond,
            ).float()

        # optional temporal score rescaling: https://arxiv.org/pdf/2510.01184
        if rescale_k is not None and rescale_sigma is not None:
            v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

        # optional kv speaker scaling
        if speaker_kv_scale is not None and t_next < speaker_kv_min_t and t >= speaker_kv_min_t:
            _multiply_kv_cache(kv_speaker_cond, 1. / speaker_kv_scale, speaker_kv_max_layers)
            kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

        x_t = x_t + v_pred * (t_next - t)

    return x_t


if __name__ == "__main__":
    model = load_model_from_hf(delete_blockwise_modules=True)
