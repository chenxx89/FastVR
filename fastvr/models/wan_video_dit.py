import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext
from typing import Tuple, Optional
from einops import rearrange
from diffsynth.core.gradient import gradient_checkpoint_forward

# ---------------------------------------------------------------------------
# Optional profiling scopes.
#
# When enabled (via set_dit_profile_scopes(True)), key regions of the DiT block
# are wrapped in torch.profiler.record_function so a profiler pass can attribute
# device time to "self-attn core / q,k,v,o projection / cross-attn / ffn".
# The default is OFF and resolves to a shared nullcontext, so the normal
# inference/training path pays zero overhead.
# ---------------------------------------------------------------------------
try:
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - profiler always present in supported torch
    _record_function = None

_WAN_DIT_PROFILE_SCOPES = False
_NULL_SCOPE = nullcontext()


def set_dit_profile_scopes(enabled: bool) -> None:
    """Toggle DiT profiling record_function scopes (used by the timing benchmark)."""
    global _WAN_DIT_PROFILE_SCOPES
    _WAN_DIT_PROFILE_SCOPES = bool(enabled)


def _prof_scope(name: str):
    if _WAN_DIT_PROFILE_SCOPES and _record_function is not None:
        return _record_function(name)
    return _NULL_SCOPE

# ---------------------------------------------------------------------------
# Constant-context fast path (cross-attention k/v caching).
# When the text embedding is FIXED for the whole denoise (e.g. a constant empty
# prompt), every cross-attention's context-derived k/v are identical across all
# chunk/tile/step forwards. Enabling this mode makes each CrossAttention compute
# k/v once (on the first row of the context) and reuse it for the rest, skipping
# the redundant k/v projection + norm_k. Bit-exact for a truly constant context.
# A generation counter, bumped on every enable, invalidates stale caches so a new
# denoise (or a changed prompt) never reuses old k/v. Default OFF -> zero effect
# on training / normal inference.
# ---------------------------------------------------------------------------
_WAN_DIT_CONST_CONTEXT = False
_WAN_DIT_CONST_CONTEXT_GEN = 0


def set_dit_const_context(enabled: bool) -> None:
    """Toggle the constant-context cross-attn k/v cache (assumes a fixed context).

    Bumps the invalidation generation on enable so caches from a prior denoise or
    a different prompt are never reused.
    """
    global _WAN_DIT_CONST_CONTEXT, _WAN_DIT_CONST_CONTEXT_GEN
    if enabled:
        _WAN_DIT_CONST_CONTEXT_GEN += 1
    _WAN_DIT_CONST_CONTEXT = bool(enabled)

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_3_func
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    _flash_attn_3_func = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    from flash_attn import flash_attn_func as _flash_attn_2_func
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    _flash_attn_2_func = None
    FLASH_ATTN_2_AVAILABLE = False

_FLASH_ATTN_BACKEND = None


def _initialize_flash_attention_backend() -> None:
    """Select the best installed backend once for the active CUDA device."""
    global _FLASH_ATTN_BACKEND
    if not torch.cuda.is_available():
        _FLASH_ATTN_BACKEND = "sdpa"
    else:
        major, _ = torch.cuda.get_device_capability()
        if FLASH_ATTN_3_AVAILABLE and major >= 9:
            _FLASH_ATTN_BACKEND = "fa3"
        elif FLASH_ATTN_2_AVAILABLE and major >= 8:
            _FLASH_ATTN_BACKEND = "fa2"
        else:
            _FLASH_ATTN_BACKEND = "sdpa"
    print(f"[FastVR] Attention backend: {_FLASH_ATTN_BACKEND}")


SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x

    if _FLASH_ATTN_BACKEND is None:
        _initialize_flash_attention_backend()

    if _FLASH_ATTN_BACKEND == "fa3":
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = _flash_attn_3_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif _FLASH_ATTN_BACKEND == "fa2":
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = _flash_attn_2_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    # Ensure position is at least 1-D for torch.outer
    if position.dim() == 0:
        position = position.unsqueeze(0)
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.float()


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def ensure_temporal_rope_length(model, required_length: int) -> None:
    """Extend the temporal RoPE table without changing existing positions."""
    current = model.freqs[0]
    if required_length <= current.shape[0]:
        return
    new_length = max(int(required_length), int(current.shape[0]) * 2)
    temporal_dim = current.shape[-1] * 2
    extended = precompute_freqs_cis(temporal_dim, new_length).to(
        device=current.device,
        dtype=current.dtype,
    )
    model.freqs = (extended, model.freqs[1], model.freqs[2])


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


def build_window_causal_chunk_ids(latent_frames: int, chunk_size: int, first_chunk_size: int, device):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if first_chunk_size <= 0:
        raise ValueError(f"first_chunk_size must be positive, got {first_chunk_size}")
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    frame_ids = torch.arange(latent_frames, device=device)
    first_end = min(first_chunk_size, latent_frames)
    return torch.where(
        frame_ids < first_end,
        torch.zeros_like(frame_ids),
        1 + (frame_ids - first_end) // chunk_size,
    )


def build_window_causal_chunk_ranges(
    latent_frames: int,
    chunk_size: int,
    first_chunk_size: int = 1,
):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if first_chunk_size <= 0:
        raise ValueError(f"first_chunk_size must be positive, got {first_chunk_size}")
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    # First chunk holds the first `first_chunk_size` latents (defaults to 1,
    # matching the training distribution where latent 0 is its own chunk).
    # Subsequent chunks are sliced by `chunk_size`.
    first_end = min(first_chunk_size, latent_frames)
    ranges = [(0, first_end)]
    start = first_end
    while start < latent_frames:
        end = min(start + chunk_size, latent_frames)
        ranges.append((start, end))
        start = end
    return ranges


def window_causal_history_frames(chunk_size: int, window_size: int) -> int:
    """Number of past latent frames every chunk may attend to.

    The history budget is counted in latent frames, not in chunks, so it stays
    at `(window_size - 1) * chunk_size` no matter how long the first chunk is.
    A longer `first_chunk_size` therefore only widens the first chunk's own
    bidirectional window; it never inflates the history seen by chunk 1.
    """
    return max(0, (int(window_size) - 1) * int(chunk_size))


def build_window_causal_chunk_bounds(
    latent_frames: int,
    chunk_size: int,
    first_chunk_size: int,
    device,
):
    """Per-latent-frame chunk id plus the [start, end) bounds of its own chunk."""
    chunk_ids = build_window_causal_chunk_ids(latent_frames, chunk_size, first_chunk_size, device)
    first_end = min(first_chunk_size, latent_frames)
    starts = torch.where(
        chunk_ids == 0,
        torch.zeros_like(chunk_ids),
        first_end + (chunk_ids - 1) * chunk_size,
    )
    ends = torch.where(
        chunk_ids == 0,
        torch.full_like(chunk_ids, first_end),
        (starts + chunk_size).clamp(max=latent_frames),
    )
    return chunk_ids, starts, ends


def trim_window_causal_kv_cache(layer_cache, max_history_tokens: int):
    """Shrink a streaming KV cache in place to the last `max_history_tokens` tokens.

    Entries are whole chunks, so the oldest entry is sliced (not just dropped)
    when it only partially fits into the history budget. This keeps the cache
    equal to the history the training-time mask exposes, which matters as soon
    as `first_chunk_size != chunk_size`.
    """
    if max_history_tokens <= 0:
        layer_cache.clear()
        return
    total = sum(item[0].shape[1] for item in layer_cache)
    while layer_cache and total - layer_cache[0][0].shape[1] >= max_history_tokens:
        total -= layer_cache[0][0].shape[1]
        layer_cache.pop(0)
    excess = total - max_history_tokens
    if excess > 0 and layer_cache:
        k, v = layer_cache[0]
        layer_cache[0] = (k[:, excess:], v[:, excess:])


def build_window_causal_self_attn_mask(
    frame_tokens: int,
    latent_frames: int,
    chunk_size: int,
    window_size: int,
    first_chunk_size: int,
    device,
):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if first_chunk_size <= 0:
        raise ValueError(f"first_chunk_size must be positive, got {first_chunk_size}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    if frame_tokens <= 0:
        raise ValueError(f"frame_tokens must be positive, got {frame_tokens}")

    # A query in chunk c sees its own chunk (bidirectionally) plus the last
    # `(window_size - 1) * chunk_size` latent frames before the chunk start.
    # For uniform chunks this is identical to a chunk-level sliding window; it
    # only differs when the first chunk is longer than `chunk_size`, where the
    # frame-level budget keeps chunk 1's context from growing with it.
    #
    # This first implementation intentionally builds a dense [S, S] boolean mask
    # to keep window causal semantics explicit. Long videos should use a future
    # block-sparse or tiled attention implementation to avoid excessive memory use.
    _, starts, ends = build_window_causal_chunk_bounds(latent_frames, chunk_size, first_chunk_size, device)
    history = window_causal_history_frames(chunk_size, window_size)
    lo = (starts - history).clamp_min(0)
    key_frame_ids = torch.arange(latent_frames, device=device)[None, :]
    visible = (key_frame_ids >= lo[:, None]) & (key_frame_ids < ends[:, None])
    visible = visible.repeat_interleave(frame_tokens, dim=0).repeat_interleave(frame_tokens, dim=1)
    return visible.unsqueeze(0).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None, causal=False):
        if attn_mask is None and not causal:
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
            return x

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal
        )
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None, kv_cache=None, return_kv=False):
        with _prof_scope("dit.sa.q"):
            q = self.q(x)
        with _prof_scope("dit.sa.k"):
            k = self.k(x)
        with _prof_scope("dit.sa.v"):
            v = self.v(x)
        with _prof_scope("dit.sa.qknorm"):
            q = self.norm_q(q)
            k = self.norm_k(k)
        with _prof_scope("dit.sa.rope"):
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
        current_kv = (k, v)
        if kv_cache:
            with _prof_scope("dit.sa.kvcat"):
                cached_k = torch.cat([item[0] for item in kv_cache], dim=1)
                cached_v = torch.cat([item[1] for item in kv_cache], dim=1)
                k = torch.cat([cached_k, k], dim=1)
                v = torch.cat([cached_v, v], dim=1)
        with _prof_scope("dit.sa.core"):
            x = self.attn(q, k, v, attn_mask=attn_mask)
        with _prof_scope("dit.sa.o"):
            x = self.o(x)
        if return_kv:
            return x, current_kv
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        # Constant-context cache: (k, v) derived from a fixed context, plus the
        # generation it was computed at. See set_dit_const_context.
        self._cc_kv = None
        self._cc_gen = -1

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        with _prof_scope("dit.ca.q"):
            q = self.q(x)
        with _prof_scope("dit.ca.qknorm"):
            q = self.norm_q(q)
        # k/v depend only on the (constant) context -> cache & reuse when enabled.
        use_cache = _WAN_DIT_CONST_CONTEXT and not self.has_image_input
        if use_cache and self._cc_gen == _WAN_DIT_CONST_CONTEXT_GEN and self._cc_kv is not None:
            k, v = self._cc_kv
        else:
            # ctx[:1]: rows are identical for a constant context, so one row
            # suffices and dedupes the batched-tile replicas.
            ctx_src = ctx[:1] if use_cache else ctx
            with _prof_scope("dit.ca.kv"):
                k = self.k(ctx_src)
                v = self.v(ctx_src)
            with _prof_scope("dit.ca.qknorm"):
                k = self.norm_k(k)
            if use_cache:
                self._cc_kv = (k, v)
                self._cc_gen = _WAN_DIT_CONST_CONTEXT_GEN
        if k.shape[0] != q.shape[0]:
            k = k.expand(q.shape[0], -1, -1)
            v = v.expand(q.shape[0], -1, -1)
        with _prof_scope("dit.ca.core"):
            x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        with _prof_scope("dit.ca.o"):
            x = self.o(x)
        return x


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, self_attn_mask=None, self_attn_kv_cache=None, return_self_attn_kv=False):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        with _prof_scope("dit.blk.mod"):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
            if has_seq:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                    shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
                )
        with _prof_scope("dit.blk.norm1"):
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_out = self.self_attn(
            input_x,
            freqs,
            attn_mask=self_attn_mask,
            kv_cache=self_attn_kv_cache,
            return_kv=return_self_attn_kv,
        )
        self_attn_kv = None
        if return_self_attn_kv:
            self_attn_out, self_attn_kv = self_attn_out
        with _prof_scope("dit.blk.gate"):
            x = self.gate(x, gate_msa, self_attn_out)
        with _prof_scope("dit.blk.norm3"):
            norm3_x = self.norm3(x)
        cross_attn_out = self.cross_attn(norm3_x, context)
        x = x + cross_attn_out
        with _prof_scope("dit.blk.norm2"):
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        with _prof_scope("dit.ffn.fc1"):
            h = self.ffn[0](input_x)
        with _prof_scope("dit.ffn.act"):
            h = self.ffn[1](h)
        with _prof_scope("dit.ffn.fc2"):
            ffn_out = self.ffn[2](h)
        with _prof_scope("dit.blk.gate"):
            x = self.gate(x, gate_mlp, ffn_out)
        if return_self_attn_kv:
            return x, self_attn_kv
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    """Wan DiT restricted to the FastVR checkpoint architecture."""

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        **unsupported_options,
    ):
        enabled = [name for name, value in unsupported_options.items() if value]
        if enabled:
            raise ValueError(
                "Unsupported non-FastVR Wan options: " + ", ".join(sorted(enabled))
            )
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(dim // num_heads)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                window_causal_attention_chunk_size: int = 3,
                window_causal_attention_window_size: int = 2,
                window_causal_attention_first_chunk_size: int = 3,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.float()))
        t_mod = self.time_projection(t).to(x.dtype).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        
        ensure_temporal_rope_length(self, f)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        self_attn_mask = build_window_causal_self_attn_mask(
            frame_tokens=h * w,
            latent_frames=f,
            chunk_size=int(window_causal_attention_chunk_size),
            window_size=int(window_causal_attention_window_size),
            first_chunk_size=int(window_causal_attention_first_chunk_size),
            device=x.device,
        )

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs, self_attn_mask
                )
            else:
                x = block(x, context, t_mod, freqs, self_attn_mask)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


def upgrade_wan_model(model):
    """Attach FastVR attention behavior to a DiffSynth 2.1.0 Wan model in place."""
    if model is None or isinstance(model, WanModel):
        return model
    model.__class__ = WanModel
    for block in model.blocks:
        block.__class__ = DiTBlock
        block.self_attn.__class__ = SelfAttention
        block.self_attn.attn.__class__ = AttentionModule
        block.cross_attn.__class__ = CrossAttention
        block.cross_attn.attn.__class__ = AttentionModule
        block.cross_attn._cc_kv = None
        block.cross_attn._cc_gen = -1
    return model
