"""FastVR training extensions for the frozen DiffSynth Wan VAE.

Stage 2 decodes one latent frame at a time and checkpoints the complete decoder
chunk. Compared with checkpointing individual decoder blocks over the whole
temporal window, this keeps high-resolution activations bounded to one latent
frame. Causal feature caches are detached between chunks, matching the original
FastVR training implementation and trading additional recomputation for a lower
peak memory footprint.
"""

from __future__ import annotations

import torch
from diffsynth.models.wan_video_vae import Decoder3d_38, unpatchify
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


def _scaled_latents(vae_model, latents: torch.Tensor, scale) -> torch.Tensor:
    if isinstance(scale[0], torch.Tensor):
        mean, inverse_std = (
            value.to(dtype=latents.dtype, device=latents.device) for value in scale
        )
        return latents / inverse_std.view(
            1, vae_model.z_dim, 1, 1, 1
        ) + mean.view(1, vae_model.z_dim, 1, 1, 1)
    scale = scale.to(dtype=latents.dtype, device=latents.device)
    return latents / scale[1] + scale[0]


def _detach_cache(cache):
    return [value.detach() if isinstance(value, torch.Tensor) else value for value in cache]


def _decode_chunk(vae_model, chunk, first_chunk, cache):
    feat_cache = list(cache)
    feat_idx = [0]
    if isinstance(vae_model.decoder, Decoder3d_38):
        output, feat_cache, _ = vae_model.decoder(
            chunk,
            feat_cache=feat_cache,
            feat_idx=feat_idx,
            first_chunk=first_chunk,
        )
    else:
        output, feat_cache, _ = vae_model.decoder(
            chunk,
            feat_cache=feat_cache,
            feat_idx=feat_idx,
        )
    return (output, *feat_cache)


def decode_wan_vae_window(
    vae,
    latents: torch.Tensor,
    *,
    device: str | torch.device,
    latent_start: int,
    latent_end: int,
    use_gradient_checkpointing: bool = True,
) -> torch.Tensor:
    """Decode ``[latent_start, latent_end)`` with a detached causal prefix.

    Prefix chunks are evaluated without gradients only to initialize causal
    caches. Target chunks are checkpointed independently, so backward never
    retains a full multi-frame decoder activation graph.
    """

    if latents.ndim != 5:
        raise ValueError(f"Wan VAE latents must use BCTHW layout, got {latents.shape}")
    latent_start = int(latent_start)
    latent_end = int(latent_end)
    if not 0 <= latent_start < latent_end <= latents.shape[2]:
        raise ValueError(
            f"Invalid latent window [{latent_start}, {latent_end}) for "
            f"{latents.shape[2]} latent frames."
        )
    vae_model = getattr(vae, "model", None)
    if vae_model is None or not hasattr(vae_model, "decoder"):
        raise TypeError("FastVR Stage 2 requires a DiffSynth Wan VAE.")

    latents = _scaled_latents(vae_model, latents.to(device), vae.scale)
    is_vae38 = isinstance(vae_model.decoder, Decoder3d_38)
    vae_model.clear_cache()
    cache = list(vae_model._feat_map)

    def run_chunk(chunk, global_index, cache_values, checkpoint_enabled):
        first_chunk = is_vae38 and global_index == 0

        def forward(current_chunk, *current_cache):
            return _decode_chunk(
                vae_model,
                current_chunk,
                first_chunk,
                current_cache,
            )

        if checkpoint_enabled and torch.is_grad_enabled() and chunk.requires_grad:
            outputs = gradient_checkpoint(
                forward,
                chunk,
                *cache_values,
                use_reentrant=False,
            )
        else:
            outputs = forward(chunk, *cache_values)
        output, *next_cache = outputs
        return output, _detach_cache(next_cache)

    try:
        if latent_start:
            with torch.no_grad():
                for index in range(latent_start):
                    chunk = vae_model.conv2(latents[:, :, index : index + 1])
                    _, cache = run_chunk(
                        chunk, index, cache, checkpoint_enabled=False
                    )

        decoded = []
        for index in range(latent_start, latent_end):
            chunk = vae_model.conv2(latents[:, :, index : index + 1])
            output, cache = run_chunk(
                chunk,
                index,
                cache,
                checkpoint_enabled=use_gradient_checkpointing,
            )
            decoded.append(output)

        video = torch.cat(decoded, dim=2)
        if is_vae38:
            video = unpatchify(video, patch_size=2)
        return video.clamp(-1, 1)
    finally:
        vae_model.clear_cache()
