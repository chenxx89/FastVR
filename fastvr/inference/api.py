"""Tensor-native public inference API shared by integrations such as ComfyUI."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F

from fastvr.inference.color_fix import TorchColorCorrector
from fastvr.inference.pipeline import (
    CAUSAL_SIZES,
    build_pipeline,
    load_prompt_embedding,
)
from fastvr.inference.streaming import FastVRStreamSession
from fastvr.inference.utils import (
    build_resolution_plan,
    crop_video_tensor_spatial,
    resize_video_tensor_spatial,
    validate_video_tensor_shape,
)
from fastvr.models import LightweightVAE
from fastvr.models.download import EMPTY_PROMPT_PATH, ensure_fastvr_checkpoint


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class FastVRInferenceConfig:
    """Small stable configuration surface for non-file FastVR integrations."""

    upscale: float = 1.0
    target_short_edge: int | None = 1024
    streaming: bool = True
    enable_denoise_tiling: bool = True
    color_fix: str | None = None
    vsr_target_timestep: float = 399.0
    denoise_tile_size: int = 64
    denoise_tile_stride: int = 52
    denoise_tile_batch_size: int = 1

    def __post_init__(self):
        if self.upscale <= 0:
            raise ValueError("upscale must be positive")
        if self.target_short_edge is not None and self.target_short_edge <= 0:
            raise ValueError("target_short_edge must be positive or None")
        if self.color_fix not in (None, "", "adain", "wavelet"):
            raise ValueError("color_fix must be None, 'adain', or 'wavelet'")
        if self.denoise_tile_size <= 0 or self.denoise_tile_stride <= 0:
            raise ValueError("denoise tile size and stride must be positive")
        if self.denoise_tile_batch_size <= 0:
            raise ValueError("denoise_tile_batch_size must be positive")


def load_fastvr_pipeline(checkpoint_dir: str, *, device=None):
    """Load the complete inference pipeline from one FastVR checkpoint directory."""

    checkpoint_dir = ensure_fastvr_checkpoint(checkpoint_dir)
    pipe = build_pipeline(SimpleNamespace(ckpt_path=checkpoint_dir), device=device)
    pipe.lightweight_vae = LightweightVAE(
        checkpoint_path=os.path.join(checkpoint_dir, "vae.safetensors")
    ).to(pipe.device)
    pipe.set_prompt_embedding(load_prompt_embedding(str(EMPTY_PROMPT_PATH)))
    return pipe


class FastVRRuntime:
    """Lazily loaded, serialized FastVR runtime suitable for graph-node reuse."""

    def __init__(self, checkpoint_dir: str, *, device=None):
        self.checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        self.device = device
        self._pipe = None
        self._lock = threading.RLock()

    @property
    def pipe(self):
        with self._lock:
            if self._pipe is None:
                self._pipe = load_fastvr_pipeline(
                    self.checkpoint_dir, device=self.device
                )
            return self._pipe

    def enhance(
        self,
        frames: torch.Tensor,
        config: FastVRInferenceConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> torch.Tensor:
        # Pipeline and VAE streaming caches are mutable per-video state. Serializing
        # calls prevents two graph executions from sharing those caches.
        with self._lock:
            return enhance_video_tensor(
                self.pipe,
                frames,
                config,
                progress_callback=progress_callback,
            )

    def unload(self) -> None:
        with self._lock:
            if self._pipe is not None:
                self._pipe.clear_fastvr_stream()
            self._pipe = None


def _validate_thwc(frames: torch.Tensor) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be a torch.Tensor, got {type(frames)!r}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "FastVR tensor input must use THWC layout with three channels, "
            f"got {tuple(frames.shape)}."
        )
    if frames.shape[0] == 0:
        raise ValueError("FastVR requires at least one input frame.")
    if not frames.is_floating_point():
        raise TypeError("FastVR tensor input must be floating point in [0, 1].")
    if not torch.isfinite(frames).all():
        raise ValueError("FastVR tensor input contains NaN or infinity.")
    return frames.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)


def _input_tchw(frames: torch.Tensor) -> torch.Tensor:
    return _validate_thwc(frames).permute(0, 3, 1, 2).contiguous()


def _resize_tchw(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] == (height, width):
        return frames
    return F.interpolate(
        frames,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).clamp_(0.0, 1.0)


def _pad_tchw(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    pad_h = height - frames.shape[-2]
    pad_w = width - frames.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Model padding cannot reduce the input resolution.")
    if pad_h == 0 and pad_w == 0:
        return frames
    return F.pad(frames, (0, pad_w, 0, pad_h), mode="replicate")


def _normalized_bcthw(frames: torch.Tensor, *, device, dtype) -> torch.Tensor:
    return (
        frames.permute(1, 0, 2, 3)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        .mul(2.0)
        .sub(1.0)
    )


def _output_thwc(video: torch.Tensor) -> torch.Tensor:
    return (
        video[0]
        .permute(1, 2, 3, 0)
        .add(1.0)
        .mul(0.5)
        .clamp(0.0, 1.0)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )


def _prepare_inputs(pipe, frames: torch.Tensor, config: FastVRInferenceConfig):
    input_count, _, input_height, input_width = frames.shape
    plan = build_resolution_plan(
        input_width=input_width,
        input_height=input_height,
        upscale=config.upscale,
        target_short_edge=config.target_short_edge,
        width_factor=pipe.width_division_factor,
        height_factor=pipe.height_division_factor,
    )
    processed = _resize_tchw(
        frames, plan.processing_height, plan.processing_width
    )
    model_input = _pad_tchw(processed, plan.model_height, plan.model_width)
    return input_count, plan, processed, model_input


def _apply_color_fix(pipe, decoded, reference, method):
    if not method:
        return decoded
    if pipe.color_corrector is None:
        pipe.color_corrector = TorchColorCorrector(levels=5)
    reference = _normalized_bcthw(
        reference, device=decoded.device, dtype=decoded.dtype
    )
    if decoded.shape != reference.shape:
        raise RuntimeError(
            "Color-fix tensors must have identical BCTHW shapes: "
            f"decoded={tuple(decoded.shape)}, reference={tuple(reference.shape)}."
        )
    return pipe.color_corrector(
        decoded,
        reference,
        clip_range=(-1, 1),
        chunk_size=16,
        method=method,
    )


def _enhance_full(pipe, frames, config, progress_callback):
    input_count, plan, processed, model_input = _prepare_inputs(pipe, frames, config)
    remainder = pipe.time_division_remainder
    factor = pipe.time_division_factor
    if input_count % factor != remainder:
        target_count = (
            (input_count - remainder + factor - 1) // factor
        ) * factor + remainder
        target_count = max(target_count, remainder)
        if target_count > input_count:
            model_input = torch.cat(
                [
                    model_input,
                    model_input[-1:].expand(
                        target_count - input_count, -1, -1, -1
                    ),
                ],
                dim=0,
            )

    if progress_callback:
        progress_callback(0, input_count)
    with pipe.lightweight_vae.stream_session():
        decoded = pipe(
            model_input,
            vsr_target_timestep=config.vsr_target_timestep,
            enable_denoise_tiling=config.enable_denoise_tiling,
            denoise_tile_size=config.denoise_tile_size,
            denoise_tile_stride=config.denoise_tile_stride,
            denoise_tile_batch_size=config.denoise_tile_batch_size,
            progress_bar_cmd=lambda iterable, **_: iterable,
        )
    decoded = decoded[:, :, :input_count].float()
    decoded = crop_video_tensor_spatial(
        decoded, plan.processing_height, plan.processing_width
    )
    validate_video_tensor_shape(
        decoded,
        label="FastVR tensor output",
        expected_frames=input_count,
        expected_height=plan.processing_height,
        expected_width=plan.processing_width,
    )
    decoded = _apply_color_fix(pipe, decoded, processed, config.color_fix)
    decoded = resize_video_tensor_spatial(
        decoded, plan.output_height, plan.output_width
    )
    if progress_callback:
        progress_callback(input_count, input_count)
    return _output_thwc(decoded)


def _enhance_streaming(pipe, frames, config, progress_callback):
    input_count, plan, processed, model_input = _prepare_inputs(pipe, frames, config)
    first_latents, latent_chunk, _ = CAUSAL_SIZES
    first_frames = 4 * first_latents
    chunk_frames = 4 * latent_chunk
    outputs = []
    latent_buffer = None
    reference_buffer = processed
    written = 0

    stream_args = SimpleNamespace(
        vsr_target_timestep=config.vsr_target_timestep,
        enable_denoise_tiling=config.enable_denoise_tiling,
        denoise_tile_size=config.denoise_tile_size,
        denoise_tile_stride=config.denoise_tile_stride,
        denoise_tile_batch_size=config.denoise_tile_batch_size,
    )
    starts = [0]
    if input_count > first_frames:
        starts.extend(range(first_frames, input_count, chunk_frames))

    if progress_callback:
        progress_callback(0, input_count)
    with FastVRStreamSession(pipe, stream_args):
        for chunk_id, start in enumerate(starts):
            size = first_frames if chunk_id == 0 else chunk_frames
            end = min(start + size, input_count)
            final_input = end == input_count
            chunk = model_input[start:end]
            if final_input:
                target_count = ((input_count - 1 + 3) // 4) * 4 + 1
                padding = target_count - input_count
                if padding:
                    chunk = torch.cat(
                        [chunk, chunk[-1:].expand(padding, -1, -1, -1)], dim=0
                    )
            model_video = _normalized_bcthw(
                chunk, device=pipe.device, dtype=pipe.torch_dtype
            )
            pipe.lightweight_vae.queue_decode_condition(
                model_video,
                target_height=plan.model_height,
                target_width=plan.model_width,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = pipe.lightweight_vae.encode_stream(
                    model_video, final=final_input, device=pipe.device
                )
            if encoded is not None:
                latent_buffer = (
                    encoded
                    if latent_buffer is None
                    else torch.cat([latent_buffer, encoded], dim=2)
                )

            while latent_buffer is not None and latent_buffer.shape[2]:
                expected = first_latents if written == 0 else latent_chunk
                if latent_buffer.shape[2] < expected and not final_input:
                    break
                take = min(expected, latent_buffer.shape[2])
                current = latent_buffer[:, :, :take]
                latent_buffer = latent_buffer[:, :, take:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    denoised = pipe.denoise_fastvr_stream_chunk(current)
                final_decode = final_input and latent_buffer.shape[2] == 0
                with torch.autocast(device_type="cuda", enabled=False):
                    decoded = pipe.lightweight_vae.decode_stream(
                        denoised, final=final_decode
                    ).float()

                keep = min(decoded.shape[2], input_count - written)
                if keep == 0:
                    continue
                decoded = crop_video_tensor_spatial(
                    decoded[:, :, :keep],
                    plan.processing_height,
                    plan.processing_width,
                )
                reference = reference_buffer[:keep]
                reference_buffer = reference_buffer[keep:]
                decoded = _apply_color_fix(
                    pipe, decoded, reference, config.color_fix
                )
                decoded = resize_video_tensor_spatial(
                    decoded, plan.output_height, plan.output_width
                )
                outputs.append(_output_thwc(decoded))
                written += keep
                if progress_callback:
                    progress_callback(written, input_count)

    if written != input_count:
        raise RuntimeError(
            f"FastVR streaming output frame mismatch: input={input_count}, output={written}."
        )
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def enhance_video_tensor(
    pipe,
    frames: torch.Tensor,
    config: FastVRInferenceConfig | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> torch.Tensor:
    """Enhance a Comfy-style THWC tensor and return a CPU THWC tensor in [0, 1]."""

    config = config or FastVRInferenceConfig()
    frames = _input_tchw(frames)
    if pipe.lightweight_vae is None:
        raise RuntimeError("FastVR inference requires pipe.lightweight_vae.")
    if config.streaming:
        return _enhance_streaming(pipe, frames, config, progress_callback)
    return _enhance_full(pipe, frames, config, progress_callback)


__all__ = [
    "FastVRInferenceConfig",
    "FastVRRuntime",
    "enhance_video_tensor",
    "load_fastvr_pipeline",
]
