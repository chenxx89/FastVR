"""Small stateless helpers shared by FastVR inference paths."""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image


def normalize_fps(value, default: float = 30.0) -> float:
    """Return a positive finite FPS without rounding fractional source rates."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return float(default)
    return fps if math.isfinite(fps) and fps > 0 else float(default)


class ResolutionPlan(NamedTuple):
    input_width: int
    input_height: int
    processing_width: int
    processing_height: int
    model_width: int
    model_height: int
    output_width: int
    output_height: int
    upscale: float
    target_short_edge: int | None
    processing_mode: str
    recommended_upscale: float | None


def log_resolution_plan(plan: ResolutionPlan) -> None:
    """Print the shared processing and output resolution plan."""

    if plan.recommended_upscale is not None:
        print(
            f"[FastVR] Notice: target_short_edge={plan.target_short_edge} "
            f"exceeds the input short edge ({min(plan.input_width, plan.input_height)}). "
            "The model will upscale the input for processing, but the final output "
            f"size is still controlled by upscale={plan.upscale:g}. To keep the processed "
            "short-edge resolution in the saved output, consider setting --upscale "
            f"to at least {plan.recommended_upscale:.3g}."
        )
    print(
        f"[FastVR] Resolution: input={plan.input_width}x{plan.input_height}, "
        f"processing={plan.processing_width}x{plan.processing_height} "
        f"({plan.processing_mode}), model={plan.model_width}x{plan.model_height}, "
        f"output={plan.output_width}x{plan.output_height}"
    )


def audio_source_path(input_path: str, video_extensions: tuple[str, ...]) -> str | None:
    """Return a source video path eligible for optional audio passthrough."""

    if os.path.isfile(input_path) and os.path.splitext(input_path)[1].lower() in video_extensions:
        return input_path
    return None


def build_resolution_plan(
    *,
    input_width: int,
    input_height: int,
    upscale: float,
    target_short_edge: int | None,
    width_factor: int,
    height_factor: int,
) -> ResolutionPlan:
    """Build the shared spatial plan used by full and streaming inference."""
    output_width = max(1, int(round(input_width * upscale)))
    output_height = max(1, int(round(input_height * upscale)))
    recommended_upscale = None
    if target_short_edge is None:
        processing_width, processing_height = output_width, output_height
        processing_mode = f"upscale={upscale:g}"
    else:
        scale = target_short_edge / min(input_width, input_height)
        processing_width = max(1, int(round(input_width * scale)))
        processing_height = max(1, int(round(input_height * scale)))
        processing_mode = f"target_short_edge={target_short_edge}"
        if scale > max(upscale, 1.0):
            recommended_upscale = scale
    model_width = (
        (processing_width + width_factor - 1) // width_factor
    ) * width_factor
    model_height = (
        (processing_height + height_factor - 1) // height_factor
    ) * height_factor
    return ResolutionPlan(
        input_width=input_width,
        input_height=input_height,
        processing_width=processing_width,
        processing_height=processing_height,
        model_width=model_width,
        model_height=model_height,
        output_width=output_width,
        output_height=output_height,
        upscale=upscale,
        target_short_edge=target_short_edge,
        processing_mode=processing_mode,
        recommended_upscale=recommended_upscale,
    )


def pad_image_to_size(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Pad an image on the right and bottom by replicating edge pixels."""

    width, height = image.size
    if width == target_width and height == target_height:
        return image
    padded = Image.new(image.mode, (target_width, target_height))
    padded.paste(image, (0, 0))
    if target_width > width:
        edge = image.crop((width - 1, 0, width, height))
        padded.paste(edge.resize((target_width - width, height)), (width, 0))
    if target_height > height:
        edge = padded.crop((0, height - 1, target_width, height))
        padded.paste(edge.resize((target_width, target_height - height)), (0, height))
    return padded


def resize_frames_to_size(
    frames: list[Image.Image],
    target_width: int,
    target_height: int,
) -> list[Image.Image]:
    """Resize input frames to one exact processing or output size."""

    if not frames:
        raise ValueError("Need at least one frame to resize.")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target frame dimensions must be positive.")
    if all(frame.size == (target_width, target_height) for frame in frames):
        return frames
    return [
        frame.resize((target_width, target_height), Image.BICUBIC)
        for frame in frames
    ]


def validate_video_tensor_shape(
    video_tensor: torch.Tensor,
    *,
    label: str,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
    expected_batch: int = 1,
    expected_channels: int = 3,
) -> None:
    """Require an exact BCTHW shape before paired frame-wise operations."""

    if video_tensor.ndim != 5:
        raise RuntimeError(
            f"{label} must be BCTHW, got shape {tuple(video_tensor.shape)}."
        )
    expected_shape = (
        expected_batch,
        expected_channels,
        expected_frames,
        expected_height,
        expected_width,
    )
    if tuple(video_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{label} shape mismatch: expected {expected_shape}, "
            f"got {tuple(video_tensor.shape)}."
        )


def crop_video_tensor_spatial(
    video_tensor: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Remove right/bottom alignment padding from a BCTHW tensor."""

    height, width = video_tensor.shape[-2:]
    if height < target_height or width < target_width:
        raise ValueError(
            f"Cannot crop {width}x{height} video to {target_width}x{target_height}."
        )
    return video_tensor[:, :, :, :target_height, :target_width]


def resize_video_tensor_spatial(
    video_tensor: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize a decoded BCTHW tensor without cropping its content."""

    batch, channels, frames, height, width = video_tensor.shape
    if height == target_height and width == target_width:
        return video_tensor
    flattened = video_tensor.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    flattened = F.interpolate(
        flattened,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return flattened.reshape(
        batch, frames, channels, target_height, target_width
    ).permute(0, 2, 1, 3, 4)
