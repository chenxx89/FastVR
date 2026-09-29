"""Compact ComfyUI nodes for FastVR video enhancement."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar

from fastvr.inference import FastVRInferenceConfig, FastVRRuntime
from fastvr.models.download import FASTVR_REQUIRED_FILES, ensure_fastvr_checkpoint


MODEL_CATEGORY = "FastVR"
MODEL_TYPE = "FASTVR_MODEL"
DEFAULT_MODEL_NAME = "FastVR"


def _model_root() -> Path:
    root = Path(folder_paths.models_dir) / MODEL_CATEGORY
    root.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.add_model_folder_path(MODEL_CATEGORY, str(root))
    except Exception:
        # Older ComfyUI builds may not expose custom model categories through
        # this helper. The canonical models/FastVR path still works directly.
        pass
    return root


def _checkpoint_choices() -> list[str]:
    root = _model_root()
    choices = [DEFAULT_MODEL_NAME]
    for directory, _, filenames in os.walk(root):
        if all(filename in filenames for filename in FASTVR_REQUIRED_FILES):
            relative = os.path.relpath(directory, root)
            name = DEFAULT_MODEL_NAME if relative == "." else relative
            if name not in choices:
                choices.append(name)
    return sorted(
        choices,
        key=lambda value: (value != DEFAULT_MODEL_NAME, value.lower()),
    )


def _checkpoint_path(name: str) -> str:
    root = _model_root()
    if not name or name == DEFAULT_MODEL_NAME:
        return str(root)
    candidate = (root / name).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError(f"Invalid FastVR checkpoint directory: {name!r}")
    return str(candidate)


class FastVRModelLoader:
    """Resolve/download a checkpoint and create a reusable lazy model handle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_checkpoint_choices(),),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("fastvr_model",)
    FUNCTION = "load_model"
    CATEGORY = "FastVR"

    def load_model(self, checkpoint: str):
        checkpoint_dir = ensure_fastvr_checkpoint(_checkpoint_path(checkpoint))
        device = model_management.get_torch_device()
        if torch.device(device).type != "cuda":
            raise RuntimeError(
                f"FastVR requires an NVIDIA CUDA device; ComfyUI selected {device}."
            )
        return (FastVRRuntime(checkpoint_dir, device=device),)


class FastVRVideoEnhancer:
    """Enhance a ComfyUI IMAGE frame batch with the fixed FastVR pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fastvr_model": (MODEL_TYPE,),
                "images": ("IMAGE",),
                "upscale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1},
                ),
                "target_short_edge": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Set to 0 to process at the final output resolution selected by upscale; 1024 is recommended for speed.",
                    },
                ),
                "streaming": ("BOOLEAN", {"default": True}),
                "enable_denoise_tiling": ("BOOLEAN", {"default": True}),
                "color_fix": (["none", "adain", "wavelet"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "enhance"
    CATEGORY = "FastVR"

    def enhance(
        self,
        fastvr_model: FastVRRuntime,
        images: torch.Tensor,
        upscale: float,
        target_short_edge: int,
        streaming: bool,
        enable_denoise_tiling: bool,
        color_fix: str,
    ):
        if not isinstance(fastvr_model, FastVRRuntime):
            raise TypeError("fastvr_model must come from FastVR Model Loader.")
        total = int(images.shape[0]) if images.ndim else 0
        progress = ProgressBar(max(total, 1))

        def update(completed: int, frame_count: int) -> None:
            progress.update_absolute(completed, max(frame_count, 1))

        config = FastVRInferenceConfig(
            upscale=float(upscale),
            target_short_edge=(int(target_short_edge) or None),
            streaming=bool(streaming),
            enable_denoise_tiling=bool(enable_denoise_tiling),
            color_fix=None if color_fix == "none" else color_fix,
        )
        output = fastvr_model.enhance(
            images,
            config,
            progress_callback=update,
        )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "FastVRModelLoader": FastVRModelLoader,
    "FastVRVideoEnhancer": FastVRVideoEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVRModelLoader": "FastVR Model Loader",
    "FastVRVideoEnhancer": "FastVR Video Enhancer",
}
