"""FastVR training module and model assembly."""

from __future__ import annotations

from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.diffusion import DiffusionTrainingModule

from fastvr.inference.pipeline import FastVRPipeline
from fastvr.models.checkpoint import dit_checkpoint_paths
from fastvr.models.download import EMPTY_PROMPT_PATH
from fastvr.training.config import TrainingConfig
from fastvr.training.losses import flow_match_sft_loss, pixel_loss


class FastVRTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        config: TrainingConfig,
        *,
        device: str | torch.device,
    ):
        super().__init__()
        dit_path = (
            list(config.dit_path)
            if isinstance(config.dit_path, tuple)
            else config.dit_path
        )
        if isinstance(dit_path, str) and Path(dit_path).is_dir():
            dit_path = dit_checkpoint_paths(dit_path)
        model_configs = [ModelConfig(path=dit_path), ModelConfig(path=config.vae_path)]

        self.pipe = FastVRPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except(["dit"])
        self.pipe.dit.train()
        self.task = config.task
        if self.pipe.vae is not None:
            self.pipe.vae.eval()

        embedding = torch.load(
            EMPTY_PROMPT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"{EMPTY_PROMPT_PATH} must contain a torch.Tensor.")
        self.pipe.set_prompt_embedding(embedding)

        self.loss_config = config.loss
        self.causal_sizes = config.causal_sizes
        if config.torch_compile:
            print("[FastVR] Compiling DiT with torch.compile...")
            self.pipe.dit = torch.compile(self.pipe.dit)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pipe.dit.train(mode)
        if self.pipe.vae is not None:
            self.pipe.vae.eval()
        return self

    def get_pipeline_inputs(self, data: dict):
        gt_tensor = data.get("hq_video")
        lq_tensor = data.get("lq_video")
        if not isinstance(gt_tensor, torch.Tensor) or not isinstance(
            lq_tensor, torch.Tensor
        ):
            raise ValueError("Training sample must contain HQ and LQ video tensors.")
        if gt_tensor.shape != lq_tensor.shape or gt_tensor.ndim != 4:
            raise ValueError(
                f"HQ/LQ tensors must have matching TCHW shapes, got "
                f"{gt_tensor.shape} and {lq_tensor.shape}."
            )

        num_frames, _, height, width = gt_tensor.shape
        first_chunk_size, chunk_size, window_size = self.causal_sizes
        inputs_shared = {
            "input_video": gt_tensor,
            "lq_video": lq_tensor,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "tiled": False,
            "tile_size": (34, 34),
            "tile_stride": (18, 16),
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
            "vsr_target_timestep": self.loss_config.vsr_target_timestep,
            "consistency_weight": self.loss_config.consistency_weight,
            "gt_pixel_tensor": gt_tensor,
            "pixel_l1_weight": self.loss_config.pixel_l1_weight,
            "dists_weight": self.loss_config.dists_weight,
            "pixel_loss_latent_window_size": self.loss_config.pixel_loss_latent_window_size,
            "pixel_loss_random_window": self.loss_config.pixel_loss_random_window,
            "vgg16_weights_path": self.loss_config.vgg16_weights_path,
            "dists_weights_path": self.loss_config.dists_weights_path,
            "window_causal_attention_first_chunk_size": first_chunk_size,
            "window_causal_attention_chunk_size": chunk_size,
            "window_causal_attention_window_size": window_size,
            "window_causal_attention_first_chunk_loss_weight": (
                self.loss_config.first_chunk_loss_weight
            ),
        }
        return inputs_shared, {}, {}

    def forward(self, data: dict):
        inputs = self._transfer_to_device(self.get_pipeline_inputs(data))
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs
        inputs_posi["context"] = self.pipe.get_prompt_embedding()
        loss_fn = flow_match_sft_loss if self.task == "sft" else pixel_loss
        return loss_fn(self.pipe, **inputs_shared, **inputs_posi)

    def _transfer_to_device(self, value):
        if isinstance(value, torch.Tensor):
            dtype = self.pipe.torch_dtype if value.is_floating_point() else value.dtype
            return value.to(device=self.pipe.device, dtype=dtype, non_blocking=True)
        if isinstance(value, tuple):
            return tuple(self._transfer_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._transfer_to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: self._transfer_to_device(item) for key, item in value.items()}
        return value
