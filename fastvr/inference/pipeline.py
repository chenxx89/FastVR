"""Complete FastVR inference pipeline and construction helpers."""

import contextlib
import io
import itertools
import os
import math
from typing import Optional, Union

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit
from diffsynth.models.wan_video_vae import WanVideoVAE

from fastvr.models.checkpoint import dit_checkpoint_paths
from fastvr.models.wan_video_dit import (
    WanModel,
    build_window_causal_chunk_ranges,
    build_window_causal_self_attn_mask,
    ensure_temporal_rope_length,
    set_dit_const_context,
    sinusoidal_embedding_1d,
    trim_window_causal_kv_cache,
    upgrade_wan_model,
    window_causal_history_frames,
)


CAUSAL_SIZES = (3, 3, 2)
INFERENCE_SHIFT = 5.0


class _LocalModelConfig(ModelConfig):
    """DiffSynth model config for local checkpoint files without downloader notices."""

    def check_download_source(self):
        # FastVR resolves and downloads its checkpoint directory before pipeline
        # construction, so DiffSynth's downloader configuration is irrelevant.
        return None


def align_time_embedding_input_dtype(time_embedding_input: torch.Tensor, time_embedding_module: torch.nn.Module) -> torch.Tensor:
    for parameter in time_embedding_module.parameters():
        return time_embedding_input.to(dtype=parameter.dtype)
    return time_embedding_input

def prepare_denoise_tiling_infos_generator(
    enable_spatial_tiling,
    enable_temporal_tiling,
    latents,
    tile_size,
    tile_stride,
    temporal_tile_size,
    temporal_tile_stride,
):
    if not enable_spatial_tiling and not enable_temporal_tiling:
        yield [slice(None), slice(None), slice(None), slice(None), slice(None)], torch.ones_like(latents)
        return

    _, _, num_frames, height, width = latents.shape

    if not enable_spatial_tiling:
        tile_size = max(height, width)
    if not enable_temporal_tiling:
        temporal_tile_size = num_frames

    def create_start_indices(size, tile_size, tile_stride):
        if size <= tile_size:
            tile_stride = tile_size
        else:
            num_tiles = (size - tile_size) // tile_stride + 1
            if (size - tile_size) % tile_stride != 0:
                num_tiles += 1
            tile_stride = math.ceil((size - tile_size) / (num_tiles - 1))
        i_list = list(range(0, max(1, size - tile_size + 1), tile_stride))
        if size >= tile_size and (size - tile_size) % tile_stride != 0:
            i_list.append(size - tile_size)
        return i_list, tile_size, tile_stride

    ti_list, t_tile_size, t_tile_stride = create_start_indices(num_frames, temporal_tile_size, temporal_tile_stride)
    hi_list, h_tile_size, h_tile_stride = create_start_indices(height, tile_size, tile_stride)
    wi_list, w_tile_size, w_tile_stride = create_start_indices(width, tile_size, tile_stride)

    def compute_valid_weights_range(i, i_end, size, tile_size, tile_stride):
        float_padding = (tile_size - tile_stride) / 2
        end = tile_size - math.floor(float_padding) if i_end < size else tile_size
        start = math.ceil(float_padding) if i > 0 else 0
        remainder = i % tile_stride
        if remainder > 0:
            start = tile_size - (math.floor(float_padding) + remainder)
        return slice(start, end)

    for ti, hi, wi in itertools.product(ti_list, hi_list, wi_list):
        ti_end = min(ti + t_tile_size, num_frames)
        hi_end = min(hi + h_tile_size, height)
        wi_end = min(wi + w_tile_size, width)
        tile_slice = [slice(None), slice(None), slice(ti, ti_end), slice(hi, hi_end), slice(wi, wi_end)]

        t_valid_slice = compute_valid_weights_range(ti, ti_end, num_frames, t_tile_size, t_tile_stride)
        h_valid_slice = compute_valid_weights_range(hi, hi_end, height, h_tile_size, h_tile_stride)
        w_valid_slice = compute_valid_weights_range(wi, wi_end, width, w_tile_size, w_tile_stride)
        weights = torch.zeros((1, 1, ti_end - ti, hi_end - hi, wi_end - wi))
        weights[:, :, t_valid_slice, h_valid_slice, w_valid_slice] = 1

        yield tile_slice, weights.to(device=latents.device, dtype=latents.dtype)

class FastVRPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.dit: Optional[WanModel] = None
        self.vae: Optional[WanVideoVAE] = None
        self.lightweight_vae: Optional[torch.nn.Module] = None
        self.fixed_prompt_embedding: Optional[torch.Tensor] = None
        self.color_corrector: Optional[torch.nn.Module] = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_LightweightVAEEncoder(),
            WanVideoUnit_LQLatentInitializer(),
        ]
        self.inference_units = tuple(self.units)
        self.post_units = []
        self.model_fn = model_fn_wan_video
        self.compilable_models = ["dit"]
        self._fastvr_stream_state = None

    def set_prompt_embedding(self, embedding: torch.Tensor) -> None:
        """Set the one fixed prompt embedding used by training and inference."""
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"Prompt embedding must be a torch.Tensor, got {type(embedding)!r}."
            )
        self.fixed_prompt_embedding = embedding.detach()

    def get_prompt_embedding(self) -> torch.Tensor:
        """Return the fixed prompt embedding on the active pipeline device."""
        if self.fixed_prompt_embedding is None:
            raise RuntimeError("FastVR prompt embedding has not been configured.")
        self.fixed_prompt_embedding = self.fixed_prompt_embedding.to(
            device=self.device, dtype=self.torch_dtype
        )
        return self.fixed_prompt_embedding



    def _slice_denoise_inputs(self, inputs, tile_slice, full_latents_shape):
        sliced_inputs = {}
        full_t, full_h, full_w = full_latents_shape[2:]
        for name, value in inputs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 5
                and value.shape[2] == full_t
                and value.shape[3] == full_h
                and value.shape[4] == full_w
            ):
                sliced_inputs[name] = value[tuple(tile_slice)]
            else:
                sliced_inputs[name] = value
        return sliced_inputs

    @staticmethod
    def _new_window_causal_cache_bank(dit, num_steps):
        return [[[] for _ in dit.blocks] for _ in range(num_steps)]

    def _window_causal_model_call(
        self,
        models,
        inputs_shared,
        inputs_cond,
        timestep,
        chunk_start,
        step_id,
        cache_bank,
        write_cache,
    ):
        inputs = dict(
            inputs_shared,
            window_causal_attention_chunk_start=chunk_start,
            window_causal_attention_kv_cache=cache_bank[step_id],
            window_causal_attention_write_kv_cache=write_cache,
        )
        return self.model_fn(**models, **inputs, **inputs_cond, timestep=timestep)

    @staticmethod
    def _cat_window_causal_inputs(items):
        merged = {}
        for key in items[0].keys():
            values = [item[key] for item in items]
            if all(isinstance(value, torch.Tensor) for value in values):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values[0]
        return merged

    @staticmethod
    def _ensure_window_causal_tile_cache(cache_bank, tile_id, step_id, dit):
        while len(cache_bank) <= tile_id:
            cache_bank.append([])
        while len(cache_bank[tile_id]) <= step_id:
            cache_bank[tile_id].append([[] for _ in dit.blocks])

    @staticmethod
    def _batch_window_causal_cache(cache_bank, tile_ids, step_id):
        layer_count = len(cache_bank[tile_ids[0]][step_id])
        batched_cache = []
        for layer_id in range(layer_count):
            layer_caches = [cache_bank[tile_id][step_id][layer_id] for tile_id in tile_ids]
            if not layer_caches[0]:
                batched_cache.append([])
                continue
            batched_cache.append([
                (
                    torch.cat([layer_cache[slot_id][0] for layer_cache in layer_caches], dim=0),
                    torch.cat([layer_cache[slot_id][1] for layer_cache in layer_caches], dim=0),
                )
                for slot_id in range(len(layer_caches[0]))
            ])
        return batched_cache

    @staticmethod
    def _commit_window_causal_batched_cache(
        cache_bank, tile_ids, step_id, batched_cache, batch_sizes, max_history_tokens
    ):
        if max_history_tokens <= 0:
            for tile_id in tile_ids:
                for layer_cache in cache_bank[tile_id][step_id]:
                    layer_cache.clear()
            return
        offsets = [0]
        for batch_size in batch_sizes:
            offsets.append(offsets[-1] + batch_size)
        for layer_id, layer_cache in enumerate(batched_cache):
            if not layer_cache:
                continue
            # `layer_cache[-1]` is the chunk model_fn just appended; it may already
            # be trimmed to the history budget, which is exactly what we cache.
            current_k, current_v = layer_cache[-1]
            for item_id, tile_id in enumerate(tile_ids):
                start, end = offsets[item_id], offsets[item_id + 1]
                tile_layer_cache = cache_bank[tile_id][step_id][layer_id]
                tile_layer_cache.append((current_k[start:end].detach(), current_v[start:end].detach()))
                trim_window_causal_kv_cache(tile_layer_cache, max_history_tokens)

    def _window_causal_predict_chunk(
        self,
        models,
        inputs_shared,
        inputs_cond,
        timestep,
        chunk_start,
        step_id,
        cache_bank,
        write_cache,
        enable_spatial_tiling,
        tile_size,
        tile_stride,
        denoise_tile_batch_size=1,
    ):
        latents = inputs_shared["latents"]
        if not enable_spatial_tiling:
            return self._window_causal_model_call(
                models,
                inputs_shared,
                inputs_cond,
                timestep,
                chunk_start,
                step_id,
                cache_bank,
                write_cache,
            )

        pred_meshgrid = torch.zeros_like(latents)
        weights_meshgrid = torch.zeros_like(latents[:, :1])
        full_latents_shape = latents.shape
        tile_records = []
        for tile_id, (tile_slice, tile_weights) in enumerate(
            prepare_denoise_tiling_infos_generator(
                True,
                False,
                latents,
                tile_size,
                tile_stride,
                latents.shape[2],
                latents.shape[2],
            )
        ):
            self._ensure_window_causal_tile_cache(
                cache_bank, tile_id, step_id, models["dit"]
            )
            inputs_shared_tile = self._slice_denoise_inputs(
                inputs_shared, tile_slice, full_latents_shape
            )
            tile_records.append({
                "tile_id": tile_id,
                "tile_slice": tile_slice,
                "tile_weights": tile_weights.to(
                    device=latents.device, dtype=latents.dtype
                ),
                "inputs_shared": inputs_shared_tile,
                "inputs_cond": self._slice_denoise_inputs(
                    inputs_cond, tile_slice, full_latents_shape
                ),
                "chunk_start": chunk_start + (tile_slice[2].start or 0),
                "signature": tuple(inputs_shared_tile["latents"].shape[1:]),
            })

        groups = {}
        for record in tile_records:
            groups.setdefault(record["signature"], []).append(record)
        batch_size = (
            len(tile_records)
            if denoise_tile_batch_size is None or denoise_tile_batch_size <= 0
            else int(denoise_tile_batch_size)
        )
        max_history_frames = window_causal_history_frames(
            int(inputs_shared.get("window_causal_attention_chunk_size", 3)),
            int(inputs_shared.get("window_causal_attention_window_size", 2)),
        )
        patch_size = getattr(models["dit"], "patch_size", (1, 2, 2))

        for group in groups.values():
            for batch_start in range(0, len(group), batch_size):
                batch = group[batch_start:batch_start + batch_size]
                tile_ids = [record["tile_id"] for record in batch]
                batch_sizes = [
                    record["inputs_shared"]["latents"].shape[0]
                    for record in batch
                ]
                tile_latents_shape = batch[0]["inputs_shared"]["latents"].shape
                tokens_per_frame = (
                    tile_latents_shape[3] // patch_size[1]
                ) * (
                    tile_latents_shape[4] // patch_size[2]
                )
                max_history_tokens = max_history_frames * tokens_per_frame
                batched_shared = self._cat_window_causal_inputs(
                    [record["inputs_shared"] for record in batch]
                )
                batched_cond = self._cat_window_causal_inputs(
                    [record["inputs_cond"] for record in batch]
                )
                batched_cache = self._batch_window_causal_cache(
                    cache_bank, tile_ids, step_id
                )
                batched_cache_bank = [None] * (step_id + 1)
                batched_cache_bank[step_id] = batched_cache
                prediction = self._window_causal_model_call(
                    models,
                    batched_shared,
                    batched_cond,
                    timestep,
                    batch[0]["chunk_start"],
                    step_id,
                    batched_cache_bank,
                    write_cache,
                )
                if write_cache:
                    self._commit_window_causal_batched_cache(
                        cache_bank,
                        tile_ids,
                        step_id,
                        batched_cache,
                        batch_sizes,
                        max_history_tokens,
                    )

                for record, pred_tile in zip(
                    batch, prediction.split(batch_sizes, dim=0)
                ):
                    tile_slice = record["tile_slice"]
                    tile_weights = record["tile_weights"]
                    pred_meshgrid[tuple(tile_slice)] += pred_tile * tile_weights
                    weights_meshgrid[tuple(tile_slice)] += tile_weights
        return pred_meshgrid / weights_meshgrid

    def window_causal_streaming_denoise(
        self,
        models,
        inputs_shared,
        inputs_cond,
        scheduler_timesteps,
        single_step_sigmas=None,
        enable_spatial_tiling=True,
        enable_temporal_tiling=False,
        tile_size=64,
        tile_stride=32,
        denoise_tile_batch_size=1,
        progress_bar_cmd=None,
        progress_bar_desc=None,
        on_chunk_done=None,
    ):
        if enable_temporal_tiling:
            raise NotImplementedError(
                "window-causal streaming does not support temporal denoise tiling"
            )
        latents = inputs_shared["latents"].clone()
        chunk_ranges = build_window_causal_chunk_ranges(
            latents.shape[2],
            int(inputs_shared.get("window_causal_attention_chunk_size", 3)),
            int(inputs_shared.get("window_causal_attention_first_chunk_size", 3)),
        )
        num_steps = len(scheduler_timesteps)
        use_tiling = bool(enable_spatial_tiling)
        cache_bank = (
            []
            if use_tiling
            else self._new_window_causal_cache_bank(models["dit"], num_steps)
        )

        chunk_iter = chunk_ranges
        if progress_bar_cmd is not None:
            chunk_iter = progress_bar_cmd(
                chunk_ranges,
                desc=progress_bar_desc or "Window causal chunks",
                total=len(chunk_ranges),
            )

        for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_iter):
            for step_id, scheduler_timestep in enumerate(scheduler_timesteps):
                inputs_shared["latents"] = latents
                chunk_slice = [
                    slice(None),
                    slice(None),
                    slice(chunk_start, chunk_end),
                    slice(None),
                    slice(None),
                ]
                chunk_inputs = self._slice_denoise_inputs(
                    inputs_shared, chunk_slice, latents.shape
                )
                timestep = scheduler_timestep.to(
                    dtype=torch.float32, device=latents.device
                ).reshape(1)
                prediction = self._window_causal_predict_chunk(
                    models,
                    chunk_inputs,
                    inputs_cond,
                    timestep,
                    chunk_start,
                    step_id,
                    cache_bank,
                    True,
                    use_tiling,
                    tile_size,
                    tile_stride,
                    denoise_tile_batch_size,
                )
                chunk_latents = chunk_inputs["latents"]
                if single_step_sigmas is None:
                    chunk_next = self.scheduler.step(
                        prediction, scheduler_timestep, chunk_latents
                    )
                else:
                    sigma = single_step_sigmas[step_id].to(
                        device=latents.device, dtype=latents.dtype
                    )
                    chunk_next = chunk_latents - sigma * prediction
                latents[:, :, chunk_start:chunk_end] = chunk_next
                if "first_frame_latents" in inputs_shared and chunk_start == 0:
                    latents[:, :, 0:1] = inputs_shared["first_frame_latents"]
            if on_chunk_done is not None:
                on_chunk_done(
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    num_chunks=len(chunk_ranges),
                    latents=latents,
                )
        inputs_shared["latents"] = latents
        return latents

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: Optional[list[ModelConfig]] = None,
        vram_limit: Optional[float] = None,
    ):
        """Load only the DiT and optional official Wan VAE/image encoder."""
        pipe = FastVRPipeline(device=device, torch_dtype=torch_dtype)
        # DiffSynth prints the complete detected model configuration. FastVR
        # reports its checkpoint path after construction, so suppress the
        # redundant third-party loader output here.
        with contextlib.redirect_stdout(io.StringIO()):
            model_pool = pipe.download_and_load_models(model_configs or [], vram_limit)
            dit = model_pool.fetch_model("wan_video_dit", index=2)
            vae = (
                model_pool.fetch_model("wan_video_vae")
                if "wan_video_vae" in model_pool.model_name
                else None
            )
        if isinstance(dit, list):
            if len(dit) != 1:
                raise ValueError("FastVR requires exactly one Wan DiT checkpoint.")
            dit = dit[0]
        pipe.dit = upgrade_wan_model(dit) if dit is not None else None
        pipe.vae = vae
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    def _prepare_fastvr_single_step_scheduler(self, vsr_target_timestep):
        self.scheduler.set_timesteps(num_inference_steps=1000, shift=INFERENCE_SHIFT)
        timestep_id = torch.argmin(
            (self.scheduler.timesteps - float(vsr_target_timestep)).abs()
        )
        matched_timestep = float(self.scheduler.timesteps[timestep_id].item())
        matched_sigma = float(self.scheduler.sigmas[timestep_id].item())
        denominator = INFERENCE_SHIFT - matched_sigma * (INFERENCE_SHIFT - 1.0)
        denoising_strength = 1.0 if denominator <= 0 else matched_sigma / denominator
        denoising_strength = min(max(denoising_strength, 0.0), 1.0)
        self.scheduler.set_timesteps(
            num_inference_steps=1,
            denoising_strength=denoising_strength,
            shift=INFERENCE_SHIFT,
        )
        return matched_timestep, matched_sigma, denoising_strength

    def initialize_fastvr_stream(
        self,
        *,
        vsr_target_timestep,
        enable_denoise_tiling,
        denoise_tile_size,
        denoise_tile_stride,
        denoise_tile_batch_size,
    ):
        """Initialize persistent DiT state for one streamed video."""
        if self._fastvr_stream_state is not None:
            raise RuntimeError("FastVR DiT stream is already active")
        first_chunk_size, chunk_size, window_size = CAUSAL_SIZES
        self._prepare_fastvr_single_step_scheduler(vsr_target_timestep)
        self.load_models_to_device(self.in_iteration_models)
        models = {
            name: getattr(self, name)
            for name in self.in_iteration_models
        }
        use_tiling = bool(enable_denoise_tiling)
        context = self.get_prompt_embedding()
        timestep = self.scheduler.timesteps[0].to(
            dtype=torch.float32, device=self.device
        ).reshape(1)
        time_embedding_input = sinusoidal_embedding_1d(models["dit"].freq_dim, timestep)
        time_embedding_input = align_time_embedding_input_dtype(
            time_embedding_input, models["dit"].time_embedding
        )
        timestep_embedding = models["dit"].time_embedding(time_embedding_input)
        timestep_modulation = models["dit"].time_projection(
            timestep_embedding
        ).unflatten(1, (6, models["dit"].dim))
        embedded_context = models["dit"].text_embedding(context)
        set_dit_const_context(True)
        self._fastvr_stream_state = {
            "models": models,
            "context": context,
            "embedded_context": embedded_context,
            "timestep_embedding": timestep_embedding,
            "timestep_modulation": timestep_modulation,
            "cache_bank": (
                [] if use_tiling
                else self._new_window_causal_cache_bank(models["dit"], 1)
            ),
            "latent_offset": 0,
            "first_chunk_size": int(first_chunk_size),
            "chunk_size": int(chunk_size),
            "window_size": int(window_size),
            "use_tiling": use_tiling,
            "tile_size": int(denoise_tile_size),
            "tile_stride": int(denoise_tile_stride),
            "tile_batch_size": int(denoise_tile_batch_size),
        }

    def clear_fastvr_stream(self):
        """Release the KV cache and all per-video DiT stream state."""
        self._fastvr_stream_state = None
        set_dit_const_context(False)

    @torch.no_grad()
    def denoise_fastvr_stream_chunk(self, latents):
        """Denoise one causal latent chunk using persistent per-video KV state."""
        state = self._fastvr_stream_state
        if state is None:
            raise RuntimeError("FastVR DiT stream is not initialized")
        expected = state["first_chunk_size"] if state["latent_offset"] == 0 else state["chunk_size"]
        if latents.shape[2] > expected:
            raise ValueError(
                f"Streaming latent chunk has {latents.shape[2]} frames; expected at most {expected}"
            )
        inputs_shared = {
            "latents": latents,
            "lq_latents": latents,
            "fastvr_embedded_context": state["embedded_context"],
            "fastvr_timestep_embedding": state["timestep_embedding"],
            "fastvr_timestep_modulation": state["timestep_modulation"],
            "window_causal_attention_first_chunk_size": state["first_chunk_size"],
            "window_causal_attention_chunk_size": state["chunk_size"],
            "window_causal_attention_window_size": state["window_size"],
        }
        timestep = self.scheduler.timesteps[0].to(
            dtype=torch.float32, device=latents.device
        ).reshape(1)
        prediction = self._window_causal_predict_chunk(
            state["models"],
            inputs_shared,
            {"context": state["context"]},
            timestep,
            state["latent_offset"],
            0,
            state["cache_bank"],
            True,
            state["use_tiling"],
            state["tile_size"],
            state["tile_stride"],
            state["tile_batch_size"],
        )
        output = self.scheduler.step(prediction, self.scheduler.timesteps[0], latents)
        state["latent_offset"] += latents.shape[2]
        return output

    @torch.no_grad()
    def __call__(
        self,
        lq_frames: Union[list[Image.Image], torch.Tensor],
        *,
        vsr_target_timestep: float = 399.0,
        enable_denoise_tiling: bool = True,
        denoise_tile_size: int = 64,
        denoise_tile_stride: int = 32,
        denoise_tile_batch_size: int = 1,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor:
        """Run the single FastVR model path and return a decoded BCTHW tensor."""

        if isinstance(lq_frames, torch.Tensor):
            if lq_frames.ndim != 4 or lq_frames.shape[1] != 3:
                raise ValueError(
                    "Tensor input must use TCHW layout with three channels, "
                    f"got {tuple(lq_frames.shape)}."
                )
            if lq_frames.shape[0] == 0:
                raise ValueError("FastVR inference requires at least one input frame.")
            num_frames, _, height, width = lq_frames.shape
        else:
            if not lq_frames:
                raise ValueError("FastVR inference requires at least one input frame.")
            width, height = lq_frames[0].size
            if any(frame.size != (width, height) for frame in lq_frames):
                raise ValueError("All FastVR model input frames must have the same size.")
            num_frames = len(lq_frames)
        if self.lightweight_vae is None:
            raise RuntimeError("FastVR inference requires pipe.lightweight_vae.")
        first_chunk_size, chunk_size, window_size = CAUSAL_SIZES

        # Match the training timestep and construct the equivalent single step.
        matched_timestep, matched_sigma, denoising_strength = (
            self._prepare_fastvr_single_step_scheduler(vsr_target_timestep)
        )

        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            inputs_shared = {
                "lq_video": lq_frames,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "tiled": False,
                "window_causal_attention_chunk_size": chunk_size,
                "window_causal_attention_window_size": window_size,
                "window_causal_attention_first_chunk_size": first_chunk_size,
            }
            unit_inputs = (inputs_shared, {}, {})
            for unit in self.inference_units:
                unit_inputs = self.unit_runner(unit, self, *unit_inputs)
            inputs_shared = unit_inputs[0]
            inputs_cond = {"context": self.get_prompt_embedding()}

            self.load_models_to_device(self.in_iteration_models)
            models = {
                name: getattr(self, name)
                for name in self.in_iteration_models
            }
            latents = self.window_causal_streaming_denoise(
                models=models,
                inputs_shared=inputs_shared,
                inputs_cond=inputs_cond,
                scheduler_timesteps=self.scheduler.timesteps,
                enable_spatial_tiling=enable_denoise_tiling,
                tile_size=denoise_tile_size,
                tile_stride=denoise_tile_stride,
                denoise_tile_batch_size=denoise_tile_batch_size,
                progress_bar_cmd=progress_bar_cmd,
                progress_bar_desc="Window causal chunks",
            )

        with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
            return self.lightweight_vae.decode(
                latents,
                condition=_preprocess_video(self, lq_frames),
            ).float()


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: FastVRPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    """Encode the HQ training video with the official Wan VAE."""

    def __init__(self):
        super().__init__(
            input_params=("input_video", "tiled", "tile_size", "tile_stride"),
            output_params=("hq_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: FastVRPipeline, input_video, tiled, tile_size, tile_stride):
        if input_video is None:
            return {}
        if pipe.vae is None:
            raise RuntimeError("FastVR training requires the official Wan VAE.")
        pipe.load_models_to_device(("vae",))
        input_video = _preprocess_video(pipe, input_video)
        with torch.no_grad():
            if tiled:
                hq_latents = pipe.vae.encode(
                    input_video,
                    device=pipe.device,
                    tiled=True,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            else:
                # WanVideoVAE.encode() is an inference-oriented batch wrapper
                # that first copies every input sample back to CPU. Training
                # tensors are already on the target device, so call the
                # tensor-native path and avoid a blocking GPU -> CPU -> GPU
                # round trip every step.
                hq_latents = pipe.vae.single_encode(input_video, device=pipe.device)
        hq_latents = hq_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"hq_latents": hq_latents}



class WanVideoUnit_LightweightVAEEncoder(PipelineUnit):
    """Encode every LQ input with FastVR's single LightweightVAE."""

    def __init__(self):
        super().__init__(
            input_params=("lq_video",),
            output_params=("lq_latents",),
            onload_model_names=("lightweight_vae",),
        )

    def process(self, pipe: FastVRPipeline, lq_video):
        if lq_video is None:
            return {}
        video_tensor = _preprocess_video(pipe, lq_video)
        if pipe.scheduler.training:
            if pipe.vae is None:
                raise RuntimeError("FastVR training requires the official Wan VAE.")
            pipe.load_models_to_device(("vae",))
            # See WanVideoUnit_InputVideoEmbedder: avoid the CPU-staging wrapper
            # for the already-device-resident training tensor.
            with torch.no_grad():
                lq_latents = pipe.vae.single_encode(
                    video_tensor, device=pipe.device
                )
        else:
            lightweight_vae = getattr(pipe, "lightweight_vae", None)
            if lightweight_vae is None:
                raise RuntimeError("FastVR inference requires pipe.lightweight_vae.")
            lq_latents = lightweight_vae.encode(video_tensor, device=pipe.device)
        return {"lq_latents": lq_latents.to(dtype=pipe.torch_dtype, device=pipe.device)}


class WanVideoUnit_LQLatentInitializer(PipelineUnit):
    """Initialize VSR denoising directly from the encoded LQ latents."""

    def __init__(self):
        super().__init__(
            input_params=("lq_latents",),
            output_params=("latents",),
        )

    def process(self, pipe: FastVRPipeline, lq_latents):
        if lq_latents is None:
            return {}
        return {"latents": lq_latents}


def _preprocess_video(pipe: FastVRPipeline, video):
    if not isinstance(video, torch.Tensor):
        return pipe.preprocess_video(video)
    if video.ndim != 4:
        raise ValueError(f"Training video tensor must use TCHW layout, got {video.shape}")
    return video.permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1)



def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fastvr_embedded_context: Optional[torch.Tensor] = None,
    fastvr_timestep_embedding: Optional[torch.Tensor] = None,
    fastvr_timestep_modulation: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    window_causal_attention_chunk_size: int = 3,
    window_causal_attention_window_size: int = 2,
    window_causal_attention_first_chunk_size: int = 3,
    window_causal_attention_chunk_start: int = 0,
    window_causal_attention_kv_cache=None,
    window_causal_attention_write_kv_cache: bool = False,
    **_ignored,
):
    """Run the sole FastVR DiT path with window-causal attention."""
    if dit is None:
        raise RuntimeError("FastVR requires a loaded Wan DiT.")

    if fastvr_timestep_embedding is not None and fastvr_timestep_modulation is not None:
        t = fastvr_timestep_embedding
        t_mod = fastvr_timestep_modulation
    elif dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros(
                (1, latents.shape[3] * latents.shape[4] // 4),
                dtype=latents.dtype,
                device=latents.device,
            ),
            torch.ones(
                (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                dtype=latents.dtype,
                device=latents.device,
            ) * timestep,
        ]).flatten()
        time_input = sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        time_input = align_time_embedding_input_dtype(time_input, dit.time_embedding)
        t = dit.time_embedding(time_input)
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        time_input = sinusoidal_embedding_1d(dit.freq_dim, timestep)
        time_input = align_time_embedding_input_dtype(time_input, dit.time_embedding)
        t = dit.time_embedding(time_input)
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = (
        fastvr_embedded_context
        if fastvr_embedded_context is not None
        else dit.text_embedding(context)
    )
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)
    x = dit.patchify(x)
    frames, height, width = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    ensure_temporal_rope_length(dit, window_causal_attention_chunk_start + frames)
    freqs = torch.cat([
        dit.freqs[0][
            window_causal_attention_chunk_start:
            window_causal_attention_chunk_start + frames
        ].view(frames, 1, 1, -1).expand(frames, height, width, -1),
        dit.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
        dit.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
    ], dim=-1).reshape(frames * height * width, 1, -1).to(x.device)

    streaming = window_causal_attention_kv_cache is not None
    attention_mask = None if streaming else build_window_causal_self_attn_mask(
        frame_tokens=height * width,
        latent_frames=frames,
        chunk_size=int(window_causal_attention_chunk_size),
        window_size=int(window_causal_attention_window_size),
        first_chunk_size=int(window_causal_attention_first_chunk_size),
        device=x.device,
    )

    new_cache = []
    for block_id, block in enumerate(dit.blocks):
        if streaming:
            x, current_kv = block(
                x,
                context,
                t_mod,
                freqs,
                attention_mask,
                self_attn_kv_cache=window_causal_attention_kv_cache[block_id],
                return_self_attn_kv=True,
            )
            new_cache.append(tuple(item.detach() for item in current_kv))
        else:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                context,
                t_mod,
                freqs,
                attention_mask,
            )

    if streaming and window_causal_attention_write_kv_cache:
        max_history_tokens = window_causal_history_frames(
            window_causal_attention_chunk_size,
            window_causal_attention_window_size,
        ) * height * width
        for layer_cache, current_kv in zip(window_causal_attention_kv_cache, new_cache):
            if max_history_tokens == 0:
                layer_cache.clear()
            else:
                layer_cache.append(current_kv)
                trim_window_causal_kv_cache(layer_cache, max_history_tokens)

    x = dit.head(x, t)
    return dit.unpatchify(x, (frames, height, width))


def load_prompt_embedding(path: str) -> torch.Tensor:
    """Load the fixed prompt embedding required by FastVR."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt embedding file does not exist: {path}")
    embedding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(
            f"Prompt embedding checkpoint must contain a torch.Tensor, got {type(embedding)!r}"
        )
    return embedding


def resolve_inference_device(device=None) -> torch.device:
    """Return the active CUDA device or fail with a clear inference error."""
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(
                f"FastVR inference requires a CUDA device, got {device}."
            )
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("FastVR inference requires a CUDA-capable GPU.")
    return torch.device("cuda", torch.cuda.current_device())


def build_pipeline(args, *, device=None):
    """Load both FastVR DiT shards together as one model through DiffSynth."""
    device = resolve_inference_device(device)
    dit_path = dit_checkpoint_paths(args.ckpt_path)
    pipe = FastVRPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[_LocalModelConfig(path=dit_path)],
    )

    # Keep DiT in its native dtype (bfloat16 from checkpoint). Training uses
    # accelerate's autocast to handle the float32 output of sinusoidal_embedding_1d
    # being fed into bfloat16 model weights. Inference uses torch.autocast too.
    pipe.dit = pipe.dit.to(device=pipe.device)

    # LightweightVAE compresses spatially by 16x and the DiT patchifies by 2x.
    pipe.height_division_factor = 32
    pipe.width_division_factor = 32

    return pipe
