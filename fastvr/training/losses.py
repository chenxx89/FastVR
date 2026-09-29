"""FastVR Stage 1 and Stage 2 training objectives."""

import os
import shutil
from typing import Any, cast

import torch
from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.diffusion.base_pipeline import BasePipeline

from fastvr.models.wan_video_dit import build_window_causal_chunk_ids
from fastvr.models.wan_vae import decode_wan_vae_window


def _window_causal_chunk_ids(inputs, latent_frames: int, device):
    """Return the fixed FastVR window-causal chunk id for each latent frame."""
    return build_window_causal_chunk_ids(
        latent_frames,
        int(inputs.get("window_causal_attention_chunk_size", 3)),
        int(inputs.get("window_causal_attention_first_chunk_size", 3)),
        device,
    )


def _first_chunk_loss_weights(inputs, chunk_ids, device):
    """Per-latent-frame loss weights that up-weight the history-free first chunk.

    Weights are normalized to mean 1 so the loss scale (and therefore the usable
    learning rate) does not change when the weight or the chunk layout changes.
    """
    weight = float(inputs.get("window_causal_attention_first_chunk_loss_weight", 1.0))
    if chunk_ids is None or weight == 1.0:
        return None
    weights = torch.ones(chunk_ids.shape[0], device=device, dtype=torch.float32)
    weights[chunk_ids == 0] = weight
    weights = weights / weights.mean()
    return weights.view(1, 1, -1, 1, 1)


def _log_per_chunk_error(log: dict, inputs, squared_error: torch.Tensor, chunk_ids):
    """Split the reconstruction error into first chunk vs. the rest, for diagnosis."""
    if chunk_ids is None:
        return
    log["first_chunk_size"] = float(inputs.get("window_causal_attention_first_chunk_size", 3))
    with torch.no_grad():
        per_frame = squared_error.detach().float().mean(dim=(0, 1, 3, 4))
        is_first = chunk_ids == 0
        log["mse_chunk0"] = per_frame[is_first].mean()
        log["mse_rest"] = torch.zeros((), device=per_frame.device)
        rest = per_frame[~is_first]
        if rest.numel():
            log["mse_rest"] = rest.mean()


def flow_match_sft_loss(pipe: BasePipeline, **inputs):
    pipe_dynamic = cast(Any, pipe)
    scheduler = cast(Any, pipe.scheduler)
    vsr_target_timestep = inputs.get("vsr_target_timestep")
    timestep_tau_id = int(torch.argmin((scheduler.timesteps - vsr_target_timestep).abs()).item())
    sigma_tau = scheduler.sigmas[timestep_tau_id].to(device=pipe.device)

    consistency_weight = float(inputs.get("consistency_weight") or 0.0)
    num_timesteps = len(scheduler.timesteps)
    timestep_id = int(torch.randint(timestep_tau_id, num_timesteps, (1,)).item())
    timestep = scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device).reshape(1)
    sigma = scheduler.sigmas[timestep_id].to(device=pipe.device)

    models = {name: getattr(pipe_dynamic, name) for name in pipe_dynamic.in_iteration_models}
    hq_latents = inputs["hq_latents"]
    lq_latents = inputs["lq_latents"]

    def predict_v(latents, timestep_value):
        inputs["latents"] = latents
        return pipe_dynamic.model_fn(**models, **inputs, timestep=timestep_value)

    w = (sigma / sigma_tau).clamp(min=0.0)

    training_target = (lq_latents - hq_latents) / sigma_tau
    latents = (1.0 - w) * hq_latents + w * lq_latents

    v_pred = predict_v(latents, timestep)

    squared_error = (v_pred.float() - training_target.float()).pow(2)
    chunk_ids = _window_causal_chunk_ids(inputs, squared_error.shape[2], squared_error.device)
    frame_weights = _first_chunk_loss_weights(inputs, chunk_ids, squared_error.device)

    def weighted_mean(error):
        return error.mean() if frame_weights is None else (error * frame_weights).mean()

    loss_mse = weighted_mean(squared_error)
    zero = torch.zeros((), device=squared_error.device, dtype=torch.float32)
    loss_consistency = zero
    teacher_timestep = timestep.detach().float()
    has_consistency = consistency_weight > 0.0 and timestep_id + 1 < num_timesteps

    if has_consistency:
        timestep_id_2 = int(torch.randint(timestep_id + 1, num_timesteps, (1,)).item())
        timestep_2 = scheduler.timesteps[timestep_id_2].to(dtype=torch.float32, device=pipe.device).reshape(1)
        sigma_2 = scheduler.sigmas[timestep_id_2].to(device=pipe.device)
        with torch.no_grad():
            latents_2 = latents + (sigma_2 - sigma) * v_pred
            v_teacher = predict_v(latents_2, timestep_2)
        loss_consistency = weighted_mean((v_pred.float() - v_teacher.float()).pow(2))
        teacher_timestep = timestep_2.detach().float()

    loss = loss_mse + consistency_weight * loss_consistency
    loss_log = {
        "loss_mse": loss_mse.detach().float(),
        "loss_consistency": loss_consistency.detach().float(),
        "consistency_weight": consistency_weight,
        "timestep_student": timestep.detach().float().mean(),
        "timestep_teacher": teacher_timestep.mean(),
    }

    _log_per_chunk_error(loss_log, inputs, squared_error, chunk_ids)
    pipe._last_flow_match_sft_loss_log = loss_log
    return loss

_dists_loss_module = None

def _prepare_dists_weights(vgg16_weights_path=None, dists_weights_path=None):
    """Optionally seed pyiqa caches from configured local weight files."""
    sources = [
        (vgg16_weights_path, os.path.expanduser("~/.cache/torch/hub/checkpoints")),
        (dists_weights_path, os.path.expanduser("~/.cache/torch/hub/pyiqa")),
    ]
    for source, destination_dir in sources:
        if not source:
            continue
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Configured metric weight does not exist: {source}")
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(destination_dir, os.path.basename(source))
        if not os.path.exists(destination) or os.path.getsize(destination) != os.path.getsize(source):
            shutil.copy2(source, destination)


def _get_dists_loss(device, vgg16_weights_path=None, dists_weights_path=None):
    """Lazily initialize and return DISTS loss module with local weights."""
    global _dists_loss_module
    if _dists_loss_module is None:
        _prepare_dists_weights(vgg16_weights_path, dists_weights_path)
        import pyiqa
        _dists_loss_module = pyiqa.create_metric("dists", device=device, as_loss=True)
        _dists_loss_module.eval().requires_grad_(False)
    return _dists_loss_module


def pixel_loss(pipe: BasePipeline, **inputs):
    pipe_dynamic = cast(Any, pipe)
    scheduler = cast(Any, pipe.scheduler)
    vsr_target_timestep = inputs.get("vsr_target_timestep")
    timestep_tau_id = int(torch.argmin((scheduler.timesteps - vsr_target_timestep).abs()).item())

    timestep = scheduler.timesteps[timestep_tau_id].to(dtype=torch.float32, device=pipe.device).reshape(1)
    sigma = scheduler.sigmas[timestep_tau_id].to(device=pipe.device)

    models = {name: getattr(pipe_dynamic, name) for name in pipe_dynamic.in_iteration_models}
    inputs["latents"] = inputs["lq_latents"]
    v_pred = pipe_dynamic.model_fn(**models, **inputs, timestep=timestep)
    x0_pred = inputs["latents"] - sigma * v_pred

    gt_pixel_tensor = inputs.get("gt_pixel_tensor")
    if gt_pixel_tensor is None:
        raise ValueError("pixel_loss requires gt_pixel_tensor for pixel supervision.")

    latent_window_size = int(inputs.get("pixel_loss_latent_window_size", 1))
    latent_window_size = max(1, min(latent_window_size, x0_pred.shape[2]))

    if inputs.get("pixel_loss_random_window", True):
        latent_start = torch.randint(
            0,
            x0_pred.shape[2] - latent_window_size + 1,
            (1,),
            device=x0_pred.device,
        ).item()
    else:
        latent_start = 0
    latent_end = latent_start + latent_window_size

    if latent_start == 0:
        frame_start = 0
        frame_count = latent_window_size * 4 - 3
    else:
        frame_start = latent_start * 4 - 3
        frame_count = latent_window_size * 4
    gt_pixel_tensor = gt_pixel_tensor[frame_start:frame_start + frame_count]

    pipe.load_models_to_device(("vae",))
    decoder = pipe.vae
    if decoder is None:
        raise RuntimeError("pixel_loss requires the official Wan VAE.")
    pred_videos = decode_wan_vae_window(
        decoder,
        x0_pred,
        device=pipe.device,
        latent_start=latent_start,
        latent_end=latent_end,
        use_gradient_checkpointing=True,
    )
    pred_videos = pred_videos[:, :, :frame_count].contiguous()
    pred_videos = pred_videos * 0.5 + 0.5

    gt_videos = (
        gt_pixel_tensor.unsqueeze(0)
        .permute(0, 2, 1, 3, 4)
        .to(device=pred_videos.device, dtype=pred_videos.dtype)
        .clamp(0.0, 1.0)
    )

    pred_videos_image = pred_videos.clamp(0.0, 1.0)

    pixel_l1_loss_raw = None
    pixel_l1_weight = float(inputs.get("pixel_l1_weight", 1.0))
    if pixel_l1_weight > 0:
        pixel_l1_loss_raw = torch.nn.functional.l1_loss(
            pred_videos.float(), gt_videos.float(), reduction="mean"
        )

    dists_loss_raw = None
    dists_weight = float(inputs.get("dists_weight", 0.0))
    if dists_weight > 0:
        dists_module = _get_dists_loss(
            pred_videos.device,
            inputs.get("vgg16_weights_path"),
            inputs.get("dists_weights_path"),
        )

        dists_scores = []
        for frame_idx in range(pred_videos.shape[2]):
            pred_frame = pred_videos_image[:, :, frame_idx, :, :].float()
            gt_frame = gt_videos[:, :, frame_idx, :, :].float()
            score = gradient_checkpoint_forward(
                dists_module,
                True,
                False,
                pred_frame,
                gt_frame,
            )
            dists_scores.append(score.float().mean())

        dists_loss_raw = torch.stack(dists_scores).mean()

    zero = torch.zeros((), device=pred_videos.device)
    pixel_l1_loss = (
        pixel_l1_weight * pixel_l1_loss_raw
        if pixel_l1_loss_raw is not None
        else zero
    )
    dists_loss = (
        dists_weight * dists_loss_raw if dists_loss_raw is not None else zero
    )

    pipe._last_pixel_loss_log = {
        "pixel_l1_loss": pixel_l1_loss.detach().float(),
        "dists_loss": dists_loss.detach().float(),
        "w_pixel_l1": pixel_l1_weight,
    }
    return pixel_l1_loss + dists_loss
