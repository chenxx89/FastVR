"""FastVR optimization loop."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm

from fastvr.training.config import TrainingConfig
from fastvr.training.logging import ModelLogger


def _unwrap_single_sample(samples):
    return samples[0]


def _build_scheduler(optimizer, config: TrainingConfig, num_steps: int, world_size: int):
    schedule = config.optimizer
    effective_steps = num_steps * world_size
    effective_warmup = schedule.lr_warmup_steps * world_size

    def warmup(step: int) -> float:
        return min(1.0, step / max(effective_warmup, 1))

    if schedule.lr_scheduler == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(effective_steps - effective_warmup, 1),
            eta_min=0.0,
        )
        if effective_warmup:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [torch.optim.lr_scheduler.LambdaLR(optimizer, warmup), cosine],
                milestones=[effective_warmup],
            )
        return cosine

    if schedule.lr_scheduler == "cosine_with_restarts":
        def cosine_with_restarts(step: int) -> float:
            if step < effective_warmup:
                return warmup(step)
            progress = (step - effective_warmup) / max(effective_steps - effective_warmup, 1)
            if progress >= 1.0:
                return 0.0
            cycle = (schedule.lr_num_cycles * max(progress, 0.0)) % 1.0
            return 0.5 * (1.0 + math.cos(math.pi * cycle))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_restarts)

    if effective_warmup:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)
    return torch.optim.lr_scheduler.ConstantLR(optimizer)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    model_logger: ModelLogger,
    config: TrainingConfig,
) -> None:
    schedule = config.optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
        betas=(0.9, 0.95),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        collate_fn=_unwrap_single_sample,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        prefetch_factor=(
            config.dataset.prefetch_factor if config.dataset.num_workers else None
        ),
        persistent_workers=config.dataset.num_workers > 0,
    )
    print(
        f"[FastVR] DataLoader: workers={config.dataset.num_workers}, "
        f"prefetch_factor={config.dataset.prefetch_factor}, "
        f"pin_memory={config.dataset.pin_memory}"
    )

    world_size = max(accelerator.num_processes, 1)
    steps_per_epoch = math.ceil(len(dataloader) / world_size)
    total_steps = schedule.max_train_steps or max(
        math.ceil(
            steps_per_epoch * schedule.num_epochs
            / schedule.gradient_accumulation_steps
        ),
        1,
    )
    epochs_to_run = schedule.num_epochs
    if schedule.max_train_steps is not None:
        updates_per_epoch = max(
            math.ceil(steps_per_epoch / schedule.gradient_accumulation_steps), 1
        )
        epochs_to_run = math.ceil(total_steps / updates_per_epoch)
    scheduler = _build_scheduler(optimizer, config, total_steps, world_size)

    if config.loss.dists_weight > 0:
        from fastvr.training.losses import _get_dists_loss

        _get_dists_loss(
            accelerator.device,
            config.loss.vgg16_weights_path,
            config.loss.dists_weights_path,
        )
        accelerator.wait_for_everyone()
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    accelerator.register_for_checkpointing(model_logger)
    if config.resume_from_checkpoint is not None:
        resume_path = Path(config.resume_from_checkpoint)
        if not resume_path.is_dir():
            raise FileNotFoundError(
                f"Accelerate training state does not exist: {resume_path}"
            )
        accelerator.load_state(str(resume_path))
        if accelerator.is_main_process:
            print(
                f"[FastVR] Resumed training state: {resume_path} "
                f"(epoch={model_logger.epoch_id}, "
                f"batch={model_logger.batch_in_epoch}, "
                f"step={model_logger.num_update_steps})"
            )
    _initialize_deepspeed_gradient_checkpointing(accelerator)

    print(
        f"[FastVR] Training: stage={config.stage}, updates={total_steps}, "
        f"gradient_accumulation={schedule.gradient_accumulation_steps}"
    )
    if model_logger.num_update_steps >= total_steps:
        if accelerator.is_main_process:
            print("[FastVR] Requested training steps are already complete.")
        return

    finished = False
    first_epoch = model_logger.epoch_id
    for epoch_id in range(first_epoch, epochs_to_run):
        skip_batches = model_logger.batch_in_epoch if epoch_id == first_epoch else 0
        epoch_dataloader = accelerator.skip_first_batches(dataloader, skip_batches)
        progress = tqdm(
            epoch_dataloader,
            disable=not accelerator.is_local_main_process,
            initial=skip_batches,
            total=len(dataloader),
        )
        for batch_id, data in enumerate(progress, start=skip_batches):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                optimizer_updated = accelerator.sync_gradients
                if optimizer_updated and schedule.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), schedule.max_grad_norm)
                optimizer.step()
                if optimizer_updated:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                model_logger.epoch_id = epoch_id
                model_logger.batch_in_epoch = batch_id + 1
                model_logger.on_step_end(
                    accelerator,
                    model,
                    schedule.save_steps,
                    optimizer_updated=optimizer_updated,
                    loss=loss,
                    dit_lr=optimizer.param_groups[0]["lr"],
                )
            if model_logger.num_update_steps >= total_steps:
                finished = True
                break

        if model_logger.batch_in_epoch >= len(dataloader):
            model_logger.epoch_id = epoch_id + 1
            model_logger.batch_in_epoch = 0
        if schedule.save_steps is None:
            model_logger.on_epoch_end(accelerator, model)
        if finished:
            break

    model_logger.on_training_end(accelerator, model, schedule.save_steps)


def _initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator) -> None:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return
    activation = plugin.deepspeed_config.get("activation_checkpointing")
    if activation is None:
        return
    import deepspeed

    deepspeed.checkpointing.configure(
        mpu_=None,
        partition_activations=activation.get("partition_activations", False),
        checkpoint_in_cpu=activation.get("cpu_checkpointing", False),
        contiguous_checkpointing=activation.get("contiguous_memory_optimization", False),
    )
