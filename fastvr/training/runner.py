"""FastVR training orchestration."""

from __future__ import annotations

import datetime
import accelerate
from diffsynth.core.device import get_available_device_type, parse_nccl_backend

from fastvr.data.dataset import FastVRVideoDataset
from fastvr.models.download import ensure_wan_model_base
from fastvr.training.checkpointing import FastVRModelLogger, write_training_manifest
from fastvr.training.config import TrainingConfig
from fastvr.training.engine import launch_training_task
from fastvr.training.module import FastVRTrainingModule


def run(config: TrainingConfig) -> None:
    """Build and run one validated FastVR training stage."""

    if config.model_base is not None:
        ensure_wan_model_base(
            config.model_base,
            include_dit=config.stage == "stage1_causal",
        )
    write_training_manifest(config.to_mapping())

    timeout_seconds = config.distributed_timeout_seconds
    timeout = datetime.timedelta(seconds=timeout_seconds)
    backend = parse_nccl_backend(get_available_device_type())
    _set_default_nccl_timeout(timeout)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=config.find_unused_parameters
            ),
            accelerate.InitProcessGroupKwargs(backend=backend, timeout=timeout),
        ],
    )
    if accelerator.is_main_process:
        print(
            f"[FastVR] Accelerator: backend={backend}, "
            f"processes={accelerator.num_processes}, timeout={timeout_seconds}s"
        )

    dataset = FastVRVideoDataset(config.dataset, config.degradation)
    if accelerator.is_main_process:
        print(f"[FastVR] Dataset: {len(dataset)} samples")

    model = FastVRTrainingModule(
        config,
        device=accelerator.device,
    )
    logger = FastVRModelLogger(
        output_path=config.output_path,
        tensorboard=config.tensorboard,
    )
    launch_training_task(accelerator, dataset, model, logger, config)


def _set_default_nccl_timeout(timeout: datetime.timedelta) -> None:
    """Apply the timeout to process groups created later by DeepSpeed."""

    try:
        import torch.distributed.constants as constants
        import torch.distributed.distributed_c10d as distributed_c10d

        constants.default_pg_nccl_timeout = timeout
        distributed_c10d.default_pg_nccl_timeout = timeout
    except Exception as error:
        print(f"[FastVR] Could not override the default NCCL timeout: {error}")
