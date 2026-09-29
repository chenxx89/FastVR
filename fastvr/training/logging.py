"""Checkpoint progress and TensorBoard logging."""

from __future__ import annotations

import os

import torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"[FastVR] TensorBoard log: {log_dir}")

    def log(self, key: str, value, step: int) -> None:
        self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class ModelLogger:
    def __init__(self, output_path: str, *, tensorboard: bool = True):
        self.output_path = output_path
        self.tensorboard_enabled = tensorboard
        self.tensorboard: TensorBoardLogger | None = None
        self.num_steps = 0
        self.num_update_steps = 0
        self.epoch_id = 0
        self.batch_in_epoch = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "num_steps": self.num_steps,
            "num_update_steps": self.num_update_steps,
            "epoch_id": self.epoch_id,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.num_steps = int(state_dict["num_steps"])
        self.num_update_steps = int(state_dict["num_update_steps"])
        self.epoch_id = int(state_dict["epoch_id"])
        self.batch_in_epoch = int(state_dict["batch_in_epoch"])

    def _ensure_tensorboard(self) -> None:
        if self.tensorboard_enabled and self.tensorboard is None:
            self.tensorboard = TensorBoardLogger(
                os.path.join(self.output_path, "tensorboard")
            )

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps: int | None = None,
        optimizer_updated: bool = True,
        **metrics,
    ) -> None:
        self.num_steps += 1
        if optimizer_updated:
            self.num_update_steps += 1
        loss = metrics.get("loss")
        if optimizer_updated and loss is not None:
            reduced_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
            reduced_details = self._consume_loss_details(accelerator, model)
            if accelerator.is_main_process:
                self._ensure_tensorboard()
                loss_value = reduced_loss.item()
                details = {
                    key: value.item() for key, value in reduced_details.items()
                }
                lr = metrics.get("dit_lr")
                lr_text = "" if lr is None else f" lr={lr:.2e}"
                detail_text = "".join(
                    f" {key}={value:.6f}" for key, value in details.items()
                )
                print(
                    f"[FastVR] step={self.num_update_steps} loss={loss_value:.6f}"
                    f"{lr_text}{detail_text}"
                )
                if self.tensorboard is not None:
                    self.tensorboard.log("loss", loss_value, self.num_update_steps)
                    if lr is not None:
                        self.tensorboard.log("learning_rate", lr, self.num_update_steps)
                    for key, value in details.items():
                        self.tensorboard.log(key, value, self.num_update_steps)

        if (
            save_steps is not None
            and optimizer_updated
            and self.num_update_steps > 0
            and self.num_update_steps % save_steps == 0
        ):
            self.save_model(accelerator, model)

    @staticmethod
    def _consume_loss_details(
        accelerator: Accelerator, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        pipe = getattr(accelerator.unwrap_model(model), "pipe", None)
        details = {}
        for attribute in ("_last_pixel_loss_log", "_last_flow_match_sft_loss_log"):
            values = getattr(pipe, attribute, None) if pipe is not None else None
            if isinstance(values, dict):
                for key, value in values.items():
                    if value is None:
                        continue
                    tensor = (
                        value.detach().float()
                        if isinstance(value, torch.Tensor)
                        else torch.tensor(float(value), device=accelerator.device)
                    )
                    details[key] = accelerator.reduce(tensor, reduction="mean")
                delattr(pipe, attribute)
        return details

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module) -> None:
        self.save_model(accelerator, model)

    def on_training_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps: int | None = None,
    ) -> None:
        if (
            save_steps is not None
            and self.num_update_steps > 0
            and self.num_update_steps % save_steps != 0
        ):
            self.save_model(accelerator, model)
        if self.tensorboard is not None:
            self.tensorboard.close()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module) -> None:
        raise NotImplementedError
