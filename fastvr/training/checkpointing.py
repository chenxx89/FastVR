"""Checkpoint I/O helpers for FastVR distributed training."""

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml

from fastvr import __version__
from fastvr.models.checkpoint import save_dit_checkpoint
from fastvr.training.logging import ModelLogger


def write_training_manifest(config: Mapping[str, Any]) -> Path | None:
    """Write the resolved training recipe and its summary once on rank zero."""

    if int(os.environ.get("RANK", "0")) != 0:
        return None
    output = config.get("output_path")
    if not output:
        return None
    output_dir = Path(str(output)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = output_dir / "fastvr_config.yaml"
    manifest_path = output_dir / "fastvr_manifest.json"
    manifest = {
        "format_version": 1,
        "fastvr_version": __version__,
        "stage": config.get("stage"),
        "dit_path": config.get("dit_path"),
        "vsr_target_timestep": (config.get("loss") or {}).get(
            "vsr_target_timestep", 399.0
        ),
        "causal_sizes": config.get("causal_sizes", [3, 3, 2]),
        "training_vae": "official_wan",
        "vae_path": config.get("vae_path"),
        "validation_during_training": False,
    }

    def atomic_write(path: Path, content: str) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        os.replace(temporary, path)

    atomic_write(resolved_path, yaml.safe_dump(dict(config), sort_keys=False))
    atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest_path


class FastVRModelLogger(ModelLogger):
    """Save only the trainable FastVR DiT weights."""

    def save_model(self, accelerator, model):
        accelerator.wait_for_everyone()
        # DeepSpeed state is required for exact resume and is saved before
        # Accelerate's model hooks run. This directory is overwritten in place
        # because only the latest full state is needed for resuming training.
        state_path = os.path.join(self.output_path, "training_states")
        accelerator.save_state(state_path, safe_serialization=True)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            self._write_progress_marker(state_path)

        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            exported = unwrapped.export_trainable_state_dict(
                state_dict, remove_prefix="pipe.dit."
            )
            snapshot_dir = os.path.join(
                self.output_path,
                "checkpoints",
                f"epoch-{self.epoch_id}-step-{self.num_update_steps}",
            )
            save_dit_checkpoint(exported, snapshot_dir)
        accelerator.wait_for_everyone()

    def _write_progress_marker(self, state_path):
        os.makedirs(state_path, exist_ok=True)
        final_path = os.path.join(state_path, "progress.json")
        temporary = os.path.join(
            state_path,
            f".progress.{uuid.uuid4().hex}.tmp",
        )
        progress = {
            "epoch": self.epoch_id,
            "step": self.num_update_steps,
            "batch_in_epoch": self.batch_in_epoch,
            "micro_step": self.num_steps,
        }
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(progress, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, final_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
