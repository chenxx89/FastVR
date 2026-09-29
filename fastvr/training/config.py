"""Typed, FastVR-only training configuration."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a FastVR training configuration is invalid."""


_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.-]*)\}")


def _lookup(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Unknown configuration reference '${{{dotted_key}}}'.")
        value = value[part]
    return value


def _expand_string(value: str, root: Mapping[str, Any]) -> Any:
    def replace_env(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise ConfigError(
                f"Environment variable {name!r} is required by the configuration."
            )
        return resolved

    expanded = _ENV_PATTERN.sub(replace_env, value)
    exact = _REF_PATTERN.fullmatch(expanded)
    if exact:
        return copy.deepcopy(_lookup(root, exact.group(1)))

    def replace_ref(match: re.Match[str]) -> str:
        referenced = _lookup(root, match.group(1))
        if isinstance(referenced, (dict, list, tuple)):
            raise ConfigError(
                f"Reference '${{{match.group(1)}}}' cannot be embedded in a string."
            )
        return str(referenced)

    return _REF_PATTERN.sub(replace_ref, expanded)


def _resolve(value: Any, root: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, root)
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, root) for key, item in value.items()}
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load YAML and resolve environment variables and config references."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("The configuration root must be a YAML mapping.")

    resolved: dict[str, Any] = loaded
    for _ in range(10):
        next_value = _resolve(resolved, resolved)
        if next_value == resolved:
            return next_value
        resolved = next_value
    raise ConfigError("Configuration references did not converge after 10 passes.")


def require(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    """Return a required value and reject missing or blank strings."""

    try:
        value = _lookup(mapping, dotted_key)
    except ConfigError as error:
        raise ConfigError(f"Missing required configuration key: {dotted_key}") from error
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError(f"Configuration key {dotted_key!r} cannot be empty.")
    return value


def as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"Configuration key {key!r} must be a boolean.")


_STAGES = {"stage1_causal", "stage2_pixel"}
_SCHEDULERS = {"constant", "cosine", "cosine_with_restarts"}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in {section}: {', '.join(unknown)}")


def _positive_int(value: Any, key: str) -> int:
    result = int(value)
    if result <= 0:
        raise ConfigError(f"Configuration key {key!r} must be positive.")
    return result


def _nonnegative_float(value: Any, key: str) -> float:
    result = float(value)
    if result < 0:
        raise ConfigError(f"Configuration key {key!r} cannot be negative.")
    return result


def _path(value: Any, base_dir: Path, key: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError(f"Configuration key {key!r} cannot be empty.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _optional_path(value: Any, base_dir: Path, key: str) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _path(value, base_dir, key)


def _model_path(value: Any, base_dir: Path) -> str | tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ConfigError("training.dit_path cannot be empty.")
        return tuple(_path(item, base_dir, "training.dit_path") for item in value)
    return _path(value, base_dir, "training.dit_path")


@dataclass(frozen=True)
class DatasetConfig:
    jsonl_path: str
    video_key: str = "Filepath"
    repeat: int = 1
    height: int | None = None
    width: int | None = None
    max_pixels: int = 1920 * 1080
    num_frames: int = 45
    interval_list: tuple[int, ...] | None = None
    num_workers: int = 1
    prefetch_factor: int = 2
    pin_memory: bool = False


@dataclass(frozen=True)
class DegradationConfig:
    config_path: str


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    num_epochs: int = 1
    max_train_steps: int | None = None
    gradient_accumulation_steps: int = 1
    save_steps: int | None = None
    max_grad_norm: float | None = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1


@dataclass(frozen=True)
class LossConfig:
    vsr_target_timestep: float = 399.0
    consistency_weight: float = 0.0
    pixel_l1_weight: float = 1.0
    dists_weight: float = 0.0
    pixel_loss_latent_window_size: int = 1
    pixel_loss_random_window: bool = True
    first_chunk_loss_weight: float = 1.0
    vgg16_weights_path: str | None = None
    dists_weights_path: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    stage: str
    dataset: DatasetConfig
    degradation: DegradationConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    dit_path: str | tuple[str, ...]
    vae_path: str
    output_path: str
    model_base: str | None = None
    causal_sizes: tuple[int, int, int] = (3, 3, 2)
    resume_from_checkpoint: str | None = None
    torch_compile: bool = False
    tensorboard: bool = True
    find_unused_parameters: bool = False
    distributed_timeout_seconds: int = 7200
    source_path: str = ""

    @property
    def task(self) -> str:
        return "sft" if self.stage == "stage1_causal" else "pixels"

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["dit_path"] = (
            list(self.dit_path) if isinstance(self.dit_path, tuple) else self.dit_path
        )
        result["causal_sizes"] = list(self.causal_sizes)
        return result


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load one config, resolving relative paths against its directory."""

    source_path = Path(path).expanduser().resolve()
    raw = load_config(source_path)
    _reject_unknown(raw, {"version", "stage", "dataset", "training"}, "config root")
    if int(raw.get("version", 1)) != 1:
        raise ConfigError("Unsupported training configuration version.")

    stage = str(raw.get("stage", ""))
    if stage not in _STAGES:
        raise ConfigError(f"training stage must be one of {sorted(_STAGES)}.")
    dataset_raw = require(raw, "dataset")
    training_raw = require(raw, "training")
    if not isinstance(dataset_raw, Mapping) or not isinstance(training_raw, Mapping):
        raise ConfigError("dataset and training must be YAML mappings.")

    _reject_unknown(
        dataset_raw,
        {
            "dataset_jsonl",
            "video_key",
            "dataset_repeat",
            "height",
            "width",
            "max_pixels",
            "num_frames",
            "interval_list",
            "num_workers",
            "prefetch_factor",
            "pin_memory",
        },
        "dataset",
    )
    _reject_unknown(
        training_raw,
        {
            "model_base",
            "dit_path",
            "vae_path",
            "degradation",
            "output_path",
            "learning_rate",
            "weight_decay",
            "num_epochs",
            "max_train_steps",
            "gradient_accumulation_steps",
            "save_steps",
            "max_grad_norm",
            "lr_scheduler",
            "lr_warmup_steps",
            "lr_num_cycles",
            "resume_from_checkpoint",
            "torch_compile",
            "tensorboard",
            "find_unused_parameters",
            "distributed_timeout_seconds",
            "vsr_target_timestep",
            "consistency_weight",
            "window_causal_attention",
            "pixel_l1_weight",
            "dists_weight",
            "pixel_loss_latent_window_size",
            "pixel_loss_random_window",
            "vgg16_weights_path",
            "dists_weights_path",
        },
        "training",
    )

    base_dir = source_path.parent
    interval = dataset_raw.get("interval_list")
    if interval is not None:
        if not isinstance(interval, (list, tuple)) or not interval:
            raise ConfigError("dataset.interval_list must be a non-empty list.")
        interval = tuple(
            _positive_int(value, "dataset.interval_list") for value in interval
        )

    height = dataset_raw.get("height")
    width = dataset_raw.get("width")
    if (height is None) != (width is None):
        raise ConfigError("dataset.height and dataset.width must be set together.")
    if height is not None and (int(height) % 32 or int(width) % 32):
        raise ConfigError("dataset.height and dataset.width must be divisible by 32.")
    dataset = DatasetConfig(
        jsonl_path=_path(
            require(dataset_raw, "dataset_jsonl"), base_dir, "dataset.dataset_jsonl"
        ),
        video_key=str(dataset_raw.get("video_key", "Filepath")),
        repeat=_positive_int(
            dataset_raw.get("dataset_repeat", 1), "dataset.dataset_repeat"
        ),
        height=None if height is None else _positive_int(height, "dataset.height"),
        width=None if width is None else _positive_int(width, "dataset.width"),
        max_pixels=_positive_int(
            dataset_raw.get("max_pixels", 1920 * 1080), "dataset.max_pixels"
        ),
        num_frames=_positive_int(
            dataset_raw.get("num_frames", 45), "dataset.num_frames"
        ),
        interval_list=interval,
        num_workers=max(0, int(dataset_raw.get("num_workers", 1))),
        prefetch_factor=_positive_int(
            dataset_raw.get("prefetch_factor", 2), "dataset.prefetch_factor"
        ),
        pin_memory=as_bool(
            dataset_raw.get("pin_memory", False), key="dataset.pin_memory"
        ),
    )
    if not dataset.video_key:
        raise ConfigError("dataset.video_key cannot be empty.")

    degradation_raw = require(training_raw, "degradation")
    if not isinstance(degradation_raw, Mapping):
        raise ConfigError("training.degradation must be a YAML mapping.")
    _reject_unknown(degradation_raw, {"config"}, "training.degradation")
    degradation = DegradationConfig(
        config_path=_path(
            require(degradation_raw, "config"),
            base_dir,
            "training.degradation.config",
        ),
    )

    scheduler = str(training_raw.get("lr_scheduler", "constant"))
    if scheduler not in _SCHEDULERS:
        raise ConfigError(f"training.lr_scheduler must be one of {sorted(_SCHEDULERS)}.")
    max_steps = training_raw.get("max_train_steps")
    save_steps = training_raw.get("save_steps")
    max_grad_norm = training_raw.get("max_grad_norm")
    optimizer = OptimizerConfig(
        learning_rate=float(training_raw.get("learning_rate", 1e-5)),
        weight_decay=_nonnegative_float(
            training_raw.get("weight_decay", 1e-2), "training.weight_decay"
        ),
        num_epochs=_positive_int(
            training_raw.get("num_epochs", 1), "training.num_epochs"
        ),
        max_train_steps=(
            None
            if max_steps is None
            else _positive_int(max_steps, "training.max_train_steps")
        ),
        gradient_accumulation_steps=_positive_int(
            training_raw.get("gradient_accumulation_steps", 1),
            "training.gradient_accumulation_steps",
        ),
        save_steps=(
            None
            if save_steps is None
            else _positive_int(save_steps, "training.save_steps")
        ),
        max_grad_norm=(
            None
            if max_grad_norm is None
            else _nonnegative_float(max_grad_norm, "training.max_grad_norm")
        ),
        lr_scheduler=scheduler,
        lr_warmup_steps=max(0, int(training_raw.get("lr_warmup_steps", 0))),
        lr_num_cycles=_positive_int(
            training_raw.get("lr_num_cycles", 1), "training.lr_num_cycles"
        ),
    )
    if optimizer.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive.")

    window = training_raw.get("window_causal_attention") or {}
    if not isinstance(window, Mapping):
        raise ConfigError("training.window_causal_attention must be a YAML mapping.")
    _reject_unknown(
        window,
        {"sizes", "first_chunk_loss_weight"},
        "training.window_causal_attention",
    )
    sizes = window.get("sizes", [3, 3, 2])
    if not isinstance(sizes, (list, tuple)) or len(sizes) != 3:
        raise ConfigError("training.window_causal_attention.sizes must contain three values.")
    causal_values = [
        _positive_int(value, "training.window_causal_attention.sizes")
        for value in sizes
    ]
    causal_sizes = (causal_values[0], causal_values[1], causal_values[2])

    loss = LossConfig(
        vsr_target_timestep=float(training_raw.get("vsr_target_timestep", 399.0)),
        consistency_weight=_nonnegative_float(
            training_raw.get("consistency_weight", 0.0),
            "training.consistency_weight",
        ),
        pixel_l1_weight=_nonnegative_float(
            training_raw.get("pixel_l1_weight", 1.0), "training.pixel_l1_weight"
        ),
        dists_weight=_nonnegative_float(
            training_raw.get("dists_weight", 0.0), "training.dists_weight"
        ),
        pixel_loss_latent_window_size=_positive_int(
            training_raw.get("pixel_loss_latent_window_size", 1),
            "training.pixel_loss_latent_window_size",
        ),
        pixel_loss_random_window=as_bool(
            training_raw.get("pixel_loss_random_window", True),
            key="training.pixel_loss_random_window",
        ),
        first_chunk_loss_weight=_nonnegative_float(
            window.get("first_chunk_loss_weight", 1.0),
            "training.window_causal_attention.first_chunk_loss_weight",
        ),
        vgg16_weights_path=_optional_path(
            training_raw.get("vgg16_weights_path"),
            base_dir,
            "training.vgg16_weights_path",
        ),
        dists_weights_path=_optional_path(
            training_raw.get("dists_weights_path"),
            base_dir,
            "training.dists_weights_path",
        ),
    )
    if stage == "stage1_causal" and any(
        key in training_raw
        for key in (
            "pixel_l1_weight",
            "dists_weight",
            "pixel_loss_latent_window_size",
            "pixel_loss_random_window",
            "vgg16_weights_path",
            "dists_weights_path",
        )
    ):
        raise ConfigError("Stage 2 pixel-loss settings cannot be used in stage1_causal.")
    if stage == "stage2_pixel" and "consistency_weight" in training_raw:
        raise ConfigError("training.consistency_weight is only valid for stage1_causal.")

    resume = training_raw.get("resume_from_checkpoint")
    return TrainingConfig(
        stage=stage,
        dataset=dataset,
        degradation=degradation,
        optimizer=optimizer,
        loss=loss,
        dit_path=_model_path(require(training_raw, "dit_path"), base_dir),
        vae_path=_path(require(training_raw, "vae_path"), base_dir, "training.vae_path"),
        output_path=_path(require(training_raw, "output_path"), base_dir, "training.output_path"),
        model_base=_optional_path(
            training_raw.get("model_base"), base_dir, "training.model_base"
        ),
        causal_sizes=causal_sizes,
        resume_from_checkpoint=(
            None
            if resume in (None, "")
            else _path(resume, base_dir, "training.resume_from_checkpoint")
        ),
        torch_compile=as_bool(
            training_raw.get("torch_compile", False), key="training.torch_compile"
        ),
        tensorboard=as_bool(
            training_raw.get("tensorboard", True), key="training.tensorboard"
        ),
        find_unused_parameters=as_bool(
            training_raw.get("find_unused_parameters", False),
            key="training.find_unused_parameters",
        ),
        distributed_timeout_seconds=_positive_int(
            training_raw.get("distributed_timeout_seconds", 7200),
            "training.distributed_timeout_seconds",
        ),
        source_path=str(source_path),
    )
