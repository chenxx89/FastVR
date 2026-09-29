"""The two-shard FastVR DiT checkpoint format shared by training and inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file


DIT_FILES = (
    "dit-00001-of-00002.safetensors",
    "dit-00002-of-00002.safetensors",
)
MAX_SHARD_BYTES = 5_000_000_000  # Decimal GB, including the safetensors header.


def dit_checkpoint_paths(directory: str | os.PathLike) -> list[str]:
    """Require both shards before handing their paths to DiffSynth."""
    directory = Path(directory).expanduser()
    paths = [directory / name for name in DIT_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FastVR DiT shard(s): {', '.join(missing)}")
    return [str(path) for path in paths]


def save_dit_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    directory: str | os.PathLike,
    *,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> list[str]:
    """Balance whole tensors across two files without changing their dtype/value.

    Check actual serialized sizes (including headers), and finish writing both
    temporary files before publishing either shard. Only one shard is copied to
    CPU at a time when exporting a GPU training state.
    """
    if not state_dict:
        raise ValueError("Cannot save an empty DiT state dict.")
    sizes = {
        name: tensor.numel() * tensor.element_size()
        for name, tensor in state_dict.items()
    }
    if max(sizes.values()) > max_shard_bytes or sum(sizes.values()) > 2 * max_shard_bytes:
        raise ValueError("DiT tensors do not fit into two shards within the size limit.")
    shards = [{}, {}]
    totals = [0, 0]
    for name in sorted(sizes, key=lambda key: (-sizes[key], key)):
        index = 0 if totals[0] <= totals[1] else 1
        shards[index][name] = state_dict[name]
        totals[index] += sizes[name]

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        for filename, shard in zip(DIT_FILES, shards):
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            temporary_paths.append(temporary)
            tensors = {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in shard.items()
            }
            save_file(tensors, temporary, metadata={"format": "pt"})
            del tensors
            if temporary.stat().st_size > max_shard_bytes:
                raise ValueError(
                    f"{filename} exceeds {max_shard_bytes} bytes including its header."
                )
        for temporary, filename in zip(temporary_paths, DIT_FILES):
            os.replace(temporary, directory / filename)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return dit_checkpoint_paths(directory)
