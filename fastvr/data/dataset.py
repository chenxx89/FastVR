"""JSONL video dataset used by FastVR training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fastvr.data.degradation import RealVSRDegradationHelper
from fastvr.data.operators import ImageCropAndResize, LoadVideo
from fastvr.training.config import DatasetConfig, DegradationConfig


class FastVRVideoDataset(torch.utils.data.Dataset):
    """Load one sampled video clip from each JSONL record."""

    def __init__(self, config: DatasetConfig, degradation: DegradationConfig):
        metadata_path = Path(config.jsonl_path)
        if metadata_path.suffix.lower() != ".jsonl":
            raise ValueError(f"FastVR training metadata must be JSONL: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError(f"Training metadata contains no samples: {metadata_path}")

        self.video_key = config.video_key
        for line_number, record in enumerate(self.records, 1):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Training metadata line {line_number} must be a JSON object."
                )
            video_path = record.get(self.video_key)
            if not video_path:
                raise ValueError(
                    f"Training metadata line {line_number} is missing "
                    f"{self.video_key!r}."
                )
            if not Path(str(video_path)).is_absolute():
                raise ValueError(
                    f"Training metadata line {line_number} must use an absolute "
                    f"{self.video_key} path: {video_path}"
                )
        self.repeat = config.repeat
        self.degradation = RealVSRDegradationHelper(degradation.config_path)
        self.video_loader = LoadVideo(
            config.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=ImageCropAndResize(
                config.height,
                config.width,
                config.max_pixels,
                height_division_factor=32,
                width_division_factor=32,
            ),
            interval_list=list(config.interval_list) if config.interval_list else None,
        )

    def __getitem__(self, data_id: int):
        last_error: Exception | None = None
        for attempt in range(5):
            index = (data_id + attempt) % len(self.records)
            record = self.records[index].copy()
            try:
                raw_path = record[self.video_key]
                video_path = str(raw_path)
                loader_input = record.copy()
                loader_input["__data_path__"] = video_path
                frames = self.video_loader(loader_input)
                frame_array = np.stack(
                    [np.asarray(frame, dtype=np.uint8) for frame in frames]
                )
                hq_video, lq_video = self.degradation(frame_array)
                return {"hq_video": hq_video, "lq_video": lq_video}
            except Exception as error:
                last_error = error
                print(
                    f"[FastVR] Failed to load sample {index} "
                    f"(attempt {attempt + 1}/5): {error}"
                )
        raise RuntimeError(
            f"Failed to load training sample {data_id} after 5 attempts."
        ) from last_error

    def __len__(self) -> int:
        return len(self.records) * self.repeat
