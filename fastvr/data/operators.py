"""Video decoding and spatial preprocessing for FastVR training."""

from __future__ import annotations

import random
import warnings

import imageio
import torchvision
from PIL import Image

try:
    import decord

    decord.bridge.set_bridge("native")
    _DECORD_AVAILABLE = True
except ImportError:
    decord = None
    _DECORD_AVAILABLE = False


def _metadata_value(metadata, *keys):
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


class ImageCropAndResize:
    def __init__(
        self,
        height=None,
        width=None,
        max_pixels=None,
        height_division_factor=1,
        width_division_factor=1,
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def __call__(self, image: Image.Image) -> Image.Image:
        target_height, target_width = self._target_size(image)
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )

    def _target_size(self, image: Image.Image) -> tuple[int, int]:
        if self.height is not None and self.width is not None:
            return self.height, self.width
        width, height = image.size
        if self.max_pixels is not None and width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor
        if height <= 0 or width <= 0:
            raise ValueError(f"Video frame is too small after alignment: {image.size}")
        return height, width


class LoadVideo:
    def __init__(
        self,
        num_frames=45,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda image: image,
        interval_list=None,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.interval_list = interval_list

    def __call__(self, data) -> list[Image.Image]:
        metadata = data if isinstance(data, dict) else None
        path = data.get("__data_path__") if isinstance(data, dict) else data
        reader = self._open_reader(path)
        try:
            start_frame, total_frames = self._frame_range(reader, metadata)
            frame_ids = [
                start_frame + frame_id
                for frame_id in self._sample_frame_ids(total_frames)
            ]
            return self._read_frames(reader, path, frame_ids)
        finally:
            if hasattr(reader, "close"):
                reader.close()

    @staticmethod
    def _open_reader(path):
        if _DECORD_AVAILABLE:
            try:
                return decord.VideoReader(path, ctx=decord.cpu(0))
            except Exception:
                pass
        return imageio.get_reader(path)

    @staticmethod
    def _frame_range(reader, metadata) -> tuple[int, int]:
        start = int(_metadata_value(metadata, "Start_Frame", "start_frame") or 0)
        if start < 0:
            raise ValueError(f"Start_Frame cannot be negative: {start}")
        count = _metadata_value(
            metadata, "Num_Frames", "num_frames", "total_frames", "Total_Frames"
        )
        if count is not None:
            return start, int(count)
        end = _metadata_value(metadata, "End_Frame", "end_frame")
        if end is not None:
            return start, max(int(end) - start, 0)
        if _DECORD_AVAILABLE and isinstance(reader, decord.VideoReader):
            return start, max(len(reader) - start, 0)
        if hasattr(reader, "count_frames"):
            try:
                return start, max(int(reader.count_frames()) - start, 0)
            except Exception:
                pass
        count = reader.get_length()
        if count == float("inf"):
            raise ValueError("Video reader cannot determine the frame count.")
        return start, max(int(count) - start, 0)

    def _aligned_length(self, available: int) -> int:
        length = min(self.num_frames, available)
        while length > 1 and length % self.time_division_factor != self.time_division_remainder:
            length -= 1
        if length <= 0:
            raise ValueError("Video contains no readable frames.")
        return length

    def _sample_frame_ids(self, total_frames: int) -> list[int]:
        length = self._aligned_length(total_frames)
        interval = random.choice(self.interval_list) if self.interval_list else 1
        required = (length - 1) * interval + 1
        while required > total_frames and interval > 1:
            interval -= 1
            required = (length - 1) * interval + 1
        if required > total_frames:
            length = self._aligned_length(total_frames)
            interval = 1
            required = length
        start = random.randint(0, max(total_frames - required, 0))
        return [start + index * interval for index in range(length)]

    def _read_frames(self, reader, path, frame_ids: list[int]) -> list[Image.Image]:
        if _DECORD_AVAILABLE and isinstance(reader, decord.VideoReader):
            try:
                batch = reader.get_batch(frame_ids).asnumpy()
                return [self.frame_processor(Image.fromarray(frame)) for frame in batch]
            except Exception as error:
                warnings.warn(
                    f"Decord failed to read {path}: {error}. Falling back to imageio."
                )
                if hasattr(reader, "close"):
                    reader.close()
                fallback = imageio.get_reader(path)
                try:
                    return [
                        self.frame_processor(Image.fromarray(fallback.get_data(frame_id)))
                        for frame_id in frame_ids
                    ]
                finally:
                    fallback.close()
        return [
            self.frame_processor(Image.fromarray(reader.get_data(frame_id)))
            for frame_id in frame_ids
        ]
