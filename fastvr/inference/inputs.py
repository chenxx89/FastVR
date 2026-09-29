"""Input discovery, metadata probing, and frame decoding for inference."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterator

import imageio
from PIL import Image

from fastvr.inference.utils import normalize_fps

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass(frozen=True)
class InferenceSample:
    """One normalized inference input and its optional FPS override."""

    path: str
    fps: float | None = None


def sample_fps_override(sample: InferenceSample, image_fps: float) -> float | None:
    """Prefer JSONL Fps; otherwise leave video FPS to source metadata."""
    if sample.fps is not None:
        return sample.fps
    if os.path.isdir(sample.path):
        return image_fps
    if os.path.splitext(sample.path)[1].lower() in VIDEO_EXTS:
        return None
    return image_fps


@dataclass(frozen=True)
class InputInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", name)]


def _image_paths(path: str) -> list[str]:
    return [
        os.path.join(path, name)
        for name in sorted(
            (
                name
                for name in os.listdir(path)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS
            ),
            key=_natural_key,
        )
    ]


def probe_input(path: str, fallback_fps: float | None = 30.0) -> InputInfo:
    """Read only metadata and the first frame required to plan a stream."""
    fallback_fps = normalize_fps(fallback_fps, default=30.0)
    path = path.rstrip("/")
    extension = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and extension in IMAGE_EXTS:
        with Image.open(path) as image:
            width, height = image.size
        return InputInfo(width, height, fallback_fps, 1)
    if os.path.isdir(path):
        paths = _image_paths(path)
        if not paths:
            raise FileNotFoundError(f"No images found in directory: {path}")
        with Image.open(paths[0]) as image:
            width, height = image.size
        return InputInfo(width, height, fallback_fps, len(paths))
    if os.path.isfile(path) and extension in VIDEO_EXTS:
        reader = imageio.get_reader(path)
        try:
            metadata = reader.get_meta_data() if hasattr(reader, "get_meta_data") else {}
            first = Image.fromarray(reader.get_data(0)).convert("RGB")
            raw_count = metadata.get("nframes")
            frame_count = int(raw_count) if isinstance(raw_count, int) and raw_count > 0 else None
            if frame_count is None and hasattr(reader, "count_frames"):
                try:
                    counted = int(reader.count_frames())
                    frame_count = counted if counted > 0 else None
                except Exception:
                    frame_count = None
            return InputInfo(
                first.width,
                first.height,
                normalize_fps(metadata.get("fps"), default=fallback_fps),
                frame_count,
            )
        finally:
            reader.close()
    raise ValueError(f"Unsupported streaming input: {path}")


def iter_input_frames(path: str) -> Iterator[tuple[Image.Image, str | None]]:
    """Yield source frames without retaining the full input video."""
    path = path.rstrip("/")
    extension = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and extension in IMAGE_EXTS:
        with Image.open(path) as image:
            yield image.convert("RGB"), os.path.splitext(os.path.basename(path))[0] + ".png"
        return
    if os.path.isdir(path):
        for image_path in _image_paths(path):
            with Image.open(image_path) as image:
                yield image.convert("RGB"), os.path.splitext(os.path.basename(image_path))[0] + ".png"
        return
    if os.path.isfile(path) and extension in VIDEO_EXTS:
        reader = imageio.get_reader(path)
        try:
            for frame in reader:
                yield Image.fromarray(frame).convert("RGB"), None
        finally:
            reader.close()
        return
    raise ValueError(f"Unsupported streaming input: {path}")


def iter_frame_chunks(path: str, first_size: int, chunk_size: int):
    """Yield bounded chunks with one-frame lookahead to identify the final chunk."""
    source = iter(iter_input_frames(path))
    pending = []
    target = first_size
    exhausted = False
    while not exhausted:
        while len(pending) < target:
            try:
                pending.append(next(source))
            except StopIteration:
                exhausted = True
                break
        if not pending:
            break
        if not exhausted:
            try:
                lookahead = next(source)
            except StopIteration:
                exhausted = True
            else:
                chunk = pending
                pending = [lookahead]
                yield chunk, False
                target = chunk_size
                continue
        yield pending, True
        break


def next_4n_plus_1(frame_count: int) -> int:
    """Return the smallest 4n+1 frame count that is not smaller than the input."""
    return 4 * ((frame_count + 2) // 4) + 1


def load_lq_frames(path: str):
    """Load input frames and metadata from a video, image, or frame directory.

    Returns ``(frames, fps, frame_count, input_height, input_width, frame_names)``.
    ``frame_names`` is populated for image and frame-directory inputs so optional
    PNG output can preserve source names. Temporal and spatial alignment are
    handled by the inference runner after loading.
    """
    path = path.rstrip("/")
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and ext in IMAGE_EXTS:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        frame_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        return [image], 30.0, 1, height, width, [frame_name]

    if os.path.isfile(path) and ext in VIDEO_EXTS:
        reader = imageio.get_reader(path)
        try:
            try:
                metadata = reader.get_meta_data() if hasattr(reader, "get_meta_data") else {}
            except Exception:
                metadata = {}
            fps = normalize_fps(metadata.get("fps"), default=30.0)
            frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise ValueError(f"No frames found in video: {path}")
        width, height = frames[0].size
        return frames, fps, len(frames), height, width, None

    if os.path.isdir(path):
        files = sorted(
            [
                name
                for name in os.listdir(path)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS
            ],
            key=_natural_key,
        )
        if not files:
            raise FileNotFoundError(f"No images found in directory: {path}")

        print(f"[FastVR] Loading image sequence from {path}: {len(files)} images")
        frames = [Image.open(os.path.join(path, name)).convert("RGB") for name in files]
        width, height = frames[0].size
        frame_names = [os.path.splitext(name)[0] + ".png" for name in files]
        return frames, 30.0, len(frames), height, width, frame_names

    raise ValueError(f"Unsupported input: {path}")


def _collect_direct_inputs(path: str):
    """Resolve *path* to a list of inputs to process.

    Behavior:
      - Single file (video or image): returns [path]
      - Directory containing image files: returns [path] (the directory itself
        is treated as one video sequence of consecutive images)
      - Directory containing only video files: returns each video as a separate
        input for batch processing
      - Directory containing both videos and images: videos are processed
        individually; if images also exist, the directory is additionally
        included as an image-sequence input
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        entries = sorted(os.listdir(path), key=_natural_key)

        video_files = [os.path.join(path, f) for f in entries
                       if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        image_files = [f for f in entries
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

        if not video_files and not image_files:
            raise FileNotFoundError(f"No video/image files in {path}")

        # If directory contains images, treat the whole directory as one
        # video-sequence input (consecutive frames → single .mp4 output)
        if image_files and not video_files:
            print(f"[FastVR] Detected {len(image_files)} images in {path}, "
                  f"treating as a single video sequence")
            return [path]

        # If directory contains videos (with or without images), process each
        # video individually for batch inference
        if video_files:
            result = list(video_files)
            if image_files:
                print(f"[FastVR] Found {len(video_files)} video(s) and "
                      f"{len(image_files)} image(s) in {path}; "
                      f"processing videos individually")
            return result

    raise ValueError(f"Input path does not exist: {path}")


def _collect_subfolder_samples(input_dir: str):
    """Scan input_dir for subfolders containing image files.

    Returns a naturally-sorted list of absolute paths to valid subfolders.
    Each entry is a dict with 'path' and optional 'fps' override.
    """
    entries = sorted(os.listdir(input_dir), key=_natural_key)
    valid_dirs = []
    for entry in entries:
        full_path = os.path.join(input_dir, entry)
        if not os.path.isdir(full_path):
            continue
        # Check if directory contains at least one image file
        has_images = any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTS
            for f in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, f))
        )
        if has_images:
            valid_dirs.append(InferenceSample(path=full_path))

    return valid_dirs


def _load_jsonl_samples(jsonl_path: str):
    """Load samples from a JSONL file.

    Each line should be a JSON object with at least 'Filepath' key pointing to
    an image-sequence directory. Optional 'Fps' key specifies the output frame rate.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        Normalized inference samples.
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[FastVR] WARNING: Skipping malformed JSON at line {line_num}: {e}")
                continue

            filepath = record.get("Filepath")
            if not filepath:
                print(f"[FastVR] WARNING: Skipping line {line_num}: missing 'Filepath' key")
                continue

            if not os.path.exists(filepath):
                print(f"[FastVR] WARNING: Skipping line {line_num}: input not found: {filepath}")
                continue

            fps_value = record.get("Fps")
            fps = normalize_fps(fps_value) if fps_value is not None else None
            samples.append(InferenceSample(path=filepath, fps=fps))

    print(f"[FastVR] Loaded {len(samples)} sample(s) from {jsonl_path}")
    return samples


def collect_samples(input_path: str):
    """Analyze one input path and return normalized inference sample records."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path) and input_path.lower().endswith(".jsonl"):
        return _load_jsonl_samples(input_path), f"JSONL: {input_path}"

    try:
        paths = _collect_direct_inputs(input_path)
        samples = [InferenceSample(path=path) for path in paths]
    except FileNotFoundError:
        if not os.path.isdir(input_path):
            raise
        samples = _collect_subfolder_samples(input_path)
    return samples, f"input: {input_path}"
