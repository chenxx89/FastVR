"""Output naming and asynchronous frame/video writers."""

import io
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from fastvr.data.ffmpeg import resolve_ffmpeg


def output_video_path(input_path: str, output_dir: str) -> str:
    """Build one MP4 output path while preserving an MP4 input's filename."""
    input_basename = os.path.basename(input_path.rstrip("/"))
    input_stem, input_ext = os.path.splitext(input_basename)
    output_filename = input_basename if input_ext.lower() == ".mp4" else f"{input_stem}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.abspath(output_path) == os.path.abspath(input_path):
        raise ValueError("Output directory would overwrite the input video; choose another OUTPUT path.")
    return output_path


def _run_audio_mux(
    ffmpeg_path: str,
    video_path: str,
    audio_source_path: str,
    output_path: str,
    duration: float,
    *,
    transcode_audio: bool,
) -> None:
    audio_args = ["-c:a", "aac", "-b:a", "192k"] if transcode_audio else ["-c:a", "copy"]
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        *audio_args,
        "-t", f"{duration:.9f}",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        output_path,
    ]
    environment = os.environ.copy()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    environment["LD_LIBRARY_PATH"] = (
        f"{ffmpeg_dir}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"FFmpeg exited with code {completed.returncode}"
        raise RuntimeError(message)


def _mux_source_audio(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    duration: float,
) -> None:
    """Mux an optional source audio stream, retrying with AAC when copy is unsupported."""
    ffmpeg_path = resolve_ffmpeg()
    try:
        _run_audio_mux(
            ffmpeg_path,
            video_path,
            audio_source_path,
            output_path,
            duration,
            transcode_audio=False,
        )
    except RuntimeError as copy_error:
        try:
            _run_audio_mux(
                ffmpeg_path,
                video_path,
                audio_source_path,
                output_path,
                duration,
                transcode_audio=True,
            )
        except RuntimeError as transcode_error:
            raise RuntimeError(
                f"Failed to mux source audio by stream copy ({copy_error}) or AAC "
                f"transcoding ({transcode_error})."
            ) from transcode_error



def _x265_pixel_format(width: int, height: int) -> str:
    return "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"


def _x265_common_args(crf: int) -> list[str]:
    return [
        "-preset", "medium",
        "-crf", str(int(crf)),
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_trc", "bt709",
        "-color_primaries", "bt709",
        "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
        "-x265-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:repeat-headers=1",
        "-tag:v", "hvc1",
    ]


def _x265_encode_args(crf: int, pixel_format: str) -> list[str]:
    """Return the single x265 encoding policy used by every video writer."""
    return [
        "-c:v", "libx265",
        "-pix_fmt", pixel_format,
        *_x265_common_args(crf),
    ]


def _x265_imageio_kwargs(crf: int, width: int, height: int) -> dict:
    return {
        "codec": "libx265",
        "pixelformat": _x265_pixel_format(width, height),
        "macro_block_size": 1,
        "ffmpeg_log_level": "error",
        # imageio's quality is a 1-10 abstraction and does not map to x265 CRF.
        # Disable it and pass the native encoder settings directly to FFmpeg.
        "quality": None,
        "output_params": [*_x265_common_args(crf), "-movflags", "+faststart"],
    }


def save_video(frames, save_path, fps, crf=18):
    """Save a complete frame sequence with FFmpeg and an imageio fallback."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    if not frames:
        raise ValueError("Cannot save a video with no frames")
    video_frames = [np.asarray(frame) for frame in frames]
    video_frames = [
        (
            (frame * 255).astype(np.uint8)
            if frame.dtype != np.uint8 and frame.max() <= 1.0
            else frame.astype(np.uint8)
        )
        for frame in video_frames
    ]
    ffmpeg_path = resolve_ffmpeg()

    height, width = video_frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in video_frames):
        raise ValueError("All video frames must have the same resolution")
    environment = os.environ.copy()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    environment["LD_LIBRARY_PATH"] = (
        f"{ffmpeg_dir}:{environment.get('LD_LIBRARY_PATH', '')}"
    )

    pixel_format = _x265_pixel_format(width, height)
    command = [
        ffmpeg_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        *_x265_encode_args(crf, pixel_format),
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-loglevel", "warning",
        save_path,
        "-y",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        for frame in tqdm(video_frames, desc=f"Saving video (CRF={crf})"):
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.wait()
        if process.returncode != 0:
            print(
                f"[FastVR] WARNING: FFmpeg exited with code {process.returncode}; "
                "falling back to imageio"
            )
            _save_video_imageio(frames, save_path, fps, crf=crf)
    except Exception as error:
        print(f"[FastVR] WARNING: FFmpeg failed ({error}); falling back to imageio")
        _save_video_imageio(frames, save_path, fps, crf=crf)


def _save_video_imageio(frames, save_path, fps, crf=18):
    """Fallback complete-video writer used when FFmpeg is unavailable."""
    import imageio
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    first = np.asarray(frames[0])
    writer = imageio.get_writer(
        save_path,
        fps=fps,
        **_x265_imageio_kwargs(crf, first.shape[1], first.shape[0]),
    )
    try:
        for frame in tqdm(frames, desc="Saving video (imageio fallback)"):
            value = np.asarray(frame) if isinstance(frame, Image.Image) else frame
            writer.append_data(value)
    finally:
        writer.close()

def _publish_file(source_path: str, target_path: str) -> None:
    """Atomically publish a file, copying only when crossing filesystems."""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    if os.stat(source_path).st_dev == os.stat(target_dir).st_dev:
        os.replace(source_path, target_path)
        return
    partial_path = f"{target_path}.part-{os.getpid()}"
    try:
        shutil.copy2(source_path, partial_path)
        os.replace(partial_path, target_path)
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)


def save_video_via_temp(
    frames,
    target_path: str,
    fps: float,
    *,
    audio_source_path: str | None = None,
):
    """Encode processed frames and preserve the source video's audio when present."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    scratch_dir = os.environ.get("FASTVR_SCRATCH_DIR", tempfile.gettempdir())
    os.makedirs(scratch_dir, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=scratch_dir) as tmp_f:
            video_path = tmp_f.name
        temporary_paths.append(video_path)
        save_video(frames, video_path, fps=fps)

        completed_path = video_path
        if audio_source_path is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=scratch_dir) as tmp_f:
                muxed_path = tmp_f.name
            temporary_paths.append(muxed_path)
            duration = len(frames) / float(fps)
            _mux_source_audio(video_path, audio_source_path, muxed_path, duration)
            completed_path = muxed_path

        _publish_file(completed_path, target_path)
    finally:
        for temporary_path in temporary_paths:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def png_frames_dir(mp4_path: str, dir_suffix: str, dir_name: str = None) -> str:
    """Frame-sequence directory that mirrors an mp4 output path.

    Example: <output>/foo.mp4 -> <output>_frames/foo/.
    ``dir_name`` can override the leaf directory used for the PNG sequence.
    """
    parent, name = os.path.split(mp4_path)
    if dir_name is None:
        dir_name = os.path.splitext(name)[0]
    return os.path.join(parent + dir_suffix, dir_name)


class AsyncFrameWriter:
    """Write decoded frames as PNG on a background thread pool.

    PNG encoding costs far more than the GPU work it follows, so it runs off the
    main thread: PIL releases the GIL inside the zlib encoder, so the writes
    overlap with the next sample's denoise. Each frame is encoded into memory
    and flushed with a single write() — the network-mounted filesystem used for cluster
    outputs does not handle the incremental writes a streaming encoder issues
    (the same reason mp4s go through save_video_via_temp).

    A sample's directory gets a DONE_MARKER once every frame has landed. Resume
    detection checks this marker, so a partially written directory is redone.
    """

    DONE_MARKER = ".done"

    def __init__(self, num_workers: int, max_pending: int):
        self._pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="pngsave")
        # Backpressure: bounds the frames held in host memory when the pool
        # drains slower than the GPU produces samples. Eight shards per node
        # each keep their own queue, so this cap is per process.
        self._slots = threading.BoundedSemaphore(max_pending)
        self._cv = threading.Condition()
        self._inflight = 0
        self._saved = 0
        self._errors = []

    def submit_frames(self, frames, out_dir: str, label: str, names=None):
        """Queue one PNG per frame into out_dir.

        names carries the LQ input file names (extension already swapped to .png)
        so a metric script can pair SR against GT by file name instead of relying
        on both trees sorting the same way. Falls back to 1-based indices when the
        names are missing, the wrong length, or not unique.
        """
        os.makedirs(out_dir, exist_ok=True)
        marker = os.path.join(out_dir, self.DONE_MARKER)
        if os.path.exists(marker):
            os.remove(marker)
        if names is not None and (len(names) != len(frames) or len(set(names)) != len(names)):
            print(
                f"[FastVR] WARNING: unusable frame names for {out_dir} "
                f"({len(names)} name(s), {len(set(names))} unique, {len(frames)} frame(s)); "
                f"falling back to sequential numbering"
            )
            names = None
        if names is None:
            names = [f"{index:06d}.png" for index in range(1, len(frames) + 1)]
        group = {"remaining": len(frames), "total": len(frames), "failed": False}
        for name, frame in zip(names, frames):
            path = os.path.join(out_dir, name)
            self._slots.acquire()
            with self._cv:
                self._inflight += 1
            self._pool.submit(self._write_one, frame, path, group, out_dir, label)

    def _write_one(self, frame, path, group, out_dir, label):
        error = None
        try:
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            with open(path, "wb") as fp:
                fp.write(buffer.getvalue())
        except Exception as exc:  # a bad frame must not kill the batch
            error = f"{path}: {exc}"

        with self._cv:
            if error is None:
                self._saved += 1
            else:
                group["failed"] = True
                self._errors.append(error)
            group["remaining"] -= 1
            write_marker = group["remaining"] == 0 and not group["failed"]

        # Marker write stays outside the lock; it touches the same slow mount.
        if write_marker:
            try:
                with open(os.path.join(out_dir, self.DONE_MARKER), "w") as fp:
                    fp.write(f"{group['total']}\n")
            except OSError as exc:
                with self._cv:
                    self._errors.append(f"{out_dir}/{self.DONE_MARKER}: {exc}")
            # Deliberately not the word "Saved": the shell progress monitor
            # counts those lines to track finished mp4s.
            print(f"[FastVR] PNG frames[{label}]: {out_dir} ({group['total']} files)")

        self._slots.release()
        # Released after the marker so wait() cannot return on an unmarked group.
        with self._cv:
            self._inflight -= 1
            self._cv.notify_all()

    def wait(self):
        """Block until every queued frame (and its marker) has been written."""
        with self._cv:
            while self._inflight:
                self._cv.wait()
            return self._saved, list(self._errors)

    def shutdown(self):
        saved, errors = self.wait()
        self._pool.shutdown(wait=True)
        return saved, errors

class StreamingVideoWriter:
    """Stream RGB frames to the FFmpeg executable selected by FastVR."""

    def __init__(self, target_path: str, fps: float, audio_source_path: str | None = None):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        scratch_dir = os.environ.get("FASTVR_SCRATCH_DIR", tempfile.gettempdir())
        os.makedirs(scratch_dir, exist_ok=True)
        self.target_path = target_path
        self.fps = float(fps)
        self.audio_source_path = audio_source_path
        self.frame_count = 0
        self._size = None
        self._ffmpeg_path = resolve_ffmpeg()
        self._process = None
        self._closed = False
        self._temporary_paths = []
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=scratch_dir) as tmp_f:
            self.video_path = tmp_f.name
        self._temporary_paths.append(self.video_path)

    def _start(self, width: int, height: int) -> None:
        self._size = (width, height)
        pixel_format = _x265_pixel_format(width, height)
        command = [
            self._ffmpeg_path,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}",
            "-r", f"{self.fps:.12g}",
            "-i", "pipe:0",
            "-an",
            *_x265_encode_args(18, pixel_format),
            "-movflags", "+faststart",
            "-map_metadata", "-1",
            self.video_path,
        ]
        environment = os.environ.copy()
        ffmpeg_dir = os.path.dirname(self._ffmpeg_path)
        environment["LD_LIBRARY_PATH"] = (
            f"{ffmpeg_dir}:{environment.get('LD_LIBRARY_PATH', '')}"
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=environment,
        )
        print(f"[FastVR] Streaming video writer: FFmpeg ({pixel_format})")

    def append(self, frames) -> None:
        import numpy as np

        for frame in frames:
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            height, width = array.shape[:2]
            if self._size is None:
                self._start(width, height)
            elif self._size != (width, height):
                raise ValueError(
                    f"Streaming video frame size changed from {self._size} to {(width, height)}"
                )
            try:
                self._process.stdin.write(array.tobytes())
            except BrokenPipeError as error:
                message = self._process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg streaming encode failed: {message}") from error
            self.frame_count += 1

    def _close_encoder(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            message = self._process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = self._process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg streaming encode failed: {message or return_code}"
                )

    def close(self, *, commit: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._close_encoder()
            if not commit:
                return
            if self.frame_count == 0:
                raise RuntimeError("Cannot save a streaming video with no frames")
            completed_path = self.video_path
            if self.audio_source_path is not None:
                scratch_dir = os.path.dirname(self.video_path)
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False, dir=scratch_dir
                ) as tmp_f:
                    muxed_path = tmp_f.name
                self._temporary_paths.append(muxed_path)
                _mux_source_audio(
                    self.video_path,
                    self.audio_source_path,
                    muxed_path,
                    self.frame_count / self.fps,
                )
                completed_path = muxed_path
            _publish_file(completed_path, self.target_path)
        finally:
            for path in self._temporary_paths:
                if os.path.exists(path):
                    os.unlink(path)


class StreamingPNGWriter:
    """Write chunks into a temporary directory and publish only when complete."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.frame_count = 0
        parent = os.path.dirname(output_dir)
        os.makedirs(parent, exist_ok=True)
        self._temporary_dir = tempfile.mkdtemp(
            prefix=f".{os.path.basename(output_dir)}.part-",
            dir=parent,
        )

    def append(self, frames, names=None) -> None:
        if names is not None and len(names) != len(frames):
            raise ValueError("PNG frame names must match the streamed frame count")
        for index, frame in enumerate(frames):
            name = names[index] if names is not None else f"{self.frame_count + 1:06d}.png"
            if not name.lower().endswith(".png"):
                name = f"{name}.png"
            frame.save(os.path.join(self._temporary_dir, name), format="PNG")
            self.frame_count += 1

    def close(self, *, commit: bool = True) -> None:
        if not os.path.isdir(self._temporary_dir):
            return
        if not commit:
            shutil.rmtree(self._temporary_dir, ignore_errors=True)
            return
        marker = os.path.join(self._temporary_dir, AsyncFrameWriter.DONE_MARKER)
        with open(marker, "w") as fp:
            fp.write(f"{self.frame_count}\n")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.replace(self._temporary_dir, self.output_dir)
