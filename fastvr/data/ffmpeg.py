"""Shared FFmpeg executable discovery for training and inference."""

from functools import lru_cache
import os
import shutil
import subprocess


def _is_usable_ffmpeg(path: str) -> bool:
    try:
        version = subprocess.run(
            [path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if version.returncode != 0:
            return False
        encoders = subprocess.run(
            [path, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return encoders.returncode == 0 and "libx265" in encoders.stdout.split()


def _report_ffmpeg(path: str) -> str:
    path = os.path.abspath(path)
    print(f"[FastVR] FFmpeg: {path}")
    return path


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str:
    """Resolve one working FFmpeg executable for all FastVR video operations."""
    configured = os.environ.get("FFMPEG_PATH")
    if configured:
        configured = os.path.abspath(os.path.expanduser(configured))
        if not os.path.isfile(configured) or not os.access(configured, os.X_OK):
            raise FileNotFoundError(
                f"FFMPEG_PATH does not point to an executable file: {configured}"
            )
        if not _is_usable_ffmpeg(configured):
            raise RuntimeError(
                "FFMPEG_PATH is not a working FFmpeg executable with the required "
                f"libx265 encoder: {configured}"
            )
        return _report_ffmpeg(configured)

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and _is_usable_ffmpeg(system_ffmpeg):
        return _report_ffmpeg(system_ffmpeg)

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        bundled_ffmpeg = None
    if bundled_ffmpeg and _is_usable_ffmpeg(bundled_ffmpeg):
        return _report_ffmpeg(bundled_ffmpeg)

    raise FileNotFoundError(
        "FastVR requires an FFmpeg build with the libx265 encoder. Install one from "
        "https://ffmpeg.org/download.html or set FFMPEG_PATH to the executable."
    )
