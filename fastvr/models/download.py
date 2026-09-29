"""Shared Hugging Face snapshot download and validation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from fastvr.models.checkpoint import DIT_FILES

PRIMARY_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"
EMPTY_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints/FastVR/empty_prompt.pt"
)

FASTVR_REPOSITORY = "chenxx89/FastVR"
FASTVR_REQUIRED_FILES = (*DIT_FILES, "vae.safetensors")

WAN_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
WAN_VAE_FILE = "Wan2.2_VAE.pth"
WAN_DIT_FILES = (
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
)


def _endpoint_is_accessible(endpoint: str, timeout: float = 5.0) -> bool:
    try:
        import requests

        response = requests.get(endpoint, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _select_download_endpoint() -> tuple[str, bool]:
    configured = os.environ.get("HF_ENDPOINT")
    if configured:
        return configured.rstrip("/"), True
    if _endpoint_is_accessible(PRIMARY_ENDPOINT):
        return PRIMARY_ENDPOINT, False
    os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
    print(
        f"[FastVR] Hugging Face is unreachable; switching to mirror: "
        f"{MIRROR_ENDPOINT}"
    )
    return MIRROR_ENDPOINT, False


def ensure_huggingface_files(
    path: str,
    *,
    repo_id: str,
    required_files: Sequence[str],
    artifact: str,
) -> str:
    """Download, resume, and validate selected files from one HF repository."""
    from filelock import FileLock

    target_dir = os.path.abspath(os.path.expanduser(path))
    required_files = tuple(required_files)
    if not required_files:
        raise ValueError("required_files cannot be empty")

    lock_path = f"{target_dir}.download.lock"
    progress_path = f"{target_dir}.download-in-progress"
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    if os.path.exists(target_dir) and not os.path.isdir(target_dir):
        raise NotADirectoryError(f"{artifact} path is not a directory: {target_dir}")

    def missing_files() -> list[str]:
        missing = []
        for filename in required_files:
            file = Path(target_dir) / filename
            if not file.is_file() or file.stat().st_size == 0:
                missing.append(filename)
            elif file.stat().st_size < 1024:
                # Cloning without an LFS smudge filter leaves a text pointer,
                # not usable model weights. Download the real file from HF.
                if file.read_bytes().startswith(
                    b"version https://git-lfs.github.com/spec/v1\n"
                ):
                    missing.append(filename)
        return missing

    if missing_files() or os.path.exists(progress_path):
        with FileLock(lock_path):
            missing = missing_files()
            if missing:
                endpoint, user_configured_endpoint = _select_download_endpoint()
                print(
                    f"[FastVR] {artifact} files are unavailable in: {target_dir}\n"
                    f"[FastVR] Downloading {repo_id} from {endpoint}..."
                )
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as error:
                    raise RuntimeError(
                        f"huggingface_hub is required to download {artifact}."
                    ) from error

                Path(progress_path).touch()
                try:
                    try:
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=target_dir,
                            allow_patterns=missing,
                            endpoint=endpoint,
                        )
                    except Exception as primary_error:
                        if user_configured_endpoint or endpoint == MIRROR_ENDPOINT:
                            raise
                        os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
                        print(
                            f"[FastVR] Download from {PRIMARY_ENDPOINT} failed; "
                            f"retrying with mirror: {MIRROR_ENDPOINT}"
                        )
                        try:
                            snapshot_download(
                                repo_id=repo_id,
                                local_dir=target_dir,
                                allow_patterns=missing,
                                endpoint=MIRROR_ENDPOINT,
                            )
                        except Exception as mirror_error:
                            raise RuntimeError(
                                f"Failed to download {artifact} from both Hugging Face "
                                f"({primary_error}) and its mirror ({mirror_error})."
                            ) from mirror_error

                    missing = missing_files()
                    if missing:
                        raise FileNotFoundError(
                            f"Downloaded {artifact} is missing required file(s): "
                            f"{', '.join(missing)}"
                        )
                    os.unlink(progress_path)
                    print(f"[FastVR] {artifact} download completed: {target_dir}")
                except Exception:
                    # Retain the marker so the next launch resumes partial files.
                    raise
            elif os.path.exists(progress_path):
                os.unlink(progress_path)

    missing = missing_files()
    if missing:
        raise FileNotFoundError(
            f"{artifact} directory is missing required file(s): {', '.join(missing)}"
        )
    return target_dir


def ensure_fastvr_checkpoint(path: str) -> str:
    """Download, resume, and validate the public inference checkpoint."""
    return ensure_huggingface_files(
        path,
        repo_id=FASTVR_REPOSITORY,
        required_files=FASTVR_REQUIRED_FILES,
        artifact="FastVR checkpoint",
    )


def ensure_wan_model_base(path: str, *, include_dit: bool) -> str:
    """Download, resume, and validate the Wan files required by training."""
    required_files = (*WAN_DIT_FILES, WAN_VAE_FILE) if include_dit else (WAN_VAE_FILE,)
    return ensure_huggingface_files(
        path,
        repo_id=WAN_REPOSITORY,
        required_files=required_files,
        artifact="Wan model",
    )
