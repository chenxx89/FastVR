"""Public FastVR inference command."""

from __future__ import annotations

import argparse
import math
import os


def parse_args(argv=None):
    """Parse and validate every public inference option in one place."""
    parser = argparse.ArgumentParser(description="FastVR video enhancement inference")

    # Model
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help=(
            "Checkpoint directory containing dit-00001-of-00002.safetensors, "
            "dit-00002-of-00002.safetensors, and vae.safetensors."
        ),
    )

    # Input and output
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Video, image, JSONL file, frame directory, directory of videos, "
            "or root containing frame-sequence subdirectories."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory. Each result keeps its input base filename.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output FPS for a single image or frame directory (default: 30). Video inputs use their source FPS unless JSONL provides Fps.",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Final output scale relative to the input resolution (default: 1).",
    )
    parser.add_argument(
        "--target_short_edge",
        type=int,
        default=None,
        help=(
            "Optional model-processing short edge. When omitted, the model processes "
            "at the resolution selected by --upscale."
        ),
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable end-to-end streaming for long videos.",
    )
    parser.add_argument(
        "--save_formats",
        default="mp4",
        help="Comma-separated output formats: mp4, png, or both.",
    )
    parser.add_argument(
        "--png_name_mode",
        choices=["source", "index"],
        default="source",
        help="Use source frame names or sequential indices for PNG output.",
    )
    parser.add_argument("--png_dir_suffix", default="_frames")
    parser.add_argument("--png_workers", type=int, default=8)
    parser.add_argument("--png_max_pending", type=int, default=64)

    # Single-step inference
    parser.add_argument(
        "--vsr_target_timestep",
        type=float,
        default=399.0,
        help="Training-aligned target timestep for single-step inference.",
    )
    parser.add_argument(
        "--enable_denoise_tiling",
        action="store_true",
        help="Enable spatial latent denoise tiling.",
    )
    parser.add_argument("--denoise_tile_size", type=int, default=64)
    parser.add_argument("--denoise_tile_stride", type=int, default=52)
    parser.add_argument("--denoise_tile_batch_size", type=int, default=1)

    # Post-processing
    parser.add_argument(
        "--color_fix",
        choices=["", "adain", "wavelet"],
        default=None,
        help="Color correction method; an empty value or omission disables it.",
    )

    # Independent sample sharding for multi-GPU inference
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))

    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be a positive finite number")
    if args.upscale <= 0:
        parser.error("--upscale must be positive")
    if args.target_short_edge is not None and args.target_short_edge <= 0:
        parser.error("--target_short_edge must be positive when provided")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error(f"invalid shard: rank={args.rank}, world_size={args.world_size}")

    args.output_formats = {
        value.strip().lower() for value in args.save_formats.split(",") if value.strip()
    }
    unknown_formats = args.output_formats - {"mp4", "png"}
    if unknown_formats or not args.output_formats:
        parser.error(f"--save_formats must be a subset of mp4,png (got {args.save_formats!r})")
    if args.png_workers < 1 or args.png_max_pending < 1:
        parser.error("--png_workers and --png_max_pending must be >= 1")

    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    from fastvr.inference.runner import run

    run(args)


if __name__ == "__main__":
    main()
