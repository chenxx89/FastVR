#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # Repository root.

# =============================================================================
# User configuration
# =============================================================================
CKPT_PATH="$ROOT/checkpoints/FastVR"  # Model directory; missing weights are downloaded automatically.
INPUT="/path/to/input.mp4"  # Video, image, frame directory, directory batch, or JSONL manifest.
OUTPUT="$ROOT/outputs/inference"  # Directory for enhanced videos and/or PNG frames.
FPS=30  # Output FPS for a single image or frame directory; video files keep source FPS.
FFMPEG_PATH=""  # Optional FFmpeg executable; empty uses PATH or imageio-ffmpeg.

NUM_GPUS=1  # Number of local GPUs used to shard independent input samples.
UPSCALE=1  # Final output scale relative to the source resolution.
TARGET_SHORT_EDGE="1024"  # Model-processing short edge; empty uses the final UPSCALE-selected output resolution. 1024 is recommended for speed.
COLOR_FIX=""  # Optional color correction: "adain", "wavelet", or empty to disable.
STREAMING=1  # Enable bounded-memory streaming with negligible throughput impact.
SAVE_FORMATS="mp4"  # Output format: "mp4", "png", or "mp4,png".
# =============================================================================

if ((NUM_GPUS < 1)); then
  echo "NUM_GPUS must be at least 1." >&2
  exit 2
fi

export FFMPEG_PATH

run_rank() {
  local rank="$1"
  local command=(
    python3 -m fastvr.infer
    --ckpt_path "$CKPT_PATH"
    --input "$INPUT"
    --output_dir "$OUTPUT"
    --fps "$FPS"
    --upscale "$UPSCALE"
    --save_formats "$SAVE_FORMATS"
    --enable_denoise_tiling
    --color_fix "$COLOR_FIX"
    --rank "$rank"
    --world_size "$NUM_GPUS"
  )
  if [[ -n "$TARGET_SHORT_EDGE" ]]; then
    command+=(--target_short_edge "$TARGET_SHORT_EDGE")
  fi
  if ((STREAMING)); then
    command+=(--streaming)
  fi

  CUDA_VISIBLE_DEVICES="$rank" "${command[@]}"
}

if ((NUM_GPUS == 1)); then
  run_rank 0
  exit $?
fi

pids=()
for ((rank = 0; rank < NUM_GPUS; rank++)); do
  run_rank "$rank" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
