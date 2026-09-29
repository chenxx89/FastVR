#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# User configuration
# =============================================================================
MODEL_BASE="$ROOT/checkpoints/Wan2.2-TI2V-5B"
DATASET_JSONL="/path/to/train.jsonl"
OUTPUT="$ROOT/results/train/stage1"
ACCELERATE_CONFIG="$ROOT/configs/accelerate/zero2.yaml"
FFMPEG_PATH=""  # Optional FFmpeg executable; empty uses PATH or imageio-ffmpeg.

NPROC_PER_NODE=8
# =============================================================================

export FASTVR_MODEL_BASE="$MODEL_BASE"
export FASTVR_DATASET_JSONL="$DATASET_JSONL"
export FASTVR_OUTPUT="$OUTPUT"
export FFMPEG_PATH

CONFIG="$ROOT/configs/training/stage1_causal.yaml"
exec accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NPROC_PER_NODE" \
  -m fastvr.train --config "$CONFIG"
