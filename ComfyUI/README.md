# ComfyUI-FastVR

[中文说明](README_zh.md)

ComfyUI nodes for FastVR video super-resolution and same-resolution restoration. The initial integration accepts and returns ComfyUI `IMAGE` frame batches.

## Installation

Link this repository's `ComfyUI` directory into the ComfyUI custom-node directory:

```bash
git clone https://github.com/chenxx89/FastVR.git
cd FastVR
pip install -e .
ln -s "$(pwd)/ComfyUI" /path/to/ComfyUI/custom_nodes/ComfyUI-FastVR
```

Run the installation command in ComfyUI's Python environment, then restart ComfyUI. FastVR requires an NVIDIA CUDA device.

## Checkpoints

The nodes use the following directory by default:

```text
ComfyUI/models/FastVR/
├── dit-00001-of-00002.safetensors
├── dit-00002-of-00002.safetensors
└── vae.safetensors
```

Both DiT shards are required and must stay in the same directory. Each shard is at most 5 GB (5,000,000,000 bytes); no manual concatenation is needed.

`FastVR Model Loader` automatically downloads missing files from `chenxx89/FastVR`. The fixed empty-prompt embedding is included in the FastVR project and requires no node setting.

## Workflow

The recommended VideoHelperSuite workflow is:

```text
VHS Load Video ── IMAGE ──> FastVR Video Enhancer ── IMAGE ──> VHS Video Combine
               ├─ AUDIO ─────────────────────────────────────> VHS Video Combine
               └─ FPS ───────────────────────────────────────> VHS Video Combine

FastVR Model Loader ───────> FastVR Video Enhancer
```

- FastVR processes frames only; it does not alter FPS or encode the output video.
- Pass audio directly from the video loader to the video-combine node.
- `upscale` controls the final output size.
- `target_short_edge` controls model processing resolution; set it to `0` to process at the final output resolution selected by `upscale`. The default `1024` is recommended for speed.
- `streaming` uses bounded GPU chunk processing by default. ComfyUI still materializes complete input and output `IMAGE` batches, so host-memory usage grows with video length.
- `color_fix=none` disables color correction.

The nodes intentionally do not expose seed, sampling steps, CFG, prompts, causal chunk sizes, or VAE selection because these are fixed parts of FastVR inference.

