<div align="center">

<img src="assets/logo.png" width="180" alt="FastVR logo">

# FastVR: Efficient Streaming Video Restoration<br>with One-Step Diffusion

**[Xiaoxu Chen](https://scholar.google.com/citations?user=-jJkyWsAAAAJ&hl=zh-CN)<sup>1,∗</sup>, Qin Yang<sup>1,2,∗</sup>, [Haoran Bai](https://csbhr.github.io/)<sup>1</sup>, Sibin Deng<sup>1</sup>, [Ying Chen](https://scholar.google.com/citations?user=NpTmcKEAAAAJ&hl=en)<sup>1,†</sup>**

<sup>1</sup>Alibaba Group &nbsp;&nbsp; <sup>2</sup>Xidian University<br>
<sup>∗</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

[![Paper](https://img.shields.io/badge/arXiv-2609.36757-b31b1b)](https://arxiv.org/abs/2609.36757)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://chenxx89.github.io/projects/fastvr/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/chenxx89/FastVR)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Nodes-blueviolet)](ComfyUI/README.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

</div>

<p align="center">
  <strong>FastVR is a one-step diffusion framework for video restoration, supporting
  arbitrary-scale super-resolution, and streaming
  inference for long videos.</strong>
</p>

<div align="center">
  <img src="assets/teaser.png" width="100%" alt="FastVR restoration quality, inference speed, and GPU memory comparison">
</div>

## 🔥 News

- **2026-09-29:** The [FastVR paper](https://arxiv.org/abs/2609.36757) is released.
- **2026-09-29:** Training and inference code are released.

## 🧩 Method Overview

<div align="center">
  <img src="assets/overview.png" width="100%" alt="Overview of FastVR inference and two-stage training">
</div>

## 🔗 More from Our Team

| Project | Highlight | Paper | Repository |
| :---: | :---: | :---: | :---: |
| **SATB-VR** | Flexible trade-off between restoration quality and inference speed. | [arXiv](https://arxiv.org/abs/2606.28677) | [GitHub](https://github.com/chenxx89/SATB-VR) |
| **Vivid-VR** (ICLR 2026) | High-quality video restoration with photorealistic detail. | [arXiv](https://arxiv.org/abs/2508.14483) | [GitHub](https://github.com/csbhr/Vivid-VR) |

## 🎨 ComfyUI

FastVR includes `FastVR Model Loader` and `FastVR Video Enhancer` nodes for
ComfyUI `IMAGE` frame batches. See [ComfyUI integration](ComfyUI/README.md) for
installation and VideoHelperSuite workflow instructions.

## 🔧 Dependencies and Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/chenxx89/FastVR.git
   cd FastVR
   ```

2. Create the environment and install dependencies. Python 3.10 or newer and a
   CUDA-capable NVIDIA GPU are required. Install the
   [PyTorch build](https://pytorch.org/get-started/locally/) matching your CUDA
   environment before installing FastVR.

   ```bash
   conda create -n fastvr python=3.10 -y
   conda activate fastvr

   pip install torch torchvision
   pip install -e .
   ```

3. Weights are not included in this repository. Inference automatically downloads
   [FastVR weights](https://huggingface.co/chenxx89/FastVR) to `CKPT_PATH`; training
   downloads [Wan weights](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) to
   `MODEL_BASE`. For manual installation, place both DiT shards and
   `vae.safetensors` in `CKPT_PATH` without merging the shards.

   The default directory layout is:

   ```text
   checkpoints/
   ├── FastVR/                       # Inference
   │   ├── dit-00001-of-00002.safetensors
   │   ├── dit-00002-of-00002.safetensors
   │   ├── vae.safetensors
   │   └── empty_prompt.pt           # Included in this repository
   └── Wan2.2-TI2V-5B/              # Training only
       ├── Wan2.2_VAE.pth
       ├── diffusion_pytorch_model-00001-of-00003.safetensors
       ├── diffusion_pytorch_model-00002-of-00003.safetensors
       └── diffusion_pytorch_model-00003-of-00003.safetensors
   ```

4. We recommend installing [FFmpeg](https://ffmpeg.org/download.html) with
   `libx265` support. If FFmpeg is not available from `PATH`, set its executable
   path:

   ```bash
   export FFMPEG_PATH=/path/to/ffmpeg
   ```

## 🚀 Quick Inference

### Shell entry point

Edit the **User configuration** block in `scripts/infer.sh`, then run:

```bash
bash scripts/infer.sh
```

### Command line

```bash
python3 -m fastvr.infer \
  --ckpt_path /path/to/FastVR \
  --input /path/to/input.mp4 \
  --output_dir outputs \
  --upscale 1 \
  --target_short_edge 1024 \
  --streaming \
  --enable_denoise_tiling
```

The shell script is the recommended editable entry point. Use
`python3 -m fastvr.infer` for direct command-line automation.

### Options

Common inference options are listed below.

| Option | Description |
|---|---|
| `--input` | Video, image, JSONL, frame directory, video directory, or a root of frame-sequence directories |
| `--output_dir` | Output directory; results keep their input base names |
| `--fps` | Output FPS for a single image or frame directory; video files use source FPS |
| `--upscale` | Final output scale relative to the source dimensions |
| `--target_short_edge` | Model-processing short edge; final dimensions still follow `--upscale` |
| `--streaming` | Enable bounded end-to-end long-video streaming |
| `--save_formats` | Save `mp4`, `png`, or both |
| `--color_fix` | Optionally apply `adain` or `wavelet` color correction |

### Input and output

1. **Input**
   - Supports videos, images, frame directories, video directories, roots
     containing multiple frame-sequence directories, and JSONL manifests.
   - Each JSONL `Filepath` may point to a video, image, or frame directory. Frame
     directories can specify their FPS:

     ```json
     {"Filepath": "/absolute/path/to/clip_frames", "Fps": 30}
     ```

   - See `configs/data/inference.example.jsonl` for a complete example.

2. **Output**
   - **Filename:** keeps the input base name and skips existing results.
   - **Resolution:** saves at the dimensions selected by `--upscale`.
   - **FPS:** videos preserve source FPS, including fractional values; images and frame directories use `--fps` (30 by default). JSONL `Fps` overrides either.
   - **Audio:** preserves available source audio in MP4 output.

### Resolution and streaming

- **`--upscale`:** controls the saved resolution. For a `W×H` source, the output
  is `round(W×upscale) × round(H×upscale)`; use `1` for same-resolution
  enhancement or `2` for 2× output.
- **`--target_short_edge`:** controls only the model-processing resolution. The
  input is resized with its aspect ratio preserved; the default short edge is
  `1024`, while the saved size still follows `--upscale`.
- **`--streaming`:** processes long videos with bounded memory and asynchronous
  I/O without changing resolution or FPS. It is enabled by default in
  `scripts/infer.sh`; set `STREAMING=0` to disable it.
- **`--enable_denoise_tiling`:** reduces peak GPU memory through spatial tiling,
  without changing the saved resolution.

## 🏋️ Training

1. Prepare a JSONL training manifest. `Filepath` must be an absolute video path;
   `Start_Frame` and `End_Frame` are optional:

   ```json
   {"Filepath": "/absolute/path/to/example.mp4", "Start_Frame": 0, "End_Frame": 121}
   ```

   See `configs/data/train.example.jsonl` for a complete example.

2. Edit the **User configuration** block in `scripts/train_stage1.sh`, then run:

   ```bash
   bash scripts/train_stage1.sh
   ```

3. Choose a Stage 1 checkpoint directory containing both DiT shards, set
   `STAGE1_CHECKPOINT` in
   `scripts/train_stage2.sh`, update the remaining paths, then run:

   ```bash
   bash scripts/train_stage2.sh
   ```

4. Model checkpoints are written to
   `OUTPUT/checkpoints/epoch-<epoch>-step-<step>/` as the same two DiT shards.
   Copy both files to the inference checkpoint directory to use trained weights.
   To continue an
   interrupted run, set `resume_from_checkpoint` in the corresponding training
   YAML to the absolute `training_states` path:

   ```yaml
   resume_from_checkpoint: /absolute/path/to/output/training_states
   ```

## 📝 Citation

If FastVR is useful for your research, please cite the [paper](https://arxiv.org/abs/2609.36757):

```bibtex
@misc{chen2026fastvrefficientstreamingvideo,
  title         = {FastVR: Efficient Streaming Video Restoration with One-Step Diffusion},
  author        = {Xiaoxu Chen and Qin Yang and Haoran Bai and Sibin Deng and Ying Chen},
  year          = {2026},
  eprint        = {2609.36757},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.36757}
}
```

## 🙏 Acknowledgements

FastVR builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2),
[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio), and video
degradation practices from
[RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR). It also uses
[DISTS/pyiqa](https://github.com/chaofengc/IQA-PyTorch) for Stage 2 perceptual
supervision and [FFmpeg](https://ffmpeg.org/) for video processing.

## 📄 License

FastVR code is released under the [Apache License 2.0](LICENSE). Third-party code,
dependencies, and the Wan2.2 base model remain subject to their respective
licenses. Users are responsible for reviewing the model licenses before
redistribution or commercial use.
