# ComfyUI-FastVR

FastVR 的 ComfyUI 节点实现。首版提供 `IMAGE` 视频帧批次的超分辨率与同分辨率增强。

## 安装

将 FastVR 仓库中的 `ComfyUI` 目录链接到 ComfyUI 的自定义节点目录：

```bash
git clone https://github.com/chenxx89/FastVR.git
cd FastVR
pip install -e .
ln -s "$(pwd)/ComfyUI" /path/to/ComfyUI/custom_nodes/ComfyUI-FastVR
```

请使用 ComfyUI 对应的 Python 环境执行安装命令，然后重新启动 ComfyUI。FastVR 仅支持 NVIDIA CUDA 设备。

## 模型权重

节点默认使用：

```text
ComfyUI/models/FastVR/
├── dit-00001-of-00002.safetensors
├── dit-00002-of-00002.safetensors
└── vae.safetensors
```

两个 DiT 分片缺一不可，需放在同一目录；每个不超过 5 GB（5,000,000,000 字节），无需手动拼接。

首次执行 `FastVR Model Loader` 时，如果文件缺失，将自动从 `chenxx89/FastVR` 下载。固定空提示词嵌入由 FastVR 项目自带，不需要在节点中设置。

## 工作流

推荐与 VideoHelperSuite 配合：

```text
VHS Load Video ── IMAGE ──> FastVR Video Enhancer ── IMAGE ──> VHS Video Combine
               ├─ AUDIO ─────────────────────────────────────> VHS Video Combine
               └─ FPS ───────────────────────────────────────> VHS Video Combine

FastVR Model Loader ───────> FastVR Video Enhancer
```

- FastVR 节点只处理帧，不修改 FPS，也不编码视频。
- 音频应从视频加载节点旁路连接至视频合成节点。
- `upscale` 控制最终输出尺寸。
- `target_short_edge` 控制模型处理尺寸；设为 `0` 时，按 `upscale` 决定的最终输出分辨率处理。默认值 `1024` 适合提高推理速度。
- `streaming` 默认启用 GPU 分块推理并限制 GPU 中间状态；ComfyUI 的输入和输出仍是完整 `IMAGE` 批次，因此主机内存占用仍随视频长度增长。
- `color_fix` 为 `none` 时不执行色彩校正。

节点不会暴露 seed、采样步数、CFG、提示词、因果分块尺寸或 VAE 选择；这些均为 FastVR 的固定推理设计。

