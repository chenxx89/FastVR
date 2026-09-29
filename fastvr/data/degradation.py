"""RealVSR-style video degradation for FastVR training.

The pipeline applies two configurable spatial degradation stages followed by
temporal degradation. Input arrays use uint8 ``[frames, height, width, channels]``
layout; returned GT and LQ tensors use normalized ``[frames, channels, height,
width]`` layout and remain aligned.

Usage:
    from fastvr.data.degradation import RealVSRDegradationHelper
    helper = RealVSRDegradationHelper("path/to/degradation_config.yaml")
    gt, lq = helper(frame_array)
"""

import numpy as np

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from fastvr.data.degradation_transforms import (
    Clip,
    RescaleToZeroOne,
    UnsharpMasking,
    img2tensor,
)
from fastvr.data.random_degradations import (
    RandomBlur,
    RandomJPEGCompression,
    RandomNoise,
    RandomResize,
    RandomTemporalDegradation,
    VideoCompressor,
)
class RealVSRDegradationHelper:
    """Build and run the fixed FastVR degradation pipeline."""

    def __init__(self, config):
        opt = OmegaConf.load(config)
        self.opt = opt

        self.random_mpeg = VideoCompressor(
            params=opt["degradation"]["random_mpeg"]["params"],
            keys=opt["degradation"]["random_mpeg"]["keys"],
        )

        # Spatial degradation stage 1.
        self.random_blur_1 = RandomBlur(
            params=opt["degradation_1"]["random_blur"]["params"],
            keys=opt["degradation_1"]["random_blur"]["keys"],
        )
        self.random_resize_1 = RandomResize(
            params=opt["degradation_1"]["random_resize"]["params"],
            keys=opt["degradation_1"]["random_resize"]["keys"],
        )
        self.random_noise_1 = RandomNoise(
            params=opt["degradation_1"]["random_noise"]["params"],
            keys=opt["degradation_1"]["random_noise"]["keys"],
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt["degradation_1"]["random_jpeg"]["params"],
            keys=opt["degradation_1"]["random_jpeg"]["keys"],
        )
        self.random_mpeg_1 = VideoCompressor(
            params=opt["degradation_1"]["random_mpeg"]["params"],
            keys=opt["degradation_1"]["random_mpeg"]["keys"],
        )

        # Spatial degradation stage 2.
        self.random_blur_2 = RandomBlur(
            params=opt["degradation_2"]["random_blur"]["params"],
            keys=opt["degradation_2"]["random_blur"]["keys"],
        )
        self.random_resize_2 = RandomResize(
            params=opt["degradation_2"]["random_resize"]["params"],
            keys=opt["degradation_2"]["random_resize"]["keys"],
        )
        self.random_noise_2 = RandomNoise(
            params=opt["degradation_2"]["random_noise"]["params"],
            keys=opt["degradation_2"]["random_noise"]["keys"],
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt["degradation_2"]["random_jpeg"]["params"],
            keys=opt["degradation_2"]["random_jpeg"]["keys"],
        )
        self.random_mpeg_2 = VideoCompressor(
            params=opt["degradation_2"]["random_mpeg"]["params"],
            keys=opt["degradation_2"]["random_mpeg"]["keys"],
        )

        # Final resize, blur, and compression.
        self.resize_final = RandomResize(
            params=opt["degradation_2"]["resize_final"]["params"],
            keys=opt["degradation_2"]["resize_final"]["keys"],
        )
        self._resize_final_uses_source_size = (
            self.resize_final.params.get("target_size") is None
        )
        self.blur_final = RandomBlur(
            params=opt["degradation_2"]["blur_final"]["params"],
            keys=opt["degradation_2"]["blur_final"]["keys"],
        )
        self.mpeg_final = VideoCompressor(
            params=opt["degradation_2"]["mpeg_final"]["params"],
            keys=opt["degradation_2"]["mpeg_final"]["keys"],
        )

        # Preprocessing and output range transforms.
        self.usm = UnsharpMasking(
            params=opt["transforms"]["usm"]["params"],
            keys=opt["transforms"]["usm"]["keys"],
        )
        self.clip = Clip(keys=opt["transforms"]["clip"]["keys"])
        self.rescale = RescaleToZeroOne(keys=opt["transforms"]["rescale"]["keys"])

        self.random_temp_deg = RandomTemporalDegradation(
            params=opt["temporal_degradation"]["params"],
            keys=opt["temporal_degradation"]["keys"],
        )

    @torch.no_grad()
    def degrade(self, videos: np.ndarray):
        if videos.ndim != 4 or videos.shape[-1] != 3:
            raise ValueError(f"Video frames must use THWC RGB layout, got {videos.shape}")
        videos = np.clip(videos, 0, 255).astype(np.uint8, copy=False)
        gt_frames = [frame for frame in videos]
        out_dict = {"lqs": [f.copy() for f in gt_frames], "gts": [f.copy() for f in gt_frames]}

        out_dict = self.usm(out_dict)
        out_dict = self.random_mpeg(out_dict)

        # Spatial degradation stage 1.
        degradation_1 = [
            self.random_blur_1,
            self.random_resize_1,
            self.random_noise_1,
            self.random_jpeg_1,
            self.random_mpeg_1,
        ]
        if np.random.uniform() < self.opt["degradation_1"].get("random_pipe_prob", 0):
            np.random.shuffle(degradation_1)
        for operation in degradation_1:
            out_dict = operation(out_dict)

        # Spatial degradation stage 2.
        degradation_2 = [
            self.random_blur_2,
            self.random_resize_2,
            self.random_noise_2,
            self.random_jpeg_2,
            self.random_mpeg_2,
        ]
        if np.random.uniform() < self.opt["degradation_2"].get("random_pipe_prob", 0):
            np.random.shuffle(degradation_2)
        for operation in degradation_2:
            out_dict = operation(out_dict)

        # Restore the source resolution before final degradation.
        if self._resize_final_uses_source_size:
            self.resize_final.params["target_size"] = videos.shape[1:3]
        degradation_final = [self.resize_final, self.blur_final, self.mpeg_final]
        if np.random.uniform() < self.opt["degradation_2"].get(
            "random_pipe_prob_final", 0
        ):
            np.random.shuffle(degradation_final)
        for operation in degradation_final:
            out_dict = operation(out_dict)

        # Convert degraded arrays back to normalized tensors.
        out_dict = self.clip(out_dict)
        out_dict = self.rescale(out_dict)

        for k in out_dict.keys():
            out_dict[k] = torch.stack(img2tensor(out_dict[k], bgr2rgb=False), dim=0)

        # Apply temporal degradation after spatial processing.
        out_dict = self.random_temp_deg(out_dict)

        if out_dict["lqs"].shape != out_dict["gts"].shape:
            out_dict["lqs"] = F.interpolate(
                out_dict["lqs"], size=out_dict["gts"].shape[2:4], mode="bicubic"
            )

        return out_dict["gts"], out_dict["lqs"]

    def __call__(self, videos: np.ndarray):
        hq_video, lq_video = self.degrade(videos)
        return hq_video.clip(0, 1), lq_video.clip(0, 1)
