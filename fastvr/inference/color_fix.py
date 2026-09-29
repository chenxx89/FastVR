"""
Color Correction Utilities for Video Super-Resolution

This module provides color correction methods (ADAIN and Wavelet) to align
the color distribution of generated high-quality videos with the input
low-quality videos. Ported from FlashVSR.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean and standard deviation of feature maps.

    Args:
        feat: Input tensor of shape (N, C, H, W).
        eps: Small value to avoid division by zero.

    Returns:
        Tuple of (mean, std), each of shape (N, C, 1, 1).
    """
    assert feat.dim() == 4, 'feat must be (N, C, H, W)'
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Adaptive Instance Normalization.

    Aligns the mean and variance of content features to match style features.

    Args:
        content_feat: Content features of shape (N, C, H, W).
        style_feat: Style features of shape (N, C, H, W).

    Returns:
        Normalized content features with style statistics.
    """
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N and C must match"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _make_gaussian3x3_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a 3x3 Gaussian kernel for wavelet blur."""
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply wavelet-style Gaussian blur with dilation.

    Args:
        x: Input tensor of shape (N, C, H, W).
        radius: Dilation radius for the convolution.

    Returns:
        Blurred tensor of same shape.
    """
    assert x.dim() == 4, 'x must be (N, C, H, W)'
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    out = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C)
    return out


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose image into high-frequency and low-frequency components.

    Args:
        x: Input tensor of shape (N, C, H, W).
        levels: Number of decomposition levels.

    Returns:
        Tuple of (high_freq, low_freq) tensors.
    """
    assert x.dim() == 4, 'x must be (N, C, H, W)'
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2 ** i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """Reconstruct image using high-freq from content and low-freq from style.

    This preserves the structural details of content while adopting the
    color/brightness characteristics of style.

    Args:
        content: Content tensor of shape (N, C, H, W).
        style: Style tensor of shape (N, C, H, W).
        levels: Number of decomposition levels.

    Returns:
        Reconstructed tensor.
    """
    # Process both inputs in one batch so each level uses one convolution launch.
    combined = torch.cat((content, style), dim=0)
    high, low = _wavelet_decompose(combined, levels=levels)
    c_high, _ = high.chunk(2, dim=0)
    _, s_low = low.chunk(2, dim=0)
    return c_high + s_low


class TorchColorCorrector(nn.Module):
    """Stateless color correction module for video frames.

    Supports two methods:
      - 'adain': Adaptive Instance Normalization (faster, good for most cases)
      - 'wavelet': Wavelet-based reconstruction (preserves more structure)

    The module processes video tensors of shape (B, C, F, H, W) and supports
    chunked processing to manage memory usage.
    """

    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Flatten temporal dimension: (B, C, F, H, W) -> (B*F, C, H, W)."""
        assert x.dim() == 5, 'Input must be (B, C, F, H, W)'
        B, C, F, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        return y, B, F

    @staticmethod
    def _unflatten_time(y: torch.Tensor, B: int, F: int) -> torch.Tensor:
        """Unflatten temporal dimension: (B*F, C, H, W) -> (B, C, F, H, W)."""
        BF, C, H, W = y.shape
        assert BF == B * F
        return y.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal['wavelet', 'adain'] = 'adain',
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply color correction to high-quality video using LQ reference.

        Args:
            hq_image: High-quality video tensor of shape (B, 3, F, H, W) in [-1, 1].
            lq_image: Low-quality reference tensor of shape (B, 3, F, H, W) in [-1, 1].
            clip_range: Value range to clamp output to.
            method: Color correction method ('adain' or 'wavelet').
            chunk_size: If set, process frames in chunks to reduce memory usage.

        Returns:
            Color-corrected video tensor of same shape as hq_image.
        """
        if hq_image.ndim != 5 or hq_image.shape[1] != 3:
            raise ValueError(
                "HQ input must be (B, 3, F, H, W), "
                f"got {tuple(hq_image.shape)}."
            )
        if lq_image.ndim != 5 or lq_image.shape[1] != 3:
            raise ValueError(
                "LQ reference must be (B, 3, F, H, W), "
                f"got {tuple(lq_image.shape)}."
            )
        if hq_image.shape != lq_image.shape:
            raise ValueError(
                "HQ and LQ color-correction shapes must match exactly: "
                f"HQ={tuple(hq_image.shape)}, LQ={tuple(lq_image.shape)}."
            )

        B, C, F, H, W = hq_image.shape

        if chunk_size is None or chunk_size >= F:
            # Process all frames at once
            hq4, B_out, F_out = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)

            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"Unknown method: {method}")

            out4 = torch.clamp(out4, *clip_range)
            out = self._unflatten_time(out4, B_out, F_out)
            return out

        # Process in chunks
        outs = []
        for start in range(0, F, chunk_size):
            end = min(start + chunk_size, F)
            hq_chunk = hq_image[:, :, start:end]
            lq_chunk = lq_image[:, :, start:end]

            hq4, B_, F_ = self._flatten_time(hq_chunk)
            lq4, _, _ = self._flatten_time(lq_chunk)

            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"Unknown method: {method}")

            out4 = torch.clamp(out4, *clip_range)
            out_chunk = self._unflatten_time(out4, B_, F_)
            outs.append(out_chunk)

        out = torch.cat(outs, dim=2)
        return out
