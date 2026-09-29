"""FastVR lightweight encoder and decoder."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _MemoryBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=False),
            conv(n_out, n_out), nn.ReLU(inplace=False),
            conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=False)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TemporalPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class _TemporalGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class _SuperMemoryBlock(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f * 2, n_f * 2, 7, padding=3, groups=n_f * 2, bias=False),
            nn.Conv2d(n_f * 2, n_f * 4, 1), nn.ReLU(inplace=False),
            nn.Conv2d(n_f * 4, n_f, 1, bias=False),
        )

    def forward(self, x, past):
        return self.conv(torch.cat([x, past], 1)) + x


class _DecoderBackbone(nn.Module):
    """Fixed decoder backbone."""

    image_channels = 3
    frames_to_trim = 3

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(48, 1024, 1, bias=False),
            _SuperMemoryBlock(1024), _SuperMemoryBlock(1024), _SuperMemoryBlock(1024),
            conv(1024, 512 * 4),
            nn.ReLU(inplace=False),
            nn.PixelShuffle(2),
            _TemporalGrow(512, 1),
            _SuperMemoryBlock(512), _SuperMemoryBlock(512), _SuperMemoryBlock(512),
            conv(512, 256 * 4),
            nn.ReLU(inplace=False),
            nn.PixelShuffle(2),
            _TemporalGrow(256, 2),
            _SuperMemoryBlock(256), _SuperMemoryBlock(256), _SuperMemoryBlock(256),
            conv(256, 128 * 4),
            nn.ReLU(inplace=False),
            nn.PixelShuffle(2),
            _TemporalGrow(128, 2),
            conv(128, self.image_channels * 4),
        )


class _SplitConv(nn.Module):
    def __init__(self, conv, n_pre):
        super().__init__()
        self.n_pre = int(n_pre)
        kw = dict(kernel_size=conv.kernel_size, padding=conv.padding)
        self.conv_pre = nn.Conv2d(n_pre, conv.out_channels,
                                  bias=conv.bias is not None, **kw)
        self.conv_new = nn.Conv2d(conv.in_channels - n_pre, conv.out_channels,
                                  bias=False, **kw)

    def forward(self, x):
        return self.conv_pre(x[:, :self.n_pre]) + self.conv_new(x[:, self.n_pre:])


class _CondConv(nn.Module):
    def __init__(self, conv, n_extra):
        super().__init__()
        self.n_old, self.n_extra = conv.in_channels, n_extra
        kw = dict(kernel_size=conv.kernel_size, padding=conv.padding)
        self.conv_pre = nn.Conv2d(conv.in_channels, conv.out_channels,
                                  bias=conv.bias is not None, **kw)
        self.conv_new = nn.Conv2d(n_extra, conv.out_channels, bias=False, **kw)
        self._fold = None
        self._cursor = 0

    def clear(self):
        self._fold = None
        self._cursor = 0

    def load(self, fold):
        if fold.shape[2] != self.n_extra:
            raise ValueError(
                f"LQ fold channels ({fold.shape[2]}) != injection slot ({self.n_extra})"
            )
        self._fold = fold
        self._cursor = 0

    def forward(self, x):
        if self._fold is None or self._cursor >= self._fold.shape[1]:
            raise RuntimeError("_CondConv has no streaming condition frame available")
        condition = self._fold[:, self._cursor].to(device=x.device, dtype=x.dtype)
        self._cursor += 1
        return self.conv_pre(x) + self.conv_new(condition)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class _CausalConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        kt, kh, kw = ((kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size)
        self.kt = kt
        self.pad = (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0)
        self.conv = nn.Conv3d(in_ch, out_ch, (kt, kh, kw))

    def forward(self, x):
        return self.conv(F.pad(x, self.pad, value=0.))

    def forward_stream(self, frame, history):
        """Apply one causal temporal step and retain only the required history."""
        history = list(history)
        context = history + [frame]
        if len(context) < self.kt:
            context = [torch.zeros_like(frame)] * (self.kt - len(context)) + context
        context = torch.cat(context[-self.kt:], dim=2)
        output = self.conv(F.pad(context, self.pad[:4] + (0, 0), value=0.0))
        keep = max(0, self.kt - 1)
        return output, (history + [frame])[-keep:] if keep else []


class _Decoder(nn.Module):
    """Fixed, temporally streaming decoder with integrated restoration."""

    def __init__(self):
        super().__init__()
        self.align = nn.Sequential(
            _RMSNorm(48),
            nn.Conv3d(48, 1024, 1),
            nn.SiLU(),
            _CausalConv3d(1024, 1024, 3),
        )
        self.align_skip = nn.Conv3d(48, 1024, 1)
        self.tc = _DecoderBackbone()

        # Entry order is [latent | condition | aligned latent]. Keeping the
        # original split modules preserves the released checkpoint keys.
        entry = nn.Conv2d(48 + 3072 + 1024, 1024, 1, bias=False)
        self.tc.decoder[0] = _SplitConv(entry, 48)
        for index, fold in ((11, (4, 8, 8)), (18, (2, 4, 4)), (22, (1, 2, 2))):
            self.tc.decoder[index] = _CondConv(
                self.tc.decoder[index], 3 * fold[0] * fold[1] * fold[2]
            )

        self.output_head = _LumaOutputHead(num_in_ch=1, ngf=4)
        self.register_buffer(
            "rgb_to_yuv_matrix",
            torch.tensor([
                [0.29900, 0.58700, 0.11400],
                [-0.14714119, -0.28886916, 0.43601035],
                [0.61497538, -0.51496512, -0.10001026],
            ], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "yuv_to_rgb_matrix",
            torch.tensor([
                [1.0, 0.0, 1.13983],
                [1.0, -0.39465, -0.58060],
                [1.0, 2.03211, 0.0],
            ], dtype=torch.float32),
            persistent=False,
        )
        self._tc_cache = None
        self._align_cache = None
        self._trim_remaining = None
        self._decoded_latent_frames = None

    def initialize_stream_cache(self):
        """Initialize decoder state for one video."""
        self._tc_cache = [None] * len(self.tc.decoder)
        self._align_cache = []
        self._trim_remaining = self.tc.frames_to_trim
        self._decoded_latent_frames = 0
        for layer in self.tc.decoder:
            if isinstance(layer, _CondConv):
                layer.clear()

    def clear_stream_cache(self):
        """Release every tensor retained by the decoder stream."""
        self._tc_cache = None
        self._align_cache = None
        self._trim_remaining = None
        self._decoded_latent_frames = None
        for layer in self.tc.decoder:
            if isinstance(layer, _CondConv):
                layer.clear()

    def _require_stream_cache(self):
        if self._tc_cache is None:
            raise RuntimeError(
                "Decoder stream cache is not initialized; use "
                "LightweightVAE.stream_session() for each video"
            )

    @staticmethod
    def _fold_condition(block, temporal_fold, spatial_fold):
        return rearrange(
            block,
            "b c (f ft) (h fh) (w fw) -> b f (c ft fh fw) h w",
            ft=temporal_fold,
            fh=spatial_fold,
            fw=spatial_fold,
        )

    def _align_frame(self, latent_frame, history):
        value = self.align[0](latent_frame)
        value = self.align[1](value)
        value = self.align[2](value)
        value, history = self.align[3].forward_stream(value, history)
        return value + self.align_skip(latent_frame), history

    def _apply_tc_frame(self, value, memory):
        """Push one latent frame through the decoder while retaining causal state."""
        batch = value.shape[0]
        outputs = []
        queue = [(value, 0)]
        while queue:
            current, index = queue.pop(0)
            if index == len(self.tc.decoder):
                outputs.append(current)
                continue
            layer = self.tc.decoder[index]
            if isinstance(layer, (_MemoryBlock, _SuperMemoryBlock)):
                past = torch.zeros_like(current) if memory[index] is None else memory[index]
                next_value = layer(current, past)
                memory[index] = current
                queue.insert(0, (next_value, index + 1))
            elif isinstance(layer, _TemporalGrow):
                grown = layer(current)
                _, channels, height, width = grown.shape
                frames = grown.reshape(batch, layer.stride, channels, height, width)
                for next_value in reversed(frames.unbind(dim=1)):
                    queue.insert(0, (next_value, index + 1))
            else:
                queue.insert(0, (layer(current), index + 1))
        return outputs

    def set_reconstruction_dtype(self, dtype: torch.dtype) -> None:
        """Set the reconstruction backbone dtype."""
        self.align.to(dtype=dtype)
        self.align_skip.to(dtype=dtype)
        self.tc.to(dtype=dtype)

    def _restore_output(self, video: torch.Tensor) -> torch.Tensor:
        """Restore decoded frames in bounded temporal batches."""
        output_dtype = video.dtype
        batch, channels, frames, height, width = video.shape
        rgb = video.float().add(1.0).div(2.0).clamp(0.0, 1.0)
        rgb = rgb.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        outputs = []
        for start in range(0, rgb.shape[0], 5):
            chunk = rgb[start:start + 5]
            yuv = torch.einsum(
                "b c h w, c k -> b k h w",
                chunk,
                self.rgb_to_yuv_matrix.transpose(0, 1),
            )
            restored_y = self.output_head(yuv[:, :1])
            restored = torch.cat((restored_y, yuv[:, 1:]), dim=1)
            restored = torch.einsum(
                "b c h w, c k -> b k h w",
                restored,
                self.yuv_to_rgb_matrix.transpose(0, 1),
            )
            restored = F.interpolate(restored, size=(height, width), mode="bicubic")
            outputs.append(restored.clamp(0.0, 1.0))
        output = torch.cat(outputs, dim=0)
        output = output.reshape(batch, frames, channels, height, width)
        output = output.permute(0, 2, 1, 3, 4)
        return output.to(dtype=output_dtype).mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def forward(self, latents, condition_blocks):
        """Decode one latent frame at a time; no full-video feature map is built."""
        if torch.is_grad_enabled():
            raise RuntimeError("Decoder is inference-only and requires a streaming session")

        self._require_stream_cache()
        restored_chunks = []

        if len(condition_blocks) != latents.shape[2]:
            raise ValueError(
                f"Decoder received {latents.shape[2]} latent frames but "
                f"{len(condition_blocks)} condition blocks"
            )

        for local_index, condition_block in enumerate(condition_blocks):
            latent_frame = latents[:, :, local_index:local_index + 1]
            aligned, self._align_cache = self._align_frame(
                latent_frame, self._align_cache
            )
            condition_block = condition_block.to(
                device=latent_frame.device, dtype=latent_frame.dtype
            )

            folded_4x16 = self._fold_condition(condition_block, 4, 16)
            self.tc.decoder[11].load(self._fold_condition(condition_block, 4, 8))
            self.tc.decoder[18].load(self._fold_condition(condition_block, 2, 4))
            self.tc.decoder[22].load(self._fold_condition(condition_block, 1, 2))

            entry = torch.cat(
                [latent_frame[:, :, 0], folded_4x16[:, 0], aligned[:, :, 0]],
                dim=1,
            )
            decoded_frames = self._apply_tc_frame(entry, self._tc_cache)
            self._decoded_latent_frames += 1
            if self._trim_remaining:
                drop = min(self._trim_remaining, len(decoded_frames))
                decoded_frames = decoded_frames[drop:]
                self._trim_remaining -= drop
            if not decoded_frames:
                continue

            decoded = torch.stack(decoded_frames, dim=1)
            decoded = F.pixel_shuffle(decoded, 2).transpose(1, 2)
            restored_chunks.append(self._restore_output(decoded))

        if not restored_chunks:
            raise RuntimeError("Decoder produced no output frames")
        return torch.cat(restored_chunks, dim=2)


def _build_encoder() -> nn.Sequential:
    return nn.Sequential(
        conv(12, 64),
        nn.ReLU(inplace=False),
        _TemporalPool(64, 2),
        conv(64, 64, stride=2, bias=False),
        _MemoryBlock(64, 64), _MemoryBlock(64, 64), _MemoryBlock(64, 64),
        _TemporalPool(64, 2),
        conv(64, 64, stride=2, bias=False),
        _MemoryBlock(64, 64), _MemoryBlock(64, 64), _MemoryBlock(64, 64),
        _TemporalPool(64, 1),
        conv(64, 64, stride=2, bias=False),
        _MemoryBlock(64, 64), _MemoryBlock(64, 64), _MemoryBlock(64, 64),
        conv(64, 48),
    )


class _OutputDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class _OutputTripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


class _OutputDownBlock(nn.Module):
    """Downscaling with avgpool then double/triple conv"""

    def __init__(self, in_channels, out_channels, num_conv=2):
        super().__init__()
        if num_conv == 2:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                _OutputDoubleConv(in_channels, out_channels)
            )
        else:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                _OutputTripleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.pool_conv(x)


class _OutputResidualBlock(nn.Module):
    """Define a mobile-version Resnet block"""

    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class _OutputUpBlock(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = _OutputDoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
            # self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = _OutputDoubleConv(in_channels, out_channels)

        self.resblock1 = _OutputResidualBlock(out_channels)
        self.subpixel = nn.PixelShuffle(2)

    def interpolate(self, x):
        tensor_temp = x
        for i in range(3):
            tensor_temp = torch.cat((tensor_temp, x), 1)
        x = tensor_temp
        x = self.subpixel(x)
        return x

    def forward(self, x1, x2):
        x1 = self.interpolate(x1)

        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)

        ##Resnet
        out1 = self.resblock1(out)

        return out1


class _LumaOutputHead(nn.Module):
    """Luma-domain output head used by the complete LightweightVAE decoder."""

    def __init__(self, num_in_ch=1, ngf=4):
        super().__init__()

        self.padder_size = 16

        self.inconv1 = _OutputDoubleConv(num_in_ch, 3*ngf)
        # downsample
        self.down0 = _OutputDownBlock(3*ngf, 5*ngf)

        self.d0_resnet = nn.Sequential(
            _OutputResidualBlock(5*ngf),
        )
        self.down1 = _OutputDownBlock(5*ngf, 9*ngf)

        self.d1_resnet = nn.Sequential(
            _OutputResidualBlock(9*ngf),
        )

        self.down2 = _OutputDownBlock(9*ngf, 16*ngf)

        self.d2_resnet = nn.Sequential(
            _OutputResidualBlock(16*ngf),
        )
        self.down3 = _OutputDownBlock(16*ngf, 28*ngf, num_conv=3)

        self.d3_resnet = nn.Sequential(
            _OutputResidualBlock(28*ngf),
        )

        # Decoder skip path.
        self.up3_resnet = _OutputUpBlock(44*ngf, 16*ngf)
        self.up2_resnet = _OutputUpBlock(25*ngf, 9*ngf)
        self.up1_resnet = _OutputUpBlock(14*ngf, 5*ngf)
        self.up0_resnet = _OutputUpBlock(8*ngf, 3*ngf)

        # Residual output projection.
        self.outconv1 = nn.Conv2d(3*ngf, num_in_ch, 3, 1, 1, bias=False)

    def forward(self, x, alpha=1.0):
        x, H, W = self.check_image_size(x)

        x0 = self.inconv1(x)
        # downsample
        x1 = self.down0(x0)  # 1/2
        x1 = self.d0_resnet(x1)
        x2 = self.down1(x1)  # 1/4
        x2 = self.d1_resnet(x2)
        x3 = self.down2(x2)  # 1/8
        x3 = self.d2_resnet(x3)
        x4 = self.down3(x3)  # 1/16
        x4 = self.d3_resnet(x4)

        x5 = self.up3_resnet(x4, x3)  # 1/8
        x5 = self.up2_resnet(x5, x2)  # 1/4
        x5 = self.up1_resnet(x5, x1)  # 1/2
        x5 = self.up0_resnet(x5, x0)  # 1
        x5 = self.outconv1(x5)
        out = x + x5
        out = (1-alpha) * x + alpha * out
        return out[..., :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x, h, w




class LightweightVAE(nn.Module):
    """Frozen streaming encoder and LQ-conditioned streaming decoder.

    The public inference model uses one fixed encoder and decoder architecture. Both paths process the temporal dimension
    incrementally so intermediate feature maps do not grow with video length.
    """

    latent_channels = 48
    upsampling_factor = 16

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = _build_encoder()
        self.decoder = _Decoder()
        self._load_unified_checkpoint(checkpoint_path)
        self.encoder.to(dtype=torch.bfloat16)
        self.decoder.set_reconstruction_dtype(torch.bfloat16)
        self.requires_grad_(False)
        self.eval()
        self._stream_active = False
        self._encoder_cache = None
        self._encoder_input_frames = 0
        self._encoder_finalized = False
        self._condition_cache = None
        self._decoder_first_condition = True

    def initialize_stream_cache(self):
        """Initialize all temporal state for one video."""
        if self._stream_active:
            raise RuntimeError("LightweightVAE stream cache is already active")
        self._encoder_cache = [None] * len(self.encoder)
        self._encoder_input_frames = 0
        self._encoder_finalized = False
        self._condition_cache = None
        self._decoder_first_condition = True
        self.decoder.initialize_stream_cache()
        self._stream_active = True

    def clear_stream_cache(self):
        """Release all encoder and decoder state retained for the current video."""
        self._encoder_cache = None
        self._encoder_input_frames = 0
        self._encoder_finalized = False
        self._condition_cache = None
        self._decoder_first_condition = True
        self.decoder.clear_stream_cache()
        self._stream_active = False

    @contextmanager
    def stream_session(self):
        """Own the cache lifecycle for one complete video enhancement."""
        self.initialize_stream_cache()
        try:
            yield self
        finally:
            self.clear_stream_cache()

    def _require_stream_cache(self):
        if not self._stream_active or self._encoder_cache is None:
            raise RuntimeError(
                "LightweightVAE stream cache is not initialized; wrap each video "
                "with lightweight_vae.stream_session()"
            )

    @staticmethod
    def _load_checkpoint(path: str):
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "vae.pt"
        path = str(checkpoint_path)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported LightweightVAE checkpoint payload: {type(state)!r}")
        return dict(state)

    def _load_unified_checkpoint(self, path: str) -> None:
        """Strictly load the single released LightweightVAE checkpoint."""
        state = self._load_checkpoint(path)
        try:
            self.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"Unified LightweightVAE checkpoint does not match the model: {error}"
            ) from error

    @staticmethod
    def _encode_frame(encoder, frame, memory):
        value = frame
        for index, layer in enumerate(encoder):
            if isinstance(layer, _MemoryBlock):
                past = torch.zeros_like(value) if memory[index] is None else memory[index]
                next_value = layer(value, past)
                memory[index] = value
                value = next_value
            elif isinstance(layer, _TemporalPool):
                buffered = [] if memory[index] is None else memory[index]
                buffered.append(value)
                memory[index] = buffered
                if len(buffered) < layer.stride:
                    return None
                batch, channels, height, width = value.shape
                grouped = torch.stack(buffered, dim=1).reshape(
                    batch * layer.stride, channels, height, width
                )
                value = layer(grouped)
                memory[index] = []
            else:
                value = layer(value)
        return value

    @torch.no_grad()
    def encode_stream(
        self,
        video: torch.Tensor,
        *,
        final: bool = False,
        device=None,
    ) -> torch.Tensor | None:
        """Encode a temporal chunk while preserving causal state across calls."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"LightweightVAE encoder expects BCTHW RGB, got {tuple(video.shape)}")
        if video.shape[-2] % 2 or video.shape[-1] % 2:
            raise ValueError("LightweightVAE encoder height and width must be divisible by 2")
        self._require_stream_cache()
        if self._encoder_finalized:
            raise RuntimeError("LightweightVAE encoder stream has already been finalized")

        device = device or next(self.parameters()).device
        dtype = next(self.encoder.parameters()).dtype
        frame_count = video.shape[2]
        outputs = []

        for frame_index in range(frame_count):
            frame = video[:, :, frame_index].to(device=device, dtype=dtype)
            frame = frame.add(1.0).mul(0.5).clamp(0.0, 1.0)
            frame = F.pixel_unshuffle(frame, 2)
            encoded = self._encode_frame(self.encoder, frame, self._encoder_cache)
            if encoded is not None:
                outputs.append(encoded)
        self._encoder_input_frames += frame_count

        if final:
            if frame_count == 0 and self._encoder_input_frames == 0:
                raise ValueError("LightweightVAE encoder requires at least one frame")
            padding = (-self._encoder_input_frames) % 4
            if padding:
                if frame_count == 0:
                    raise ValueError("The final encoder chunk must include the last video frame")
                frame = video[:, :, -1].to(device=device, dtype=dtype)
                frame = frame.add(1.0).mul(0.5).clamp(0.0, 1.0)
                frame = F.pixel_unshuffle(frame, 2)
                for _ in range(padding):
                    encoded = self._encode_frame(self.encoder, frame, self._encoder_cache)
                    if encoded is not None:
                        outputs.append(encoded)
            self._encoder_finalized = True

        if not outputs:
            return None
        return torch.stack(outputs, dim=2)

    @torch.no_grad()
    def encode(self, video: torch.Tensor, device=None) -> torch.Tensor:
        """Encode one complete video using the streaming encoder."""
        encoded = self.encode_stream(video, final=True, device=device)
        if encoded is None:
            raise RuntimeError("LightweightVAE encoder produced no latent frames")
        return encoded

    @staticmethod
    def _resize_condition(condition, target_height, target_width):
        if condition.ndim != 5 or condition.shape[1] != 3:
            raise ValueError(
                f"LightweightVAE condition expects BCTHW RGB, got {tuple(condition.shape)}"
            )
        if condition.shape[-2:] == (target_height, target_width):
            return condition
        return torch.stack(
            [
                F.interpolate(
                    condition[:, :, index],
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                )
                for index in range(condition.shape[2])
            ],
            dim=2,
        )

    def queue_decode_condition(self, condition, *, target_height, target_width):
        condition = self._resize_condition(condition, target_height, target_width)
        if self._condition_cache is None:
            self._condition_cache = condition
        else:
            if self._condition_cache.shape[:2] != condition.shape[:2] or self._condition_cache.shape[-2:] != condition.shape[-2:]:
                raise ValueError("Decoder condition chunks must have matching BCTHW dimensions")
            self._condition_cache = torch.cat([self._condition_cache, condition], dim=2)

    def _take_condition_blocks(self, latent_count):
        blocks = []
        for _ in range(latent_count):
            required = 1 if self._decoder_first_condition else 4
            available = 0 if self._condition_cache is None else self._condition_cache.shape[2]
            if available < required:
                raise RuntimeError(
                    f"Decoder condition cache has {available} frame(s), needs {required}"
                )
            if self._decoder_first_condition:
                block = self._condition_cache[:, :, :1].expand(-1, -1, 4, -1, -1)
                self._decoder_first_condition = False
            else:
                block = self._condition_cache[:, :, :4]
            self._condition_cache = self._condition_cache[:, :, required:]
            blocks.append(block)
        return blocks

    @torch.no_grad()
    def decode_stream(self, latents, condition=None, *, final: bool = False):
        """Decode a latent chunk while preserving decoder state across calls."""
        if latents.ndim != 5 or latents.shape[1] != 48:
            raise ValueError(f"LightweightVAE decoder expects BCTHW latents, got {tuple(latents.shape)}")
        self._require_stream_cache()
        device = next(self.decoder.parameters()).device
        dtype = next(self.decoder.parameters()).dtype
        latents = latents.to(device=device, dtype=dtype)
        target_height = latents.shape[-2] * 16
        target_width = latents.shape[-1] * 16
        if condition is not None:
            self.queue_decode_condition(
                condition,
                target_height=target_height,
                target_width=target_width,
            )
        condition_blocks = self._take_condition_blocks(latents.shape[2])
        decoded = self.decoder(latents, condition_blocks)
        if final and self._condition_cache is not None and self._condition_cache.shape[2]:
            raise RuntimeError(
                f"Decoder condition cache still contains {self._condition_cache.shape[2]} frame(s)"
            )
        return decoded

    @torch.no_grad()
    def decode(self, latents, condition):
        """Decode one complete video using the streaming decoder."""
        return self.decode_stream(latents, condition=condition, final=True)
