# Copyright (c) OpenMMLab. All rights reserved.
"""Random image and video degradations used by FastVR training."""

import logging
import math
import random
import subprocess
import threading
from functools import lru_cache

import cv2
import numpy as np
import torch

from fastvr.data.ffmpeg import resolve_ffmpeg
from .degradation_kernels import random_mixed_kernels


logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _available_ffmpeg_encoders(ffmpeg: str) -> frozenset[str]:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to query FFmpeg encoders:\n{completed.stdout}")
    return frozenset(
        line.split()[1]
        for line in completed.stdout.splitlines()
        if len(line.split()) >= 2 and line.lstrip().startswith("V")
    )


class VideoCompressor:
    """Apply an FFmpeg encode/decode round trip without blocking on pipes."""

    _FORMATS = {
        "mpeg4": "m4v",
        "libx264": "h264",
        "libx265": "hevc",
    }
    _MAX_CONSECUTIVE_FAILURES = 5
    _FAILURE_RATE_MIN_ATTEMPTS = 20
    _MAX_FAILURE_RATE = 0.2

    def __init__(self, params, keys):
        self.params = params
        self.keys = keys
        self.codecs = list(params["codec"])
        probabilities = np.asarray(params["codec_prob"], dtype=np.float64)
        if len(probabilities) != len(self.codecs):
            raise ValueError("codec and codec_prob must have the same length")
        if any(codec not in self._FORMATS for codec in self.codecs):
            raise ValueError(f"Unsupported video codec: {self.codecs}")
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            raise ValueError("codec_prob must have a positive sum")
        self.codec_probabilities = probabilities / probability_sum
        self.presets = list(params["preset_list"])
        self.gops = list(params.get("gop_list", [12, 25, 50, 250]))
        self.threads = str(params.get("ffmpeg_threads", 1))
        self.timeout = float(params.get("ffmpeg_timeout", 120))
        self._attempts = 0
        self._failures = 0
        self._consecutive_failures = 0
        available = _available_ffmpeg_encoders(resolve_ffmpeg())
        missing = sorted(set(self.codecs) - available)
        if missing:
            raise RuntimeError(
                f"FFmpeg does not provide the configured video encoders: {missing}"
            )

    def _record_failure(self, error: Exception) -> None:
        self._failures += 1
        self._consecutive_failures += 1
        if self._failures <= 3 or self._failures % 20 == 0:
            logger.warning(
                "Video compression failed (%d/%d, consecutive=%d): %s",
                self._failures,
                self._attempts,
                self._consecutive_failures,
                error,
            )
        failure_rate = self._failures / self._attempts
        if (
            self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES
            or (
                self._attempts >= self._FAILURE_RATE_MIN_ATTEMPTS
                and failure_rate > self._MAX_FAILURE_RATE
            )
        ):
            raise RuntimeError(
                "Video compression is repeatedly failing: "
                f"{self._failures}/{self._attempts} attempts failed "
                f"({failure_rate:.1%}), including "
                f"{self._consecutive_failures} consecutive failures."
            ) from error

    def _build_commands(
        self,
        width,
        height,
        fps,
        codec,
        crf,
        preset,
        blur_sigma,
        gop,
    ):
        ffmpeg = resolve_ffmpeg()
        encode = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-threads",
            self.threads,
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if blur_sigma > 1e-3:
            encode.extend(["-vf", f"gblur=sigma={blur_sigma}"])

        encode.extend(["-c:v", codec, "-pix_fmt", "yuv420p"])
        if codec == "mpeg4":
            quality = int(np.clip((crf - 15) * 0.5 + 2, 2, 31))
            encode.extend(["-q:v", str(quality)])
        else:
            encode.extend(
                ["-crf", f"{crf:.1f}", "-preset", preset, "-g", str(gop)]
            )
        if codec == "libx265":
            encode.extend(["-x265-params", "pools=1:frame-threads=1:log-level=none"])
        encode.extend(
            ["-threads", self.threads, "-f", self._FORMATS[codec], "pipe:1"]
        )

        decode = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-threads",
            self.threads,
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            self.threads,
            "pipe:1",
        ]
        return encode, decode

    @staticmethod
    def _feed_stdin(pipe, data, errors):
        try:
            pipe.write(data)
        except (BrokenPipeError, OSError) as error:
            errors.append(error)
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    @staticmethod
    def _drain_pipe(pipe, chunks):
        try:
            chunks.append(pipe.read())
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def apply_compression(self, value, crf=30, fps=25, blur_sigma=0.0):
        encoder = decoder = None
        feeder = stderr_reader = None
        self._attempts += 1
        try:
            if isinstance(value, torch.Tensor):
                frames = (
                    value.clamp(0, 1)
                    .permute(0, 2, 3, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            elif isinstance(value, np.ndarray):
                frames = np.clip(value, 0, 255).astype(np.uint8, copy=False)
            elif isinstance(value, list):
                frames = np.clip(np.stack(value), 0, 255).astype(np.uint8, copy=False)
            else:
                raise TypeError(f"Unsupported video type: {type(value).__name__}")

            frame_count, input_height, input_width, channels = frames.shape
            output_height = math.ceil(input_height / 2) * 2
            output_width = math.ceil(input_width / 2) * 2
            pad_height = output_height - input_height
            pad_width = output_width - input_width
            if pad_height or pad_width:
                frames = np.pad(
                    frames,
                    ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
                    mode="reflect",
                )
            frames = np.ascontiguousarray(frames)

            codec = str(np.random.choice(self.codecs, p=self.codec_probabilities))
            preset = str(np.random.choice(self.presets))
            gop = int(np.random.choice(self.gops))
            if codec == "libx265" and preset == "slow":
                preset = "medium"
            if "264" in codec:
                crf -= self.params.get("264_crf_shift", 0)
            encode_command, decode_command = self._build_commands(
                output_width,
                output_height,
                fps,
                codec,
                crf,
                preset,
                blur_sigma,
                gop,
            )

            encoder = subprocess.Popen(
                encode_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            decoder = subprocess.Popen(
                decode_command,
                stdin=encoder.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            encoder.stdout.close()

            feed_errors = []
            encoder_errors = []
            feeder = threading.Thread(
                target=self._feed_stdin,
                args=(encoder.stdin, memoryview(frames), feed_errors),
                daemon=True,
            )
            stderr_reader = threading.Thread(
                target=self._drain_pipe,
                args=(encoder.stderr, encoder_errors),
                daemon=True,
            )
            feeder.start()
            stderr_reader.start()

            try:
                output_bytes, decoder_error = decoder.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"FFmpeg timed out after {self.timeout:g}s "
                    f"({codec}, {frame_count}x{output_height}x{output_width})"
                ) from error

            encoder.wait(timeout=10)
            feeder.join(timeout=2)
            stderr_reader.join(timeout=2)
            if feeder.is_alive() or stderr_reader.is_alive():
                raise RuntimeError("FFmpeg pipe worker did not terminate")
            if feed_errors:
                raise RuntimeError(f"FFmpeg input pipe failed: {feed_errors[0]}")
            if encoder.returncode != 0:
                message = b"".join(encoder_errors).decode(errors="replace")
                raise RuntimeError(f"FFmpeg encode failed:\n{message}")
            if decoder.returncode != 0:
                raise RuntimeError(
                    "FFmpeg decode failed:\n" + decoder_error.decode(errors="replace")
                )

            expected_size = frame_count * output_height * output_width * channels
            if len(output_bytes) != expected_size:
                decoded_frames = len(output_bytes) / (output_height * output_width * channels)
                raise RuntimeError(
                    f"FFmpeg decoded {decoded_frames:g} frames; expected {frame_count}"
                )

            output = np.frombuffer(output_bytes, np.uint8).reshape(
                frame_count, output_height, output_width, channels
            )
            if pad_height or pad_width:
                output = output[:, :input_height, :input_width, :]

            if isinstance(value, torch.Tensor):
                result = (
                    torch.from_numpy(output.copy())
                    .permute(0, 3, 1, 2)
                    .float()
                    .div_(255)
                )
            elif isinstance(value, np.ndarray):
                result = output.astype(np.float32)
            else:
                result = [
                    output[index].astype(np.float32) for index in range(frame_count)
                ]
            self._consecutive_failures = 0
            return result
        except Exception as error:
            self._record_failure(error)
            return value
        finally:
            for process in (decoder, encoder):
                if process is not None and process.poll() is None:
                    process.kill()
            for process in (decoder, encoder):
                if process is not None and process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            for worker in (feeder, stderr_reader):
                if worker is not None and worker.is_alive():
                    worker.join(timeout=2)

    def __call__(self, results, fps=25):
        if np.random.uniform() > self.params.get("prob", 1):
            return results

        crf = np.random.uniform(*self.params["crf_range"])
        blur_sigma = np.random.uniform(*self.params["gblur_sigma_range"])
        for key in self.keys:
            results[key] = self.apply_compression(results[key], crf, fps, blur_sigma)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, keys={self.keys})"


class RandomTemporalDegradation:
    """Randomly repeat, drop, or interpolate frames while preserving length."""

    def __init__(self, params, keys):
        self.params = params
        self.keys = keys

    @staticmethod
    def _apply(frames, indices, corruption_type):
        frame_count = frames.shape[0]
        if corruption_type == "repeat":
            for index in indices:
                frames[index] = frames[index - 1]
        elif corruption_type == "drop_pad_last":
            keep_mask = np.ones(frame_count, dtype=bool)
            keep_mask[indices] = False
            frames = frames[keep_mask]
            padding = frames[-1:].expand(frame_count - frames.shape[0], *frames.shape[1:])
            frames = torch.cat((frames, padding), dim=0)
        elif corruption_type == "interpolate":
            kept_indices = np.setdiff1d(np.arange(frame_count), indices)
            for index in indices:
                insert_at = np.searchsorted(kept_indices, index)
                if insert_at == 0:
                    frames[index] = frames[kept_indices[0]]
                elif insert_at == len(kept_indices):
                    frames[index] = frames[kept_indices[-1]]
                else:
                    previous_index = kept_indices[insert_at - 1]
                    next_index = kept_indices[insert_at]
                    weight = (index - previous_index) / float(
                        next_index - previous_index
                    )
                    frames[index] = (
                        (1.0 - weight) * frames[previous_index]
                        + weight * frames[next_index]
                    )
        else:
            raise ValueError(f"Unsupported temporal degradation: {corruption_type}")
        return frames

    def __call__(self, results):
        if np.random.uniform() > self.params.get("prob", 1):
            return results

        frame_count = results[self.keys[0]].shape[0]
        ratio = self.params.get("corruption_ratio", [0.1, 0.3])
        selected_count = int(np.random.uniform(*ratio) * frame_count)
        indices = np.random.choice(
            np.arange(1, frame_count), selected_count, replace=False
        )
        corruption_type = np.random.choice(
            self.params["corruption_type"], p=self.params["corruption_prob"]
        )
        for key in self.keys:
            if key != "gts" or corruption_type != "interpolate":
                results[key] = self._apply(results[key], indices, corruption_type)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, keys={self.keys})"


class RandomBlur:
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def get_kernel(self, num_kernels: int):
        """This is the function to create kernel.

        Args:
            num_kernels (int): the number of kernels

        Returns:
            _type_: _description_
        """
        kernel_type = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_x_range = self.params.get('sigma_x', [0, 0])
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_x_step = self.params.get('sigma_x_step', 0)

        sigma_y_range = self.params.get('sigma_y', [0, 0])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        sigma_y_step = self.params.get('sigma_y_step', 0)

        rotate_angle_range = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0],
                                         rotate_angle_range[1])
        rotate_angle_step = self.params.get('rotate_angle_step', 0)

        beta_gau_range = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get('beta_gaussian_step', 0)

        beta_pla_range = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get('beta_plateau_step', 0)

        omega_range = self.params.get('omega', None)
        omega_step = self.params.get('omega_step', 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3., np.pi]
            else:
                omega_range = [np.pi / 5., np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma_x, sigma_x],
                [sigma_y, sigma_y],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)

            # update kernel parameters
            sigma_x += np.random.uniform(-sigma_x_step, sigma_x_step)
            sigma_y += np.random.uniform(-sigma_y_step, sigma_y_step)
            rotate_angle += np.random.uniform(-rotate_angle_step,
                                              rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma_x = np.clip(sigma_x, sigma_x_range[0], sigma_x_range[1])
            sigma_y = np.clip(sigma_y, sigma_y_range[0], sigma_y_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0],
                                   rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels

    def _apply_random_blur(self, imgs):
        """This is the function to apply blur operation on images.

        Args:
            imgs (Tensor): images

        Returns:
            Tensor: Images applied blur
        """
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernels = self.get_kernel(num_kernels=len(imgs))
        imgs = [
            cv2.filter2D(img, -1, kernel)
            for img, kernel in zip(imgs, kernels)
        ]

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_blur(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomJPEGCompression:
    """Apply random JPEG compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
        bgr2rgb (str): Whether change channel order. Default: False.
    """

    def __init__(self, params, keys, color_type='color', bgr2rgb=False):
        self.keys = keys
        self.params = params
        self.color_type = color_type
        self.bgr2rgb = bgr2rgb

    def _apply_random_compression(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # determine initial compression level and the step size
        quality = self.params['quality']
        quality_step = self.params.get('quality_step', 0)
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))

        # apply jpeg compression
        outputs = []
        for img in imgs:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
            img = np.clip(img, 0, 255).round().astype(np.uint8)
            if self.bgr2rgb and self.color_type == 'color':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.jpg', img, encode_param)

            if self.color_type == 'color':
                img_encoded = cv2.imdecode(img_encoded, 1)
                if self.bgr2rgb:
                    img_encoded = cv2.cvtColor(img_encoded, cv2.COLOR_BGR2RGB)
                outputs.append(img_encoded)
            else:
                outputs.append(cv2.imdecode(img_encoded, 0))

            # update compression level
            jpeg_param += np.random.uniform(-quality_step, quality_step)
            jpeg_param = round(np.clip(jpeg_param, quality[0], quality[1]))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomNoise:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):
        """This is the function used to apply gaussian noise on images.

        Args:
            imgs (Tensor): images

        Returns:
            Tensor: images applied gaussian noise
        """
        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            outputs.append(img + noise)

            # update noise level
            sigma += np.random.uniform(-sigma_step, sigma_step)
            sigma = np.clip(sigma, sigma_range[0], sigma_range[1])

        return outputs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        gray_noise_prob = self.params['poisson_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(img.copy())
            if is_gray_noise:
                noise = cv2.cvtColor(noise[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
                noise = noise[..., np.newaxis]
            noise = np.clip((noise).round(), 0, 255)
            unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val).astype(np.float32) \
                / unique_val - noise

            outputs.append(img + noise * scale)

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs

    def _apply_random_noise(self, imgs):
        """This is the function used to apply random noise on images.

        Args:
            imgs (Tensor): training images

        Returns:
            _type_: _description_
        """
        noise_type = np.random.choice(
            self.params['noise_type'], p=self.params['noise_prob'])

        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        if noise_type.lower() == 'gaussian':
            imgs = self._apply_gaussian_noise(imgs)
        elif noise_type.lower() == 'poisson':
            imgs = self._apply_poisson_noise(imgs)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is '
                                      'not implemented.')

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomResize:
    """Randomly resize the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA,
            lanczos=cv2.INTER_LANCZOS4)

    def _random_resize(self, imgs):
        """This is the function used to randomly resize images for training
        augmentation.

        Args:
            imgs (Tensor): training images.

        Returns:
            Tensor: images after randomly resized
        """
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        h, w = imgs[0].shape[:2]

        resize_opt = self.params['resize_opt']
        resize_prob = self.params['resize_prob']
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt = self.resize_dict[resize_opt]

        resize_step = self.params.get('resize_step', 0)

        # determine the target size, if not provided
        target_size = self.params.get('target_size', None)
        if target_size is None:
            resize_mode = np.random.choice(['up', 'down', 'keep'],
                                           p=self.params['resize_mode_prob'])
            resize_scale = self.params['resize_scale']
            if resize_mode == 'up':
                scale_factor = np.random.uniform(1, resize_scale[1])
            elif resize_mode == 'down':
                scale_factor = np.random.uniform(resize_scale[0], 1)
            else:
                scale_factor = 1

            # determine output size
            h_out, w_out = h * scale_factor, w * scale_factor
            if self.params.get('is_size_even', False):
                h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
            target_size = (int(h_out), int(w_out))
        else:
            resize_step = 0

        # resize the input
        if resize_step == 0:  # same target_size for all input images
            outputs = [
                cv2.resize(img, target_size[::-1], interpolation=resize_opt)
                for img in imgs
            ]
        else:  # different target_size for each input image
            outputs = []
            for img in imgs:
                img = cv2.resize(
                    img, target_size[::-1], interpolation=resize_opt)
                outputs.append(img)

                # update scale
                scale_factor += np.random.uniform(-resize_step, resize_step)
                scale_factor = np.clip(scale_factor, resize_scale[0],
                                       resize_scale[1])

                # determine output size
                h_out, w_out = h * scale_factor, w * scale_factor
                if self.params.get('is_size_even', False):
                    h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
                target_size = (int(h_out), int(w_out))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._random_resize(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str
