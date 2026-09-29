"""End-to-end streaming inference for long videos and frame sequences."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from tqdm import tqdm

from fastvr.inference.inputs import (
    VIDEO_EXTS,
    iter_frame_chunks,
    next_4n_plus_1,
    probe_input,
)
from fastvr.inference.utils import (
    audio_source_path,
    build_resolution_plan,
    crop_video_tensor_spatial,
    log_resolution_plan,
    normalize_fps,
    pad_image_to_size,
    resize_frames_to_size,
    resize_video_tensor_spatial,
    validate_video_tensor_shape,
)
from fastvr.inference.output import StreamingPNGWriter, StreamingVideoWriter, png_frames_dir
from fastvr.inference.pipeline import CAUSAL_SIZES

@dataclass(frozen=True)
class _WriterStop:
    commit: bool


class FastVRStreamSession:
    """Own all model-side cache state for one streamed video."""

    def __init__(self, pipe, args):
        self.pipe = pipe
        self.args = args
        self._vae_context = None

    def __enter__(self):
        self.pipe.initialize_fastvr_stream(
            vsr_target_timestep=self.args.vsr_target_timestep,
            enable_denoise_tiling=self.args.enable_denoise_tiling,
            denoise_tile_size=self.args.denoise_tile_size,
            denoise_tile_stride=self.args.denoise_tile_stride,
            denoise_tile_batch_size=self.args.denoise_tile_batch_size,
        )
        try:
            self._vae_context = self.pipe.lightweight_vae.stream_session()
            self._vae_context.__enter__()
        except Exception:
            self.pipe.clear_fastvr_stream()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._vae_context is not None:
                self._vae_context.__exit__(exc_type, exc_value, traceback)
        finally:
            self.pipe.clear_fastvr_stream()


def _next_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def _frames_to_pinned_video(frames):
    import numpy as np

    array = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def _tensor_to_images(video_tensor):
    from PIL import Image

    frames = video_tensor[0].permute(1, 2, 3, 0)
    frames = frames.add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def run_streaming_inference(pipe, input_path, output_path, args, fps=None, png_dir_name=None):
    """Process one input with asynchronous reading and writing around GPU work."""
    if args.color_fix and pipe.color_corrector is None:
        raise RuntimeError(
            "Color correction is enabled but no color corrector is configured."
        )
    info = probe_input(input_path, fallback_fps=fps)
    output_fps = normalize_fps(fps, default=info.fps)
    plan = build_resolution_plan(
        input_width=info.width,
        input_height=info.height,
        upscale=args.upscale,
        target_short_edge=args.target_short_edge,
        width_factor=pipe.width_division_factor,
        height_factor=pipe.height_division_factor,
    )
    output_width, output_height = plan.output_width, plan.output_height
    processing_width, processing_height = plan.processing_width, plan.processing_height
    model_width, model_height = plan.model_width, plan.model_height
    log_resolution_plan(plan)

    first_latents, latent_chunk, _ = CAUSAL_SIZES
    first_input_frames = 4 * first_latents
    input_chunk_frames = 4 * latent_chunk
    print(
        f"[FastVR] Streaming enabled, input and output I/O run asynchronously"
    )

    audio_source = audio_source_path(input_path, VIDEO_EXTS)
    video_writer = (
        StreamingVideoWriter(output_path, output_fps, audio_source)
        if "mp4" in args.output_formats
        else None
    )
    png_writer = (
        StreamingPNGWriter(
            png_frames_dir(output_path, args.png_dir_suffix, png_dir_name)
        )
        if "png" in args.output_formats
        else None
    )

    def prepared_chunks():
        total = 0
        for items, is_last in iter_frame_chunks(
            input_path, first_input_frames, input_chunk_frames
        ):
            source_frames = [frame for frame, _ in items]
            source_names = [name for _, name in items]
            total += len(source_frames)
            processed = resize_frames_to_size(
                source_frames, processing_width, processing_height
            )
            model_frames = list(processed)
            if is_last:
                model_frame_count = next_4n_plus_1(total)
                padding = model_frame_count - total
                if padding:
                    model_frames.extend([model_frames[-1]] * padding)
                    print(
                        f"[FastVR] Streaming temporal padding: {total} -> "
                        f"{model_frame_count} frames (4n+1)"
                    )
            if (model_height, model_width) != (processing_height, processing_width):
                model_frames = [
                    pad_image_to_size(frame, model_width, model_height)
                    for frame in model_frames
                ]
            yield (
                _frames_to_pinned_video(model_frames),
                processed if args.color_fix else None,
                source_names,
                len(source_frames),
                is_last,
            )

    write_queue = queue.Queue(maxsize=2)
    writer_errors = []

    def writer_worker():
        commit = False
        try:
            while True:
                item = write_queue.get()
                if isinstance(item, _WriterStop):
                    commit = item.commit and not writer_errors
                    break
                if writer_errors:
                    continue
                try:
                    cpu_tensor, ready_event, names = item
                    if ready_event is not None:
                        ready_event.synchronize()
                    frames = _tensor_to_images(cpu_tensor)
                    if video_writer is not None:
                        video_writer.append(frames)
                    if png_writer is not None:
                        png_writer.append(frames, names=names)
                except Exception as error:
                    writer_errors.append(error)
        finally:
            try:
                if video_writer is not None:
                    video_writer.close(commit=commit and not writer_errors)
            except Exception as error:
                writer_errors.append(error)
            try:
                if png_writer is not None:
                    png_writer.close(commit=commit and not writer_errors)
            except Exception as error:
                writer_errors.append(error)

    writer_thread = threading.Thread(
        target=writer_worker, name="fastvr-writer", daemon=True
    )
    writer_thread.start()

    reference_frames = []
    reference_names = []
    latent_buffer = None
    input_count = 0
    written_count = 0
    writer_stopped = False
    progress = tqdm(
        total=info.frame_count,
        desc="FastVR streaming",
        unit="frame",
        dynamic_ncols=True,
    )
    reader = iter(prepared_chunks())
    h2d_stream = torch.cuda.Stream(device=pipe.device) if torch.cuda.is_available() else None
    d2h_stream = torch.cuda.Stream(device=pipe.device) if torch.cuda.is_available() else None

    try:
        with FastVRStreamSession(pipe, args), ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fastvr-reader",
        ) as pool:
            pending_read = pool.submit(_next_chunk, reader)
            while True:
                prepared = pending_read.result()
                if prepared is None:
                    break
                pending_read = pool.submit(_next_chunk, reader)
                cpu_model_video, processed, source_names, source_count, is_last = prepared
                input_count += source_count
                if processed is not None:
                    reference_frames.extend(processed)
                reference_names.extend(source_names)

                if h2d_stream is not None:
                    with torch.cuda.stream(h2d_stream):
                        model_video = cpu_model_video.to(
                            device=pipe.device,
                            dtype=pipe.torch_dtype,
                            non_blocking=True,
                        )
                        model_video = model_video.mul(2.0 / 255.0).sub(1.0)
                        h2d_ready = torch.cuda.Event()
                        h2d_ready.record(h2d_stream)
                    torch.cuda.current_stream(pipe.device).wait_event(h2d_ready)
                    model_video.record_stream(torch.cuda.current_stream(pipe.device))
                else:
                    model_video = cpu_model_video.to(
                        device=pipe.device,
                        dtype=pipe.torch_dtype,
                    ).mul(2.0 / 255.0).sub(1.0)
                pipe.lightweight_vae.queue_decode_condition(
                    model_video,
                    target_height=model_height,
                    target_width=model_width,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    encoded = pipe.lightweight_vae.encode_stream(
                        model_video,
                        final=is_last,
                        device=pipe.device,
                    )
                if encoded is not None:
                    latent_buffer = encoded if latent_buffer is None else torch.cat(
                        [latent_buffer, encoded], dim=2
                    )

                while latent_buffer is not None and latent_buffer.shape[2]:
                    expected = (
                        first_latents
                        if pipe._fastvr_stream_state["latent_offset"] == 0
                        else latent_chunk
                    )
                    if latent_buffer.shape[2] < expected and not is_last:
                        break
                    take = min(expected, latent_buffer.shape[2])
                    latent_chunk_tensor = latent_buffer[:, :, :take]
                    latent_buffer = latent_buffer[:, :, take:]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        denoised = pipe.denoise_fastvr_stream_chunk(
                            latent_chunk_tensor
                        )
                    with torch.autocast(device_type="cuda", enabled=False):
                        decoded = pipe.lightweight_vae.decode_stream(
                            denoised,
                            final=is_last and latent_buffer.shape[2] == 0,
                        ).float()

                    remaining = input_count - written_count if is_last else decoded.shape[2]
                    keep = min(decoded.shape[2], remaining)
                    decoded = decoded[:, :, :keep]
                    if keep == 0:
                        continue
                    decoded = crop_video_tensor_spatial(
                        decoded,
                        target_height=processing_height,
                        target_width=processing_width,
                    )
                    chunk_references = reference_frames[:keep] if args.color_fix else None
                    chunk_names = reference_names[:keep]
                    if args.color_fix:
                        del reference_frames[:keep]
                    del reference_names[:keep]

                    validate_video_tensor_shape(
                        decoded,
                        label="Streaming decoded chunk",
                        expected_frames=keep,
                        expected_height=processing_height,
                        expected_width=processing_width,
                    )
                    if args.color_fix:
                        reference = pipe.preprocess_video(chunk_references).to(
                            device=decoded.device, dtype=decoded.dtype
                        )
                        validate_video_tensor_shape(
                            reference,
                            label="Streaming color reference",
                            expected_frames=keep,
                            expected_height=processing_height,
                            expected_width=processing_width,
                        )
                        decoded = pipe.color_corrector(
                            decoded,
                            reference,
                            clip_range=(-1, 1),
                            chunk_size=16,
                            method=args.color_fix,
                        )
                    decoded = resize_video_tensor_spatial(
                        decoded,
                        target_height=output_height,
                        target_width=output_width,
                    )

                    if d2h_stream is not None and decoded.device.type == "cuda":
                        d2h_stream.wait_stream(torch.cuda.current_stream(decoded.device))
                        cpu_tensor = torch.empty_like(
                            decoded, device="cpu", pin_memory=True
                        )
                        with torch.cuda.stream(d2h_stream):
                            cpu_tensor.copy_(decoded, non_blocking=True)
                            ready_event = torch.cuda.Event()
                            ready_event.record(d2h_stream)
                            decoded.record_stream(d2h_stream)
                    else:
                        cpu_tensor = decoded.cpu()
                        ready_event = None
                    names = (
                        chunk_names
                        if args.png_name_mode == "source" and all(chunk_names)
                        else None
                    )
                    write_queue.put((cpu_tensor, ready_event, names))
                    written_count += keep
                    progress.update(keep)
                    if writer_errors:
                        raise RuntimeError(f"Streaming writer failed: {writer_errors[0]}")
        if written_count != input_count:
            raise RuntimeError(
                f"Streaming output frame mismatch: input={input_count}, output={written_count}"
            )
        write_queue.put(_WriterStop(commit=True))
        writer_stopped = True
        writer_thread.join()
        if writer_errors:
            raise RuntimeError(f"Streaming writer failed: {writer_errors[0]}")
    except Exception:
        if not writer_stopped:
            write_queue.put(_WriterStop(commit=False))
            writer_thread.join()
        raise
    finally:
        progress.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    label = "lightweight_vae" + (f"/{args.color_fix}" if args.color_fix else "")
    if video_writer is not None:
        print(f"[FastVR] Saved[{label}]: {output_path} (fps={output_fps:.6g})")
    if png_writer is not None:
        print(f"[FastVR] PNG frames[{label}]: {png_writer.output_dir} ({written_count} files)")
