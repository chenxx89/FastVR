"""FastVR video enhancement inference runner.

Features:
  - One input path with automatic video, directory, frame-sequence, or JSONL detection
  - Separate model-processing resolution from the final upscale-selected output resolution
  - One output per input, with the same base filename
  - Optional color correction applied before that single output is saved
  - MP4 and/or lossless PNG output, with PNG writes handed to a background pool
  - Resume support: skip already-processed samples
  - Source video FPS or configurable image-sequence FPS, with JSONL overrides
"""

import os
import sys
import torch

from fastvr.models.wan_video_dit import set_dit_const_context
from fastvr.models.download import EMPTY_PROMPT_PATH, ensure_fastvr_checkpoint
from fastvr.inference.color_fix import TorchColorCorrector
from fastvr.inference.inputs import VIDEO_EXTS, collect_samples, load_lq_frames, sample_fps_override
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
from fastvr.inference.pipeline import build_pipeline, load_prompt_embedding
from fastvr.models import LightweightVAE
from fastvr.inference.output import (
    AsyncFrameWriter,
    output_video_path,
    png_frames_dir,
    save_video_via_temp,
)

# ============================================================
# Inference
# ============================================================

def run_single_inference(
    pipe,
    input_path,
    output_path,
    args,
    fps: float | None = None,
    frame_writer = None,
    png_dir_name: str = None,
):
    if args.streaming:
        from fastvr.inference.streaming import run_streaming_inference

        return run_streaming_inference(
            pipe,
            input_path,
            output_path,
            args,
            fps=fps,
            png_dir_name=png_dir_name,
        )

    # All supported input types share one loading path. Alignment happens below.
    (
        input_frames,
        detected_fps,
        input_frame_count,
        input_height,
        input_width,
        frame_names,
    ) = load_lq_frames(input_path)
    output_fps = normalize_fps(fps, default=detected_fps)
    if args.png_name_mode != "source":
        frame_names = None

    plan = build_resolution_plan(
        input_width=input_width,
        input_height=input_height,
        upscale=args.upscale,
        target_short_edge=args.target_short_edge,
        width_factor=pipe.width_division_factor,
        height_factor=pipe.height_division_factor,
    )
    output_width, output_height = plan.output_width, plan.output_height
    processing_width, processing_height = plan.processing_width, plan.processing_height
    model_width, model_height = plan.model_width, plan.model_height
    processing_frames = resize_frames_to_size(
        input_frames, processing_width, processing_height
    )
    log_resolution_plan(plan)

    color_reference_frames = list(processing_frames) if args.color_fix else None
    if color_reference_frames is not None:
        if len(color_reference_frames) != input_frame_count:
            raise RuntimeError(
                "Color reference frame count mismatch before inference: "
                f"expected {input_frame_count}, got {len(color_reference_frames)}."
            )
        mismatched_reference_sizes = [
            (index, frame.size)
            for index, frame in enumerate(color_reference_frames)
            if frame.size != (processing_width, processing_height)
        ]
        if mismatched_reference_sizes:
            raise RuntimeError(
                "Color reference frames are not spatially aligned to "
                f"{processing_width}x{processing_height}: "
                f"{mismatched_reference_sizes[:5]}"
            )

    # Pad the processing frames on the right/bottom for the model only.
    height_factor = pipe.height_division_factor
    width_factor = pipe.width_division_factor
    model_input_frames = processing_frames

    if (model_height, model_width) != (processing_height, processing_width):
        model_input_frames = [
            pad_image_to_size(frame, model_width, model_height)
            for frame in processing_frames
        ]
        print(
            f"[FastVR] Model padding: {processing_width}x{processing_height} -> "
            f"{model_width}x{model_height} ({height_factor}/{width_factor}-aligned)"
        )

    # Align temporal dim UP to 4n+1 via last-frame padding
    t_factor = pipe.time_division_factor
    t_remainder = pipe.time_division_remainder
    if input_frame_count % t_factor != t_remainder:
        target_frame_count = (
            (input_frame_count - t_remainder + t_factor - 1) // t_factor
        ) * t_factor + t_remainder
        if target_frame_count < t_remainder:
            target_frame_count = t_remainder
        if target_frame_count > input_frame_count:
            padding_frame_count = target_frame_count - input_frame_count
            model_input_frames = model_input_frames + [
                model_input_frames[input_frame_count - 1]
            ] * padding_frame_count
            print(
                f"[FastVR] Temporal padding: {input_frame_count} -> "
                f"{len(model_input_frames)} frames (4n+1)"
            )
    model_frame_count = len(model_input_frames)

    print(
        f"[FastVR] Frames: input={input_frame_count}, model={model_frame_count}; "
        f"fps={output_fps}"
    )
    device = pipe.device

    set_dit_const_context(True)
    try:
        with pipe.lightweight_vae.stream_session():
            video_tensor = pipe(
                model_input_frames,
                vsr_target_timestep=args.vsr_target_timestep,
                enable_denoise_tiling=args.enable_denoise_tiling,
                denoise_tile_size=args.denoise_tile_size,
                denoise_tile_stride=args.denoise_tile_stride,
                denoise_tile_batch_size=args.denoise_tile_batch_size,
            )
    finally:
        set_dit_const_context(False)

    def _emit(frames, decoder_label):
        """Write the single selected output in every requested format."""
        if "mp4" in args.output_formats:
            source_audio = audio_source_path(input_path, VIDEO_EXTS)
            save_video_via_temp(
                frames,
                output_path,
                fps=output_fps,
                audio_source_path=source_audio,
            )
            print(
                f"[FastVR] Saved[{decoder_label}]: {output_path} "
                f"(fps={output_fps:.6g})"
            )
        if "png" in args.output_formats and frame_writer is not None:
            frame_writer.submit_frames(
                frames,
                png_frames_dir(output_path, args.png_dir_suffix, png_dir_name),
                decoder_label,
                names=frame_names,
            )

    def _save_decoder_pass(video_tensor, decoder_label):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_tensor = video_tensor
        output_label = decoder_label

        if args.color_fix:
            color_corrector = getattr(pipe, "color_corrector", None)
            if color_corrector is None:
                raise RuntimeError("Color correction is enabled but no color corrector is available.")
            hq_tensor_for_fix = video_tensor.to(device=device)
            lq_tensor_for_fix = pipe.preprocess_video(color_reference_frames).to(
                device=device,
                dtype=hq_tensor_for_fix.dtype,
            )
            validate_video_tensor_shape(
                hq_tensor_for_fix,
                label="Decoded color-fix input",
                expected_frames=input_frame_count,
                expected_height=processing_height,
                expected_width=processing_width,
            )
            validate_video_tensor_shape(
                lq_tensor_for_fix,
                label="LQ color reference",
                expected_frames=input_frame_count,
                expected_height=processing_height,
                expected_width=processing_width,
            )
            if hq_tensor_for_fix.shape != lq_tensor_for_fix.shape:
                raise RuntimeError(
                    "Color-fix tensors must have identical BCTHW shapes: "
                    f"decoded={tuple(hq_tensor_for_fix.shape)}, "
                    f"reference={tuple(lq_tensor_for_fix.shape)}."
                )
            print(
                f"[FastVR] Color fix aligned: frames={input_frame_count}, "
                f"resolution={processing_width}x{processing_height}, "
                f"dtype={hq_tensor_for_fix.dtype}"
            )
            output_tensor = color_corrector(
                hq_tensor_for_fix,
                lq_tensor_for_fix,
                clip_range=(-1, 1),
                chunk_size=16,
                method=args.color_fix,
            )
            validate_video_tensor_shape(
                output_tensor,
                label="Color-corrected output",
                expected_frames=input_frame_count,
                expected_height=processing_height,
                expected_width=processing_width,
            )
            output_label = f"{decoder_label}/{args.color_fix}"
            print(
                f"[FastVR] Color fix applied[{decoder_label}]: "
                f"method={args.color_fix}"
            )

        if (processing_height, processing_width) != (output_height, output_width):
            print(
                f"[FastVR] Output resize: {processing_width}x{processing_height} -> "
                f"{output_width}x{output_height}"
            )
        output_tensor = resize_video_tensor_spatial(
            output_tensor,
            target_height=output_height,
            target_width=output_width,
        )
        validate_video_tensor_shape(
            output_tensor,
            label="Final output",
            expected_frames=input_frame_count,
            expected_height=output_height,
            expected_width=output_width,
        )
        output_video = pipe.vae_output_to_video(output_tensor.float().cpu())
        _emit(output_video, output_label)


    if video_tensor.ndim != 5:
        raise RuntimeError(
            f"LightweightVAE output must be BCTHW, got {tuple(video_tensor.shape)}."
        )
    if video_tensor.shape[2] < input_frame_count:
        raise RuntimeError(
            "LightweightVAE returned fewer frames than the input: "
            f"decoded={video_tensor.shape[2]}, input={input_frame_count}."
        )
    video_tensor = video_tensor[:, :, :input_frame_count].float()
    video_tensor = crop_video_tensor_spatial(
        video_tensor,
        target_height=processing_height,
        target_width=processing_width,
    )
    validate_video_tensor_shape(
        video_tensor,
        label="Decoded video",
        expected_frames=input_frame_count,
        expected_height=processing_height,
        expected_width=processing_width,
    )
    _save_decoder_pass(video_tensor, "lightweight_vae")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Runner
# ============================================================

def validate_unique_output_paths(samples, output_dir):
    """Reject batches whose inputs would publish to the same output file."""
    output_to_inputs = {}
    for sample in samples:
        target = os.path.abspath(output_video_path(sample.path, output_dir))
        output_to_inputs.setdefault(target, []).append(sample.path)
    collisions = {
        target: paths for target, paths in output_to_inputs.items() if len(paths) > 1
    }
    if collisions:
        details = "; ".join(
            f"{target} <- {paths}" for target, paths in sorted(collisions.items())
        )
        raise ValueError(f"Multiple inputs resolve to the same output path: {details}")


def run(args):
    samples, source_desc = collect_samples(args.input)
    print(f"\n[FastVR] Found {len(samples)} sample(s) from {source_desc}\n")
    if not samples:
        print("[FastVR] No valid samples found. Exiting.")
        return

    validate_unique_output_paths(samples, args.output_dir)

    checkpoint_dir = ensure_fastvr_checkpoint(args.ckpt_path)
    args.ckpt_path = checkpoint_dir
    print(f"[FastVR] Checkpoint: {checkpoint_dir}")
    prompt_embedding = load_prompt_embedding(EMPTY_PROMPT_PATH)

    # Fixed embeddings make tokenizer and text encoder unnecessary.
    pipe = build_pipeline(args)

    vae_checkpoint = os.path.join(checkpoint_dir, "vae.safetensors")
    lightweight_vae = LightweightVAE(
        checkpoint_path=vae_checkpoint,
    ).to(pipe.device)
    pipe.lightweight_vae = lightweight_vae

    pipe.set_prompt_embedding(prompt_embedding)

    if args.color_fix:
        pipe.color_corrector = TorchColorCorrector(levels=5)
        print(f"[FastVR] Color correction enabled: method={args.color_fix}")

    frame_writer = None
    if "png" in args.output_formats and not args.streaming:
        frame_writer = AsyncFrameWriter(
            num_workers=args.png_workers,
            max_pending=args.png_max_pending,
        )
    print(
        f"[FastVR] Output Formats: {','.join(sorted(args.output_formats))}"
        + (f" (png: {args.png_workers} writer threads, "
           f"lossless, dir_suffix='{args.png_dir_suffix}')"
           if frame_writer is not None else "")
    )

    rank_tag = f"[rank {args.rank}/{args.world_size}] " if args.world_size > 1 else ""
    indexed_samples = list(enumerate(samples, 1))
    if args.world_size > 1:
        indexed_samples = [
            (idx, sample) for idx, sample in indexed_samples
            if (idx - 1) % args.world_size == args.rank
        ]
        print(
            f"[FastVR] {rank_tag}shard: {len(indexed_samples)}/{len(samples)} sample(s)"
        )
        if not indexed_samples:
            print(f"[FastVR] {rank_tag}empty shard, nothing to do.")
            sys.exit(0)

    # Process each sample
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    for local_idx, (idx, sample) in enumerate(indexed_samples, 1):
        sample_path = sample.path
        sample_fps = sample_fps_override(sample, args.fps)
        sample_name = os.path.splitext(os.path.basename(sample_path.rstrip("/")))[0]

        output_path = output_video_path(sample_path, args.output_dir)
        png_dir_name = os.path.splitext(os.path.basename(output_path))[0]

        # The PNG check targets the directory marker, not the frames, so a run that
        # died mid-sequence is redone rather than skipped.
        requested_outputs = []
        for path in (output_path,):
            if "mp4" in args.output_formats:
                requested_outputs.append(path)
            if "png" in args.output_formats:
                requested_outputs.append(
                    os.path.join(
                        png_frames_dir(path, args.png_dir_suffix, png_dir_name),
                        AsyncFrameWriter.DONE_MARKER,
                    )
                )

        print(f"{'='*60}")
        print(
            f"[FastVR] {rank_tag}[local {local_idx}/{len(indexed_samples)} | "
            f"global {idx}/{len(samples)}] {sample_name} "
            f"(fps={sample_fps if sample_fps is not None else 'source'})"
        )
        print(f"{'='*60}")

        # Existing complete outputs are always skipped.
        if all(os.path.exists(path) for path in requested_outputs):
            print(f"[FastVR] {rank_tag}Skipping (outputs exist): {', '.join(requested_outputs)}")
            skipped_count += 1
            continue

        try:
            run_single_inference(
                pipe,
                sample_path,
                output_path,
                args,
                fps=sample_fps,
                frame_writer=frame_writer,
                png_dir_name=png_dir_name,
            )
            processed_count += 1
        except Exception as e:
            failed_count += 1
            print(f"[FastVR] {rank_tag}ERROR processing {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # The last samples' frames are still in the writer queue; drain before the
    # summary so a failed write is reported instead of silently lost.
    png_errors = []
    if frame_writer is not None:
        print(f"[FastVR] {rank_tag}Waiting for queued PNG writes...")
        png_saved, png_errors = frame_writer.shutdown()
        print(f"[FastVR] {rank_tag}PNG frames written: {png_saved}, failed: {len(png_errors)}")
        for message in png_errors[:10]:
            print(f"[FastVR] {rank_tag}PNG write failed: {message}")

    print(f"\n{'='*60}")
    print(
        f"[FastVR] {rank_tag}Done! Processed: {processed_count}, "
        f"Skipped: {skipped_count}, Shard total: {len(indexed_samples)}"
    )
    print(f"{'='*60}")
    if png_errors or failed_count:
        raise RuntimeError(
            f"Batch inference failed for {failed_count} sample(s) and "
            f"{len(png_errors)} PNG write(s)."
        )
