"""Isolated rollout runner for causal-forcing student keyframe experiments."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from einops import rearrange
from omegaconf import OmegaConf
import torch
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.misc import set_seed

from .boundaries import build_chunk_boundaries, write_chunk_boundaries
from .metrics import MetricAdapter, VideoMetrics, evaluate_video_metrics
from .runner import RolloutResult, infer_num_chunks, load_prompt_set
from .schedules import total_requested_steps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/default_config.yaml"


@dataclass(frozen=True)
class StudentRolloutSpec:
    experiment_name: str
    prompt_id: str
    prompt_text: str
    seed: int
    checkpoint: str
    model_config_path: str
    schedule_name: str
    steps_per_chunk: List[int]
    num_output_frames: int
    output_root: str
    use_ema: bool = False
    fps: int = 16
    dtype: str = "bfloat16"
    suffix_window_latent: int = 32
    latent_to_visible_ratio: int = 4
    save_chunk_boundaries: bool = False


def _as_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _slugify(value: str, max_len: int = 96) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return (normalized or "item")[:max_len]


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {sorted(mapping)}.")
    return mapping[key]


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_student_config(config_path: str):
    config = OmegaConf.load(str(_as_repo_path(config_path)))
    default_config = OmegaConf.load(str(DEFAULT_CONFIG_PATH))
    return OmegaConf.merge(default_config, config)


def _build_rollout_dir(spec: StudentRolloutSpec) -> Path:
    return (
        _as_repo_path(spec.output_root)
        / spec.experiment_name
        / "rollouts"
        / _slugify(spec.prompt_id)
        / str(spec.seed)
        / _slugify(spec.schedule_name)
    )


def _cleanup_student_pipeline(pipeline: Optional[torch.nn.Module]) -> None:
    if pipeline is not None:
        for attr in ("kv_cache1", "crossattn_cache"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, None)
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
            try:
                pipeline.vae.model.clear_cache()
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def load_student_pipeline(
    config_path: str,
    checkpoint_path: str,
    use_ema: bool,
    device: torch.device,
    dtype: str = "bfloat16",
) -> Tuple[CausalInferencePipeline, Any]:
    config = load_student_config(config_path)
    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = torch.load(str(_as_repo_path(checkpoint_path)), map_location="cpu")
    key = "generator_ema" if use_ema else "generator"
    generator_state = state_dict.get(key, state_dict)
    try:
        pipeline.generator.load_state_dict(generator_state)
    except RuntimeError:
        fixed = {}
        for k, v in generator_state.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            if k.startswith("_fsdp_wrapped_module."):
                k = k.replace("_fsdp_wrapped_module.", "", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

    parsed_dtype = _parse_dtype(dtype)
    pipeline = pipeline.to(dtype=parsed_dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline, config


def _validate_student_steps(steps_per_chunk: List[int], max_steps: int) -> List[int]:
    out: List[int] = []
    for i, value in enumerate(steps_per_chunk):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"steps_per_chunk[{i}] must be an integer, got {type(value).__name__}.")
        if value <= 0 or value > max_steps:
            raise ValueError(f"steps_per_chunk[{i}] must be in [1, {max_steps}], got {value}.")
        out.append(value)
    return out


def student_inference_with_step_budget(
    pipeline: CausalInferencePipeline,
    noise: torch.Tensor,
    text_prompts: List[str],
    steps_per_chunk: List[int],
    initial_latent: Optional[torch.Tensor] = None,
    start_frame_index: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    batch_size, num_frames, num_channels, height, width = noise.shape
    if not pipeline.independent_first_frame or (pipeline.independent_first_frame and initial_latent is not None):
        if num_frames % pipeline.num_frame_per_block != 0:
            raise ValueError("num_frames must be divisible by num_frame_per_block.")
        num_blocks = num_frames // pipeline.num_frame_per_block
    else:
        if (num_frames - 1) % pipeline.num_frame_per_block != 0:
            raise ValueError("(num_frames - 1) must be divisible by num_frame_per_block.")
        num_blocks = (num_frames - 1) // pipeline.num_frame_per_block

    all_num_frames = [pipeline.num_frame_per_block] * num_blocks
    if pipeline.independent_first_frame and initial_latent is None:
        all_num_frames = [1] + all_num_frames
    if len(steps_per_chunk) != len(all_num_frames):
        raise ValueError(f"steps_per_chunk length mismatch: expected {len(all_num_frames)}, got {len(steps_per_chunk)}")
    requested_steps_per_chunk = _validate_student_steps(steps_per_chunk, max_steps=len(pipeline.denoising_step_list))

    num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
    num_output_frames = num_frames + num_input_frames
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)
    output = torch.zeros([batch_size, num_output_frames, num_channels, height, width], device=noise.device, dtype=noise.dtype)

    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        pipeline._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
    else:
        for block_index in range(pipeline.num_transformer_blocks):
            pipeline.crossattn_cache[block_index]["is_init"] = False
        for block_index in range(len(pipeline.kv_cache1)):
            pipeline.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
            pipeline.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

    current_start_frame = int(start_frame_index)
    if initial_latent is not None:
        timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
        if pipeline.independent_first_frame:
            if (num_input_frames - 1) % pipeline.num_frame_per_block != 0:
                raise ValueError("(num_input_frames - 1) must be divisible by num_frame_per_block.")
            num_input_blocks = (num_input_frames - 1) // pipeline.num_frame_per_block
            output[:, :1] = initial_latent[:, :1]
            pipeline.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            current_start_frame += 1
        else:
            if num_input_frames % pipeline.num_frame_per_block != 0:
                raise ValueError("num_input_frames must be divisible by num_frame_per_block.")
            num_input_blocks = num_input_frames // pipeline.num_frame_per_block

        for _ in range(num_input_blocks):
            current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + pipeline.num_frame_per_block]
            output[:, current_start_frame:current_start_frame + pipeline.num_frame_per_block] = current_ref_latents
            pipeline.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            current_start_frame += pipeline.num_frame_per_block

    chunk_logs: List[Dict[str, Any]] = []
    for chunk_idx, current_num_frames in enumerate(all_num_frames):
        chunk_start_time = time.perf_counter()
        requested_steps = requested_steps_per_chunk[chunk_idx]
        noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
        active_timesteps = pipeline.denoising_step_list[:requested_steps]
        denoised_pred = noisy_input

        for index, current_timestep in enumerate(active_timesteps):
            timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            if index < len(active_timesteps) - 1:
                next_timestep = active_timesteps[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
        context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )

        chunk_logs.append(
            {
                "chunk_idx": chunk_idx,
                "num_frames": int(current_num_frames),
                "requested_num_steps": int(requested_steps),
                "actual_num_steps": int(len(active_timesteps)),
                "runtime_sec": time.perf_counter() - chunk_start_time,
                "timestep_values": [float(x.item()) for x in active_timesteps],
            }
        )
        current_start_frame += current_num_frames

    video = pipeline.vae.decode_to_pixel(output, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video, output, chunk_logs


def _failure_meta(spec: StudentRolloutSpec, rollout_dir: Path, error: str, trace: str) -> Dict[str, Any]:
    return {
        "experiment_name": spec.experiment_name,
        "prompt_id": spec.prompt_id,
        "prompt_text": spec.prompt_text,
        "seed": spec.seed,
        "backend": "causal_forcing_student",
        "checkpoint": spec.checkpoint,
        "model_config_path": spec.model_config_path,
        "schedule_name": spec.schedule_name,
        "steps_per_chunk": spec.steps_per_chunk,
        "num_chunks": len(spec.steps_per_chunk),
        "status": "failed",
        "error": error,
        "traceback": trace,
        "video_path": str(rollout_dir / "full.mp4"),
        "metrics_path": str(rollout_dir / "metrics.json"),
    }


def run_student_rollout(
    spec: StudentRolloutSpec,
    pipeline: Optional[CausalInferencePipeline] = None,
    loaded_config: Optional[Any] = None,
    device: Optional[torch.device] = None,
    force: bool = False,
    metric_adapters: Optional[Iterable[MetricAdapter]] = None,
    raise_on_error: bool = False,
) -> RolloutResult:
    rollout_dir = _build_rollout_dir(spec)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rollout_meta_path = rollout_dir / "rollout_meta.json"
    metrics_path = rollout_dir / "metrics.json"
    video_path = rollout_dir / "full.mp4"
    boundaries_path = rollout_dir / "chunk_boundaries.json"

    if rollout_meta_path.exists() and not force:
        existing = _read_json(rollout_meta_path)
        if existing.get("status") == "success" and video_path.exists():
            return RolloutResult(
                experiment_name=existing.get("experiment_name", spec.experiment_name),
                prompt_id=existing.get("prompt_id", spec.prompt_id),
                seed=int(existing.get("seed", spec.seed)),
                schedule_name=existing.get("schedule_name", spec.schedule_name),
                status="success",
                video_path=str(video_path),
                metrics_path=str(metrics_path),
                rollout_meta_path=str(rollout_meta_path),
                wall_clock_sec=float(existing.get("wall_clock_sec", 0.0)),
                total_nfe=int(existing.get("total_nfe", 0)),
                score=existing.get("score"),
            )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pipeline is None:
        pipeline, loaded_config = load_student_pipeline(
            config_path=spec.model_config_path,
            checkpoint_path=spec.checkpoint,
            use_ema=spec.use_ema,
            device=device,
            dtype=spec.dtype,
        )
    elif loaded_config is None:
        loaded_config = pipeline.args

    try:
        num_chunks = infer_num_chunks(
            num_output_frames=spec.num_output_frames,
            num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
            independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
        )
        if len(spec.steps_per_chunk) != num_chunks:
            raise ValueError(f"steps_per_chunk length mismatch: expected {num_chunks}, got {len(spec.steps_per_chunk)}")

        set_seed(spec.seed)
        dtype = _parse_dtype(spec.dtype)
        sampled_noise = torch.randn([1, spec.num_output_frames, 16, 60, 104], device=device, dtype=dtype)
        rollout_start = time.perf_counter()
        with torch.inference_mode():
            video, latents, chunk_logs = student_inference_with_step_budget(
                pipeline=pipeline,
                noise=sampled_noise,
                text_prompts=[spec.prompt_text],
                steps_per_chunk=spec.steps_per_chunk,
                start_frame_index=0,
            )
        wall_clock_sec = time.perf_counter() - rollout_start

        out_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        out_video = (255.0 * out_video).clamp(0, 255).to(torch.uint8)
        write_video(str(video_path), out_video[0], fps=spec.fps)
        if hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()

        metric_result: VideoMetrics = evaluate_video_metrics(video_path=video_path, suffix_start_frame=0, adapters=metric_adapters)
        _write_json_atomic(metrics_path, metric_result.values)

        total_nfe = int(sum(int(item.get("actual_num_steps", 0)) for item in chunk_logs))
        requested_total_steps = total_requested_steps(spec.steps_per_chunk)
        step_mismatch_chunks = [
            int(item.get("chunk_idx", -1))
            for item in chunk_logs
            if int(item.get("actual_num_steps", -1)) != int(item.get("requested_num_steps", -2))
        ]
        decoded_total_frames = int(out_video.shape[1])
        boundary_payload = build_chunk_boundaries(
            num_output_frames=spec.num_output_frames,
            num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
            independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
            latent_to_visible_ratio=spec.latent_to_visible_ratio,
            decoded_total_frames=decoded_total_frames,
            fps=spec.fps,
            suffix_window_latent=spec.suffix_window_latent,
        )
        if spec.save_chunk_boundaries:
            write_chunk_boundaries(boundaries_path, boundary_payload)

        rollout_meta: Dict[str, Any] = {
            "experiment_name": spec.experiment_name,
            "prompt_id": spec.prompt_id,
            "prompt_text": spec.prompt_text,
            "seed": spec.seed,
            "backend": "causal_forcing_student",
            "checkpoint": spec.checkpoint,
            "model_config_path": spec.model_config_path,
            "schedule_name": spec.schedule_name,
            "steps_per_chunk": list(spec.steps_per_chunk),
            "num_chunks": num_chunks,
            "fast_steps": min(spec.steps_per_chunk),
            "heavy_steps": max(spec.steps_per_chunk),
            "target_avg_steps": float(sum(spec.steps_per_chunk) / len(spec.steps_per_chunk)),
            "wall_clock_sec": wall_clock_sec,
            "total_nfe": total_nfe,
            "requested_total_steps": requested_total_steps,
            "decoded_total_frames": decoded_total_frames,
            "fps": spec.fps,
            "suffix_window_latent": spec.suffix_window_latent,
            "latent_to_visible_ratio": spec.latent_to_visible_ratio,
            "video_path": str(video_path),
            "metrics_path": str(metrics_path),
            "chunk_boundaries_path": str(boundaries_path),
            "status": "success",
            "score": metric_result.score,
            "chunk_logs": chunk_logs,
            "latent_shape": list(latents.shape),
            "step_mismatch_chunks": step_mismatch_chunks,
            "step_match_ok": len(step_mismatch_chunks) == 0,
            "budget_match_ok": requested_total_steps == total_nfe,
            "chunk_boundaries": boundary_payload,
        }
        _write_json_atomic(rollout_meta_path, rollout_meta)
        _cleanup_student_pipeline(pipeline)
        return RolloutResult(
            experiment_name=spec.experiment_name,
            prompt_id=spec.prompt_id,
            seed=spec.seed,
            schedule_name=spec.schedule_name,
            status="success",
            video_path=str(video_path),
            metrics_path=str(metrics_path),
            rollout_meta_path=str(rollout_meta_path),
            wall_clock_sec=wall_clock_sec,
            total_nfe=total_nfe,
            score=metric_result.score,
            error=None,
        )
    except Exception as exc:
        trace = traceback.format_exc()
        _write_json_atomic(rollout_meta_path, _failure_meta(spec, rollout_dir, str(exc), trace))
        _cleanup_student_pipeline(pipeline)
        if raise_on_error:
            raise
        return RolloutResult(
            experiment_name=spec.experiment_name,
            prompt_id=spec.prompt_id,
            seed=spec.seed,
            schedule_name=spec.schedule_name,
            status="failed",
            video_path=str(video_path),
            metrics_path=str(metrics_path),
            rollout_meta_path=str(rollout_meta_path),
            wall_clock_sec=0.0,
            total_nfe=0,
            score=None,
            error=str(exc),
        )
