"""Core rollout runner for keyframe-budget experiments."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

from einops import rearrange
from omegaconf import OmegaConf
import torch
from torchvision.io import write_video

from pipeline import CausalDiffusionInferencePipeline
from utils.misc import set_seed

from .metrics import MetricAdapter, VideoMetrics, evaluate_video_metrics
from .schedules import total_requested_steps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/default_config.yaml"


@dataclass(frozen=True)
class RolloutSpec:
    experiment_name: str
    prompt_id: str
    prompt_text: str
    seed: int
    checkpoint: str
    config_path: str
    schedule_name: str
    steps_per_chunk: List[int]
    num_output_frames: int
    output_root: str
    use_ema: bool = False
    fps: int = 16
    dtype: str = "bfloat16"
    strict_step_match: bool = True
    debug_cache_logs: bool = False
    start_frame_index: int = 0


@dataclass(frozen=True)
class RolloutResult:
    experiment_name: str
    prompt_id: str
    seed: int
    schedule_name: str
    status: str
    video_path: str
    metrics_path: str
    rollout_meta_path: str
    wall_clock_sec: float
    total_nfe: int
    score: Optional[float] = None
    error: Optional[str] = None


def _as_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _slugify(value: str, max_len: int = 96) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = "item"
    return normalized[:max_len]


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
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_rollout_dir(spec: RolloutSpec) -> Path:
    root = _as_repo_path(spec.output_root)
    return (
        root
        / spec.experiment_name
        / "rollouts"
        / _slugify(spec.prompt_id)
        / str(spec.seed)
        / _slugify(spec.schedule_name)
    )


def infer_num_chunks(
    num_output_frames: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
) -> int:
    if independent_first_frame:
        if (num_output_frames - 1) % num_frame_per_block != 0:
            raise ValueError(
                "(num_output_frames - 1) must be divisible by num_frame_per_block "
                f"when independent_first_frame=true. Got frames={num_output_frames}, "
                f"num_frame_per_block={num_frame_per_block}."
            )
        return 1 + (num_output_frames - 1) // num_frame_per_block
    if num_output_frames % num_frame_per_block != 0:
        raise ValueError(
            "num_output_frames must be divisible by num_frame_per_block. "
            f"Got frames={num_output_frames}, num_frame_per_block={num_frame_per_block}."
        )
    return num_output_frames // num_frame_per_block


def load_prompt_set(prompt_set_path: str | Path, max_prompts: Optional[int] = None) -> List[Dict[str, str]]:
    path = _as_repo_path(prompt_set_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    prompts: List[Dict[str, str]] = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, str):
                prompts.append({"prompt_id": f"prompt_{idx:04d}", "prompt_text": item})
            elif isinstance(item, dict):
                prompt_id = item.get("prompt_id", f"prompt_{idx:04d}")
                prompt_text = item.get("prompt_text", "")
                if not prompt_text:
                    raise ValueError(f"Prompt item {idx} missing non-empty prompt_text.")
                prompts.append({"prompt_id": str(prompt_id), "prompt_text": str(prompt_text)})
            else:
                raise TypeError(f"Unsupported prompt payload type at index {idx}: {type(item).__name__}")
    else:
        raise TypeError("Prompt set must be a JSON list of strings or objects.")

    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError(f"max_prompts must be > 0, got {max_prompts}")
        if max_prompts > len(prompts):
            raise ValueError(
                f"Requested max_prompts={max_prompts}, but prompt set only has {len(prompts)} entries."
            )
        prompts = prompts[:max_prompts]
    return prompts


def load_pipeline(
    config_path: str,
    checkpoint_path: str,
    use_ema: bool,
    device: torch.device,
    dtype: str = "bfloat16",
    strict_step_match: bool = True,
    debug_cache_logs: bool = False,
) -> Tuple[CausalDiffusionInferencePipeline, Any]:
    config = load_merged_config(config_path)
    config.enforce_chunk_step_match = strict_step_match
    config.debug_cache_logs = debug_cache_logs

    pipeline = CausalDiffusionInferencePipeline(config, device=device)
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
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

    parsed_dtype = _parse_dtype(dtype)
    pipeline = pipeline.to(dtype=parsed_dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    if hasattr(pipeline, "vae"):
        pipeline.vae.to(device=device)
    return pipeline, config


def load_merged_config(config_path: str):
    config = OmegaConf.load(str(_as_repo_path(config_path)))
    default_config = OmegaConf.load(str(DEFAULT_CONFIG_PATH))
    return OmegaConf.merge(default_config, config)


def _cleanup_pipeline_memory(pipeline: Optional[CausalDiffusionInferencePipeline]) -> None:
    if pipeline is not None:
        for attr in ("kv_cache_pos", "kv_cache_neg", "crossattn_cache_pos", "crossattn_cache_neg"):
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


def _create_rollout_failure_meta(
    spec: RolloutSpec,
    rollout_dir: Path,
    error: str,
    trace: str,
) -> Dict[str, Any]:
    return {
        "experiment_name": spec.experiment_name,
        "prompt_id": spec.prompt_id,
        "prompt_text": spec.prompt_text,
        "seed": spec.seed,
        "checkpoint": spec.checkpoint,
        "config_path": spec.config_path,
        "schedule_name": spec.schedule_name,
        "steps_per_chunk": spec.steps_per_chunk,
        "num_chunks": len(spec.steps_per_chunk),
        "wall_clock_sec": 0.0,
        "total_nfe": 0,
        "video_path": str(rollout_dir / "video.mp4"),
        "thumbnail_dir": str(rollout_dir / "thumbs"),
        "metrics_path": str(rollout_dir / "metrics.json"),
        "status": "failed",
        "error": error,
        "traceback": trace,
    }


def run_rollout(
    spec: RolloutSpec,
    pipeline: Optional[CausalDiffusionInferencePipeline] = None,
    loaded_config: Optional[Any] = None,
    device: Optional[torch.device] = None,
    force: bool = False,
    metric_adapters: Optional[Iterable[MetricAdapter]] = None,
    raise_on_error: bool = False,
) -> RolloutResult:
    """
    Run a single deterministic rollout under a fixed prompt/seed/schedule.

    Invariants enforced by design:
    - one checkpoint per rollout,
    - one prompt per rollout,
    - one seed per rollout,
    - only steps_per_chunk changes across schedule variants.
    """
    rollout_dir = _build_rollout_dir(spec)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rollout_meta_path = rollout_dir / "rollout_meta.json"
    metrics_path = rollout_dir / "metrics.json"
    video_path = rollout_dir / "video.mp4"

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
                error=None,
            )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pipeline is None:
        pipeline, loaded_config = load_pipeline(
            config_path=spec.config_path,
            checkpoint_path=spec.checkpoint,
            use_ema=spec.use_ema,
            device=device,
            dtype=spec.dtype,
            strict_step_match=spec.strict_step_match,
            debug_cache_logs=spec.debug_cache_logs,
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
            raise ValueError(
                f"steps_per_chunk length mismatch for rollout: "
                f"expected {num_chunks}, got {len(spec.steps_per_chunk)}"
            )

        set_seed(spec.seed)
        dtype = _parse_dtype(spec.dtype)
        sampled_noise = torch.randn(
            [1, spec.num_output_frames, 16, 60, 104],
            device=device,
            dtype=dtype,
        )
        rollout_start = time.perf_counter()
        with torch.inference_mode():
            video, latents, chunk_logs = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[spec.prompt_text],
                return_latents=True,
                start_frame_index=spec.start_frame_index,
                steps_per_chunk=spec.steps_per_chunk,
                return_chunk_logs=True,
            )
        wall_clock_sec = time.perf_counter() - rollout_start

        out_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        out_video = (255.0 * out_video).clamp(0, 255).to(torch.uint8)
        write_video(str(video_path), out_video[0], fps=spec.fps)

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()

        metric_result: VideoMetrics = evaluate_video_metrics(
            video_path=video_path,
            suffix_start_frame=0,
            adapters=metric_adapters,
        )
        _write_json_atomic(metrics_path, metric_result.values)

        total_nfe = int(sum(int(item.get("actual_num_steps", 0)) for item in chunk_logs))
        step_mismatch_chunks = [
            int(item.get("chunk_idx", -1))
            for item in chunk_logs
            if int(item.get("actual_num_steps", -1)) != int(item.get("requested_num_steps", -2))
        ]
        rollout_meta: Dict[str, Any] = {
            "experiment_name": spec.experiment_name,
            "prompt_id": spec.prompt_id,
            "prompt_text": spec.prompt_text,
            "seed": spec.seed,
            "checkpoint": spec.checkpoint,
            "config_path": spec.config_path,
            "schedule_name": spec.schedule_name,
            "steps_per_chunk": list(spec.steps_per_chunk),
            "num_chunks": num_chunks,
            "fast_steps": min(spec.steps_per_chunk),
            "heavy_steps": max(spec.steps_per_chunk),
            "target_avg_steps": float(sum(spec.steps_per_chunk) / len(spec.steps_per_chunk)),
            "wall_clock_sec": wall_clock_sec,
            "total_nfe": total_nfe,
            "requested_total_steps": total_requested_steps(spec.steps_per_chunk),
            "video_path": str(video_path),
            "thumbnail_dir": str(rollout_dir / "thumbs"),
            "metrics_path": str(metrics_path),
            "status": "success",
            "score": metric_result.score,
            "chunk_logs": chunk_logs,
            "latent_shape": list(latents.shape),
            "step_mismatch_chunks": step_mismatch_chunks,
            "step_match_ok": len(step_mismatch_chunks) == 0,
        }
        _write_json_atomic(rollout_meta_path, rollout_meta)
        _cleanup_pipeline_memory(pipeline)
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
        error = str(exc)
        trace = traceback.format_exc()
        failure_meta = _create_rollout_failure_meta(
            spec=spec,
            rollout_dir=rollout_dir,
            error=error,
            trace=trace,
        )
        _write_json_atomic(rollout_meta_path, failure_meta)
        _cleanup_pipeline_memory(pipeline)
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
            error=error,
        )
