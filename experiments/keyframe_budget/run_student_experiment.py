"""CLI entrypoint for isolated causal-forcing student keyframe experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from omegaconf import OmegaConf
import torch

from .aggregate import (
    aggregate_prompt_seed_results,
    append_gain_map_jsonl,
    append_policy_results_jsonl,
    save_prompt_seed_aggregate,
)
from .oracle import build_oracle_from_single_heavy_results
from .runner import RolloutResult, infer_num_chunks, load_prompt_set
from .schedules import all_fast, all_heavy, single_heavy_at_i
from .student_runner import StudentRolloutSpec, load_student_config, run_student_rollout


def _parse_optional_indices(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [int(item) for item in value]
    raise TypeError(f"single_heavy_indices must be a comma string or sequence, got {type(value).__name__}.")


def _validate_expected_checkpoint(checkpoint: str, expected_basename: str) -> None:
    if expected_basename and os.path.basename(str(checkpoint)) != expected_basename:
        raise ValueError(
            "Checkpoint guard failed: "
            f"expected basename '{expected_basename}', got '{os.path.basename(str(checkpoint))}' from '{checkpoint}'."
        )


def _reset_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _rollout_for_schedules(
    base_spec: StudentRolloutSpec,
    schedules: Mapping[str, Sequence[int]],
    force: bool,
    device: torch.device,
) -> Dict[str, RolloutResult]:
    out: Dict[str, RolloutResult] = {}
    for schedule_name, schedule in schedules.items():
        spec = StudentRolloutSpec(**{**base_spec.__dict__, "schedule_name": schedule_name, "steps_per_chunk": list(schedule)})
        out[schedule_name] = run_student_rollout(spec, force=force, device=device)
    return out


def _exhaustive_single_heavy_sweep(
    base_spec: StudentRolloutSpec,
    fast_steps: int,
    heavy_steps: int,
    single_heavy_indices: Optional[Sequence[int]],
    force: bool,
    device: torch.device,
) -> Dict[str, RolloutResult]:
    total_chunks = len(base_spec.steps_per_chunk)
    indices = list(range(total_chunks)) if single_heavy_indices is None else []
    if single_heavy_indices is not None:
        seen = set()
        for idx in single_heavy_indices:
            idx = int(idx)
            if idx in seen:
                continue
            if idx < 0 or idx >= total_chunks:
                raise IndexError(f"single_heavy index out of range: {idx} for total_chunks={total_chunks}")
            indices.append(idx)
            seen.add(idx)

    schedules: Dict[str, List[int]] = {
        "all_fast": all_fast(total_chunks, fast_steps),
        "all_heavy": all_heavy(total_chunks, heavy_steps),
    }
    for idx in indices:
        schedules[f"single_heavy_{idx:02d}"] = single_heavy_at_i(total_chunks, fast_steps, heavy_steps, idx)
    return _rollout_for_schedules(base_spec=base_spec, schedules=schedules, force=force, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    exp_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    experiment_name = str(exp_cfg["experiment_name"])
    output_root = str(exp_cfg.get("output_root", "outputs/causal_forcing_student_long_keyframe_budget"))
    model_config_path = str(exp_cfg["model_config_path"])
    checkpoint = str(exp_cfg["checkpoint"])
    _validate_expected_checkpoint(checkpoint, str(exp_cfg.get("expected_checkpoint_basename", "causal_forcing.pt")))

    fast_steps = int(exp_cfg["fast_steps"])
    heavy_steps = int(exp_cfg["heavy_steps"])
    if fast_steps <= 0 or heavy_steps <= 0 or heavy_steps <= fast_steps:
        raise ValueError(f"Expected 0 < fast_steps < heavy_steps, got fast={fast_steps}, heavy={heavy_steps}.")
    num_output_frames = int(exp_cfg.get("num_output_frames", 120))
    fps = int(exp_cfg.get("fps", 16))
    seeds = [int(x) for x in exp_cfg["seeds"]]
    use_ema = bool(exp_cfg.get("use_ema", False))
    dtype = str(exp_cfg.get("dtype", "bfloat16"))
    suffix_window_latent = int(exp_cfg.get("suffix_window_latent", 32))
    latent_to_visible_ratio = int(exp_cfg.get("latent_to_visible_ratio", 4))
    save_chunk_boundaries = bool(exp_cfg.get("save_chunk_boundaries", True))
    single_heavy_indices = _parse_optional_indices(exp_cfg.get("single_heavy_indices"))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    prompt_set = load_prompt_set(exp_cfg["prompt_set"], max_prompts=exp_cfg.get("num_prompts"))
    loaded_config = load_student_config(model_config_path)
    total_chunks = infer_num_chunks(
        num_output_frames=num_output_frames,
        num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
        independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
    )
    max_student_steps = len(getattr(loaded_config, "denoising_step_list"))
    if heavy_steps > max_student_steps:
        raise ValueError(f"heavy_steps={heavy_steps} exceeds student denoising_step_list length={max_student_steps}.")

    aggregate_root = Path(output_root) / experiment_name / "aggregates"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    _reset_if_exists(aggregate_root / "gain_maps.jsonl")
    _reset_if_exists(aggregate_root / "policy_results.jsonl")

    processed_pairs = 0
    success_pairs = 0
    for prompt in prompt_set:
        prompt_id = prompt["prompt_id"]
        prompt_text = prompt["prompt_text"]
        for seed in seeds:
            processed_pairs += 1
            base_spec = StudentRolloutSpec(
                experiment_name=experiment_name,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                seed=seed,
                checkpoint=checkpoint,
                model_config_path=model_config_path,
                schedule_name="all_fast",
                steps_per_chunk=all_fast(total_chunks, fast_steps),
                num_output_frames=num_output_frames,
                output_root=output_root,
                use_ema=use_ema,
                fps=fps,
                dtype=dtype,
                suffix_window_latent=suffix_window_latent,
                latent_to_visible_ratio=latent_to_visible_ratio,
                save_chunk_boundaries=save_chunk_boundaries,
            )
            rollout_results = _exhaustive_single_heavy_sweep(
                base_spec=base_spec,
                fast_steps=fast_steps,
                heavy_steps=heavy_steps,
                single_heavy_indices=single_heavy_indices,
                force=args.force,
                device=device,
            )
            fast_score = rollout_results.get("all_fast").score if "all_fast" in rollout_results else None
            heavy_score = rollout_results.get("all_heavy").score if "all_heavy" in rollout_results else None
            if fast_score is None or heavy_score is None:
                print(f"[student-keyframe] Skipping prompt_id={prompt_id}, seed={seed}; baseline score missing.")
                continue
            oracle_records = build_oracle_from_single_heavy_results(rollout_results)
            aggregate_payload = aggregate_prompt_seed_results(
                experiment_name=experiment_name,
                prompt_id=prompt_id,
                seed=seed,
                rollout_results=rollout_results,
                oracle_gain_records=oracle_records,
                suffix_scores={},
            )
            save_prompt_seed_aggregate(aggregate_root, aggregate_payload)
            if oracle_records:
                append_gain_map_jsonl(aggregate_root, experiment_name, prompt_id, seed, oracle_records)
            append_policy_results_jsonl(aggregate_root, aggregate_payload)
            success_pairs += 1

    summary = {
        "experiment_name": experiment_name,
        "checkpoint": checkpoint,
        "model_config_path": model_config_path,
        "processed_prompt_seed_pairs": processed_pairs,
        "successful_prompt_seed_pairs": success_pairs,
        "num_output_frames": num_output_frames,
        "num_chunks": total_chunks,
        "fast_steps": fast_steps,
        "heavy_steps": heavy_steps,
        "single_heavy_indices": single_heavy_indices,
    }
    (aggregate_root / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[student-keyframe] wrote: {aggregate_root}")


if __name__ == "__main__":
    main()
