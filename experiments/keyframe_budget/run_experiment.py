"""CLI entrypoint for keyframe-budget experiments (including long-rollout stages)."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import OmegaConf
import torch

from .aggregate import (
    aggregate_prompt_seed_results,
    append_gain_map_jsonl,
    append_policy_results_jsonl,
    load_oracle_source,
    save_prompt_seed_aggregate,
)
from .oracle import GainRecord, build_oracle_from_single_heavy_results, exhaustive_single_heavy_sweep
from .runner import (
    RolloutResult,
    RolloutSpec,
    infer_num_chunks,
    load_merged_config,
    load_prompt_set,
    run_rollout,
)
from .schedules import (
    all_fast,
    all_heavy,
    assert_equal_total_steps,
    compute_mixed_heavy_count,
    oracle_top_m,
    prefix_top_m,
    random_top_m,
    uniform_top_m,
)
from .visualize import (
    generate_best_vs_median_gain_plot,
    generate_gain_heatmap,
    generate_oracle_position_histogram,
    generate_recovery_barplot,
    generate_suffix_gain_curves,
)


def _stable_prompt_hash(value: str) -> int:
    total = 0
    for i, ch in enumerate(value):
        total += (i + 1) * ord(ch)
    return total


def _reset_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _stage_gate_dir(output_root: str, experiment_name: str) -> Path:
    return Path(output_root) / experiment_name / "stage_gate"


def _write_stage_gate_artifact(
    output_root: str,
    experiment_name: str,
    stage_name: str,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    gate_dir = _stage_gate_dir(output_root, experiment_name)
    gate_dir.mkdir(parents=True, exist_ok=True)
    path = gate_dir / f"{stage_name}.json"
    content = {"stage_name": stage_name, "status": status, **dict(payload)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=True, sort_keys=False)


def _validate_stage_requirements(
    stage_name: str,
    run_exhaustive: bool,
    run_policy: bool,
    run_baseline_only: bool,
    prompt_count: int,
    seed_count: int,
) -> None:
    if not stage_name:
        return
    stage = stage_name.lower()
    if stage == "stage0":
        if not run_baseline_only:
            raise ValueError("stage0 requires run_baseline_only=true for quick backend sanity checks.")
    elif stage in {"stage1", "stage2"}:
        if not run_exhaustive:
            raise ValueError(f"{stage_name} requires run_exhaustive_single_heavy=true.")
        if prompt_count != 1 or seed_count != 1:
            raise ValueError(f"{stage_name} requires exactly one prompt and one seed.")
    elif stage == "stage3":
        if not run_exhaustive:
            raise ValueError("stage3 requires run_exhaustive_single_heavy=true.")
        if run_policy:
            raise ValueError("stage3 discovery should disable run_policy_comparison.")


def _enforce_stage_success_criteria(
    stage_name: str,
    aggregate_root: Path,
    run_exhaustive: bool,
    run_policy: bool,
) -> None:
    if not stage_name:
        return
    gain_map_path = aggregate_root / "gain_maps.jsonl"
    policy_path = aggregate_root / "policy_results.jsonl"

    if run_exhaustive and (not gain_map_path.exists() or gain_map_path.stat().st_size == 0):
        raise RuntimeError(
            f"Stage {stage_name} failed success criteria: missing/non-empty gain_maps.jsonl at {gain_map_path}"
        )
    if run_policy and (not policy_path.exists() or policy_path.stat().st_size == 0):
        raise RuntimeError(
            f"Stage {stage_name} failed success criteria: missing/non-empty policy_results.jsonl at {policy_path}"
        )


def _build_policy_schedules(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    target_avg_steps: int,
    random_seed: int,
    oracle_ranked_indices: Optional[Sequence[int]] = None,
    include_baselines: bool = True,
) -> Dict[str, List[int]]:
    m = compute_mixed_heavy_count(
        total_chunks=total_chunks,
        fast_steps=fast_steps,
        heavy_steps=heavy_steps,
        target_avg_steps=target_avg_steps,
    )

    schedules: Dict[str, List[int]] = {
        "uniform_top_m": uniform_top_m(total_chunks, fast_steps, heavy_steps, m),
        "random_top_m": random_top_m(total_chunks, fast_steps, heavy_steps, m, seed=random_seed),
        "prefix_top_m": prefix_top_m(total_chunks, fast_steps, heavy_steps, m),
    }
    if include_baselines:
        schedules = {
            "all_fast": all_fast(total_chunks, fast_steps),
            "all_heavy": all_heavy(total_chunks, heavy_steps),
            **schedules,
        }

    if oracle_ranked_indices:
        schedules["oracle_top_m"] = oracle_top_m(
            total_chunks=total_chunks,
            fast_steps=fast_steps,
            heavy_steps=heavy_steps,
            m=m,
            ranked_chunk_indices=oracle_ranked_indices,
        )

    mixed_only = {
        key: value
        for key, value in schedules.items()
        if key in {"uniform_top_m", "random_top_m", "prefix_top_m", "oracle_top_m"}
    }
    if mixed_only:
        assert_equal_total_steps(mixed_only)
    return schedules


def _rollout_for_schedules(
    base_spec: RolloutSpec,
    schedules: Mapping[str, Sequence[int]],
    force: bool,
    device: torch.device,
) -> Dict[str, RolloutResult]:
    out: Dict[str, RolloutResult] = {}
    for schedule_name, schedule in schedules.items():
        spec = replace(base_spec, schedule_name=schedule_name, steps_per_chunk=list(schedule))
        out[schedule_name] = run_rollout(spec, device=device, force=force)
    return out


def _suffix_payload_from_oracle(prompt_id: str, seed: int, records: Sequence[GainRecord]) -> Dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "chunk_indices": [r.chunk_idx for r in records],
        "suffix_gains": [r.gain for r in records],
    }


def _make_rollout_spec(base_kwargs: Mapping[str, Any]) -> RolloutSpec:
    """
    Forward/backward compatible RolloutSpec construction.
    This lets run_experiment pass long-rollout metadata fields when available,
    while still running against older RolloutSpec schemas.
    """
    allowed = {f.name for f in fields(RolloutSpec)}
    filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
    return RolloutSpec(**filtered)


def _parse_optional_indices(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in raw]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [int(item) for item in value]
    raise TypeError(f"single_heavy_indices must be a comma string or sequence, got {type(value).__name__}.")


def _validate_expected_checkpoint(checkpoint: str, expected_basename: str) -> None:
    if not expected_basename:
        return
    actual = os.path.basename(str(checkpoint))
    if actual != expected_basename:
        raise ValueError(
            "Checkpoint guard failed: "
            f"expected basename '{expected_basename}', got '{actual}' from '{checkpoint}'."
        )


def _validate_forbidden_backend(backend: str, forbidden_backend: str) -> None:
    if not forbidden_backend:
        return
    if backend.lower() == forbidden_backend.lower():
        raise ValueError(f"Backend guard failed: backend '{backend}' is forbidden for this experiment.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Experiment config YAML path.")
    parser.add_argument("--force", action="store_true", help="Rerun rollouts even if already successful.")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    exp_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)

    experiment_name = str(exp_cfg["experiment_name"])
    output_root = str(exp_cfg.get("output_root", "outputs/keyframe_budget"))
    config_path = str(exp_cfg.get("model_config_path", "configs/ar_diffusion_tf_chunkwise.yaml"))
    checkpoint = str(exp_cfg["checkpoint"])
    backend = str(exp_cfg.get("backend", "default"))
    fast_steps = int(exp_cfg["fast_steps"])
    heavy_steps = int(exp_cfg["heavy_steps"])
    target_avg_steps = int(exp_cfg["target_avg_steps"])
    num_output_frames = int(exp_cfg.get("num_output_frames", 21))
    fps = int(exp_cfg.get("fps", 16))
    suffix_window_latent = int(exp_cfg.get("suffix_window_latent", 32))
    latent_to_visible_ratio = int(exp_cfg.get("latent_to_visible_ratio", 4))
    save_chunk_boundaries = bool(exp_cfg.get("save_chunk_boundaries", False))
    strict_step_match = bool(exp_cfg.get("strict_step_match", True))
    enforce_budget_parity = bool(exp_cfg.get("enforce_budget_parity", False))
    deterministic_algorithms = bool(exp_cfg.get("deterministic_algorithms", False))
    debug_cache_logs = bool(exp_cfg.get("debug_cache_logs", False))
    seeds = [int(x) for x in exp_cfg["seeds"]]
    run_exhaustive = bool(exp_cfg.get("run_exhaustive_single_heavy", True))
    run_policy = bool(exp_cfg.get("run_policy_comparison", True))
    run_baseline_only = bool(exp_cfg.get("run_baseline_only", False))
    use_ema = bool(exp_cfg.get("use_ema", False))
    dtype = str(exp_cfg.get("dtype", "bfloat16"))
    expected_checkpoint_basename = str(exp_cfg.get("expected_checkpoint_basename", ""))
    forbidden_backend = str(exp_cfg.get("forbidden_backend", ""))
    single_heavy_indices = _parse_optional_indices(exp_cfg.get("single_heavy_indices"))
    stage_name = str(exp_cfg.get("stage_name", ""))
    require_stage_gate = bool(exp_cfg.get("require_stage_gate", False))
    stage_prerequisites = [str(x) for x in exp_cfg.get("stage_prerequisites", [])]

    if run_baseline_only:
        run_exhaustive = False
        run_policy = False
    if not run_exhaustive and not run_policy and not run_baseline_only:
        raise ValueError(
            "At least one of run_exhaustive_single_heavy or run_policy_comparison must be true "
            "unless run_baseline_only=true."
        )
    # Budget parity checks are enforced on mixed policies when schedules are built.
    # Baselines (all_fast/all_heavy) are intentionally allowed to differ.
    _ = enforce_budget_parity
    _validate_expected_checkpoint(checkpoint=checkpoint, expected_basename=expected_checkpoint_basename)
    _validate_forbidden_backend(backend=backend, forbidden_backend=forbidden_backend)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)

    prompt_set = load_prompt_set(exp_cfg["prompt_set"], max_prompts=exp_cfg.get("num_prompts"))
    _validate_stage_requirements(
        stage_name=stage_name,
        run_exhaustive=run_exhaustive,
        run_policy=run_policy,
        run_baseline_only=run_baseline_only,
        prompt_count=len(prompt_set),
        seed_count=len(seeds),
    )

    if require_stage_gate and stage_prerequisites:
        for prereq in stage_prerequisites:
            prereq_path = _stage_gate_dir(output_root, experiment_name) / f"{prereq}.json"
            if not prereq_path.exists():
                raise RuntimeError(f"Missing stage prerequisite gate artifact: {prereq_path}")
            with prereq_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("status") != "success":
                raise RuntimeError(f"Stage prerequisite not successful: {prereq_path}")

    loaded_config = load_merged_config(config_path)
    total_chunks = infer_num_chunks(
        num_output_frames=num_output_frames,
        num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
        independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
    )
    default_steps = all_fast(total_chunks=total_chunks, fast_steps=fast_steps)

    oracle_source_mapping: Dict[str, List[int]] = {}
    oracle_source = exp_cfg.get("oracle_source")
    if oracle_source:
        oracle_source_mapping = load_oracle_source(output_root=output_root, oracle_source_experiment=oracle_source)

    suffix_payloads: List[Dict[str, object]] = []
    aggregate_root = Path(output_root) / experiment_name / "aggregates"
    plot_root = Path(output_root) / experiment_name / "plots"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    _reset_if_exists(aggregate_root / "gain_maps.jsonl")
    _reset_if_exists(aggregate_root / "policy_results.jsonl")

    processed_pairs = 0
    success_pairs = 0
    try:
        for prompt in prompt_set:
            prompt_id = prompt["prompt_id"]
            prompt_text = prompt["prompt_text"]
            for seed in seeds:
                processed_pairs += 1
                spec_kwargs: Dict[str, Any] = {
                    "experiment_name": experiment_name,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "config_path": config_path,
                    "model_config_path": config_path,  # for newer RolloutSpec schemas
                    "backend": backend,
                    "schedule_name": "all_fast",
                    "steps_per_chunk": default_steps,
                    "num_output_frames": num_output_frames,
                    "output_root": output_root,
                    "use_ema": use_ema,
                    "fps": fps,
                    "dtype": dtype,
                    "strict_step_match": strict_step_match,
                    "debug_cache_logs": debug_cache_logs,
                    "suffix_window_latent": suffix_window_latent,
                    "latent_to_visible_ratio": latent_to_visible_ratio,
                    "save_chunk_boundaries": save_chunk_boundaries,
                }
                base_spec = _make_rollout_spec(spec_kwargs)

                rollout_results: Dict[str, RolloutResult] = {}
                oracle_records: List[GainRecord] = []

                if run_baseline_only:
                    schedules = {
                        "all_fast": all_fast(total_chunks=total_chunks, fast_steps=fast_steps),
                        "all_heavy": all_heavy(total_chunks=total_chunks, heavy_steps=heavy_steps),
                    }
                    rollout_results = _rollout_for_schedules(
                        base_spec=base_spec,
                        schedules=schedules,
                        force=args.force,
                        device=device,
                    )
                elif run_exhaustive:
                    rollout_results = exhaustive_single_heavy_sweep(
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
                        print(
                            f"[keyframe-budget] Skipping prompt_id={prompt_id}, seed={seed} "
                            "because baseline rollout did not produce scores "
                            "(likely upstream rollout failure such as OOM)."
                        )
                        continue
                    oracle_records = build_oracle_from_single_heavy_results(rollout_results)
                else:
                    key = f"{prompt_id}::{seed}"
                    oracle_ranked = oracle_source_mapping.get(key)
                    schedules = _build_policy_schedules(
                        total_chunks=total_chunks,
                        fast_steps=fast_steps,
                        heavy_steps=heavy_steps,
                        target_avg_steps=target_avg_steps,
                        random_seed=seed + _stable_prompt_hash(prompt_id) % 100000,
                        oracle_ranked_indices=oracle_ranked,
                        include_baselines=True,
                    )
                    rollout_results = _rollout_for_schedules(
                        base_spec=base_spec,
                        schedules=schedules,
                        force=args.force,
                        device=device,
                    )
                    if "all_fast" in rollout_results and "all_heavy" in rollout_results and oracle_ranked:
                        oracle_records = [
                            GainRecord(chunk_idx=chunk_idx, gain=0.0, gain_norm=0.0, rank=rank)
                            for rank, chunk_idx in enumerate(oracle_ranked)
                        ]

                if run_policy:
                    ranked = [row.chunk_idx for row in oracle_records] if run_exhaustive else oracle_source_mapping.get(
                        f"{prompt_id}::{seed}"
                    )
                    schedules = _build_policy_schedules(
                        total_chunks=total_chunks,
                        fast_steps=fast_steps,
                        heavy_steps=heavy_steps,
                        target_avg_steps=target_avg_steps,
                        random_seed=seed + _stable_prompt_hash(prompt_id) % 100000,
                        oracle_ranked_indices=ranked,
                        include_baselines=not run_exhaustive,
                    )
                    policy_results = _rollout_for_schedules(
                        base_spec=base_spec,
                        schedules=schedules,
                        force=args.force,
                        device=device,
                    )
                    rollout_results.update(policy_results)

                aggregate_payload = aggregate_prompt_seed_results(
                    experiment_name=experiment_name,
                    prompt_id=prompt_id,
                    seed=seed,
                    rollout_results=rollout_results,
                    oracle_gain_records=oracle_records,
                    suffix_scores={},
                )
                save_prompt_seed_aggregate(aggregate_root=aggregate_root, aggregate_payload=aggregate_payload)
                if oracle_records:
                    append_gain_map_jsonl(
                        aggregate_root=aggregate_root,
                        experiment_name=experiment_name,
                        prompt_id=prompt_id,
                        seed=seed,
                        oracle_gain_records=oracle_records,
                    )
                    suffix_payloads.append(_suffix_payload_from_oracle(prompt_id, seed, oracle_records))
                append_policy_results_jsonl(aggregate_root=aggregate_root, aggregate_payload=aggregate_payload)
                success_pairs += 1

        gain_map_path = aggregate_root / "gain_maps.jsonl"
        policy_path = aggregate_root / "policy_results.jsonl"

        if gain_map_path.exists() and gain_map_path.stat().st_size > 0:
            generate_gain_heatmap(gain_map_path, plot_root / "gain_heatmap.png")
            generate_best_vs_median_gain_plot(gain_map_path, plot_root / "best_vs_median_gain.png")
            generate_oracle_position_histogram(gain_map_path, plot_root / "oracle_position_histogram.png")
            if suffix_payloads:
                generate_suffix_gain_curves(suffix_payloads, plot_root / "suffix_gain_curves.png")
        if policy_path.exists() and policy_path.stat().st_size > 0:
            generate_recovery_barplot(policy_path, plot_root / "recovery_barplot.png")

        _enforce_stage_success_criteria(
            stage_name=stage_name,
            aggregate_root=aggregate_root,
            run_exhaustive=run_exhaustive and not run_baseline_only,
            run_policy=run_policy,
        )
        if stage_name:
            _write_stage_gate_artifact(
                output_root=output_root,
                experiment_name=experiment_name,
                stage_name=stage_name,
                status="success",
                payload={
                    "processed_prompt_seed_pairs": processed_pairs,
                    "successful_prompt_seed_pairs": success_pairs,
                    "backend": backend,
                    "num_output_frames": num_output_frames,
                    "single_heavy_indices": single_heavy_indices,
                },
            )
    except Exception as exc:
        if stage_name:
            _write_stage_gate_artifact(
                output_root=output_root,
                experiment_name=experiment_name,
                stage_name=stage_name,
                status="failed",
                payload={"error": str(exc), "processed_prompt_seed_pairs": processed_pairs},
            )
        raise


if __name__ == "__main__":
    main()
