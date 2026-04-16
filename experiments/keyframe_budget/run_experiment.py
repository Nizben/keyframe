"""CLI entrypoint for v3.5 keyframe-budget experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    # Deterministic across Python processes unlike built-in hash().
    total = 0
    for i, ch in enumerate(value):
        total += (i + 1) * ord(ch)
    return total


def _reset_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


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
        out[schedule_name] = run_rollout(
            spec,
            device=device,
            force=force,
        )
    return out


def _suffix_payload_from_oracle(prompt_id: str, seed: int, records) -> Dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "chunk_indices": [r.chunk_idx for r in records],
        "suffix_gains": [r.gain for r in records],  # TODO: replace with true suffix metrics.
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Experiment config YAML path.")
    parser.add_argument("--force", action="store_true", help="Rerun rollouts even if already successful.")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    exp_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)

    experiment_name = exp_cfg["experiment_name"]
    output_root = exp_cfg.get("output_root", "outputs/keyframe_budget")
    config_path = exp_cfg.get("model_config_path", "configs/ar_diffusion_tf_chunkwise.yaml")
    checkpoint = exp_cfg["checkpoint"]
    fast_steps = int(exp_cfg["fast_steps"])
    heavy_steps = int(exp_cfg["heavy_steps"])
    target_avg_steps = int(exp_cfg["target_avg_steps"])
    num_output_frames = int(exp_cfg.get("num_output_frames", 21))
    seeds = [int(x) for x in exp_cfg["seeds"]]
    run_exhaustive = bool(exp_cfg.get("run_exhaustive_single_heavy", True))
    run_policy = bool(exp_cfg.get("run_policy_comparison", True))
    if not run_exhaustive and not run_policy:
        raise ValueError("At least one of run_exhaustive_single_heavy or run_policy_comparison must be true.")
    use_ema = bool(exp_cfg.get("use_ema", False))
    dtype = str(exp_cfg.get("dtype", "bfloat16"))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)

    prompt_set = load_prompt_set(
        exp_cfg["prompt_set"],
        max_prompts=exp_cfg.get("num_prompts"),
    )

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

    for prompt in prompt_set:
        prompt_id = prompt["prompt_id"]
        prompt_text = prompt["prompt_text"]
        for seed in seeds:
            base_spec = RolloutSpec(
                experiment_name=experiment_name,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                seed=seed,
                checkpoint=checkpoint,
                config_path=config_path,
                schedule_name="all_fast",
                steps_per_chunk=default_steps,
                num_output_frames=num_output_frames,
                output_root=output_root,
                use_ema=use_ema,
                dtype=dtype,
            )

            rollout_results: Dict[str, RolloutResult] = {}
            oracle_records = []
            if run_exhaustive:
                rollout_results = exhaustive_single_heavy_sweep(
                    base_spec=base_spec,
                    fast_steps=fast_steps,
                    heavy_steps=heavy_steps,
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
                    pseudo_records = []
                    for rank, chunk_idx in enumerate(oracle_ranked):
                        pseudo_records.append(GainRecord(chunk_idx=chunk_idx, gain=0.0, gain_norm=0.0, rank=rank))
                    oracle_records = pseudo_records

            if run_policy:
                if run_exhaustive:
                    ranked = [row.chunk_idx for row in oracle_records]
                else:
                    ranked = oracle_source_mapping.get(f"{prompt_id}::{seed}")
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


if __name__ == "__main__":
    main()
