#!/usr/bin/env python3
"""Generate AR-long Amax/S ablation router rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import torch

from experiments.keyframe_budget.runner import (
    RolloutSpec,
    infer_num_chunks,
    load_merged_config,
    load_pipeline,
    load_prompt_set,
    run_rollout,
)
from experiments.keyframe_budget.schedules import (
    all_fast,
    all_heavy,
    oracle_top_m,
    random_top_m,
    total_requested_steps,
)


DEFAULT_SCORES = Path("outputs/ar_teacher_long_router_ablation/compression_vbench_with_solver_instability.parquet")
DEFAULT_OUTPUT_ROOT = Path("outputs/ar_teacher_long_router_ablation")
DEFAULT_PROMPTS = Path("experiments/keyframe_budget/prompts/motion_rich_24.json")
DEFAULT_CONFIG = "configs/ar_diffusion_tf_chunkwise.yaml"
DEFAULT_CHECKPOINT = "checkpoints/chunkwise/ar_diffusion.pt"
TEMPORAL_METRICS = ("background_consistency", "subject_consistency", "motion_smoothness", "overall_consistency")
QUALITY_METRICS = ("imaging_quality", "aesthetic_quality")
ALL_METRICS = (
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
    "aesthetic_quality",
    "subject_consistency",
    "background_consistency",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def selected_indices(steps_per_chunk: Sequence[int], fast_steps: int) -> List[int]:
    return [idx for idx, steps in enumerate(steps_per_chunk) if int(steps) != int(fast_steps)]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def top_indices(group: pd.DataFrame, column: str, k: int) -> List[int]:
    work = group.dropna(subset=[column]).copy()
    if len(work) < k:
        raise RuntimeError(f"Not enough valid rows for {column} top-{k}: got {len(work)}")
    ordered = work.sort_values([column, "heavy_idx"], ascending=[False, True])
    return [int(x) for x in ordered.head(k)["heavy_idx"].tolist()]


def metric_oracle_score(group: pd.DataFrame, metric_cols: Sequence[str]) -> pd.Series:
    cols = [col for col in metric_cols if col in group]
    if not cols:
        raise RuntimeError(f"No metric columns available for oracle: {metric_cols}")
    z_cols = [zscore(group[col].astype(float)) for col in cols]
    return pd.concat(z_cols, axis=1).mean(axis=1)


def load_score_group(scores_path: Path, prompt_id: str, seed: int) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    group = scores[
        (scores["prompt_id"].astype(str) == str(prompt_id)) & (scores["seed"].astype(int) == int(seed))
    ].copy()
    if len(group) != 40:
        raise RuntimeError(f"Expected 40 score rows for {prompt_id} seed={seed}, got {len(group)}")
    required = {"A_max", "S_instability", "AplusS", "AtimesS"}
    missing = sorted(required - set(group.columns))
    if missing:
        raise RuntimeError(f"Scores table is missing router columns {missing}. Run merge_solver_instability_with_compression.py first.")
    group["heavy_idx"] = group["heavy_idx"].astype(int)
    if "global_mean_oracle" not in group:
        group["global_mean_oracle"] = metric_oracle_score(group, [f"{m}_delta" for m in ALL_METRICS])
    if "imaging_oracle" not in group:
        group["imaging_oracle"] = metric_oracle_score(group, [f"{m}_delta" for m in QUALITY_METRICS])
    if "temporal_oracle" not in group:
        group["temporal_oracle"] = metric_oracle_score(group, [f"{m}_delta" for m in TEMPORAL_METRICS])
    return group


def schedule_from_indices(total_chunks: int, fast_steps: int, heavy_steps: int, indices: Sequence[int]) -> List[int]:
    return oracle_top_m(
        total_chunks=total_chunks,
        fast_steps=fast_steps,
        heavy_steps=heavy_steps,
        m=len(indices),
        ranked_chunk_indices=list(indices),
    )


def build_ablation_schedules(
    group: pd.DataFrame,
    total_chunks: int,
    fast_steps: int,
    prompt_id: str,
    seed: int,
    k: int,
    random_replicates: int,
    router_columns: Sequence[str],
) -> Dict[str, List[int]]:
    schedules: Dict[str, List[int]] = {
        "all_fast": all_fast(total_chunks, fast_steps),
        "all_heavy": all_heavy(total_chunks, 24),
    }
    for rep in range(random_replicates):
        random_seed = stable_int("router_ablation_random", prompt_id, seed, k, rep) % (2**31 - 1)
        schedules[f"random_k{k:02d}_h24_r{rep:02d}"] = random_top_m(
            total_chunks, fast_steps, 24, k, seed=random_seed
        )

    amax_indices = top_indices(group, "A_max", k)
    schedules[f"Amax_k{k:02d}_h24"] = schedule_from_indices(total_chunks, fast_steps, 24, amax_indices)
    router_name = {
        "A_max": f"Amax_k{k:02d}_h24",
        "S_instability": f"S_top_k{k:02d}_h24",
        "AplusS": f"AplusS_top_k{k:02d}_h24",
        "AtimesS": f"AtimesS_top_k{k:02d}_h24",
    }
    for column in router_columns:
        if column == "A_max":
            continue
        if column not in group:
            raise RuntimeError(f"Requested router column not found: {column}")
        if column not in router_name:
            raise RuntimeError(f"No policy-name mapping for router column: {column}")
        schedules[router_name[column]] = schedule_from_indices(
            total_chunks, fast_steps, 24, top_indices(group, column, k)
        )
    for heavy_steps in (12, 16, 20):
        schedules[f"Amax_k{k:02d}_h{heavy_steps:02d}"] = schedule_from_indices(
            total_chunks, fast_steps, heavy_steps, amax_indices
        )
    schedules[f"global_mean_oracle_k{k:02d}_h24"] = schedule_from_indices(
        total_chunks, fast_steps, 24, top_indices(group, "global_mean_oracle", k)
    )
    schedules[f"imaging_oracle_k{k:02d}_h24"] = schedule_from_indices(
        total_chunks, fast_steps, 24, top_indices(group, "imaging_oracle", k)
    )
    schedules[f"temporal_oracle_k{k:02d}_h24"] = schedule_from_indices(
        total_chunks, fast_steps, 24, top_indices(group, "temporal_oracle", k)
    )
    return schedules


def assert_ablation_schedules(schedules: Mapping[str, Sequence[int]], total_chunks: int, fast_steps: int, k: int) -> None:
    for name, schedule in schedules.items():
        if len(schedule) != total_chunks:
            raise ValueError(f"{name}: expected {total_chunks} chunks, got {len(schedule)}")
        if name == "all_fast":
            if any(int(x) != fast_steps for x in schedule):
                raise ValueError("all_fast contains non-fast chunks")
            continue
        if name == "all_heavy":
            if any(int(x) != 24 for x in schedule):
                raise ValueError("all_heavy contains non-24-step chunks")
            continue
        if len(selected_indices(schedule, fast_steps)) != k:
            raise ValueError(f"{name}: expected {k} selected chunks, got {selected_indices(schedule, fast_steps)}")
    amax_selected = selected_indices(schedules[f"Amax_k{k:02d}_h24"], fast_steps)
    for heavy_steps in (12, 16, 20):
        name = f"Amax_k{k:02d}_h{heavy_steps:02d}"
        if selected_indices(schedules[name], fast_steps) != amax_selected:
            raise ValueError(f"{name}: selected chunks differ from Amax h24")


def write_policy_manifest(
    output_root: Path,
    experiment_name: str,
    prompt_id: str,
    seed: int,
    schedules: Mapping[str, Sequence[int]],
    fast_steps: int,
) -> None:
    rows = []
    for name, schedule in sorted(schedules.items()):
        rows.append(
            {
                "experiment_name": experiment_name,
                "prompt_id": prompt_id,
                "seed": seed,
                "policy_name": name,
                "selected_chunks": selected_indices(schedule, fast_steps),
                "steps_per_chunk": [int(x) for x in schedule],
                "total_nfe_requested": total_requested_steps(schedule),
            }
        )
    write_json(output_root / experiment_name / "router_ablation_manifest.json", {"policies": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True, help="0..71, prompt_index * 3 + seed_index")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt_set", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--num_prompts", type=int, default=24)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fast_steps", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--random_replicates", type=int, default=20)
    parser.add_argument("--router_columns", type=str, default="A_max,S_instability,AplusS,AtimesS")
    parser.add_argument("--num_output_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--backend", type=str, default="default")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.k != 5:
        raise ValueError("This targeted ablation is intentionally k=5 only.")
    if os.path.basename(args.checkpoint) != "ar_diffusion.pt":
        raise ValueError(f"Refusing non-AR checkpoint: {args.checkpoint}")
    if args.backend.lower() == "long_video":
        raise ValueError("Refusing backend=long_video for router ablation")

    prompt_index = args.task_id // 3
    seed = args.task_id % 3
    prompts = load_prompt_set(args.prompt_set, max_prompts=args.num_prompts)
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(f"Prompt index {prompt_index} out of bounds for {len(prompts)} prompts")
    prompt = prompts[prompt_index]
    prompt_id = str(prompt["prompt_id"])
    prompt_text = str(prompt["prompt_text"])
    experiment_name = f"ar_teacher_long_router_ablation_p{prompt_index:03d}_s{seed}"

    loaded_config = load_merged_config(args.config_path)
    total_chunks = infer_num_chunks(
        num_output_frames=args.num_output_frames,
        num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
        independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
    )
    if total_chunks != 40:
        raise ValueError(f"Expected 40 chunks for AR-long ablation, got {total_chunks}")

    router_columns = [item.strip() for item in args.router_columns.split(",") if item.strip()]
    score_group = load_score_group(args.scores, prompt_id=prompt_id, seed=seed)
    schedules = build_ablation_schedules(
        score_group,
        total_chunks=total_chunks,
        fast_steps=args.fast_steps,
        prompt_id=prompt_id,
        seed=seed,
        k=args.k,
        random_replicates=args.random_replicates,
        router_columns=router_columns,
    )
    assert_ablation_schedules(schedules, total_chunks, args.fast_steps, args.k)
    write_policy_manifest(args.output_root, experiment_name, prompt_id, seed, schedules, args.fast_steps)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline, _ = load_pipeline(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        use_ema=False,
        device=device,
        dtype=args.dtype,
        strict_step_match=True,
        debug_cache_logs=False,
        backend=args.backend,
    )

    results = []
    try:
        for schedule_name, schedule in sorted(schedules.items()):
            spec = RolloutSpec(
                experiment_name=experiment_name,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                seed=seed,
                checkpoint=args.checkpoint,
                config_path=args.config_path,
                model_config_path=args.config_path,
                backend=args.backend,
                schedule_name=schedule_name,
                steps_per_chunk=list(schedule),
                num_output_frames=args.num_output_frames,
                output_root=str(args.output_root),
                use_ema=False,
                fps=args.fps,
                dtype=args.dtype,
                strict_step_match=True,
                debug_cache_logs=False,
                suffix_window_latent=32,
                latent_to_visible_ratio=4,
                save_chunk_boundaries=True,
            )
            result = run_rollout(spec, pipeline=pipeline, loaded_config=loaded_config, device=device, force=args.force)
            results.append(asdict(result))
            if result.status != "success":
                raise RuntimeError(f"Rollout failed for {schedule_name}: {result.error}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(args.output_root / experiment_name / "router_ablation_results.json", {"results": results})
    print(f"[router-ablation] wrote {len(results)} policy rollouts for {experiment_name}")


if __name__ == "__main__":
    main()
