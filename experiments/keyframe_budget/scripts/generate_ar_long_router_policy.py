#!/usr/bin/env python3
"""Generate AR-long router-policy rollouts from compression-transition scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    uniform_top_m,
    uniform_top_m_indices,
)


DEFAULT_SCORES = Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet")
DEFAULT_OUTPUT_ROOT = Path("outputs/ar_teacher_long_router_policy")
DEFAULT_PROMPTS = Path("experiments/keyframe_budget/prompts/motion_rich_24.json")
DEFAULT_CONFIG = "configs/ar_diffusion_tf_chunkwise.yaml"
DEFAULT_CHECKPOINT = "checkpoints/chunkwise/ar_diffusion.pt"
POLICY_KS = (5, 10)
RANDOM_REPLICATES = (0, 1, 2)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def stable_int(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def selected_indices(steps_per_chunk: Sequence[int], heavy_steps: int) -> List[int]:
    return [idx for idx, steps in enumerate(steps_per_chunk) if int(steps) == int(heavy_steps)]


def load_score_group(scores_path: Path, prompt_id: str, seed: int) -> pd.DataFrame:
    table = pd.read_parquet(scores_path)
    group = table[(table["prompt_id"].astype(str) == str(prompt_id)) & (table["seed"].astype(int) == int(seed))].copy()
    if len(group) != 40:
        raise RuntimeError(f"Expected 40 score rows for {prompt_id} seed={seed}, got {len(group)}")
    return group


def top_indices(group: pd.DataFrame, column: str, k: int, exclude_chunk0: bool = False) -> List[int]:
    work = group.copy()
    if exclude_chunk0:
        work = work[work["heavy_idx"].astype(int) != 0]
    work = work.dropna(subset=[column])
    if len(work) < k:
        raise RuntimeError(f"Not enough valid rows for {column} top-{k}: got {len(work)}")
    ordered = work.sort_values([column, "heavy_idx"], ascending=[False, True])
    return [int(x) for x in ordered.head(k)["heavy_idx"].tolist()]


def build_router_schedules(
    group: pd.DataFrame,
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    prompt_id: str,
    seed: int,
    random_replicates: int,
    include_baselines: bool,
    include_no_chunk0: bool,
    oracle_prefix: str,
) -> Dict[str, List[int]]:
    schedules: Dict[str, List[int]] = {}
    if include_baselines:
        schedules["all_fast"] = all_fast(total_chunks, fast_steps)
        schedules["all_heavy"] = all_heavy(total_chunks, heavy_steps)
    random_width = 2 if random_replicates >= 10 else 1
    for k in POLICY_KS:
        schedules[f"periodic_k{k:02d}"] = uniform_top_m(total_chunks, fast_steps, heavy_steps, k)
        for rep in range(random_replicates):
            random_seed = stable_int("router_random", prompt_id, seed, k, rep) % (2**31 - 1)
            schedules[f"random_k{k:02d}_r{rep:0{random_width}d}"] = random_top_m(
                total_chunks, fast_steps, heavy_steps, k, seed=random_seed
            )
        schedules[f"Amax_top_k{k:02d}"] = oracle_top_m(
            total_chunks,
            fast_steps,
            heavy_steps,
            k,
            ranked_chunk_indices=top_indices(group, "A_max", k),
        )
        if include_no_chunk0:
            schedules[f"Amax_top_k{k:02d}_no_chunk0"] = oracle_top_m(
                total_chunks,
                fast_steps,
                heavy_steps,
                k,
                ranked_chunk_indices=top_indices(group, "A_max", k, exclude_chunk0=True),
            )
        schedules[f"{oracle_prefix}_k{k:02d}"] = oracle_top_m(
            total_chunks,
            fast_steps,
            heavy_steps,
            k,
            ranked_chunk_indices=top_indices(group, "delta_temp", k),
        )
    return schedules


def assert_router_schedules(
    schedules: Mapping[str, Sequence[int]],
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
) -> None:
    for name, schedule in schedules.items():
        if len(schedule) != total_chunks:
            raise ValueError(f"{name}: expected {total_chunks} chunks, got {len(schedule)}")
        if any(int(x) not in {fast_steps, heavy_steps} for x in schedule):
            raise ValueError(f"{name}: schedule contains steps outside {{{fast_steps}, {heavy_steps}}}")
        if name == "all_fast":
            if any(int(x) != fast_steps for x in schedule):
                raise ValueError("all_fast contains a heavy chunk")
            continue
        if name == "all_heavy":
            if any(int(x) != heavy_steps for x in schedule):
                raise ValueError("all_heavy contains a fast chunk")
            continue
        if name.startswith("periodic_k"):
            k = int(name.split("_k", 1)[1])
            if selected_indices(schedule, heavy_steps) != uniform_top_m_indices(total_chunks, k):
                raise ValueError(f"{name}: periodic indices are not the expected uniform positions")
        if "no_chunk0" in name and int(schedule[0]) == heavy_steps:
            raise ValueError(f"{name}: no_chunk0 policy selected chunk 0")
    totals_by_k: Dict[str, set[int]] = {}
    for name, schedule in schedules.items():
        if name in {"all_fast", "all_heavy"}:
            continue
        k_token = name.split("_k", 1)[1].split("_", 1)[0]
        totals_by_k.setdefault(k_token, set()).add(total_requested_steps(schedule))
    bad = {k: totals for k, totals in totals_by_k.items() if len(totals) != 1}
    if bad:
        raise ValueError(f"Router schedules are not compute-matched within k: {bad}")


def write_policy_manifest(
    output_root: Path,
    experiment_name: str,
    prompt_id: str,
    seed: int,
    schedules: Mapping[str, Sequence[int]],
    heavy_steps: int,
) -> None:
    rows = []
    for name, schedule in sorted(schedules.items()):
        rows.append(
            {
                "experiment_name": experiment_name,
                "prompt_id": prompt_id,
                "seed": seed,
                "policy_name": name,
                "selected_chunks": selected_indices(schedule, heavy_steps),
                "steps_per_chunk": [int(x) for x in schedule],
                "total_nfe_requested": total_requested_steps(schedule),
            }
        )
    path = output_root / experiment_name / "router_policy_manifest.json"
    write_json(path, {"policies": rows})


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
    parser.add_argument("--heavy_steps", type=int, default=24)
    parser.add_argument("--num_output_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--backend", type=str, default="default")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--random_replicates", type=int, default=len(RANDOM_REPLICATES))
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--skip_no_chunk0", action="store_true")
    parser.add_argument("--oracle_prefix", type=str, default="oracle_top")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.random_replicates < 0:
        raise ValueError("--random_replicates must be non-negative")

    prompt_index = args.task_id // 3
    seed = args.task_id % 3
    prompts = load_prompt_set(args.prompt_set, max_prompts=args.num_prompts)
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(f"Prompt index {prompt_index} out of bounds for {len(prompts)} prompts")
    prompt = prompts[prompt_index]
    prompt_id = str(prompt["prompt_id"])
    prompt_text = str(prompt["prompt_text"])
    experiment_name = f"ar_teacher_long_router_p{prompt_index:03d}_s{seed}"

    if os.path.basename(args.checkpoint) != "ar_diffusion.pt":
        raise ValueError(f"Refusing non-AR checkpoint: {args.checkpoint}")
    if args.backend.lower() == "long_video":
        raise ValueError("Refusing backend=long_video for router policy experiment")

    loaded_config = load_merged_config(args.config_path)
    total_chunks = infer_num_chunks(
        num_output_frames=args.num_output_frames,
        num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
        independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
    )
    if total_chunks != 40:
        raise ValueError(f"Expected 40 chunks for AR-long router experiment, got {total_chunks}")

    score_group = load_score_group(args.scores, prompt_id=prompt_id, seed=seed)
    schedules = build_router_schedules(
        score_group,
        total_chunks=total_chunks,
        fast_steps=args.fast_steps,
        heavy_steps=args.heavy_steps,
        prompt_id=prompt_id,
        seed=seed,
        random_replicates=args.random_replicates,
        include_baselines=args.include_baselines,
        include_no_chunk0=not args.skip_no_chunk0,
        oracle_prefix=args.oracle_prefix,
    )
    assert_router_schedules(schedules, total_chunks, args.fast_steps, args.heavy_steps)
    write_policy_manifest(args.output_root, experiment_name, prompt_id, seed, schedules, args.heavy_steps)

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

    write_json(args.output_root / experiment_name / "router_policy_results.json", {"results": results})
    print(f"[router-policy] wrote {len(results)} policy rollouts for {experiment_name}")


if __name__ == "__main__":
    main()
