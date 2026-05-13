#!/usr/bin/env python3
"""Compute per-chunk fast-solver instability for AR-long router ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
import torch

from experiments.keyframe_budget.runner import (
    infer_num_chunks,
    load_merged_config,
    load_pipeline,
    load_prompt_set,
)
from utils.misc import set_seed


DEFAULT_OUTPUT_ROOT = Path("outputs/ar_teacher_long_router_ablation")
DEFAULT_PROMPTS = Path("experiments/keyframe_budget/prompts/motion_rich_24.json")
DEFAULT_CONFIG = "configs/ar_diffusion_tf_chunkwise.yaml"
DEFAULT_CHECKPOINT = "checkpoints/chunkwise/ar_diffusion.pt"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def parse_dtype(name: str) -> torch.dtype:
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
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True, help="0..71, prompt_index * 3 + seed_index")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt_set", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--num_prompts", type=int, default=24)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fast_steps", type=int, default=8)
    parser.add_argument("--num_output_frames", type=int, default=120)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--backend", type=str, default="default")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prompt_index = args.task_id // 3
    seed = args.task_id % 3
    prompts = load_prompt_set(args.prompt_set, max_prompts=args.num_prompts)
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(f"Prompt index {prompt_index} out of bounds for {len(prompts)} prompts")
    prompt = prompts[prompt_index]
    prompt_id = str(prompt["prompt_id"])
    prompt_text = str(prompt["prompt_text"])
    experiment_name = f"ar_teacher_long_router_ablation_p{prompt_index:03d}_s{seed}"

    out_dir = args.output_root / "solver_instability" / experiment_name
    out_parquet = out_dir / "solver_instability.parquet"
    out_json = out_dir / "solver_instability.json"
    if out_parquet.exists() and out_json.exists() and not args.force:
        print(f"[solver-instability] cached: {out_parquet}")
        return

    if os.path.basename(args.checkpoint) != "ar_diffusion.pt":
        raise ValueError(f"Refusing non-AR checkpoint: {args.checkpoint}")
    if args.backend.lower() == "long_video":
        raise ValueError("Refusing backend=long_video for solver-instability ablation")

    loaded_config = load_merged_config(args.config_path)
    total_chunks = infer_num_chunks(
        num_output_frames=args.num_output_frames,
        num_frame_per_block=int(getattr(loaded_config, "num_frame_per_block", 1)),
        independent_first_frame=bool(getattr(loaded_config, "independent_first_frame", False)),
    )
    if total_chunks != 40:
        raise ValueError(f"Expected 40 chunks for AR-long ablation, got {total_chunks}")

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

    rows: List[Dict[str, Any]] = []
    try:
        set_seed(seed)
        noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=parse_dtype(args.dtype),
        )
        steps_per_chunk = [int(args.fast_steps)] * total_chunks
        with torch.inference_mode():
            _, chunk_logs = pipeline.inference(
                noise=noise,
                text_prompts=[prompt_text],
                return_video=False,
                steps_per_chunk=steps_per_chunk,
                return_chunk_logs=True,
                record_solver_instability=True,
            )
        for item in chunk_logs:
            step_changes = [float(x) for x in item.get("solver_step_rel_changes", [])]
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "heavy_idx": int(item["chunk_idx"]),
                    "preview_steps": int(args.fast_steps),
                    "S_instability": float(item.get("solver_instability", 0.0)),
                    "solver_step_rel_changes_json": json.dumps(step_changes),
                    "actual_num_steps": int(item.get("actual_num_steps", 0)),
                }
            )
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    write_json(
        out_json,
        {
            "experiment_name": experiment_name,
            "prompt_id": prompt_id,
            "seed": seed,
            "preview_steps": int(args.fast_steps),
            "rows": rows,
        },
    )
    print(f"[solver-instability] rows={len(rows)} wrote={out_parquet}")


if __name__ == "__main__":
    main()
