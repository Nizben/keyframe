#!/usr/bin/env python3
"""Replay AR-long rollouts and save final clean chunk latents.

The existing AR-long videos did not persist pre-VAE latents. This script uses
the completed rollout metadata to deterministically replay each schedule without
VAE decoding and saves one tensor per rollout. In lean mode, all-fast/all-heavy
remain full rollouts while single-heavy schedules save only the heavy chunk and
its immediate neighbors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch

from experiments.keyframe_budget.runner import (
    infer_num_chunks,
    load_merged_config,
    load_pipeline,
)
from utils.misc import set_seed


DEFAULT_RUNS_DIR = Path("outputs/ar_teacher_long_keyframe_budget")
DEFAULT_OUT_DIR = DEFAULT_RUNS_DIR / "compression_transition"
EXPECTED_SCHEDULES = ["all_fast", "all_heavy"] + [f"single_heavy_{i:02d}" for i in range(40)]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
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


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "item"


def discover_experiments(runs_dir: Path, task_id: Optional[int]) -> List[Path]:
    if task_id is not None:
        prompt_idx = task_id // 3
        seed = task_id % 3
        path = runs_dir / f"ar_teacher_long_p{prompt_idx:03d}_s{seed}"
        if not path.exists():
            raise FileNotFoundError(f"Expected experiment directory not found: {path}")
        return [path]
    return sorted(p for p in runs_dir.glob("ar_teacher_long_p*_s*") if p.is_dir())


def find_schedule_meta(exp_dir: Path, schedule: str) -> Path:
    matches = list((exp_dir / "rollouts").glob(f"*/*/{schedule}/rollout_meta.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one metadata file for {exp_dir.name}/{schedule}, got {len(matches)}")
    return matches[0]


def validate_meta(meta: Dict[str, Any], schedule: str) -> None:
    if meta.get("status") != "success":
        raise RuntimeError(f"Cannot replay failed rollout for schedule={schedule}: {meta.get('error')}")
    if meta.get("schedule_name") != schedule:
        raise RuntimeError(f"Metadata schedule mismatch: expected={schedule}, got={meta.get('schedule_name')}")
    if os.path.basename(str(meta.get("checkpoint", ""))) != "ar_diffusion.pt":
        raise RuntimeError(f"Refusing non-AR checkpoint: {meta.get('checkpoint')}")
    if str(meta.get("backend", "default")).lower() == "long_video":
        raise RuntimeError(f"Refusing forbidden backend=long_video in {schedule}")


def saved_chunk_indices_for_schedule(schedule: str, num_chunks: int, save_mode: str) -> Optional[List[int]]:
    if save_mode == "full" or not schedule.startswith("single_heavy_"):
        return None
    heavy_idx = int(schedule.rsplit("_", 1)[1])
    return [idx for idx in (heavy_idx - 1, heavy_idx, heavy_idx + 1) if 0 <= idx < num_chunks]


def is_compatible_cached_meta(
    out_meta: Dict[str, Any],
    schedule: str,
    num_chunks: int,
    num_frame_per_block: int,
    save_mode: str,
) -> bool:
    full_shape = [num_chunks, num_frame_per_block, 16, 60, 104]
    if out_meta.get("shape") == full_shape:
        return True
    wanted_indices = saved_chunk_indices_for_schedule(schedule, num_chunks, save_mode)
    if wanted_indices is None:
        return False
    wanted_shape = [len(wanted_indices), num_frame_per_block, 16, 60, 104]
    return out_meta.get("shape") == wanted_shape and out_meta.get("saved_chunk_indices") == wanted_indices


def manifest_row_from_cached_meta(
    out_meta: Dict[str, Any],
    schedule: str,
    num_chunks: int,
    num_frame_per_block: int,
) -> Dict[str, Any]:
    row = dict(out_meta["manifest_row"])
    if "latent_save_mode" in row and "saved_chunk_indices_json" in row:
        return row
    full_shape = [num_chunks, num_frame_per_block, 16, 60, 104]
    shape = out_meta.get("shape")
    saved_indices = out_meta.get("saved_chunk_indices")
    if shape == full_shape or saved_indices is None:
        row["latent_save_mode"] = "full"
        row["saved_chunk_indices_json"] = json.dumps(list(range(num_chunks)))
    else:
        row["latent_save_mode"] = "lean_neighbors" if schedule.startswith("single_heavy_") else "full"
        row["saved_chunk_indices_json"] = json.dumps([int(x) for x in saved_indices])
    return row


def save_latents_for_experiment(
    exp_dir: Path,
    out_dir: Path,
    device: torch.device,
    dtype_name: str,
    force: bool,
    schedules: Iterable[str],
    save_mode: str,
) -> List[Dict[str, Any]]:
    first_meta = read_json(find_schedule_meta(exp_dir, "all_fast"))
    config_path = str(first_meta.get("model_config_path") or first_meta.get("config_path"))
    checkpoint = str(first_meta["checkpoint"])
    use_ema = bool(first_meta.get("use_ema", False))
    backend = str(first_meta.get("backend", "default"))
    dtype = parse_dtype(dtype_name or str(first_meta.get("dtype", "bfloat16")))
    loaded_config = load_merged_config(config_path)
    num_output_frames = int(first_meta.get("latent_shape", [1, first_meta["num_chunks"] * 3])[1])
    num_frame_per_block = int(getattr(loaded_config, "num_frame_per_block", 1))
    independent_first_frame = bool(getattr(loaded_config, "independent_first_frame", False))
    num_chunks = infer_num_chunks(num_output_frames, num_frame_per_block, independent_first_frame)

    pipeline, _ = load_pipeline(
        config_path=config_path,
        checkpoint_path=checkpoint,
        use_ema=use_ema,
        device=device,
        dtype=dtype_name,
        strict_step_match=True,
        debug_cache_logs=False,
        backend=backend,
    )

    rows: List[Dict[str, Any]] = []
    try:
        for schedule in schedules:
            meta_path = find_schedule_meta(exp_dir, schedule)
            meta = read_json(meta_path)
            validate_meta(meta, schedule)
            steps_per_chunk = [int(x) for x in meta["steps_per_chunk"]]
            if len(steps_per_chunk) != num_chunks:
                raise RuntimeError(f"Chunk count mismatch in {meta_path}: {len(steps_per_chunk)} != {num_chunks}")

            prompt_id = str(meta["prompt_id"])
            seed = int(meta["seed"])
            latent_dir = out_dir / "latents" / exp_dir.name / slug(prompt_id) / str(seed) / schedule
            latent_path = latent_dir / "chunk_latents.pt"
            meta_out_path = latent_dir / "chunk_latents_meta.json"
            if latent_path.exists() and meta_out_path.exists() and not force:
                out_meta = read_json(meta_out_path)
                if is_compatible_cached_meta(out_meta, schedule, num_chunks, num_frame_per_block, save_mode):
                    rows.append(manifest_row_from_cached_meta(out_meta, schedule, num_chunks, num_frame_per_block))
                    continue

            set_seed(seed)
            noise = torch.randn([1, num_output_frames, 16, 60, 104], device=device, dtype=dtype)
            with torch.inference_mode():
                latents, chunk_logs = pipeline.inference(
                    noise=noise,
                    text_prompts=[str(meta["prompt_text"])],
                    return_video=False,
                    return_chunk_logs=True,
                    steps_per_chunk=steps_per_chunk,
                    start_frame_index=int(meta.get("start_frame_index", 0)),
                )
            chunk_latents_full = latents[0].reshape(num_chunks, num_frame_per_block, 16, 60, 104).detach().cpu()
            saved_chunk_indices = saved_chunk_indices_for_schedule(schedule, num_chunks, save_mode)
            if saved_chunk_indices is None:
                chunk_latents = chunk_latents_full
                manifest_saved_indices = list(range(num_chunks))
                latent_save_mode = "full"
            else:
                chunk_latents = chunk_latents_full[saved_chunk_indices].contiguous()
                manifest_saved_indices = saved_chunk_indices
                latent_save_mode = "lean_neighbors"
            latent_dir.mkdir(parents=True, exist_ok=True)
            torch.save(chunk_latents, latent_path)

            heavy_idx = None
            if schedule.startswith("single_heavy_"):
                heavy_idx = int(schedule.rsplit("_", 1)[1])
            row = {
                "experiment_name": exp_dir.name,
                "prompt_id": prompt_id,
                "seed": seed,
                "schedule": schedule,
                "run_type": "single_heavy" if heavy_idx is not None else schedule,
                "heavy_idx": heavy_idx,
                "chunk_idx": None,
                "latent_path": str(latent_path),
                "latent_save_mode": latent_save_mode,
                "saved_chunk_indices_json": json.dumps(manifest_saved_indices),
                "num_chunks": num_chunks,
                "chunk_frames": num_frame_per_block,
                "checkpoint": checkpoint,
                "config_path": config_path,
                "use_ema": use_ema,
                "source_rollout_meta": str(meta_path),
            }
            write_json(
                meta_out_path,
                {
                    "shape": list(chunk_latents.shape),
                    "dtype": str(chunk_latents.dtype),
                    "latent_save_mode": latent_save_mode,
                    "saved_chunk_indices": manifest_saved_indices,
                    "chunk_logs": chunk_logs,
                    "manifest_row": row,
                },
            )
            rows.append(row)
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task_id", type=int, default=None, help="Optional 0..71 prompt-seed task id.")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--schedules", type=str, default=",".join(EXPECTED_SCHEDULES))
    parser.add_argument(
        "--save_mode",
        choices=("lean", "full"),
        default="lean",
        help="lean saves full all_fast/all_heavy and only i-1,i,i+1 for each single_heavy_i.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedules = [x.strip() for x in args.schedules.split(",") if x.strip()]
    experiments = discover_experiments(args.runs_dir, args.task_id)
    if not experiments:
        raise RuntimeError(f"No AR-long experiment directories found under {args.runs_dir}")

    all_rows: List[Dict[str, Any]] = []
    for exp_dir in experiments:
        all_rows.extend(
            save_latents_for_experiment(
                exp_dir=exp_dir,
                out_dir=args.out_dir,
                device=device,
                dtype_name=args.dtype,
                force=args.force,
                schedules=schedules,
                save_mode=args.save_mode,
            )
        )

    manifest_dir = args.out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"manifest_task_{args.task_id:03d}.parquet" if args.task_id is not None else "manifest_all.parquet"
    out_path = manifest_dir / shard_name
    pd.DataFrame(all_rows).to_parquet(out_path, index=False)
    print(f"[compression-latents] wrote {len(all_rows)} rows: {out_path}")


if __name__ == "__main__":
    main()
