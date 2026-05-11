#!/usr/bin/env python3
"""Compute DeltaTokens-style transition scores from saved chunk latents."""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch


METRIC_EPS = 1e-8


def load_manifest(paths: List[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise RuntimeError("No manifest parquet files found.")
    return pd.concat(frames, ignore_index=True)


def load_flat_latents(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.ndim != 5:
        raise ValueError(f"Expected [chunks,frames,C,H,W], got {tuple(tensor.shape)} from {path}")
    return tensor.to(torch.float32).reshape(tensor.shape[0], -1)


def parse_saved_chunk_indices(row: object, fallback_num_chunks: Optional[int] = None) -> List[int]:
    value = getattr(row, "saved_chunk_indices_json", None)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if fallback_num_chunks is None:
            return []
        return list(range(fallback_num_chunks))
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    if isinstance(value, Sequence):
        return [int(x) for x in value]
    raise ValueError(f"Unsupported saved_chunk_indices_json value: {value!r}")


def load_single_chunk_flat(row: object, chunk_idx: int, fallback_num_chunks: int) -> torch.Tensor:
    tensor = torch.load(getattr(row, "latent_path"), map_location="cpu", weights_only=True)
    if tensor.ndim != 5:
        raise ValueError(f"Expected [chunks,frames,C,H,W], got {tuple(tensor.shape)} from {getattr(row, 'latent_path')}")
    indices = parse_saved_chunk_indices(row, fallback_num_chunks=fallback_num_chunks)
    if tensor.shape[0] == fallback_num_chunks and indices == list(range(fallback_num_chunks)):
        return tensor[chunk_idx].to(torch.float32).reshape(-1)
    if chunk_idx not in indices:
        raise RuntimeError(
            f"Lean latent file does not contain chunk {chunk_idx}: {getattr(row, 'latent_path')} "
            f"contains {indices}"
        )
    local_idx = indices.index(chunk_idx)
    return tensor[local_idx].to(torch.float32).reshape(-1)


def scalar(value: torch.Tensor) -> float:
    out = float(value.item())
    return out if math.isfinite(out) else float("nan")


def online_pca_residual_from_gram(
    gram: torch.Tensor,
    cross: torch.Tensor,
    current_energy: torch.Tensor,
    rank: int,
    eps: float,
) -> float:
    """Residual of current delta after projection onto prior-delta PCA basis.

    This avoids materializing float64 copies of the high-dimensional prior
    delta matrix for every transition. The eigendecomposition is only over the
    small prior-delta Gram matrix.
    """
    denom = current_energy.clamp_min(eps)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    valid = eigvals > eps
    if not bool(valid.any()):
        return 1.0
    k = min(rank, int(valid.sum().item()))
    eigvals = eigvals[:k]
    eigvecs = eigvecs[:, :k]
    coeff = eigvecs.T @ cross / torch.sqrt(eigvals.clamp_min(eps))
    proj_energy = torch.sum(coeff * coeff)
    residual = 1.0 - proj_energy / denom
    return float(torch.clamp(residual, 0.0, 1.0).item())


def transition_arrays(h_fast: torch.Tensor, rank: int, eps: float) -> Dict[str, List[float]]:
    n = h_fast.shape[0]
    deltas = h_fast[1:] - h_fast[:-1]
    delta_gram = deltas @ deltas.T
    delta_energy = torch.diag(delta_gram)
    d_scores = [float("nan")] * n
    a_scores = [float("nan")] * n
    r_scores = [float("nan")] * n
    for c in range(1, n):
        d = deltas[c - 1]
        prev = h_fast[c - 1]
        d_scores[c] = scalar(torch.dot(d, d) / torch.dot(prev, prev).clamp_min(eps))
        if c >= 2:
            accel = deltas[c - 1] - deltas[c - 2]
            a_scores[c] = scalar(torch.dot(accel, accel))
            r_scores[c] = online_pca_residual_from_gram(
                gram=delta_gram[: c - 1, : c - 1],
                cross=delta_gram[: c - 1, c - 1],
                current_energy=delta_energy[c - 1],
                rank=rank,
                eps=eps,
            )
    return {"D": d_scores, "A": a_scores, "R": r_scores}


def pick(values: List[float], idx: Optional[int]) -> float:
    if idx is None or idx < 0 or idx >= len(values):
        return float("nan")
    return float(values[idx])


def nanmax(a: float, b: float) -> float:
    vals = [x for x in (a, b) if math.isfinite(x)]
    return max(vals) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_glob", type=str, default="outputs/ar_teacher_long_keyframe_budget/compression_transition/manifests/*.parquet")
    parser.add_argument("--out", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_scores.parquet"))
    parser.add_argument("--pca_rank", type=int, default=8)
    parser.add_argument("--eps", type=float, default=METRIC_EPS)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--task_id", type=int, default=None, help="Optional shard id over prompt-seed groups.")
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of prompt-seed shards when --task_id is set.")
    parser.add_argument("--shard_dir", type=Path, default=None)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait_timeout_sec", type=int, default=7200)
    args = parser.parse_args()

    shard_dir = args.shard_dir or (args.out.parent / "transition_score_shards")
    if args.merge_only:
        expected = [shard_dir / f"transition_scores_task_{idx:03d}.parquet" for idx in range(args.num_tasks)]
        start = time.time()
        while True:
            missing = [path for path in expected if not path.exists()]
            if not missing:
                break
            if not args.wait or time.time() - start > args.wait_timeout_sec:
                raise RuntimeError(f"Missing {len(missing)} shard files; first missing: {missing[:5]}")
            print(f"[compression-scores] waiting for {len(missing)} shards", flush=True)
            time.sleep(60)
        merged = pd.concat([pd.read_parquet(path) for path in expected], ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(args.out, index=False)
        print(f"[compression-scores] merged rows={len(merged)} wrote={args.out}")
        return

    manifest_paths = [Path(path) for path in sorted(glob.glob(args.manifest_glob))]
    manifest = load_manifest(manifest_paths)
    required = {"prompt_id", "seed", "schedule", "heavy_idx", "latent_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

    rows = []
    all_groups = list(manifest.groupby(["prompt_id", "seed"], sort=True))
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= args.num_tasks:
            raise ValueError(f"task_id must be in [0, {args.num_tasks}), got {args.task_id}")
        groups = [group for idx, group in enumerate(all_groups) if idx % args.num_tasks == args.task_id]
        print(
            f"[compression-scores] shard task_id={args.task_id} num_tasks={args.num_tasks} "
            f"samples={len(groups)}/{len(all_groups)}",
            flush=True,
        )
    else:
        groups = all_groups
    for sample_idx, ((prompt_id, seed), group) in enumerate(groups, start=1):
        if args.progress_every > 0 and (sample_idx == 1 or sample_idx % args.progress_every == 0):
            print(f"[compression-scores] sample {sample_idx}/{len(groups)} prompt={prompt_id} seed={seed}", flush=True)
        by_schedule = {str(row.schedule): row for row in group.itertuples(index=False)}
        if "all_fast" not in by_schedule or "all_heavy" not in by_schedule:
            raise RuntimeError(f"Missing all_fast/all_heavy for {prompt_id} seed={seed}")
        h_fast = load_flat_latents(by_schedule["all_fast"].latent_path)
        h_heavy = load_flat_latents(by_schedule["all_heavy"].latent_path)
        if h_fast.shape != h_heavy.shape:
            raise RuntimeError(f"Shape mismatch for {prompt_id} seed={seed}")
        n = h_fast.shape[0]
        scores = transition_arrays(h_fast, rank=args.pca_rank, eps=args.eps)
        deltas_f = h_fast[1:] - h_fast[:-1]
        deltas_h = h_heavy[1:] - h_heavy[:-1]
        c_f = scalar(torch.sum(deltas_f * deltas_f))
        c_h = scalar(torch.sum(deltas_h * deltas_h))
        for i in range(n):
            sched = f"single_heavy_{i:02d}"
            if sched not in by_schedule:
                raise RuntimeError(f"Missing {sched} for {prompt_id} seed={seed}")
            h_i_chunk = load_single_chunk_flat(by_schedule[sched], i, fallback_num_chunks=n)
            q = scalar(torch.sum((h_i_chunk - h_fast[i]) ** 2))
            row = {
                "prompt_id": prompt_id,
                "seed": int(seed),
                "heavy_idx": i,
                "Q": q,
                "C_F": c_f,
                "C_H": c_h,
                "C_H_minus_C_F": c_h - c_f,
            }
            for prefix, arr in scores.items():
                in_score = pick(arr, i) if i > 0 else float("nan")
                out_score = pick(arr, i + 1) if i + 1 < n else float("nan")
                row[f"{prefix}_in"] = in_score
                row[f"{prefix}_out"] = out_score
                row[f"{prefix}_max"] = nanmax(in_score, out_score)
            rows.append(row)

    if args.task_id is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_path = shard_dir / f"transition_scores_task_{args.task_id:03d}.parquet"
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_path = args.out
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[compression-scores] rows={len(rows)} wrote={out_path}")


if __name__ == "__main__":
    main()
