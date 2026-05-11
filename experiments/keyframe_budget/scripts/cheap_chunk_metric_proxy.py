#!/usr/bin/env python3
"""Compute cheap per-chunk metrics and compare them with oracle hard chunks.

This is an offline long-rollout analysis. It reads existing all-fast videos and
the merged discovery gain maps; it does not generate videos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on cluster env.
    cv2 = None


METRIC_NAMES = (
    # Original local chunk-window features.
    "frame_l1",
    "edge_l1",
    "flow_mag",
    # Same cheap features on fixed suffix windows starting at chunk i.
    "suffix_frame_l1",
    "suffix_edge_l1",
    "suffix_flow_mag",
    # One-step boundary discontinuity at the chunk start.
    "boundary_frame_l1",
    "boundary_edge_l1",
    "boundary_flow_mag",
    # Acceleration: current chunk motion minus previous chunk motion.
    "frame_l1_accel",
    "edge_l1_accel",
    "flow_mag_accel",
    # Suffix-local contrast: suffix motion minus local chunk motion.
    "suffix_minus_local_frame_l1",
    "suffix_minus_local_edge_l1",
    "suffix_minus_local_flow_mag",
)

BASE_COMBO_FEATURES = (
    "frame_l1",
    "edge_l1",
    "flow_mag",
    "suffix_frame_l1",
    "suffix_edge_l1",
    "suffix_flow_mag",
    "boundary_frame_l1",
    "boundary_edge_l1",
    "boundary_flow_mag",
    "frame_l1_accel",
    "edge_l1_accel",
    "flow_mag_accel",
    "suffix_minus_local_frame_l1",
    "suffix_minus_local_edge_l1",
    "suffix_minus_local_flow_mag",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=False))
            f.write("\n")


def load_samples(aggregate_root: Path) -> List[Tuple[str, int]]:
    prompt_seed_dir = aggregate_root / "prompt_seed_aggregates"
    samples: List[Tuple[str, int]] = []
    for path in sorted(prompt_seed_dir.glob("*.json")):
        payload = read_json(path)
        samples.append((str(payload["prompt_id"]), int(payload["seed"])))
    if not samples:
        raise FileNotFoundError(f"No prompt/seed aggregate files found under {prompt_seed_dir}")
    return samples


def load_gain_records(aggregate_root: Path) -> Dict[Tuple[str, int], Dict[int, Dict[str, Any]]]:
    out: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
    for prompt_id, seed in load_samples(aggregate_root):
        path = aggregate_root / "prompt_seed_aggregates" / f"{prompt_id}_seed{seed}.json"
        payload = read_json(path)
        rows = {}
        for row in payload.get("oracle_gain_records", []):
            chunk_idx = int(row["chunk_idx"])
            rows[chunk_idx] = {
                "gain": float(row["gain"]),
                "gain_norm": float(row["gain_norm"]),
                "oracle_rank": int(row["rank"]),
            }
        if len(rows) != 40:
            raise ValueError(f"Expected 40 oracle gain records in {path}, got {len(rows)}")
        out[(prompt_id, seed)] = rows
    return out


def find_all_fast_dir(input_root: Path, prompt_id: str, seed: int) -> Path:
    shard = input_root / f"long_rollout_discovery_p{int(prompt_id.split('_')[-1]) - 1:03d}_s{seed}"
    path = shard / "rollouts" / prompt_id / str(seed) / "all_fast"
    if not path.exists():
        raise FileNotFoundError(f"Missing all_fast rollout dir: {path}")
    return path


def to_float_video(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return video.to(torch.float32) / 255.0
    return video.to(torch.float32).clamp(0.0, 1.0)


def grayscale(video: torch.Tensor) -> torch.Tensor:
    # Input [T,H,W,C], output [T,1,H,W].
    chw = video.permute(0, 3, 1, 2).contiguous()
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=chw.dtype, device=chw.device).view(1, 3, 1, 1)
    return (chw * weights).sum(dim=1, keepdim=True)


def chunk_slice(video: torch.Tensor, start: int, end: int, pad_prev: bool) -> torch.Tensor:
    start = max(0, int(start))
    end = min(int(end), int(video.shape[0]))
    if pad_prev and start > 0:
        start -= 1
    if end <= start:
        end = min(start + 1, int(video.shape[0]))
    return video[start:end]


def suffix_slice(video: torch.Tensor, start: int, horizon: int, pad_prev: bool) -> torch.Tensor:
    return chunk_slice(video, start=start, end=start + max(1, horizon), pad_prev=pad_prev)


def boundary_slice(video: torch.Tensor, boundary_frame: int) -> torch.Tensor:
    boundary_frame = int(boundary_frame)
    if boundary_frame <= 0:
        return chunk_slice(video, start=0, end=2, pad_prev=False)
    start = max(0, boundary_frame - 1)
    end = min(int(video.shape[0]), boundary_frame + 1)
    return chunk_slice(video, start=start, end=end, pad_prev=False)


def frame_l1(frames: torch.Tensor) -> float:
    if frames.shape[0] < 2:
        return float("nan")
    diff = (frames[1:] - frames[:-1]).abs().mean()
    return float(diff.item())


def edge_l1(frames: torch.Tensor) -> float:
    if frames.shape[0] < 2:
        return float("nan")
    gray = grayscale(frames)
    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    edge_x = (grad_x[1:] - grad_x[:-1]).abs().mean()
    edge_y = (grad_y[1:] - grad_y[:-1]).abs().mean()
    return float((0.5 * (edge_x + edge_y)).item())


def _resize_for_flow(gray: torch.Tensor, max_side: int) -> np.ndarray:
    # Input [T,1,H,W] float [0,1], output uint8 [T,h,w].
    h, w = int(gray.shape[-2]), int(gray.shape[-1])
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        new_h = max(2, int(round(h * scale)))
        new_w = max(2, int(round(w * scale)))
        gray = F.interpolate(gray, size=(new_h, new_w), mode="bilinear", align_corners=False)
    arr = (gray[:, 0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def flow_mag(frames: torch.Tensor, max_side: int) -> float:
    if cv2 is None or frames.shape[0] < 2:
        return float("nan")
    arr = _resize_for_flow(grayscale(frames), max_side=max_side)
    mags: List[float] = []
    for i in range(arr.shape[0] - 1):
        flow = cv2.calcOpticalFlowFarneback(
            arr[i],
            arr[i + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(np.mean(mag)))
    return float(np.mean(mags)) if mags else float("nan")


def compute_sample_rows(
    input_root: Path,
    aggregate_root: Path,
    prompt_id: str,
    seed: int,
    flow_max_side: int,
    pad_prev_frame: bool,
) -> List[Dict[str, Any]]:
    gains = load_gain_records(aggregate_root)[(prompt_id, seed)]
    rollout_dir = find_all_fast_dir(input_root, prompt_id, seed)
    video_path = rollout_dir / "full.mp4"
    boundaries_path = rollout_dir / "chunk_boundaries.json"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing all_fast video: {video_path}")
    boundaries = read_json(boundaries_path)

    video, _, info = read_video(str(video_path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Decoded empty video: {video_path}")
    video_f = to_float_video(video)
    suffix_horizon = int(boundaries.get("suffix_window_visible_nominal", 128))

    rows: List[Dict[str, Any]] = []
    prev_local: Optional[Dict[str, float]] = None
    for chunk in boundaries["chunks"]:
        chunk_idx = int(chunk["chunk_idx"])
        local_frames = chunk_slice(
            video_f,
            start=int(chunk["visible_start"]),
            end=int(chunk["visible_end"]),
            pad_prev=pad_prev_frame,
        )
        suffix_frames = suffix_slice(
            video_f,
            start=int(chunk["visible_start"]),
            horizon=suffix_horizon,
            pad_prev=pad_prev_frame,
        )
        boundary_frames = boundary_slice(video_f, boundary_frame=int(chunk["visible_start"]))

        local = {
            "frame_l1": frame_l1(local_frames),
            "edge_l1": edge_l1(local_frames),
            "flow_mag": flow_mag(local_frames, max_side=flow_max_side),
        }
        suffix = {
            "suffix_frame_l1": frame_l1(suffix_frames),
            "suffix_edge_l1": edge_l1(suffix_frames),
            "suffix_flow_mag": flow_mag(suffix_frames, max_side=flow_max_side),
        }
        boundary = {
            "boundary_frame_l1": frame_l1(boundary_frames),
            "boundary_edge_l1": edge_l1(boundary_frames),
            "boundary_flow_mag": flow_mag(boundary_frames, max_side=flow_max_side),
        }
        accel = {
            "frame_l1_accel": (
                local["frame_l1"] - prev_local["frame_l1"] if prev_local is not None else float("nan")
            ),
            "edge_l1_accel": (
                local["edge_l1"] - prev_local["edge_l1"] if prev_local is not None else float("nan")
            ),
            "flow_mag_accel": (
                local["flow_mag"] - prev_local["flow_mag"] if prev_local is not None else float("nan")
            ),
        }
        suffix_minus_local = {
            "suffix_minus_local_frame_l1": suffix["suffix_frame_l1"] - local["frame_l1"],
            "suffix_minus_local_edge_l1": suffix["suffix_edge_l1"] - local["edge_l1"],
            "suffix_minus_local_flow_mag": suffix["suffix_flow_mag"] - local["flow_mag"],
        }
        cheap = {**local, **suffix, **boundary, **accel, **suffix_minus_local}
        target = gains[chunk_idx]
        rows.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "chunk_idx": chunk_idx,
                "visible_start": int(chunk["visible_start"]),
                "visible_end": int(chunk["visible_end"]),
                "suffix_visible_start": int(chunk["visible_start"]),
                "suffix_visible_end": min(int(chunk["visible_start"]) + suffix_horizon, int(video_f.shape[0])),
                "video_path": str(video_path),
                "oracle_gain": target["gain"],
                "oracle_gain_norm": target["gain_norm"],
                "oracle_rank": target["oracle_rank"],
                **cheap,
            }
        )
        prev_local = local
    return rows


def finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x, y = finite_pairs(xs, ys)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x, y = finite_pairs(xs, ys)
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def topk_recall(rows: Sequence[Dict[str, Any]], metric_name: str, k: int) -> Optional[float]:
    valid = [r for r in rows if math.isfinite(float(r.get(metric_name, float("nan"))))]
    if len(valid) < k:
        return None
    oracle_top = {int(r["chunk_idx"]) for r in sorted(valid, key=lambda r: int(r["oracle_rank"]))[:k]}
    pred_top = {int(r["chunk_idx"]) for r in sorted(valid, key=lambda r: float(r[metric_name]), reverse=True)[:k]}
    return float(len(oracle_top & pred_top) / k)


def rank_desc_scores(rows: Sequence[Dict[str, Any]], metric_name: str) -> Dict[int, float]:
    values = []
    for r in rows:
        value = float(r.get(metric_name, float("nan")))
        if math.isfinite(value):
            values.append((int(r["chunk_idx"]), value))
    if not values:
        return {}
    arr = np.asarray([v for _, v in values], dtype=np.float64)
    ranks = average_ranks(arr)
    # Convert ascending ranks to descending "larger is harder" scores.
    max_rank = float(len(values) - 1)
    return {chunk_idx: max_rank - float(rank) for (chunk_idx, _), rank in zip(values, ranks)}


def add_combo_scores(
    rows: Sequence[Dict[str, Any]],
    metric_weights: Sequence[Tuple[str, float]],
    output_name: str,
) -> List[Dict[str, Any]]:
    by_sample: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault((str(row["prompt_id"]), int(row["seed"])), []).append(dict(row))

    out: List[Dict[str, Any]] = []
    for sample_rows in by_sample.values():
        rank_maps = {
            metric: rank_desc_scores(sample_rows, metric)
            for metric, weight in metric_weights
            if weight != 0
        }
        for row in sample_rows:
            chunk_idx = int(row["chunk_idx"])
            score = 0.0
            used = 0
            for metric, weight in metric_weights:
                if weight == 0:
                    continue
                metric_rank = rank_maps.get(metric, {}).get(chunk_idx)
                if metric_rank is None:
                    continue
                score += weight * metric_rank
                used += 1
            new_row = dict(row)
            new_row[output_name] = float(score) if used else float("nan")
            out.append(new_row)
    return out


def mean_topk_for_metric(rows: Sequence[Dict[str, Any]], metric_name: str, topk: int) -> Optional[float]:
    values = []
    for key in sorted({(r["prompt_id"], int(r["seed"])) for r in rows}):
        sample_rows = [r for r in rows if (r["prompt_id"], int(r["seed"])) == key]
        recall = topk_recall(sample_rows, metric_name, topk)
        if recall is not None:
            values.append(recall)
    return float(np.mean(values)) if values else None


def mean_spearman_for_metric(rows: Sequence[Dict[str, Any]], metric_name: str) -> Optional[float]:
    values = []
    for key in sorted({(r["prompt_id"], int(r["seed"])) for r in rows}):
        sample_rows = [r for r in rows if (r["prompt_id"], int(r["seed"])) == key]
        rho = spearman(
            [float(r.get(metric_name, float("nan"))) for r in sample_rows],
            [-float(r["oracle_rank"]) for r in sample_rows],
        )
        if rho is not None:
            values.append(rho)
    return float(np.mean(values)) if values else None


def candidate_weight_sets(features: Sequence[str]) -> List[List[Tuple[str, float]]]:
    candidates: List[List[Tuple[str, float]]] = []
    for metric in features:
        candidates.append([(metric, 1.0)])
    groups = [
        ("local_motion", ["frame_l1", "edge_l1", "flow_mag"]),
        ("suffix_motion", ["suffix_frame_l1", "suffix_edge_l1", "suffix_flow_mag"]),
        ("boundary_motion", ["boundary_frame_l1", "boundary_edge_l1", "boundary_flow_mag"]),
        ("acceleration", ["frame_l1_accel", "edge_l1_accel", "flow_mag_accel"]),
        (
            "suffix_minus_local",
            ["suffix_minus_local_frame_l1", "suffix_minus_local_edge_l1", "suffix_minus_local_flow_mag"],
        ),
        ("all_motion", list(features)),
    ]
    for _, group in groups:
        candidates.append([(metric, 1.0) for metric in group])
    # A few fixed, interpretable mixtures. This is a tiny grid over hand features,
    # not a learned visual encoder.
    candidates.extend(
        [
            [("frame_l1", 1.0), ("suffix_frame_l1", 1.0), ("boundary_frame_l1", 1.0)],
            [("edge_l1", 1.0), ("suffix_edge_l1", 1.0), ("boundary_edge_l1", 1.0)],
            [("flow_mag", 1.0), ("suffix_flow_mag", 1.0), ("boundary_flow_mag", 1.0)],
            [("suffix_frame_l1", 2.0), ("boundary_frame_l1", 1.0), ("frame_l1_accel", 1.0)],
            [("suffix_edge_l1", 2.0), ("boundary_edge_l1", 1.0), ("edge_l1_accel", 1.0)],
            [("suffix_flow_mag", 2.0), ("boundary_flow_mag", 1.0), ("flow_mag_accel", 1.0)],
        ]
    )
    return candidates


def combo_name(metric_weights: Sequence[Tuple[str, float]]) -> str:
    return "combo__" + "__".join(f"{metric}x{weight:g}" for metric, weight in metric_weights)


def leave_one_prompt_out_combos(
    rows: Sequence[Dict[str, Any]],
    features: Sequence[str],
    topk: int,
) -> Dict[str, Any]:
    prompts = sorted({str(r["prompt_id"]) for r in rows})
    candidates = candidate_weight_sets(features)
    fold_rows: List[Dict[str, Any]] = []
    chosen_names: List[str] = []
    for prompt in prompts:
        train = [r for r in rows if str(r["prompt_id"]) != prompt]
        test = [r for r in rows if str(r["prompt_id"]) == prompt]
        scored_candidates = []
        for weights in candidates:
            name = combo_name(weights)
            train_scored = add_combo_scores(train, weights, name)
            train_topk = mean_topk_for_metric(train_scored, name, topk)
            train_spearman = mean_spearman_for_metric(train_scored, name)
            scored_candidates.append(
                (
                    -1.0 if train_topk is None else train_topk,
                    -1.0 if train_spearman is None else train_spearman,
                    name,
                    weights,
                )
            )
        scored_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        train_topk, train_spearman, chosen_name, chosen_weights = scored_candidates[0]
        chosen_names.append(chosen_name)
        test_scored = add_combo_scores(test, chosen_weights, "lopo_rank_combo")
        test_topk = mean_topk_for_metric(test_scored, "lopo_rank_combo", topk)
        test_spearman = mean_spearman_for_metric(test_scored, "lopo_rank_combo")
        fold_rows.append(
            {
                "heldout_prompt_id": prompt,
                "chosen_combo": chosen_name,
                "chosen_weights": [{"metric": m, "weight": w} for m, w in chosen_weights],
                "train_mean_topk_recall": train_topk,
                "train_mean_spearman": train_spearman,
                "test_mean_topk_recall": test_topk,
                "test_mean_spearman": test_spearman,
            }
        )

    test_topks = [float(r["test_mean_topk_recall"]) for r in fold_rows if r["test_mean_topk_recall"] is not None]
    test_rhos = [float(r["test_mean_spearman"]) for r in fold_rows if r["test_mean_spearman"] is not None]
    combo_counts: Dict[str, int] = {}
    for name in chosen_names:
        combo_counts[name] = combo_counts.get(name, 0) + 1
    return {
        "features": list(features),
        "candidate_count": len(candidates),
        "folds": fold_rows,
        "chosen_combo_counts": dict(sorted(combo_counts.items(), key=lambda item: (-item[1], item[0]))),
        f"mean_lopo_top{topk}_recall": float(np.mean(test_topks)) if test_topks else None,
        f"median_lopo_top{topk}_recall": float(np.median(test_topks)) if test_topks else None,
        "mean_lopo_spearman": float(np.mean(test_rhos)) if test_rhos else None,
        "median_lopo_spearman": float(np.median(test_rhos)) if test_rhos else None,
    }


def summarize(rows: Sequence[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_rows": len(rows),
        "num_prompt_seeds": len({(r["prompt_id"], int(r["seed"])) for r in rows}),
        "metrics": {},
    }
    for metric in METRIC_NAMES:
        metric_values = [float(r.get(metric, float("nan"))) for r in rows]
        gains = [float(r["oracle_gain"]) for r in rows]
        gain_norms = [float(r["oracle_gain_norm"]) for r in rows]
        ranks_for_positive_corr = [-float(r["oracle_rank"]) for r in rows]
        per_sample_spearman = []
        per_sample_topk = []
        for key in sorted({(r["prompt_id"], int(r["seed"])) for r in rows}):
            sample_rows = [r for r in rows if (r["prompt_id"], int(r["seed"])) == key]
            rho = spearman(
                [float(r.get(metric, float("nan"))) for r in sample_rows],
                [-float(r["oracle_rank"]) for r in sample_rows],
            )
            if rho is not None:
                per_sample_spearman.append(rho)
            recall = topk_recall(sample_rows, metric, topk)
            if recall is not None:
                per_sample_topk.append(recall)
        summary["metrics"][metric] = {
            "global_spearman_vs_gain": spearman(metric_values, gains),
            "global_spearman_vs_gain_norm": spearman(metric_values, gain_norms),
            "global_spearman_vs_negative_rank": spearman(metric_values, ranks_for_positive_corr),
            "global_pearson_vs_gain": pearson(metric_values, gains),
            "mean_within_sample_spearman_vs_negative_rank": (
                float(np.mean(per_sample_spearman)) if per_sample_spearman else None
            ),
            "median_within_sample_spearman_vs_negative_rank": (
                float(np.median(per_sample_spearman)) if per_sample_spearman else None
            ),
            f"mean_top{topk}_recall": float(np.mean(per_sample_topk)) if per_sample_topk else None,
            f"median_top{topk}_recall": float(np.median(per_sample_topk)) if per_sample_topk else None,
            "finite_count": int(np.isfinite(np.asarray(metric_values, dtype=np.float64)).sum()),
        }
    summary["leave_one_prompt_out_rank_combo"] = leave_one_prompt_out_combos(
        rows=rows,
        features=BASE_COMBO_FEATURES,
        topk=topk,
    )
    return summary


def cmd_compute(args: argparse.Namespace) -> None:
    aggregate_root = Path(args.aggregate_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    samples = load_samples(aggregate_root)
    if args.task_index < 0 or args.task_index >= len(samples):
        raise IndexError(f"task_index must be in [0,{len(samples)-1}], got {args.task_index}")
    prompt_id, seed = samples[args.task_index]
    shard_path = output_root / "shards" / f"{args.task_index:03d}_{prompt_id}_seed{seed}.jsonl"
    done_path = output_root / "shards" / f"{args.task_index:03d}_{prompt_id}_seed{seed}.done"
    if args.resume and shard_path.exists() and shard_path.stat().st_size > 0 and done_path.exists():
        print(f"[cheap-metrics] existing shard found, skip: {shard_path}")
        return
    rows = compute_sample_rows(
        input_root=input_root,
        aggregate_root=aggregate_root,
        prompt_id=prompt_id,
        seed=seed,
        flow_max_side=args.flow_max_side,
        pad_prev_frame=not args.no_pad_prev_frame,
    )
    append_jsonl(shard_path, rows)
    done_path.write_text("done\n", encoding="utf-8")
    print(f"[cheap-metrics] wrote {len(rows)} rows: {shard_path}")


def cmd_merge(args: argparse.Namespace) -> None:
    aggregate_root = Path(args.aggregate_root)
    output_root = Path(args.output_root)
    expected = len(load_samples(aggregate_root))
    shard_dir = output_root / "shards"
    if args.wait:
        import time

        deadline = time.time() + args.wait_timeout_sec
        while time.time() < deadline:
            done_count = len(list(shard_dir.glob("*.done")))
            if done_count >= expected:
                break
            print(f"[cheap-metrics] waiting for shards: {done_count}/{expected}")
            time.sleep(args.wait_poll_sec)
    done_files = sorted(shard_dir.glob("*.done"))
    if len(done_files) != expected:
        raise RuntimeError(f"Expected {expected} done files, found {len(done_files)} in {shard_dir}")

    rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    rows_path = output_root / "cheap_chunk_metrics.jsonl"
    append_jsonl(rows_path, rows)
    summary = summarize(rows, topk=args.topk)
    summary.update(
        {
            "input_root": args.input_root,
            "aggregate_root": args.aggregate_root,
            "output_root": args.output_root,
            "metric_names": list(METRIC_NAMES),
            "flow_available": cv2 is not None,
            "flow_max_side": args.flow_max_side,
            "topk": args.topk,
        }
    )
    write_json(output_root / "summary.json", summary)
    print(f"[cheap-metrics] merged rows: {rows_path}")
    print(f"[cheap-metrics] summary: {output_root / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", default="outputs/long_keyframe_budget")
    parser.add_argument("--aggregate_root", default="outputs/long_keyframe_budget/long_rollout_discovery/aggregates")
    parser.add_argument(
        "--output_root",
        default="outputs/long_keyframe_budget/cheap_chunk_metrics/suffix_boundary_accel_rank_combo",
    )
    parser.add_argument("--flow_max_side", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)
    sub = parser.add_subparsers(dest="cmd", required=True)

    compute = sub.add_parser("compute", help="Compute one prompt/seed shard.")
    compute.add_argument("--task_index", type=int, required=True)
    compute.add_argument("--resume", action="store_true")
    compute.add_argument("--no_pad_prev_frame", action="store_true")
    compute.set_defaults(func=cmd_compute)

    merge = sub.add_parser("merge", help="Merge shard rows and write correlations.")
    merge.add_argument("--wait", action="store_true")
    merge.add_argument("--wait_timeout_sec", type=int, default=21600)
    merge.add_argument("--wait_poll_sec", type=int, default=60)
    merge.set_defaults(func=cmd_merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
