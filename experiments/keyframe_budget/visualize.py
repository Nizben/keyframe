"""Plotting utilities for keyframe-budget experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def generate_gain_heatmap(gain_maps_jsonl: str | Path, out_path: str | Path) -> Path:
    rows = _read_jsonl(Path(gain_maps_jsonl))
    if not rows:
        raise ValueError(f"No gain map rows found in {gain_maps_jsonl}")

    key_to_values: Dict[Tuple[str, int], Dict[int, float]] = {}
    max_chunk_idx = 0
    for row in rows:
        key = (row["prompt_id"], int(row["seed"]))
        chunk_idx = int(row["chunk_idx"])
        key_to_values.setdefault(key, {})[chunk_idx] = float(row["gain_norm"])
        max_chunk_idx = max(max_chunk_idx, chunk_idx)

    matrix = []
    labels = []
    for (prompt_id, seed), value_map in sorted(key_to_values.items()):
        labels.append(f"{prompt_id}:{seed}")
        matrix.append([value_map.get(idx, 0.0) for idx in range(max_chunk_idx + 1)])
    matrix_np = np.array(matrix, dtype=np.float32)

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(8, 0.35 * (max_chunk_idx + 1)), max(4, 0.35 * len(labels))))
    plt.imshow(matrix_np, aspect="auto", interpolation="nearest")
    plt.colorbar(label="g_i_norm")
    plt.xlabel("Chunk index")
    plt.ylabel("Prompt:Seed")
    plt.title("Gain Heatmap")
    plt.yticks(np.arange(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def generate_recovery_barplot(policy_results_jsonl: str | Path, out_path: str | Path) -> Path:
    rows = _read_jsonl(Path(policy_results_jsonl))
    if not rows:
        raise ValueError(f"No policy rows found in {policy_results_jsonl}")

    by_policy: Dict[str, List[float]] = {}
    for row in rows:
        policy = row["policy_name"]
        recovery = row.get("recovery")
        if recovery is None:
            continue
        by_policy.setdefault(policy, []).append(float(recovery))

    labels = sorted(by_policy)
    means = [float(np.mean(by_policy[name])) for name in labels]
    stds = [float(np.std(by_policy[name])) for name in labels]

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Recovery ratio")
    plt.title("Policy Recovery Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def generate_best_vs_median_gain_plot(gain_maps_jsonl: str | Path, out_path: str | Path) -> Path:
    rows = _read_jsonl(Path(gain_maps_jsonl))
    if not rows:
        raise ValueError(f"No gain map rows found in {gain_maps_jsonl}")

    by_video: Dict[Tuple[str, int], List[float]] = {}
    for row in rows:
        key = (row["prompt_id"], int(row["seed"]))
        by_video.setdefault(key, []).append(float(row["gain"]))

    best_vals = []
    median_vals = []
    labels = []
    for key, gains in sorted(by_video.items()):
        labels.append(f"{key[0]}:{key[1]}")
        best_vals.append(float(np.max(gains)))
        median_vals.append(float(np.median(gains)))

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(10, 0.35 * len(labels)), 4))
    x = np.arange(len(labels))
    width = 0.4
    plt.bar(x - width / 2, best_vals, width=width, label="Best chunk gain")
    plt.bar(x + width / 2, median_vals, width=width, label="Median chunk gain")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Best vs Median Chunk Gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def generate_oracle_position_histogram(gain_maps_jsonl: str | Path, out_path: str | Path) -> Path:
    rows = _read_jsonl(Path(gain_maps_jsonl))
    if not rows:
        raise ValueError(f"No gain map rows found in {gain_maps_jsonl}")

    by_video_best_chunk: List[int] = []
    by_video: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for row in rows:
        key = (row["prompt_id"], int(row["seed"]))
        by_video.setdefault(key, []).append((int(row["chunk_idx"]), float(row["gain"])))

    for entries in by_video.values():
        best = sorted(entries, key=lambda x: x[1], reverse=True)[0]
        by_video_best_chunk.append(best[0])

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    plt.hist(by_video_best_chunk, bins=max(by_video_best_chunk) + 1, edgecolor="black")
    plt.xlabel("Best chunk index")
    plt.ylabel("Count")
    plt.title("Oracle Chunk Position Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def generate_suffix_gain_curves(
    suffix_payloads: Sequence[Mapping[str, object]],
    out_path: str | Path,
) -> Path:
    if not suffix_payloads:
        raise ValueError("suffix_payloads cannot be empty.")

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 4))

    for payload in suffix_payloads:
        prompt_id = str(payload["prompt_id"])
        seed = int(payload["seed"])
        xs = list(payload["chunk_indices"])
        ys = list(payload["suffix_gains"])
        plt.plot(xs, ys, alpha=0.5, label=f"{prompt_id}:{seed}")

    plt.xlabel("Intervention chunk index")
    plt.ylabel("Suffix gain")
    plt.title("Suffix Gain Curves")
    if len(suffix_payloads) <= 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path

