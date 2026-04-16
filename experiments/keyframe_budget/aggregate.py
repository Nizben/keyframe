"""Aggregation utilities for keyframe-budget experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .oracle import GainRecord
from .runner import RolloutResult


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def recovery_ratio(policy_score: float, all_fast_score: float, all_heavy_score: float, eps: float = 1e-8) -> float:
    return (policy_score - all_fast_score) / (all_heavy_score - all_fast_score + eps)


def gini_coefficient(values: Sequence[float]) -> float:
    positives = [max(v, 0.0) for v in values]
    if not positives:
        return 0.0
    total = sum(positives)
    if total <= 0:
        return 0.0
    sorted_vals = sorted(positives)
    n = len(sorted_vals)
    weighted = 0.0
    for i, x in enumerate(sorted_vals, start=1):
        weighted += (2 * i - n - 1) * x
    return weighted / (n * total)


def gain_concentration_stats(gains: Sequence[float]) -> Dict[str, float]:
    if not gains:
        raise ValueError("gains must not be empty.")
    sorted_desc = sorted(gains, reverse=True)
    top3 = sorted_desc[:3]
    top3_mass = sum(top3)
    return {
        "max_gain": float(max(gains)),
        "mean_gain": float(sum(gains) / len(gains)),
        "median_gain": float(median(gains)),
        "top3_gain_mass": float(top3_mass),
        "gini_positive_gain": float(gini_coefficient(gains)),
        "rank_gap_best_minus_median": float(sorted_desc[0] - median(gains)),
    }


def aggregate_prompt_seed_results(
    experiment_name: str,
    prompt_id: str,
    seed: int,
    rollout_results: Mapping[str, RolloutResult],
    oracle_gain_records: Sequence[GainRecord],
    suffix_scores: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    if "all_fast" not in rollout_results or "all_heavy" not in rollout_results:
        raise KeyError("rollout_results must include all_fast and all_heavy.")
    all_fast = rollout_results["all_fast"]
    all_heavy = rollout_results["all_heavy"]
    if all_fast.score is None or all_heavy.score is None:
        raise ValueError("all_fast and all_heavy must have valid scores.")

    policy_scores = {
        name: float(result.score)
        for name, result in rollout_results.items()
        if result.score is not None
    }
    policy_recovery = {
        name: recovery_ratio(
            policy_score=score,
            all_fast_score=float(all_fast.score),
            all_heavy_score=float(all_heavy.score),
        )
        for name, score in policy_scores.items()
    }

    gains = [row.gain for row in oracle_gain_records]
    gain_norms = [row.gain_norm for row in oracle_gain_records]
    concentration = gain_concentration_stats(gains) if gains else {}

    return {
        "experiment_name": experiment_name,
        "prompt_id": prompt_id,
        "seed": seed,
        "all_fast_score": float(all_fast.score),
        "all_heavy_score": float(all_heavy.score),
        "policy_scores": policy_scores,
        "policy_recovery": policy_recovery,
        "oracle_gain_records": [asdict(row) for row in oracle_gain_records],
        "gain_concentration": concentration,
        "gain_norm_values": gain_norms,
        "suffix_scores": dict(suffix_scores or {}),
    }


def save_prompt_seed_aggregate(
    aggregate_root: str | Path,
    aggregate_payload: Dict[str, Any],
) -> Path:
    root = Path(aggregate_root)
    out_path = root / "prompt_seed_aggregates" / (
        f"{aggregate_payload['prompt_id']}_seed{aggregate_payload['seed']}.json"
    )
    _write_json(out_path, aggregate_payload)
    return out_path


def append_gain_map_jsonl(
    aggregate_root: str | Path,
    experiment_name: str,
    prompt_id: str,
    seed: int,
    oracle_gain_records: Sequence[GainRecord],
) -> Path:
    path = Path(aggregate_root) / "gain_maps.jsonl"
    rows = [
        {
            "experiment_name": experiment_name,
            "prompt_id": prompt_id,
            "seed": seed,
            "chunk_idx": row.chunk_idx,
            "gain": row.gain,
            "gain_norm": row.gain_norm,
            "rank": row.rank,
        }
        for row in oracle_gain_records
    ]
    _append_jsonl(path, rows)
    return path


def append_policy_results_jsonl(
    aggregate_root: str | Path,
    aggregate_payload: Mapping[str, Any],
) -> Path:
    path = Path(aggregate_root) / "policy_results.jsonl"
    rows = []
    for policy_name, score in aggregate_payload["policy_scores"].items():
        rows.append(
            {
                "experiment_name": aggregate_payload["experiment_name"],
                "prompt_id": aggregate_payload["prompt_id"],
                "seed": aggregate_payload["seed"],
                "policy_name": policy_name,
                "score": score,
                "recovery": aggregate_payload["policy_recovery"].get(policy_name),
            }
        )
    _append_jsonl(path, rows)
    return path


def load_oracle_source(
    output_root: str | Path,
    oracle_source_experiment: str,
) -> Dict[str, List[int]]:
    """
    Load oracle chunk rankings from previous discovery split.

    Returns:
        mapping[(prompt_id, seed)] -> ranked chunk index list
        represented as {"prompt_id::seed": [idx0, idx1, ...]}
    """
    root = Path(output_root)
    aggregate_dir = root / oracle_source_experiment / "aggregates" / "prompt_seed_aggregates"
    mapping: Dict[str, List[int]] = {}
    if not aggregate_dir.exists():
        raise FileNotFoundError(f"Oracle source directory not found: {aggregate_dir}")

    for path in sorted(aggregate_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        prompt_id = payload["prompt_id"]
        seed = int(payload["seed"])
        records = payload.get("oracle_gain_records", [])
        ranked = [int(row["chunk_idx"]) for row in sorted(records, key=lambda x: x["rank"])]
        mapping[f"{prompt_id}::{seed}"] = ranked
    return mapping

