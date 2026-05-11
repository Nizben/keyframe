"""Exhaustive sweep and oracle construction for keyframe-budget experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .runner import RolloutResult, RolloutSpec, run_rollout
from .schedules import single_heavy_at_i


@dataclass(frozen=True)
class GainRecord:
    chunk_idx: int
    gain: float
    gain_norm: float
    rank: int


def _safe_gap(heavy_score: float, fast_score: float, eps: float) -> float:
    gap = heavy_score - fast_score
    if abs(gap) < eps:
        return eps if gap >= 0 else -eps
    return gap


def build_oracle_from_single_heavy_results(
    result_dict: Dict[str, RolloutResult],
    eps: float = 1e-8,
) -> List[GainRecord]:
    if "all_fast" not in result_dict:
        raise KeyError("result_dict must contain 'all_fast'.")
    if "all_heavy" not in result_dict:
        raise KeyError("result_dict must contain 'all_heavy'.")

    fast = result_dict["all_fast"]
    heavy = result_dict["all_heavy"]
    if fast.score is None or heavy.score is None:
        raise ValueError("all_fast and all_heavy must include scores.")

    denom = _safe_gap(heavy.score, fast.score, eps=eps)
    rows: List[Tuple[int, float, float]] = []
    for schedule_name, result in result_dict.items():
        if not schedule_name.startswith("single_heavy_"):
            continue
        if result.score is None:
            continue
        suffix = schedule_name.replace("single_heavy_", "", 1)
        chunk_idx = int(suffix)
        gain = result.score - fast.score
        rows.append((chunk_idx, gain, gain / denom))

    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    return [
        GainRecord(chunk_idx=chunk_idx, gain=gain, gain_norm=gain_norm, rank=rank)
        for rank, (chunk_idx, gain, gain_norm) in enumerate(rows)
    ]


def exhaustive_single_heavy_sweep(
    base_spec: RolloutSpec,
    fast_steps: int,
    heavy_steps: int,
    single_heavy_indices: Optional[Sequence[int]] = None,
    force: bool = False,
    pipeline=None,
    loaded_config=None,
    device=None,
) -> Dict[str, RolloutResult]:
    total_chunks = len(base_spec.steps_per_chunk)
    if total_chunks <= 0:
        raise ValueError("base_spec.steps_per_chunk must not be empty.")
    if single_heavy_indices is None:
        heavy_indices = list(range(total_chunks))
    else:
        heavy_indices = []
        seen = set()
        for idx in single_heavy_indices:
            idx = int(idx)
            if idx in seen:
                continue
            if idx < 0 or idx >= total_chunks:
                raise IndexError(f"single_heavy index out of range: {idx} for total_chunks={total_chunks}")
            heavy_indices.append(idx)
            seen.add(idx)

    out: Dict[str, RolloutResult] = {}
    all_fast_spec = RolloutSpec(
        **{**base_spec.__dict__, "schedule_name": "all_fast", "steps_per_chunk": [fast_steps] * total_chunks}
    )
    all_heavy_spec = RolloutSpec(
        **{**base_spec.__dict__, "schedule_name": "all_heavy", "steps_per_chunk": [heavy_steps] * total_chunks}
    )
    out["all_fast"] = run_rollout(
        all_fast_spec,
        force=force,
        pipeline=pipeline,
        loaded_config=loaded_config,
        device=device,
    )
    out["all_heavy"] = run_rollout(
        all_heavy_spec,
        force=force,
        pipeline=pipeline,
        loaded_config=loaded_config,
        device=device,
    )

    for chunk_idx in heavy_indices:
        schedule = single_heavy_at_i(
            total_chunks=total_chunks,
            fast_steps=fast_steps,
            heavy_steps=heavy_steps,
            heavy_chunk_idx=chunk_idx,
        )
        schedule_name = f"single_heavy_{chunk_idx:02d}"
        sweep_spec = RolloutSpec(
            **{**base_spec.__dict__, "schedule_name": schedule_name, "steps_per_chunk": schedule}
        )
        out[schedule_name] = run_rollout(
            sweep_spec,
            force=force,
            pipeline=pipeline,
            loaded_config=loaded_config,
            device=device,
        )

    return out
