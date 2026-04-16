"""Step-schedule constructors for keyframe-budget experiments."""

from __future__ import annotations

from math import floor
import random
from typing import Dict, Iterable, List, Optional, Sequence


def _ensure_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _validate_steps(fast_steps: int, heavy_steps: int) -> None:
    _ensure_positive_int(fast_steps, "fast_steps")
    _ensure_positive_int(heavy_steps, "heavy_steps")
    if heavy_steps <= fast_steps:
        raise ValueError(
            f"heavy_steps must be larger than fast_steps, got {heavy_steps} <= {fast_steps}."
        )


def _validate_total_chunks(total_chunks: int) -> None:
    _ensure_positive_int(total_chunks, "total_chunks")


def _materialize_schedule(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    heavy_indices: Iterable[int],
) -> List[int]:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    schedule = [fast_steps] * total_chunks
    for idx in heavy_indices:
        if idx < 0 or idx >= total_chunks:
            raise IndexError(f"chunk index out of range: {idx} for total_chunks={total_chunks}")
        schedule[idx] = heavy_steps
    return schedule


def all_fast(total_chunks: int, fast_steps: int) -> List[int]:
    _validate_total_chunks(total_chunks)
    _ensure_positive_int(fast_steps, "fast_steps")
    return [fast_steps] * total_chunks


def all_heavy(total_chunks: int, heavy_steps: int) -> List[int]:
    _validate_total_chunks(total_chunks)
    _ensure_positive_int(heavy_steps, "heavy_steps")
    return [heavy_steps] * total_chunks


def single_heavy_at_i(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    heavy_chunk_idx: int,
) -> List[int]:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    if heavy_chunk_idx < 0 or heavy_chunk_idx >= total_chunks:
        raise IndexError(
            f"heavy_chunk_idx out of range: {heavy_chunk_idx} for total_chunks={total_chunks}"
        )
    return _materialize_schedule(
        total_chunks=total_chunks,
        fast_steps=fast_steps,
        heavy_steps=heavy_steps,
        heavy_indices=[heavy_chunk_idx],
    )


def compute_mixed_heavy_count(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    target_avg_steps: int,
) -> int:
    """Compute m from roadmap equation m = floor(T * (B - L) / (H - L))."""
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    _ensure_positive_int(target_avg_steps, "target_avg_steps")
    if target_avg_steps < fast_steps or target_avg_steps > heavy_steps:
        raise ValueError(
            "target_avg_steps must be within [fast_steps, heavy_steps], "
            f"got {target_avg_steps} with fast={fast_steps} and heavy={heavy_steps}."
        )
    return floor(total_chunks * (target_avg_steps - fast_steps) / (heavy_steps - fast_steps))


def _dedupe_and_pad_indices(indices: List[int], total_chunks: int, target_count: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for idx in indices:
        if idx not in seen and 0 <= idx < total_chunks:
            out.append(idx)
            seen.add(idx)

    if len(out) == target_count:
        return sorted(out)

    for idx in range(total_chunks):
        if idx in seen:
            continue
        out.append(idx)
        seen.add(idx)
        if len(out) == target_count:
            break
    return sorted(out)


def uniform_top_m_indices(total_chunks: int, m: int) -> List[int]:
    _validate_total_chunks(total_chunks)
    if m < 0 or m > total_chunks:
        raise ValueError(f"m must be within [0, total_chunks], got m={m}.")
    if m == 0:
        return []
    if m == total_chunks:
        return list(range(total_chunks))
    if m == 1:
        return [total_chunks // 2]

    stride = (total_chunks - 1) / (m - 1)
    raw = [round(i * stride) for i in range(m)]
    return _dedupe_and_pad_indices(raw, total_chunks=total_chunks, target_count=m)


def uniform_top_m(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    m: int,
) -> List[int]:
    indices = uniform_top_m_indices(total_chunks=total_chunks, m=m)
    return _materialize_schedule(total_chunks, fast_steps, heavy_steps, indices)


def random_top_m(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    m: int,
    seed: int,
) -> List[int]:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    if m < 0 or m > total_chunks:
        raise ValueError(f"m must be within [0, total_chunks], got m={m}.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(list(range(total_chunks)), k=m))
    return _materialize_schedule(total_chunks, fast_steps, heavy_steps, indices)


def prefix_top_m(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    m: int,
) -> List[int]:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    if m < 0 or m > total_chunks:
        raise ValueError(f"m must be within [0, total_chunks], got m={m}.")
    return _materialize_schedule(
        total_chunks=total_chunks,
        fast_steps=fast_steps,
        heavy_steps=heavy_steps,
        heavy_indices=list(range(m)),
    )


def oracle_top_m(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    m: int,
    ranked_chunk_indices: Sequence[int],
) -> List[int]:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    if m < 0 or m > total_chunks:
        raise ValueError(f"m must be within [0, total_chunks], got m={m}.")
    if not ranked_chunk_indices:
        raise ValueError("ranked_chunk_indices cannot be empty for oracle_top_m.")

    picked: List[int] = []
    seen = set()
    for idx in ranked_chunk_indices:
        if idx in seen:
            continue
        if 0 <= idx < total_chunks:
            picked.append(idx)
            seen.add(idx)
        if len(picked) == m:
            break

    if len(picked) < m:
        raise ValueError(
            "Not enough valid oracle indices to build schedule: "
            f"needed {m}, got {len(picked)}."
        )
    return _materialize_schedule(total_chunks, fast_steps, heavy_steps, picked)


def total_requested_steps(steps_per_chunk: Sequence[int]) -> int:
    if not steps_per_chunk:
        raise ValueError("steps_per_chunk cannot be empty.")
    total = 0
    for i, value in enumerate(steps_per_chunk):
        _ensure_positive_int(value, f"steps_per_chunk[{i}]")
        total += value
    return total


def expected_total_steps_for_mixed_policy(
    total_chunks: int,
    fast_steps: int,
    heavy_steps: int,
    m: int,
) -> int:
    _validate_total_chunks(total_chunks)
    _validate_steps(fast_steps, heavy_steps)
    if m < 0 or m > total_chunks:
        raise ValueError(f"m must be within [0, total_chunks], got m={m}.")
    return (total_chunks - m) * fast_steps + m * heavy_steps


def assert_equal_total_steps(policy_to_schedule: Dict[str, Sequence[int]]) -> None:
    if not policy_to_schedule:
        raise ValueError("policy_to_schedule cannot be empty.")

    totals = {name: total_requested_steps(schedule) for name, schedule in policy_to_schedule.items()}
    unique_totals = set(totals.values())
    if len(unique_totals) != 1:
        raise ValueError(
            "Policies are not compute-matched. "
            + ", ".join(f"{name}={value}" for name, value in sorted(totals.items()))
        )

