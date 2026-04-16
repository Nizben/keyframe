"""Keyframe-budget experiment package for v3.5 validation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .schedules import (
    all_fast,
    all_heavy,
    single_heavy_at_i,
    uniform_top_m,
    random_top_m,
    prefix_top_m,
    oracle_top_m,
    compute_mixed_heavy_count,
)

if TYPE_CHECKING:
    from .runner import RolloutResult, RolloutSpec

__all__ = [
    "RolloutSpec",
    "RolloutResult",
    "run_rollout",
    "all_fast",
    "all_heavy",
    "single_heavy_at_i",
    "uniform_top_m",
    "random_top_m",
    "prefix_top_m",
    "oracle_top_m",
    "compute_mixed_heavy_count",
]


def __getattr__(name: str) -> Any:
    """Lazily import GPU-heavy runner symbols only when needed."""
    if name in {"RolloutSpec", "RolloutResult", "run_rollout"}:
        from .runner import RolloutResult, RolloutSpec, run_rollout

        lazy_exports = {
            "RolloutSpec": RolloutSpec,
            "RolloutResult": RolloutResult,
            "run_rollout": run_rollout,
        }
        return lazy_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
