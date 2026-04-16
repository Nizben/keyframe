"""Metrics utilities for keyframe-budget experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

import torch
from torchvision.io import read_video


class MetricAdapter(Protocol):
    """Pluggable adapter for external benchmark suites (VBench, VisionReward, etc.)."""

    name: str

    def evaluate(self, video: torch.Tensor, fps: float) -> Dict[str, float]:
        """Return adapter-specific scalar metrics."""


@dataclass(frozen=True)
class VideoMetrics:
    score: float
    values: Dict[str, float]


def _ensure_video_exists(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")


def _to_float(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return video.to(torch.float32) / 255.0
    return video.to(torch.float32).clamp(0.0, 1.0)


def _to_chw(video: torch.Tensor) -> torch.Tensor:
    # read_video returns [T, H, W, C]
    return video.permute(0, 3, 1, 2).contiguous()


def compute_proxy_metrics(video: torch.Tensor) -> Dict[str, float]:
    """
    Compute lightweight in-repo proxy metrics.

    These are not replacements for VBench/Dynamic Degree, but they provide:
    - deterministic local scores for gain/recovery plumbing,
    - a default Score(.) implementation before external adapters are available.
    """
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor [T,H,W,C], got shape={tuple(video.shape)}")
    if video.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute motion-aware metrics.")

    video = _to_float(video)
    chw = _to_chw(video)

    # Temporal motion energy (frame-to-frame absolute difference).
    temporal_diff = (chw[1:] - chw[:-1]).abs().mean()

    # Spatial sharpness proxy (mean gradient magnitude on grayscale).
    gray = chw.mean(dim=1, keepdim=True)  # [T,1,H,W]
    grad_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean()
    grad_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean()
    sharpness = 0.5 * (grad_x + grad_y)

    # Mild regularizer for exposure consistency.
    brightness_std = gray.mean(dim=(-2, -1)).std()

    # Composite score for rollout comparisons.
    score = float((0.60 * temporal_diff + 0.35 * sharpness - 0.05 * brightness_std).item())

    return {
        "score": score,
        "proxy_motion_l1": float(temporal_diff.item()),
        "proxy_sharpness": float(sharpness.item()),
        "proxy_brightness_std": float(brightness_std.item()),
    }


def evaluate_video_metrics(
    video_path: str | Path,
    suffix_start_frame: int = 0,
    adapters: Optional[Iterable[MetricAdapter]] = None,
) -> VideoMetrics:
    video_path = Path(video_path)
    _ensure_video_exists(video_path)
    video, _, info = read_video(str(video_path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Decoded empty video: {video_path}")

    if suffix_start_frame < 0:
        raise ValueError(f"suffix_start_frame must be >= 0, got {suffix_start_frame}")
    if suffix_start_frame >= video.shape[0]:
        raise ValueError(
            "suffix_start_frame must be smaller than frame count, "
            f"got {suffix_start_frame} with {video.shape[0]} frames."
        )
    if suffix_start_frame > 0:
        video = video[suffix_start_frame:]

    metrics = compute_proxy_metrics(video)
    fps = float(info.get("video_fps", 16.0))

    adapter_failures: List[str] = []
    if adapters:
        for adapter in adapters:
            try:
                adapter_metrics = adapter.evaluate(video=video, fps=fps)
                for key, value in adapter_metrics.items():
                    metrics[f"{adapter.name}.{key}"] = float(value)
            except Exception as exc:  # pragma: no cover - external adapters are optional.
                adapter_failures.append(f"{adapter.name}: {exc}")

    if adapter_failures:
        metrics["adapter_failure_count"] = float(len(adapter_failures))

    return VideoMetrics(score=float(metrics["score"]), values=metrics)

