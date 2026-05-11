#!/usr/bin/env python3
"""Aggregate long-rollout VBench outputs and generate analysis plots.

This script is intended to run on Jean Zay, where the VBench
``evaluation_results`` directory is visible. It aggregates the two current
long-rollout VBench result trees:

- full videos:
  ``VBench/evaluation_results/keyframe_budget_long_rollout_full36/long_rollout_full_36gpus``
- sparse suffixes:
  ``VBench/evaluation_results/keyframe_budget_long_rollout_suffix36/long_rollout_suffix_sparse_grid_4metrics_36gpus``

VBench stores each result as ``{metric: [aggregate_score, per_video_rows]}``
for the metrics used here. We aggregate from ``per_video_rows[*].video_results``
only. This avoids mixing official per-video scores with top-level aggregates or
metric-specific metadata such as ``video_sim`` and ``cnt_per_video``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FULL_METRICS = (
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
    "aesthetic_quality",
    "subject_consistency",
    "background_consistency",
)

SUFFIX_METRICS = (
    "dynamic_degree",
    "motion_smoothness",
    "aesthetic_quality",
    "subject_consistency",
)


@dataclass(frozen=True)
class EvalRecord:
    split: str
    metric: str
    cohort: str
    chunk: str
    score: float
    score_count: int
    eval_file: str


@dataclass(frozen=True)
class VideoEvalRecord:
    split: str
    metric: str
    cohort: str
    chunk: str
    prompt_id: str
    seed: int
    schedule: str
    heavy_idx: Optional[int]
    suffix_start: Optional[int]
    suffix_tag: Optional[str]
    score: float
    video_path: str
    eval_file: str


@dataclass(frozen=True)
class CohortSummary:
    split: str
    metric: str
    cohort: str
    n_files: int
    n_values: int
    mean_score: float
    median_score: float
    std_score: float
    schedule: str
    heavy_idx: Optional[int]
    suffix_start: Optional[int]
    suffix_tag: Optional[str]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def numeric_values(value: Any) -> List[float]:
    out: List[float] = []
    number = as_float(value)
    if number is not None:
        return [number]
    if isinstance(value, Mapping):
        for child in value.values():
            out.extend(numeric_values(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(numeric_values(child))
    return out


def extract_metric_scores(payload: Any, metric: str) -> List[float]:
    """Compatibility fallback returning numeric values associated with metric.

    Prefer ``extract_video_records`` for all current long-rollout analyses.
    This function is intentionally retained only as a fallback for unexpected
    VBench schemas.
    """

    found: List[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) == metric:
                    found.extend(numeric_values(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return [x for x in found if math.isfinite(x)]


def parse_sample_key_from_video_path(video_path: str) -> Tuple[str, int]:
    """Parse symlink names like ``000000_motion_001__0.mp4``."""

    name = Path(video_path).name
    stem = Path(name).stem
    match = re.match(r"^\d+_(.+)__(-?\d+)$", stem)
    if not match:
        raise ValueError(f"Could not parse prompt/seed from VBench video path: {video_path}")
    return match.group(1), int(match.group(2))


def coerce_video_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return as_float(value)


def extract_video_records(payload: Any, metric: str) -> List[Tuple[str, float]]:
    """Extract ``(video_path, score)`` rows from a VBench metric JSON."""

    if not isinstance(payload, Mapping) or metric not in payload:
        return []
    metric_payload = payload[metric]
    rows: Any = None
    if isinstance(metric_payload, list) and len(metric_payload) >= 2:
        rows = metric_payload[1]
    elif isinstance(metric_payload, Mapping):
        rows = metric_payload.get("video_results") or metric_payload.get("results")

    if not isinstance(rows, list):
        return []

    out: List[Tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        video_path = row.get("video_path")
        score = coerce_video_score(row.get("video_results"))
        if isinstance(video_path, str) and score is not None and math.isfinite(score):
            out.append((video_path, score))
    return out


def parse_context(eval_file: Path, root: Path) -> Tuple[str, str, str]:
    rel = eval_file.relative_to(root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected VBench result path under {root}: {rel}")
    metric = parts[0]
    cohort = parts[1]
    chunk = parts[2]
    return metric, cohort, chunk


def load_records(
    root: Path,
    split: str,
    metrics: Sequence[str],
) -> Tuple[List[EvalRecord], List[VideoEvalRecord], List[str]]:
    records: List[EvalRecord] = []
    video_records: List[VideoEvalRecord] = []
    warnings: List[str] = []
    if not root.exists():
        warnings.append(f"Missing {split} root: {root}")
        return records, video_records, warnings

    metric_set = set(metrics)
    for eval_file in sorted(root.rglob("*_eval_results.json")):
        try:
            metric, cohort, chunk = parse_context(eval_file, root)
        except Exception as exc:
            warnings.append(str(exc))
            continue
        if metric not in metric_set:
            continue

        try:
            payload = read_json(eval_file)
        except Exception as exc:
            warnings.append(f"Could not read {eval_file}: {exc}")
            continue

        schedule, heavy_idx, suffix_start, suffix_tag = parse_cohort(cohort)
        video_rows = extract_video_records(payload, metric)
        for video_path, score in video_rows:
            try:
                prompt_id, seed = parse_sample_key_from_video_path(video_path)
            except Exception as exc:
                warnings.append(str(exc))
                continue
            video_records.append(
                VideoEvalRecord(
                    split=split,
                    metric=metric,
                    cohort=cohort,
                    chunk=chunk,
                    prompt_id=prompt_id,
                    seed=seed,
                    schedule=schedule,
                    heavy_idx=heavy_idx,
                    suffix_start=suffix_start,
                    suffix_tag=suffix_tag,
                    score=float(score),
                    video_path=video_path,
                    eval_file=str(eval_file),
                )
            )

        values = [score for _, score in video_rows]
        if not values:
            values = extract_metric_scores(payload, metric)
        if not values:
            warnings.append(f"No metric-specific numeric values for {metric}: {eval_file}")
            continue

        records.append(
            EvalRecord(
                split=split,
                metric=metric,
                cohort=cohort,
                chunk=chunk,
                score=float(mean(values)),
                score_count=len(values),
                eval_file=str(eval_file),
            )
        )
    return records, video_records, warnings


def parse_cohort(cohort: str) -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
    suffix_start: Optional[int] = None
    suffix_tag: Optional[str] = None
    schedule = cohort

    if "__" in cohort:
        schedule, suffix_tag = cohort.split("__", 1)
        m_suffix = re.match(r"suffix_from_(\d+)$", suffix_tag)
        if m_suffix:
            suffix_start = int(m_suffix.group(1))

    heavy_idx: Optional[int] = None
    m_heavy = re.match(r"single_heavy_(\d+)$", schedule)
    if m_heavy:
        heavy_idx = int(m_heavy.group(1))

    return schedule, heavy_idx, suffix_start, suffix_tag


def summarize(records: Sequence[EvalRecord]) -> List[CohortSummary]:
    grouped: Dict[Tuple[str, str, str], List[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.split, record.metric, record.cohort)].append(record)

    summaries: List[CohortSummary] = []
    for (split, metric, cohort), rows in sorted(grouped.items()):
        scores = [row.score for row in rows]
        schedule, heavy_idx, suffix_start, suffix_tag = parse_cohort(cohort)
        summaries.append(
            CohortSummary(
                split=split,
                metric=metric,
                cohort=cohort,
                n_files=len(rows),
                n_values=sum(row.score_count for row in rows),
                mean_score=float(mean(scores)),
                median_score=float(median(scores)),
                std_score=float(pstdev(scores)) if len(scores) > 1 else 0.0,
                schedule=schedule,
                heavy_idx=heavy_idx,
                suffix_start=suffix_start,
                suffix_tag=suffix_tag,
            )
        )
    return summaries


def summarize_video_records(records: Sequence[VideoEvalRecord]) -> List[CohortSummary]:
    grouped: Dict[Tuple[str, str, str], List[VideoEvalRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.split, record.metric, record.cohort)].append(record)

    summaries: List[CohortSummary] = []
    for (split, metric, cohort), rows in sorted(grouped.items()):
        scores = [row.score for row in rows]
        schedule, heavy_idx, suffix_start, suffix_tag = parse_cohort(cohort)
        summaries.append(
            CohortSummary(
                split=split,
                metric=metric,
                cohort=cohort,
                n_files=len({row.eval_file for row in rows}),
                n_values=len(rows),
                mean_score=float(mean(scores)),
                median_score=float(median(scores)),
                std_score=float(pstdev(scores)) if len(scores) > 1 else 0.0,
                schedule=schedule,
                heavy_idx=heavy_idx,
                suffix_start=suffix_start,
                suffix_tag=suffix_tag,
            )
        )
    return summaries


def normalized_recoveries(summaries: Sequence[CohortSummary]) -> List[Dict[str, Any]]:
    """Normalize single-heavy scores between all_fast and all_heavy per metric."""

    by_metric: Dict[Tuple[str, str], Dict[str, CohortSummary]] = defaultdict(dict)
    for row in summaries:
        if row.split != "full":
            continue
        by_metric[(row.split, row.metric)][row.schedule] = row

    out: List[Dict[str, Any]] = []
    for (_, metric), rows in sorted(by_metric.items()):
        fast = rows.get("all_fast")
        heavy = rows.get("all_heavy")
        if fast is None or heavy is None:
            continue
        denom = heavy.mean_score - fast.mean_score
        for schedule, row in sorted(rows.items()):
            if row.heavy_idx is None:
                continue
            recovery = float("nan") if abs(denom) < 1e-12 else (row.mean_score - fast.mean_score) / denom
            out.append(
                {
                    "metric": metric,
                    "schedule": schedule,
                    "heavy_idx": row.heavy_idx,
                    "mean_score": row.mean_score,
                    "all_fast_score": fast.mean_score,
                    "all_heavy_score": heavy.mean_score,
                    "recovery": recovery,
                }
            )
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_plot_modules() -> Tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Plot generation requires matplotlib and numpy in the active environment. "
            "Run this inside the VBench/analysis environment on Jean Zay or install them."
        ) from exc
    return plt, np


def plot_full_scores(summaries: Sequence[CohortSummary], out_dir: Path) -> None:
    plt, _ = get_plot_modules()
    ensure_dir(out_dir)
    full = [row for row in summaries if row.split == "full"]
    for metric in sorted({row.metric for row in full}):
        rows = [row for row in full if row.metric == metric]
        fast = next((row.mean_score for row in rows if row.schedule == "all_fast"), None)
        heavy = next((row.mean_score for row in rows if row.schedule == "all_heavy"), None)
        singles = sorted((row for row in rows if row.heavy_idx is not None), key=lambda x: x.heavy_idx or -1)
        if not singles:
            continue

        xs = [row.heavy_idx for row in singles]
        ys = [row.mean_score for row in singles]
        plt.figure(figsize=(11, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.6, label="single_heavy_i")
        if fast is not None:
            plt.axhline(fast, linestyle="--", color="tab:red", label="all_fast")
        if heavy is not None:
            plt.axhline(heavy, linestyle="--", color="tab:green", label="all_heavy")
        plt.xlabel("Heavy chunk index")
        plt.ylabel(metric)
        plt.title(f"Full-video VBench: {metric}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"full_{metric}_single_heavy_curve.png", dpi=180)
        plt.close()


def plot_full_recovery(recoveries: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    plt, _ = get_plot_modules()
    ensure_dir(out_dir)
    by_metric: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in recoveries:
        rec = as_float(row.get("recovery"))
        if rec is not None and math.isfinite(rec):
            by_metric[str(row["metric"])].append(row)

    for metric, rows in sorted(by_metric.items()):
        rows = sorted(rows, key=lambda x: int(x["heavy_idx"]))
        plt.figure(figsize=(11, 4))
        plt.plot([r["heavy_idx"] for r in rows], [r["recovery"] for r in rows], marker="o", linewidth=1.6)
        plt.axhline(0.0, linestyle="--", color="tab:red", label="all_fast")
        plt.axhline(1.0, linestyle="--", color="tab:green", label="all_heavy")
        plt.xlabel("Heavy chunk index")
        plt.ylabel("Recovery ratio")
        plt.title(f"Full-video normalized recovery: {metric}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"full_{metric}_recovery_curve.png", dpi=180)
        plt.close()


def plot_suffix_heatmaps(summaries: Sequence[CohortSummary], out_dir: Path) -> None:
    plt, np = get_plot_modules()
    ensure_dir(out_dir)
    suffix = [row for row in summaries if row.split == "suffix" and row.suffix_start is not None]
    for metric in sorted({row.metric for row in suffix}):
        rows = [row for row in suffix if row.metric == metric]
        starts = sorted({int(row.suffix_start) for row in rows if row.suffix_start is not None})
        heavy_indices = sorted({int(row.heavy_idx) for row in rows if row.heavy_idx is not None})
        if not starts or not heavy_indices:
            continue

        value_map = {
            (int(row.heavy_idx), int(row.suffix_start)): row.mean_score
            for row in rows
            if row.heavy_idx is not None and row.suffix_start is not None
        }
        mat = np.full((len(heavy_indices), len(starts)), np.nan, dtype=np.float64)
        for i, heavy_idx in enumerate(heavy_indices):
            for j, start in enumerate(starts):
                if (heavy_idx, start) in value_map:
                    mat[i, j] = value_map[(heavy_idx, start)]

        plt.figure(figsize=(max(7, 0.55 * len(starts)), max(5, 0.35 * len(heavy_indices))))
        im = plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(im, label=metric)
        plt.xticks(np.arange(len(starts)), starts, rotation=45)
        plt.yticks(np.arange(len(heavy_indices)), heavy_indices)
        plt.xlabel("Suffix start chunk")
        plt.ylabel("Heavy chunk index")
        plt.title(f"Sparse suffix VBench heatmap: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"suffix_{metric}_heavy_by_suffix_heatmap.png", dpi=180)
        plt.close()


def plot_suffix_against_baselines(summaries: Sequence[CohortSummary], out_dir: Path) -> None:
    plt, _ = get_plot_modules()
    ensure_dir(out_dir)
    suffix = [row for row in summaries if row.split == "suffix" and row.suffix_start is not None]
    for metric in sorted({row.metric for row in suffix}):
        rows = [row for row in suffix if row.metric == metric]
        starts = sorted({int(row.suffix_start) for row in rows if row.suffix_start is not None})
        if not starts:
            continue

        def series_for(schedule: str) -> List[float]:
            by_start = {row.suffix_start: row.mean_score for row in rows if row.schedule == schedule}
            return [float(by_start.get(start, float("nan"))) for start in starts]

        plt.figure(figsize=(11, 4))
        for schedule, style in (("all_fast", "--"), ("all_heavy", "--")):
            ys = series_for(schedule)
            if any(math.isfinite(y) for y in ys):
                plt.plot(starts, ys, linestyle=style, marker="o", label=schedule)

        for row in sorted((r for r in rows if r.heavy_idx is not None), key=lambda x: (x.heavy_idx, x.suffix_start)):
            pass
        heavy_indices = sorted({row.heavy_idx for row in rows if row.heavy_idx is not None})
        for heavy_idx in heavy_indices:
            by_start = {row.suffix_start: row.mean_score for row in rows if row.heavy_idx == heavy_idx}
            ys = [float(by_start.get(start, float("nan"))) for start in starts]
            plt.plot(starts, ys, alpha=0.35, linewidth=1.0)

        plt.xlabel("Suffix start chunk")
        plt.ylabel(metric)
        plt.title(f"Sparse suffix VBench curves: {metric}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"suffix_{metric}_curves_with_baselines.png", dpi=180)
        plt.close()


def build_summary_payload(
    records: Sequence[EvalRecord],
    video_records: Sequence[VideoEvalRecord],
    summaries: Sequence[CohortSummary],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    by_split_metric: Dict[str, int] = defaultdict(int)
    for row in summaries:
        by_split_metric[f"{row.split}/{row.metric}"] += 1
    return {
        "record_count": len(records),
        "video_record_count": len(video_records),
        "cohort_summary_count": len(summaries),
        "cohort_counts_by_split_metric": dict(sorted(by_split_metric.items())),
        "warnings_count": len(warnings),
        "warnings_preview": list(warnings[:50]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full_root",
        type=Path,
        default=Path(
            "/lustre/fswork/projects/rech/hdy/ujc37rw/VBench/evaluation_results/"
            "keyframe_budget_long_rollout_full36/long_rollout_full_36gpus"
        ),
        help="Root containing full-video VBench metric directories.",
    )
    parser.add_argument(
        "--suffix_root",
        type=Path,
        default=Path(
            "/lustre/fswork/projects/rech/hdy/ujc37rw/VBench/evaluation_results/"
            "keyframe_budget_long_rollout_suffix36/long_rollout_suffix_sparse_grid_4metrics_36gpus"
        ),
        help="Root containing sparse-suffix VBench metric directories.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/long_keyframe_budget/vbench_analysis"),
        help="Directory where CSV summaries and plots are written.",
    )
    parser.add_argument("--skip_full", action="store_true", help="Do not aggregate full-video results.")
    parser.add_argument("--skip_suffix", action="store_true", help="Do not aggregate suffix results.")
    parser.add_argument("--no_plots", action="store_true", help="Write CSV/JSON summaries but skip plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: List[EvalRecord] = []
    video_records: List[VideoEvalRecord] = []
    warnings: List[str] = []

    if not args.skip_full:
        full_records, full_video_records, full_warnings = load_records(args.full_root, "full", FULL_METRICS)
        records.extend(full_records)
        video_records.extend(full_video_records)
        warnings.extend(full_warnings)
    if not args.skip_suffix:
        suffix_records, suffix_video_records, suffix_warnings = load_records(args.suffix_root, "suffix", SUFFIX_METRICS)
        records.extend(suffix_records)
        video_records.extend(suffix_video_records)
        warnings.extend(suffix_warnings)

    summaries = summarize_video_records(video_records) if video_records else summarize(records)
    recoveries = normalized_recoveries(summaries)

    write_csv(
        args.output_root / "vbench_eval_records.csv",
        [asdict(row) for row in records],
        ["split", "metric", "cohort", "chunk", "score", "score_count", "eval_file"],
    )
    write_csv(
        args.output_root / "vbench_video_records.csv",
        [asdict(row) for row in video_records],
        [
            "split",
            "metric",
            "cohort",
            "chunk",
            "prompt_id",
            "seed",
            "schedule",
            "heavy_idx",
            "suffix_start",
            "suffix_tag",
            "score",
            "video_path",
            "eval_file",
        ],
    )
    write_csv(
        args.output_root / "vbench_cohort_summary.csv",
        [asdict(row) for row in summaries],
        [
            "split",
            "metric",
            "cohort",
            "n_files",
            "n_values",
            "mean_score",
            "median_score",
            "std_score",
            "schedule",
            "heavy_idx",
            "suffix_start",
            "suffix_tag",
        ],
    )
    write_csv(
        args.output_root / "full_normalized_recovery.csv",
        recoveries,
        ["metric", "schedule", "heavy_idx", "mean_score", "all_fast_score", "all_heavy_score", "recovery"],
    )
    write_json(args.output_root / "summary.json", build_summary_payload(records, video_records, summaries, warnings))
    if warnings:
        (args.output_root / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")

    if not args.no_plots and summaries:
        plots_dir = args.output_root / "plots"
        plot_full_scores(summaries, plots_dir)
        plot_full_recovery(recoveries, plots_dir)
        plot_suffix_heatmaps(summaries, plots_dir)
        plot_suffix_against_baselines(summaries, plots_dir)

    print(f"[aggregate-vbench] records: {len(records)}")
    print(f"[aggregate-vbench] video records: {len(video_records)}")
    print(f"[aggregate-vbench] cohort summaries: {len(summaries)}")
    print(f"[aggregate-vbench] warnings: {len(warnings)}")
    print(f"[aggregate-vbench] wrote: {args.output_root}")


if __name__ == "__main__":
    main()
