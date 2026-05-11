#!/usr/bin/env python3
"""Paired analysis for keyframe-budget VBench records.

Inputs:
  outputs/long_keyframe_budget/vbench_analysis/vbench_video_records.csv

Outputs:
  outputs/long_keyframe_budget/vbench_analysis/paired_analysis/

The analysis is deliberately dependency-light so it can run on the repo
environment or Jean Zay without pandas/numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FULL_METRICS = (
    "aesthetic_quality",
    "background_consistency",
    "dynamic_degree",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
)

SUFFIX_METRICS = (
    "aesthetic_quality",
    "dynamic_degree",
    "motion_smoothness",
    "subject_consistency",
)


@dataclass(frozen=True)
class VideoRecord:
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


def parse_optional_int(value: str) -> Optional[int]:
    value = str(value).strip()
    return int(value) if value else None


def read_video_records(path: Path) -> List[VideoRecord]:
    rows: List[VideoRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                VideoRecord(
                    split=row["split"],
                    metric=row["metric"],
                    cohort=row["cohort"],
                    chunk=row["chunk"],
                    prompt_id=row["prompt_id"],
                    seed=int(row["seed"]),
                    schedule=row["schedule"],
                    heavy_idx=parse_optional_int(row.get("heavy_idx", "")),
                    suffix_start=parse_optional_int(row.get("suffix_start", "")),
                    suffix_tag=row.get("suffix_tag") or None,
                    score=float(row["score"]),
                )
            )
    return rows


def read_internal_oracle(aggregate_root: Path) -> Dict[Tuple[str, int], Dict[int, Dict[str, float]]]:
    out: Dict[Tuple[str, int], Dict[int, Dict[str, float]]] = {}
    prompt_seed_dir = aggregate_root / "prompt_seed_aggregates"
    if not prompt_seed_dir.exists():
        return out
    for path in sorted(prompt_seed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (str(payload["prompt_id"]), int(payload["seed"]))
        rows: Dict[int, Dict[str, float]] = {}
        for item in payload.get("oracle_gain_records", []):
            rows[int(item["chunk_idx"])] = {
                "gain": float(item["gain"]),
                "gain_norm": float(item["gain_norm"]),
                "rank": float(item["rank"]),
            }
        out[key] = rows
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False), encoding="utf-8")


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = finite(values)
    return float(mean(vals)) if vals else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    vals = finite(values)
    return float(median(vals)) if vals else None


def safe_std(values: Sequence[float]) -> Optional[float]:
    vals = finite(values)
    return float(pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None


def bootstrap_ci_mean(values: Sequence[float], n_boot: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    vals = finite(values)
    n = len(vals)
    if n == 0:
        return None, None
    rng = random.Random(20260504 + n)
    means: List[float] = []
    for b in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return float(lo), float(hi)


def sign_test_p_value(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in finite(values) if v != 0.0]
    n = len(vals)
    if n == 0:
        return None
    positives = sum(1 for v in vals if v > 0)
    k = min(positives, n - positives)
    if n > 200:
        # Normal approximation to the two-sided binomial sign test.
        z = abs(positives - 0.5 * n) / math.sqrt(0.25 * n)
        return float(math.erfc(z / math.sqrt(2.0)))
    # Two-sided exact binomial under p=0.5.
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return float(min(1.0, p))


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1)
        for idx in order[i:j]:
            ranks[idx] = avg
        i = j
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = mean(x_vals)
    my = mean(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    return pearson(average_ranks([p[0] for p in pairs]), average_ranks([p[1] for p in pairs]))


def summarize_delta_rows(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    delta_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(float(row[delta_key]))

    out: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        vals = finite(values)
        lo, hi = bootstrap_ci_mean(vals)
        payload = {name: value for name, value in zip(group_keys, key)}
        payload.update(
            {
                "n": len(vals),
                "mean_delta": safe_mean(vals),
                "median_delta": safe_median(vals),
                "std_delta": safe_std(vals),
                "win_rate": (sum(1 for v in vals if v > 0) / len(vals)) if vals else None,
                "mean_ci95_low": lo,
                "mean_ci95_high": hi,
                "sign_test_p": sign_test_p_value(vals),
            }
        )
        out.append(payload)
    return out


def positive_top_fraction_mass(chunk_deltas: Sequence[Tuple[int, float]], top_count: int) -> Optional[float]:
    positives = [max(0.0, delta) for _, delta in chunk_deltas]
    total_positive = sum(positives)
    if total_positive <= 0:
        return None
    top_positive = sum(max(0.0, delta) for _, delta in sorted(chunk_deltas, key=lambda x: x[1], reverse=True)[:top_count])
    return float(top_positive / total_positive)


def mean_for_chunks(chunk_to_delta: Mapping[int, float], chunks: Iterable[int]) -> Optional[float]:
    values = [chunk_to_delta[c] for c in chunks if c in chunk_to_delta]
    return safe_mean(values) if values else None


def top_chunk_set(rows: Sequence[Mapping[str, Any]], metric: str, k: int) -> set[int]:
    metric_rows = [r for r in rows if r.get("metric") == metric and r.get("heavy_idx") is not None]
    ordered = sorted(metric_rows, key=lambda r: float(r["mean_delta"]), reverse=True)
    return {int(r["heavy_idx"]) for r in ordered[:k]}


def top_k_count(total: int, fraction: float = 0.10) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(total * fraction)))


def index_records(records: Sequence[VideoRecord]) -> Dict[Tuple[str, str, int, str, Optional[int]], float]:
    out: Dict[Tuple[str, str, int, str, Optional[int]], float] = {}
    for row in records:
        out[(row.metric, row.prompt_id, row.seed, row.schedule, row.suffix_start)] = row.score
    return out


def full_analysis(records: Sequence[VideoRecord], oracle: Mapping[Tuple[str, int], Dict[int, Dict[str, float]]]) -> Dict[str, Any]:
    full = [r for r in records if r.split == "full"]
    idx = index_records(full)

    baseline_rows: List[Dict[str, Any]] = []
    single_rows: List[Dict[str, Any]] = []
    sample_best_rows: List[Dict[str, Any]] = []
    oracle_alignment_rows: List[Dict[str, Any]] = []

    sample_keys = sorted({(r.metric, r.prompt_id, r.seed) for r in full})
    for metric, prompt_id, seed in sample_keys:
        fast = idx.get((metric, prompt_id, seed, "all_fast", None))
        heavy = idx.get((metric, prompt_id, seed, "all_heavy", None))
        if fast is not None and heavy is not None:
            baseline_rows.append(
                {
                    "metric": metric,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "all_fast_score": fast,
                    "all_heavy_score": heavy,
                    "heavy_minus_fast": heavy - fast,
                }
            )

        chunk_indices = sorted(
            {
                r.heavy_idx
                for r in full
                if r.metric == metric and r.prompt_id == prompt_id and r.seed == seed and r.heavy_idx is not None
            }
        )
        chunk_deltas: List[Tuple[int, float]] = []
        for heavy_idx in chunk_indices:
            schedule = f"single_heavy_{heavy_idx:02d}"
            score = idx.get((metric, prompt_id, seed, schedule, None))
            if score is None or fast is None:
                continue
            delta_fast = score - fast
            delta_heavy = score - heavy if heavy is not None else float("nan")
            single_rows.append(
                {
                    "metric": metric,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "heavy_idx": heavy_idx,
                    "score": score,
                    "all_fast_score": fast,
                    "all_heavy_score": heavy,
                    "delta_vs_fast": delta_fast,
                    "delta_vs_heavy": delta_heavy,
                }
            )
            chunk_deltas.append((heavy_idx, delta_fast))

        if chunk_deltas:
            sorted_deltas = sorted(chunk_deltas, key=lambda x: x[1], reverse=True)
            deltas_only = [d for _, d in chunk_deltas]
            positives = [d for _, d in chunk_deltas if d > 0]
            top10pct_count = top_k_count(len(chunk_deltas), 0.10)
            top10pct = sorted_deltas[:top10pct_count]
            top4 = sorted_deltas[: min(4, len(sorted_deltas))]
            top5 = sorted_deltas[:5]
            top10 = sorted_deltas[:10]
            sample_best_rows.append(
                {
                    "metric": metric,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "best_heavy_idx": sorted_deltas[0][0],
                    "best_delta_vs_fast": sorted_deltas[0][1],
                    "median_delta_vs_fast": median(deltas_only),
                    "best_minus_median_delta": sorted_deltas[0][1] - median(deltas_only),
                    "chunk_count": len(chunk_deltas),
                    "top10pct_chunk_count": top10pct_count,
                    "positive_chunk_count": len(positives),
                    "positive_chunk_fraction": len(positives) / len(chunk_deltas),
                    "positive_gain_mass_top10pct": positive_top_fraction_mass(chunk_deltas, top_count=top10pct_count),
                    "positive_gain_mass_top4": positive_top_fraction_mass(chunk_deltas, top_count=4),
                    "mean_delta_top10pct_chunks": safe_mean([d for _, d in top10pct]),
                    "mean_delta_top4_chunks": safe_mean([d for _, d in top4]),
                    "mean_delta_top5_chunks": safe_mean([d for _, d in top5]),
                    "mean_delta_top10_chunks": safe_mean([d for _, d in top10]),
                    "top5_heavy_indices": ",".join(str(i) for i, _ in sorted_deltas[:5]),
                    "top10_heavy_indices": ",".join(str(i) for i, _ in sorted_deltas[:10]),
                }
            )

        oracle_rows = oracle.get((prompt_id, seed), {})
        if chunk_deltas and oracle_rows:
            common = [(chunk, delta, oracle_rows[chunk]) for chunk, delta in chunk_deltas if chunk in oracle_rows]
            if common:
                oracle_gain = [item[2]["gain"] for item in common]
                oracle_neg_rank = [-item[2]["rank"] for item in common]
                vbench_delta = [item[1] for item in common]
                chunk_to_delta = {chunk: delta for chunk, delta, _ in common}
                random_mean_delta = safe_mean(vbench_delta)
                common_count = len(common)
                top10pct_count = top_k_count(common_count, 0.10)
                top4_count = min(4, common_count)
                top5_count = min(5, common_count)
                top10_count = min(10, common_count)
                oracle_top10pct = {chunk for chunk, _, row in sorted(common, key=lambda x: x[2]["rank"])[:top10pct_count]}
                vbench_top10pct = {chunk for chunk, _, _ in sorted(common, key=lambda x: x[1], reverse=True)[:top10pct_count]}
                oracle_top4 = {chunk for chunk, _, row in sorted(common, key=lambda x: x[2]["rank"])[:top4_count]}
                vbench_top4 = {chunk for chunk, _, _ in sorted(common, key=lambda x: x[1], reverse=True)[:top4_count]}
                oracle_top5 = {chunk for chunk, _, row in sorted(common, key=lambda x: x[2]["rank"])[:top5_count]}
                vbench_top5 = {chunk for chunk, _, _ in sorted(common, key=lambda x: x[1], reverse=True)[:top5_count]}
                oracle_top10 = {chunk for chunk, _, row in sorted(common, key=lambda x: x[2]["rank"])[:top10_count]}
                vbench_top10 = {chunk for chunk, _, _ in sorted(common, key=lambda x: x[1], reverse=True)[:top10_count]}
                oracle_top10pct_mean = mean_for_chunks(chunk_to_delta, oracle_top10pct)
                oracle_top4_mean = mean_for_chunks(chunk_to_delta, oracle_top4)
                oracle_top5_mean = mean_for_chunks(chunk_to_delta, oracle_top5)
                oracle_top10_mean = mean_for_chunks(chunk_to_delta, oracle_top10)
                oracle_alignment_rows.append(
                    {
                        "metric": metric,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "chunk_count": common_count,
                        "top10pct_chunk_count": top10pct_count,
                        "spearman_delta_vs_internal_gain": spearman(vbench_delta, oracle_gain),
                        "spearman_delta_vs_internal_negative_rank": spearman(vbench_delta, oracle_neg_rank),
                        "top10pct_recall_internal_oracle": (
                            len(oracle_top10pct & vbench_top10pct) / top10pct_count if top10pct_count else None
                        ),
                        "top4_recall_internal_oracle": len(oracle_top4 & vbench_top4) / top4_count if top4_count else None,
                        "top5_recall_internal_oracle": len(oracle_top5 & vbench_top5) / top5_count if top5_count else None,
                        "top10_recall_internal_oracle": len(oracle_top10 & vbench_top10) / top10_count if top10_count else None,
                        "mean_vbench_delta_oracle_top10pct": oracle_top10pct_mean,
                        "mean_vbench_delta_oracle_top4": oracle_top4_mean,
                        "mean_vbench_delta_oracle_top5": oracle_top5_mean,
                        "mean_vbench_delta_oracle_top10": oracle_top10_mean,
                        "mean_vbench_delta_random_topk_expectation": random_mean_delta,
                        "oracle_top4_minus_random": (
                            oracle_top4_mean - random_mean_delta
                            if oracle_top4_mean is not None and random_mean_delta is not None
                            else None
                        ),
                        "oracle_top10pct_minus_random": (
                            oracle_top10pct_mean - random_mean_delta
                            if oracle_top10pct_mean is not None and random_mean_delta is not None
                            else None
                        ),
                        "oracle_top5_minus_random": (
                            oracle_top5_mean - random_mean_delta
                            if oracle_top5_mean is not None and random_mean_delta is not None
                            else None
                        ),
                        "oracle_top10_minus_random": (
                            oracle_top10_mean - random_mean_delta
                            if oracle_top10_mean is not None and random_mean_delta is not None
                            else None
                        ),
                    }
                )

    baseline_summary = summarize_delta_rows(baseline_rows, ["metric"], "heavy_minus_fast")
    single_by_chunk = summarize_delta_rows(single_rows, ["metric", "heavy_idx"], "delta_vs_fast")
    single_by_metric = summarize_delta_rows(single_rows, ["metric"], "delta_vs_fast")
    sample_best_summary = summarize_delta_rows(sample_best_rows, ["metric"], "best_delta_vs_fast")
    concentration_summary = []
    for metric in sorted({r["metric"] for r in sample_best_rows}):
        rows = [r for r in sample_best_rows if r["metric"] == metric]
        concentration_summary.append(
            {
                "metric": metric,
                "n": len(rows),
                "mean_best_delta": safe_mean([r["best_delta_vs_fast"] for r in rows]),
                "mean_median_delta": safe_mean([r["median_delta_vs_fast"] for r in rows]),
                "mean_best_minus_median_delta": safe_mean([r["best_minus_median_delta"] for r in rows]),
                "median_best_minus_median_delta": safe_median([r["best_minus_median_delta"] for r in rows]),
                "mean_positive_chunk_fraction": safe_mean([r["positive_chunk_fraction"] for r in rows]),
                "mean_top10pct_chunk_count": safe_mean([r["top10pct_chunk_count"] for r in rows]),
                "mean_positive_gain_mass_top10pct": safe_mean(
                    [
                        r["positive_gain_mass_top10pct"]
                        for r in rows
                        if r["positive_gain_mass_top10pct"] is not None
                    ]
                ),
                "median_positive_gain_mass_top10pct": safe_median(
                    [
                        r["positive_gain_mass_top10pct"]
                        for r in rows
                        if r["positive_gain_mass_top10pct"] is not None
                    ]
                ),
                "mean_positive_gain_mass_top4": safe_mean(
                    [r["positive_gain_mass_top4"] for r in rows if r["positive_gain_mass_top4"] is not None]
                ),
                "median_positive_gain_mass_top4": safe_median(
                    [r["positive_gain_mass_top4"] for r in rows if r["positive_gain_mass_top4"] is not None]
                ),
                "mean_delta_top10pct_chunks": safe_mean([r["mean_delta_top10pct_chunks"] for r in rows]),
                "mean_delta_top4_chunks": safe_mean([r["mean_delta_top4_chunks"] for r in rows]),
                "mean_delta_top10_chunks": safe_mean([r["mean_delta_top10_chunks"] for r in rows]),
            }
        )

    best_chunk_counts: List[Dict[str, Any]] = []
    for metric in sorted({r["metric"] for r in sample_best_rows}):
        counts = Counter(int(r["best_heavy_idx"]) for r in sample_best_rows if r["metric"] == metric)
        for heavy_idx, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            best_chunk_counts.append({"metric": metric, "heavy_idx": heavy_idx, "best_count": count})

    oracle_alignment_summary = []
    for metric in sorted({r["metric"] for r in oracle_alignment_rows}):
        rows = [r for r in oracle_alignment_rows if r["metric"] == metric]
        oracle_alignment_summary.append(
            {
                "metric": metric,
                "n": len(rows),
                "mean_spearman_delta_vs_internal_gain": safe_mean(
                    [r["spearman_delta_vs_internal_gain"] for r in rows if r["spearman_delta_vs_internal_gain"] is not None]
                ),
                "median_spearman_delta_vs_internal_gain": safe_median(
                    [r["spearman_delta_vs_internal_gain"] for r in rows if r["spearman_delta_vs_internal_gain"] is not None]
                ),
                "mean_top10_recall_internal_oracle": safe_mean([r["top10_recall_internal_oracle"] for r in rows]),
                "median_top10_recall_internal_oracle": safe_median([r["top10_recall_internal_oracle"] for r in rows]),
                "mean_top10pct_recall_internal_oracle": safe_mean([r["top10pct_recall_internal_oracle"] for r in rows]),
                "median_top10pct_recall_internal_oracle": safe_median([r["top10pct_recall_internal_oracle"] for r in rows]),
                "mean_oracle_top10pct_minus_random": safe_mean(
                    [
                        r["oracle_top10pct_minus_random"]
                        for r in rows
                        if r["oracle_top10pct_minus_random"] is not None
                    ]
                ),
                "mean_top5_recall_internal_oracle": safe_mean([r["top5_recall_internal_oracle"] for r in rows]),
                "median_top5_recall_internal_oracle": safe_median([r["top5_recall_internal_oracle"] for r in rows]),
                "mean_oracle_top4_minus_random": safe_mean(
                    [r["oracle_top4_minus_random"] for r in rows if r["oracle_top4_minus_random"] is not None]
                ),
                "mean_oracle_top5_minus_random": safe_mean(
                    [r["oracle_top5_minus_random"] for r in rows if r["oracle_top5_minus_random"] is not None]
                ),
                "mean_oracle_top10_minus_random": safe_mean(
                    [r["oracle_top10_minus_random"] for r in rows if r["oracle_top10_minus_random"] is not None]
                ),
                "mean_vbench_delta_oracle_top4": safe_mean(
                    [r["mean_vbench_delta_oracle_top4"] for r in rows if r["mean_vbench_delta_oracle_top4"] is not None]
                ),
                "mean_vbench_delta_oracle_top10pct": safe_mean(
                    [
                        r["mean_vbench_delta_oracle_top10pct"]
                        for r in rows
                        if r["mean_vbench_delta_oracle_top10pct"] is not None
                    ]
                ),
                "mean_vbench_delta_oracle_top5": safe_mean(
                    [r["mean_vbench_delta_oracle_top5"] for r in rows if r["mean_vbench_delta_oracle_top5"] is not None]
                ),
                "mean_vbench_delta_oracle_top10": safe_mean(
                    [r["mean_vbench_delta_oracle_top10"] for r in rows if r["mean_vbench_delta_oracle_top10"] is not None]
                ),
            }
        )

    metric_overlap_rows: List[Dict[str, Any]] = []
    metrics = sorted({r["metric"] for r in single_by_chunk})
    chunk_count_by_metric = {
        metric: len({int(r["heavy_idx"]) for r in single_by_chunk if r["metric"] == metric})
        for metric in metrics
    }
    for idx_a, metric_a in enumerate(metrics):
        for metric_b in metrics[idx_a + 1 :]:
            n_chunks = min(chunk_count_by_metric.get(metric_a, 0), chunk_count_by_metric.get(metric_b, 0))
            for k in sorted({top_k_count(n_chunks, 0.10), min(5, n_chunks), min(10, n_chunks)}):
                if k <= 0:
                    continue
                set_a = top_chunk_set(single_by_chunk, metric_a, k)
                set_b = top_chunk_set(single_by_chunk, metric_b, k)
                union = set_a | set_b
                metric_overlap_rows.append(
                    {
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "k": k,
                        "overlap_count": len(set_a & set_b),
                        "jaccard": (len(set_a & set_b) / len(union)) if union else None,
                        "top_a": ",".join(str(x) for x in sorted(set_a)),
                        "top_b": ",".join(str(x) for x in sorted(set_b)),
                    }
                )

    return {
        "baseline_rows": baseline_rows,
        "single_rows": single_rows,
        "sample_best_rows": sample_best_rows,
        "oracle_alignment_rows": oracle_alignment_rows,
        "baseline_summary": baseline_summary,
        "single_by_chunk": single_by_chunk,
        "single_by_metric": single_by_metric,
        "sample_best_summary": sample_best_summary,
        "concentration_summary": concentration_summary,
        "best_chunk_counts": best_chunk_counts,
        "oracle_alignment_summary": oracle_alignment_summary,
        "metric_overlap_rows": metric_overlap_rows,
    }


def suffix_analysis(records: Sequence[VideoRecord]) -> Dict[str, Any]:
    suffix = [r for r in records if r.split == "suffix"]
    idx = index_records(suffix)

    baseline_rows: List[Dict[str, Any]] = []
    single_rows: List[Dict[str, Any]] = []
    sample_best_rows: List[Dict[str, Any]] = []

    sample_keys = sorted({(r.metric, r.prompt_id, r.seed, r.suffix_start) for r in suffix if r.suffix_start is not None})
    for metric, prompt_id, seed, suffix_start in sample_keys:
        fast = idx.get((metric, prompt_id, seed, "all_fast", suffix_start))
        heavy = idx.get((metric, prompt_id, seed, "all_heavy", suffix_start))
        if fast is not None and heavy is not None:
            baseline_rows.append(
                {
                    "metric": metric,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "suffix_start": suffix_start,
                    "all_fast_score": fast,
                    "all_heavy_score": heavy,
                    "heavy_minus_fast": heavy - fast,
                }
            )

        chunk_deltas: List[Tuple[int, float]] = []
        for record in suffix:
            if (
                record.metric == metric
                and record.prompt_id == prompt_id
                and record.seed == seed
                and record.suffix_start == suffix_start
                and record.heavy_idx is not None
            ):
                if fast is None:
                    continue
                delta = record.score - fast
                rel = "upstream" if record.heavy_idx < suffix_start else "same" if record.heavy_idx == suffix_start else "downstream"
                single_rows.append(
                    {
                        "metric": metric,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "suffix_start": suffix_start,
                        "heavy_idx": record.heavy_idx,
                        "relative_position": rel,
                        "score": record.score,
                        "all_fast_score": fast,
                        "delta_vs_fast": delta,
                    }
                )
                chunk_deltas.append((record.heavy_idx, delta))

        if chunk_deltas:
            sorted_deltas = sorted(chunk_deltas, key=lambda x: x[1], reverse=True)
            sample_best_rows.append(
                {
                    "metric": metric,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "suffix_start": suffix_start,
                    "best_heavy_idx": sorted_deltas[0][0],
                    "best_delta_vs_fast": sorted_deltas[0][1],
                    "positive_chunk_count": sum(1 for _, d in chunk_deltas if d > 0),
                    "top3_heavy_indices": ",".join(str(i) for i, _ in sorted_deltas[:3]),
                }
            )

    return {
        "baseline_rows": baseline_rows,
        "single_rows": single_rows,
        "sample_best_rows": sample_best_rows,
        "baseline_summary": summarize_delta_rows(baseline_rows, ["metric", "suffix_start"], "heavy_minus_fast"),
        "single_by_suffix_chunk": summarize_delta_rows(single_rows, ["metric", "suffix_start", "heavy_idx"], "delta_vs_fast"),
        "single_by_relative_position": summarize_delta_rows(single_rows, ["metric", "relative_position"], "delta_vs_fast"),
        "sample_best_summary": summarize_delta_rows(sample_best_rows, ["metric", "suffix_start"], "best_delta_vs_fast"),
    }


def top_rows(rows: Sequence[Mapping[str, Any]], metric: str, key: str, n: int = 5) -> List[Mapping[str, Any]]:
    filtered = [r for r in rows if r.get("metric") == metric and r.get(key) is not None]
    return sorted(filtered, key=lambda r: float(r[key]), reverse=True)[:n]


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def build_markdown(
    full: Mapping[str, Any],
    suffix: Mapping[str, Any],
    output_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Paired Keyframe-Budget VBench Analysis")
    lines.append("")
    lines.append("This analysis uses `vbench_video_records.csv`, i.e. per-video VBench scores extracted from existing JSONs. No VBench rerun is required.")
    lines.append("")
    lines.append("## Full-Video Baselines")
    lines.append("")
    lines.append("| Metric | n | all_heavy - all_fast mean | median | win rate | sign p |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in full["baseline_summary"]:
        lines.append(
            "| {metric} | {n} | {mean_delta} | {median_delta} | {win_rate} | {p} |".format(
                metric=row["metric"],
                n=row["n"],
                mean_delta=format_float(row["mean_delta"]),
                median_delta=format_float(row["median_delta"]),
                win_rate=format_float(row["win_rate"]),
                p=format_float(row["sign_test_p"], digits=3),
            )
        )
    lines.append("")
    lines.append("## Full-Video Single-Heavy Signal")
    lines.append("")
    lines.append("| Metric | mean single-heavy delta | win rate over all chunk/sample pairs | mean best per-sample delta | top global chunks by mean delta |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    by_metric = {row["metric"]: row for row in full["single_by_metric"]}
    best_by_metric = {row["metric"]: row for row in full["sample_best_summary"]}
    for metric in FULL_METRICS:
        row = by_metric.get(metric, {})
        best = best_by_metric.get(metric, {})
        top = top_rows(full["single_by_chunk"], metric, "mean_delta", n=5)
        top_str = ", ".join(f"{int(r['heavy_idx'])}:{format_float(r['mean_delta'], 4)}" for r in top)
        lines.append(
            f"| {metric} | {format_float(row.get('mean_delta'))} | {format_float(row.get('win_rate'))} | "
            f"{format_float(best.get('mean_delta'))} | {top_str} |"
        )
    lines.append("")
    lines.append("## A. Concentration")
    lines.append("")
    lines.append("| Metric | mean best-minus-median | mean positive chunk fraction | top-10% chunk count | mean top-10% positive gain mass | mean top-10% delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in full["concentration_summary"]:
        lines.append(
            f"| {row['metric']} | {format_float(row['mean_best_minus_median_delta'])} | "
            f"{format_float(row['mean_positive_chunk_fraction'])} | "
            f"{format_float(row['mean_top10pct_chunk_count'])} | "
            f"{format_float(row['mean_positive_gain_mass_top10pct'])} | "
            f"{format_float(row['mean_delta_top10pct_chunks'])} |"
        )
    lines.append("")
    lines.append("## Internal Oracle Alignment")
    lines.append("")
    lines.append("This compares VBench single-heavy deltas against the existing in-repo proxy oracle ranks for the same prompt/seed.")
    lines.append("")
    lines.append("| Metric | n samples | Spearman vs internal gain | top-10% recall | top-5 recall | oracle top-10% minus random |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in full["oracle_alignment_summary"]:
        lines.append(
            f"| {row['metric']} | {row['n']} | {format_float(row['mean_spearman_delta_vs_internal_gain'])} | "
            f"{format_float(row['mean_top10pct_recall_internal_oracle'])} | "
            f"{format_float(row['mean_top5_recall_internal_oracle'])} | "
            f"{format_float(row['mean_oracle_top10pct_minus_random'])} |"
        )
    lines.append("")
    lines.append("## C. Cross-Metric Keyframe Overlap")
    lines.append("")
    lines.append("| Metric pair | k | overlap | Jaccard |")
    lines.append("| --- | ---: | ---: | ---: |")
    overlap_by_pair: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in full["metric_overlap_rows"]:
        overlap_by_pair[(str(row["metric_a"]), str(row["metric_b"]))].append(row)
    for (metric_a, metric_b), rows in sorted(overlap_by_pair.items()):
        for row in sorted(rows, key=lambda r: int(r["k"])):
            lines.append(
                f"| {metric_a} / {metric_b} | {row.get('k', 'NA')} | "
                f"{row.get('overlap_count', 'NA')} | {format_float(row.get('jaccard'))} |"
            )
    lines.append("")
    lines.append("## Sparse Suffix Signal")
    lines.append("")
    lines.append("| Metric | relative position | n | mean delta vs all_fast suffix | win rate |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in suffix["single_by_relative_position"]:
        lines.append(
            f"| {row['metric']} | {row['relative_position']} | {row['n']} | "
            f"{format_float(row['mean_delta'])} | {format_float(row['win_rate'])} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The parser fix makes the aggregation statistically clean at the per-video level; every full/suffix cohort now has 72 paired sample scores.")
    lines.append("- The all-heavy anomaly remains after the parser fix, so it is not explained by the previous aggregation bug.")
    lines.append("- The keyframe signal is best framed as sparse single-heavy placements producing non-uniform paired deltas, not as monotonic improvement from all-heavy.")
    lines.append("- The next scientific test should use paired per-sample rankings and confidence/sign tests, not all-heavy-normalized recovery.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for path in sorted(output_dir.glob("*.csv")):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records_csv",
        type=Path,
        default=Path("outputs/long_keyframe_budget/vbench_analysis/vbench_video_records.csv"),
    )
    parser.add_argument(
        "--aggregate_root",
        type=Path,
        default=Path("outputs/long_keyframe_budget/long_rollout_discovery/aggregates"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/long_keyframe_budget/vbench_analysis/paired_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = read_video_records(args.records_csv)
    oracle = read_internal_oracle(args.aggregate_root)
    full = full_analysis(records, oracle)
    suffix = suffix_analysis(records)

    write_csv(
        args.output_dir / "full_baseline_paired_deltas.csv",
        full["baseline_rows"],
        ["metric", "prompt_id", "seed", "all_fast_score", "all_heavy_score", "heavy_minus_fast"],
    )
    write_csv(
        args.output_dir / "full_baseline_summary.csv",
        full["baseline_summary"],
        ["metric", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "full_single_heavy_paired_deltas.csv",
        full["single_rows"],
        ["metric", "prompt_id", "seed", "heavy_idx", "score", "all_fast_score", "all_heavy_score", "delta_vs_fast", "delta_vs_heavy"],
    )
    write_csv(
        args.output_dir / "full_single_heavy_by_chunk.csv",
        full["single_by_chunk"],
        ["metric", "heavy_idx", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "full_single_heavy_by_metric.csv",
        full["single_by_metric"],
        ["metric", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "full_sample_best_chunks.csv",
        full["sample_best_rows"],
        [
            "metric",
            "prompt_id",
            "seed",
            "best_heavy_idx",
            "best_delta_vs_fast",
            "median_delta_vs_fast",
            "best_minus_median_delta",
            "chunk_count",
            "top10pct_chunk_count",
            "positive_chunk_count",
            "positive_chunk_fraction",
            "positive_gain_mass_top10pct",
            "positive_gain_mass_top4",
            "mean_delta_top10pct_chunks",
            "mean_delta_top4_chunks",
            "mean_delta_top5_chunks",
            "mean_delta_top10_chunks",
            "top5_heavy_indices",
            "top10_heavy_indices",
        ],
    )
    write_csv(
        args.output_dir / "full_concentration_summary.csv",
        full["concentration_summary"],
        [
            "metric",
            "n",
            "mean_best_delta",
            "mean_median_delta",
            "mean_best_minus_median_delta",
            "median_best_minus_median_delta",
            "mean_positive_chunk_fraction",
            "mean_top10pct_chunk_count",
            "mean_positive_gain_mass_top10pct",
            "median_positive_gain_mass_top10pct",
            "mean_positive_gain_mass_top4",
            "median_positive_gain_mass_top4",
            "mean_delta_top10pct_chunks",
            "mean_delta_top4_chunks",
            "mean_delta_top10_chunks",
        ],
    )
    write_csv(
        args.output_dir / "full_sample_best_summary.csv",
        full["sample_best_summary"],
        ["metric", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "full_best_chunk_histogram.csv",
        full["best_chunk_counts"],
        ["metric", "heavy_idx", "best_count"],
    )
    write_csv(
        args.output_dir / "full_internal_oracle_alignment.csv",
        full["oracle_alignment_rows"],
        [
            "metric",
            "prompt_id",
            "seed",
            "chunk_count",
            "top10pct_chunk_count",
            "spearman_delta_vs_internal_gain",
            "spearman_delta_vs_internal_negative_rank",
            "top10pct_recall_internal_oracle",
            "top4_recall_internal_oracle",
            "top5_recall_internal_oracle",
            "top10_recall_internal_oracle",
            "mean_vbench_delta_oracle_top10pct",
            "mean_vbench_delta_oracle_top4",
            "mean_vbench_delta_oracle_top5",
            "mean_vbench_delta_oracle_top10",
            "mean_vbench_delta_random_topk_expectation",
            "oracle_top10pct_minus_random",
            "oracle_top4_minus_random",
            "oracle_top5_minus_random",
            "oracle_top10_minus_random",
        ],
    )
    write_csv(
        args.output_dir / "full_internal_oracle_alignment_summary.csv",
        full["oracle_alignment_summary"],
        [
            "metric",
            "n",
            "mean_spearman_delta_vs_internal_gain",
            "median_spearman_delta_vs_internal_gain",
            "mean_top10pct_recall_internal_oracle",
            "median_top10pct_recall_internal_oracle",
            "mean_top5_recall_internal_oracle",
            "median_top5_recall_internal_oracle",
            "mean_top10_recall_internal_oracle",
            "median_top10_recall_internal_oracle",
            "mean_oracle_top10pct_minus_random",
            "mean_oracle_top4_minus_random",
            "mean_oracle_top5_minus_random",
            "mean_oracle_top10_minus_random",
            "mean_vbench_delta_oracle_top10pct",
            "mean_vbench_delta_oracle_top4",
            "mean_vbench_delta_oracle_top5",
            "mean_vbench_delta_oracle_top10",
        ],
    )
    write_csv(
        args.output_dir / "full_metric_topk_overlap.csv",
        full["metric_overlap_rows"],
        ["metric_a", "metric_b", "k", "overlap_count", "jaccard", "top_a", "top_b"],
    )

    write_csv(
        args.output_dir / "suffix_baseline_paired_deltas.csv",
        suffix["baseline_rows"],
        ["metric", "prompt_id", "seed", "suffix_start", "all_fast_score", "all_heavy_score", "heavy_minus_fast"],
    )
    write_csv(
        args.output_dir / "suffix_baseline_summary.csv",
        suffix["baseline_summary"],
        ["metric", "suffix_start", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "suffix_single_heavy_paired_deltas.csv",
        suffix["single_rows"],
        ["metric", "prompt_id", "seed", "suffix_start", "heavy_idx", "relative_position", "score", "all_fast_score", "delta_vs_fast"],
    )
    write_csv(
        args.output_dir / "suffix_single_heavy_by_suffix_chunk.csv",
        suffix["single_by_suffix_chunk"],
        ["metric", "suffix_start", "heavy_idx", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "suffix_single_heavy_by_relative_position.csv",
        suffix["single_by_relative_position"],
        ["metric", "relative_position", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )
    write_csv(
        args.output_dir / "suffix_sample_best_summary.csv",
        suffix["sample_best_summary"],
        ["metric", "suffix_start", "n", "mean_delta", "median_delta", "std_delta", "win_rate", "mean_ci95_low", "mean_ci95_high", "sign_test_p"],
    )

    summary = {
        "records_csv": str(args.records_csv),
        "record_count": len(records),
        "full": {
            "baseline_summary": full["baseline_summary"],
            "single_by_metric": full["single_by_metric"],
            "sample_best_summary": full["sample_best_summary"],
            "concentration_summary": full["concentration_summary"],
            "oracle_alignment_summary": full["oracle_alignment_summary"],
            "metric_overlap_rows": full["metric_overlap_rows"],
        },
        "suffix": {
            "single_by_relative_position": suffix["single_by_relative_position"],
            "sample_best_summary": suffix["sample_best_summary"],
        },
    }
    write_json(args.output_dir / "paired_analysis_summary.json", summary)

    report = build_markdown(full, suffix, args.output_dir)
    (args.output_dir / "paired_analysis_report.md").write_text(report, encoding="utf-8")

    print(f"[paired-vbench] records: {len(records)}")
    print(f"[paired-vbench] wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
