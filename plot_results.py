#!/usr/bin/env python3
"""Generate decision-oriented plots and summaries for keyframe-budget runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import matplotlib  # type: ignore[import-not-found]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'matplotlib'. Activate your experiment environment "
        "(e.g. conda activate deepforcing) before running plot_results.py."
    ) from exc

try:
    import numpy as np  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'numpy'. Activate your experiment environment "
        "(e.g. conda activate deepforcing) before running plot_results.py."
    ) from exc


DEFAULT_EXPERIMENTS = ("discovery_split", "policy_eval_100")
PREFERRED_POLICY_ORDER = ("uniform_top_m", "random_top_m", "prefix_top_m", "oracle_top_m")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _generate_recovery_barplot_builtin(policy_results_jsonl: Path, out_path: Path) -> None:
    rows = _read_jsonl(policy_results_jsonl)
    if not rows:
        raise ValueError(f"No policy rows in {policy_results_jsonl}")

    by_policy: Dict[str, List[float]] = {}
    for row in rows:
        policy = str(row["policy_name"])
        recovery = row.get("recovery")
        if recovery is None:
            continue
        by_policy.setdefault(policy, []).append(float(recovery))

    labels = sorted(by_policy.keys())
    means = [float(np.mean(by_policy[name])) for name in labels]
    stds = [float(np.std(by_policy[name])) for name in labels]

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


def _generate_gain_heatmap_builtin(gain_maps_jsonl: Path, out_path: Path) -> None:
    rows = _read_jsonl(gain_maps_jsonl)
    if not rows:
        raise ValueError(f"No gain rows in {gain_maps_jsonl}")

    key_to_values: Dict[Tuple[str, int], Dict[int, float]] = {}
    max_chunk_idx = 0
    for row in rows:
        prompt_id = str(row["prompt_id"])
        seed = int(row["seed"])
        chunk_idx = int(row["chunk_idx"])
        gain_norm = float(row["gain_norm"])
        key_to_values.setdefault((prompt_id, seed), {})[chunk_idx] = gain_norm
        max_chunk_idx = max(max_chunk_idx, chunk_idx)

    labels: List[str] = []
    matrix: List[List[float]] = []
    for (prompt_id, seed), value_map in sorted(key_to_values.items()):
        labels.append(f"{prompt_id}:{seed}")
        matrix.append([value_map.get(idx, 0.0) for idx in range(max_chunk_idx + 1)])

    arr = np.array(matrix, dtype=np.float32)
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(8, 0.35 * (max_chunk_idx + 1)), max(4, 0.35 * len(labels))))
    plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.colorbar(label="g_i_norm")
    plt.xlabel("Chunk index")
    plt.ylabel("Prompt:Seed")
    plt.title("Gain Heatmap")
    plt.yticks(np.arange(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _generate_best_vs_median_gain_plot_builtin(gain_maps_jsonl: Path, out_path: Path) -> None:
    rows = _read_jsonl(gain_maps_jsonl)
    if not rows:
        raise ValueError(f"No gain rows in {gain_maps_jsonl}")

    by_video: Dict[Tuple[str, int], List[float]] = {}
    for row in rows:
        key = (str(row["prompt_id"]), int(row["seed"]))
        by_video.setdefault(key, []).append(float(row["gain"]))

    labels: List[str] = []
    best_vals: List[float] = []
    med_vals: List[float] = []
    for key, gains in sorted(by_video.items()):
        labels.append(f"{key[0]}:{key[1]}")
        best_vals.append(float(np.max(gains)))
        med_vals.append(float(np.median(gains)))

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(10, 0.35 * len(labels)), 4))
    x = np.arange(len(labels))
    width = 0.4
    plt.bar(x - width / 2, best_vals, width=width, label="Best chunk gain")
    plt.bar(x + width / 2, med_vals, width=width, label="Median chunk gain")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Best vs Median Chunk Gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _generate_oracle_position_histogram_builtin(gain_maps_jsonl: Path, out_path: Path) -> None:
    rows = _read_jsonl(gain_maps_jsonl)
    if not rows:
        raise ValueError(f"No gain rows in {gain_maps_jsonl}")

    by_video: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for row in rows:
        key = (str(row["prompt_id"]), int(row["seed"]))
        by_video.setdefault(key, []).append((int(row["chunk_idx"]), float(row["gain"])))

    best_chunk_positions: List[int] = []
    for entries in by_video.values():
        best_chunk_positions.append(sorted(entries, key=lambda x: x[1], reverse=True)[0][0])

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    plt.hist(best_chunk_positions, bins=max(best_chunk_positions) + 1, edgecolor="black")
    plt.xlabel("Best chunk index")
    plt.ylabel("Count")
    plt.title("Oracle Chunk Position Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(mean(values))


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def _clip(values: Iterable[float], low: float, high: float) -> List[float]:
    out: List[float] = []
    for v in values:
        out.append(float(min(max(v, low), high)))
    return out


def _load_prompt_seed_rows(experiment_root: Path) -> List[Dict[str, object]]:
    aggregate_dir = experiment_root / "aggregates" / "prompt_seed_aggregates"
    if not aggregate_dir.exists():
        raise FileNotFoundError(f"Missing aggregate directory: {aggregate_dir}")

    rows: List[Dict[str, object]] = []
    for path in sorted(aggregate_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "prompt_id": str(payload["prompt_id"]),
                "seed": int(payload["seed"]),
                "all_fast_score": float(payload["all_fast_score"]),
                "all_heavy_score": float(payload["all_heavy_score"]),
                "policy_scores": dict(payload.get("policy_scores", {})),
                "policy_recovery": dict(payload.get("policy_recovery", {})),
                "source_file": str(path),
            }
        )
    return rows


def _discover_policies(rows: Sequence[Mapping[str, object]]) -> List[str]:
    policy_names: set[str] = set()
    for row in rows:
        scores = row["policy_scores"]
        assert isinstance(scores, dict)
        for name in scores:
            if name.startswith("all_") or name.startswith("single_heavy_"):
                continue
            policy_names.add(str(name))

    ordered: List[str] = []
    for name in PREFERRED_POLICY_ORDER:
        if name in policy_names:
            ordered.append(name)
    for name in sorted(policy_names):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _get_policy_score(row: Mapping[str, object], policy: str) -> Optional[float]:
    scores = row["policy_scores"]
    assert isinstance(scores, dict)
    value = scores.get(policy)
    if value is None:
        return None
    return float(value)


def _policy_deltas_vs_fast(rows: Sequence[Mapping[str, object]], policy: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        score = _get_policy_score(row, policy)
        if score is None:
            continue
        vals.append(score - float(row["all_fast_score"]))
    return vals


def _policy_deltas_vs_heavy(rows: Sequence[Mapping[str, object]], policy: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        score = _get_policy_score(row, policy)
        if score is None:
            continue
        vals.append(score - float(row["all_heavy_score"]))
    return vals


def _policy_recoveries(rows: Sequence[Mapping[str, object]], policy: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        recovery = row["policy_recovery"]
        assert isinstance(recovery, dict)
        value = recovery.get(policy)
        if value is None:
            continue
        vals.append(float(value))
    return vals


def _prompt_sort_key(prompt_id: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in prompt_id if ch.isdigit())
    if digits:
        return (int(digits), prompt_id)
    return (10**9, prompt_id)


def _plot_gap_hist(rows: Sequence[Mapping[str, object]], out_path: Path, title: str) -> None:
    gaps = [float(r["all_heavy_score"]) - float(r["all_fast_score"]) for r in rows]
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    plt.hist(gaps, bins=30, color="#4C78A8", edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel("all_heavy - all_fast")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_fast_vs_heavy_scatter(rows: Sequence[Mapping[str, object]], out_path: Path, title: str) -> None:
    fast = np.array([float(r["all_fast_score"]) for r in rows], dtype=np.float32)
    heavy = np.array([float(r["all_heavy_score"]) for r in rows], dtype=np.float32)
    colors = np.where(heavy >= fast, "#2ca02c", "#d62728")

    lo = float(min(np.min(fast), np.min(heavy)))
    hi = float(max(np.max(fast), np.max(heavy)))
    span = hi - lo
    pad = max(1e-4, 0.05 * span)

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 6))
    plt.scatter(fast, heavy, c=colors, alpha=0.7, s=18)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("all_fast score")
    plt.ylabel("all_heavy score")
    plt.title(title)
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_policy_box(
    data_by_policy: Mapping[str, Sequence[float]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    labels: List[str] = []
    data: List[Sequence[float]] = []
    for policy, vals in data_by_policy.items():
        if not vals:
            continue
        labels.append(policy)
        data.append(vals)
    if not data:
        return

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(8, 1.4 * len(labels)), 4.8))
    plt.boxplot(data, labels=labels, showmeans=True, meanline=True)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_best_policy_counts(rows: Sequence[Mapping[str, object]], policies: Sequence[str], out_path: Path) -> None:
    counts = {name: 0 for name in policies}
    for row in rows:
        candidates: Dict[str, float] = {}
        for policy in policies:
            value = _get_policy_score(row, policy)
            if value is not None:
                candidates[policy] = value
        if not candidates:
            continue
        winner = max(candidates.items(), key=lambda item: item[1])[0]
        counts[winner] += 1

    labels = [name for name in policies if counts.get(name, 0) > 0]
    values = [counts[name] for name in labels]
    if not labels:
        return

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(8, 1.4 * len(labels)), 4.4))
    bars = plt.bar(labels, values, color="#4C78A8")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, float(v) + 0.3, str(v), ha="center", va="bottom", fontsize=8)
    plt.ylabel("Prompt-seed count")
    plt.title("Best policy count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _pairwise_win_rate(
    rows: Sequence[Mapping[str, object]],
    policy_a: str,
    policy_b: str,
) -> Tuple[Optional[float], int]:
    wins = 0
    total = 0
    for row in rows:
        a = _get_policy_score(row, policy_a)
        b = _get_policy_score(row, policy_b)
        if a is None or b is None:
            continue
        total += 1
        if a > b:
            wins += 1
    if total == 0:
        return None, 0
    return float(wins / total), total


def _plot_pairwise_win_heatmap(rows: Sequence[Mapping[str, object]], policies: Sequence[str], out_path: Path) -> None:
    n = len(policies)
    if n == 0:
        return
    mat = np.full((n, n), np.nan, dtype=np.float32)
    cnt = np.zeros((n, n), dtype=np.int32)
    for i, a in enumerate(policies):
        for j, b in enumerate(policies):
            if i == j:
                mat[i, j] = 1.0
                cnt[i, j] = 0
                continue
            rate, total = _pairwise_win_rate(rows, a, b)
            if rate is not None:
                mat[i, j] = rate
                cnt[i, j] = total

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(max(7, 1.1 * n), max(6, 1.1 * n)))
    im = plt.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Win rate")
    plt.xticks(np.arange(n), policies, rotation=30, ha="right")
    plt.yticks(np.arange(n), policies)
    plt.title("Pairwise win-rate matrix (row beats column)")

    for i in range(n):
        for j in range(n):
            if math.isnan(float(mat[i, j])):
                continue
            label = f"{mat[i, j]:.2f}"
            if i != j and cnt[i, j] > 0:
                label += f"\n(n={cnt[i, j]})"
            plt.text(j, i, label, ha="center", va="center", color="white", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_oracle_coverage(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    # Build matrix prompt x seed with value=1 if oracle_top_m exists in policy_scores.
    prompts = sorted({str(r["prompt_id"]) for r in rows}, key=_prompt_sort_key)
    seeds = sorted({int(r["seed"]) for r in rows})
    if not prompts or not seeds:
        return

    prompt_index = {p: i for i, p in enumerate(prompts)}
    seed_index = {s: i for i, s in enumerate(seeds)}
    mat = np.zeros((len(prompts), len(seeds)), dtype=np.float32)

    for row in rows:
        p = str(row["prompt_id"])
        s = int(row["seed"])
        has_oracle = _get_policy_score(row, "oracle_top_m") is not None
        mat[prompt_index[p], seed_index[s]] = 1.0 if has_oracle else 0.0

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(6, max(5, 0.18 * len(prompts))))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="magma")
    plt.colorbar(label="oracle_top_m available")
    plt.xticks(np.arange(len(seeds)), [str(s) for s in seeds])
    y_step = max(1, len(prompts) // 24)
    tick_idx = np.arange(0, len(prompts), y_step)
    plt.yticks(tick_idx, [prompts[i] for i in tick_idx])
    plt.xlabel("Seed")
    plt.ylabel("Prompt id")
    plt.title("Oracle policy availability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_oracle_vs_prefix(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        prefix = _get_policy_score(row, "prefix_top_m")
        oracle = _get_policy_score(row, "oracle_top_m")
        if prefix is None or oracle is None:
            continue
        base = float(row["all_fast_score"])
        xs.append(prefix - base)
        ys.append(oracle - base)
    if not xs:
        return

    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    span = hi - lo
    pad = max(1e-4, 0.05 * span)

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(5.8, 5.6))
    plt.scatter(xs, ys, alpha=0.7, s=20, color="#4C78A8")
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("prefix_top_m - all_fast")
    plt.ylabel("oracle_top_m - all_fast")
    plt.title("Oracle vs Prefix (overlap subset)")
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _summarize_experiment(rows: Sequence[Mapping[str, object]], policies: Sequence[str]) -> Dict[str, object]:
    gaps = [float(r["all_heavy_score"]) - float(r["all_fast_score"]) for r in rows]
    heavy_better = sum(1 for g in gaps if g > 0)
    heavy_worse = sum(1 for g in gaps if g < 0)
    heavy_equal = len(gaps) - heavy_better - heavy_worse

    policy_stats: Dict[str, Dict[str, object]] = {}
    best_counts = {p: 0 for p in policies}

    for row in rows:
        candidates: Dict[str, float] = {}
        for policy in policies:
            value = _get_policy_score(row, policy)
            if value is not None:
                candidates[policy] = value
        if candidates:
            winner = max(candidates.items(), key=lambda item: item[1])[0]
            best_counts[winner] += 1

    for policy in policies:
        deltas_fast = _policy_deltas_vs_fast(rows, policy)
        deltas_heavy = _policy_deltas_vs_heavy(rows, policy)
        recoveries = _policy_recoveries(rows, policy)
        stats = {
            "count": len(deltas_fast),
            "delta_vs_fast_mean": _safe_mean(deltas_fast),
            "delta_vs_fast_median": _safe_median(deltas_fast),
            "delta_vs_fast_min": float(min(deltas_fast)) if deltas_fast else None,
            "delta_vs_fast_max": float(max(deltas_fast)) if deltas_fast else None,
            "delta_vs_heavy_mean": _safe_mean(deltas_heavy),
            "delta_vs_heavy_median": _safe_median(deltas_heavy),
            "delta_vs_heavy_min": float(min(deltas_heavy)) if deltas_heavy else None,
            "delta_vs_heavy_max": float(max(deltas_heavy)) if deltas_heavy else None,
            "recovery_mean": _safe_mean(recoveries),
            "recovery_median": _safe_median(recoveries),
            "better_than_fast_count": int(sum(1 for v in deltas_fast if v > 0)),
            "better_than_heavy_count": int(sum(1 for v in deltas_heavy if v > 0)),
        }
        policy_stats[policy] = stats

    pairwise: Dict[str, Dict[str, object]] = {}
    for a in policies:
        pairwise[a] = {}
        for b in policies:
            if a == b:
                continue
            rate, total = _pairwise_win_rate(rows, a, b)
            pairwise[a][b] = {"win_rate": rate, "support": total}

    with_oracle = sum(1 for r in rows if _get_policy_score(r, "oracle_top_m") is not None)

    return {
        "num_prompt_seed": len(rows),
        "heavy_vs_fast": {
            "mean_gap": _safe_mean(gaps),
            "median_gap": _safe_median(gaps),
            "min_gap": float(min(gaps)) if gaps else None,
            "max_gap": float(max(gaps)) if gaps else None,
            "heavy_better_count": heavy_better,
            "heavy_worse_count": heavy_worse,
            "heavy_equal_count": heavy_equal,
        },
        "oracle_policy_coverage": {
            "count_with_oracle_top_m": with_oracle,
            "count_without_oracle_top_m": len(rows) - with_oracle,
        },
        "best_policy_counts": best_counts,
        "policy_stats": policy_stats,
        "pairwise_win_rates": pairwise,
    }


def _write_markdown_summary(
    experiment_name: str,
    summary: Mapping[str, object],
    policies: Sequence[str],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# {experiment_name} decision summary")
    lines.append("")

    hvf = summary["heavy_vs_fast"]
    assert isinstance(hvf, dict)
    lines.append("## Heavy vs Fast")
    lines.append(
        f"- Count: {summary['num_prompt_seed']} prompt-seed pairs "
        f"(heavy better: {hvf['heavy_better_count']}, heavy worse: {hvf['heavy_worse_count']})."
    )
    lines.append(
        f"- Gap (all_heavy - all_fast): mean={hvf['mean_gap']:.6f}, "
        f"median={hvf['median_gap']:.6f}, min={hvf['min_gap']:.6f}, max={hvf['max_gap']:.6f}."
    )
    lines.append("")

    coverage = summary["oracle_policy_coverage"]
    assert isinstance(coverage, dict)
    lines.append("## Oracle Availability")
    lines.append(
        f"- oracle_top_m present in {coverage['count_with_oracle_top_m']} / "
        f"{summary['num_prompt_seed']} prompt-seed pairs."
    )
    lines.append("")

    lines.append("## Policy Deltas (vs all_fast)")
    policy_stats = summary["policy_stats"]
    assert isinstance(policy_stats, dict)
    ranking: List[Tuple[str, float]] = []
    for p in policies:
        stats = policy_stats.get(p, {})
        assert isinstance(stats, dict)
        mu = stats.get("delta_vs_fast_mean")
        if mu is not None:
            ranking.append((p, float(mu)))
        lines.append(
            f"- {p}: mean={stats.get('delta_vs_fast_mean')}, "
            f"median={stats.get('delta_vs_fast_median')}, "
            f"better_than_fast={stats.get('better_than_fast_count')}/{stats.get('count')}."
        )
    lines.append("")

    if ranking:
        ranking = sorted(ranking, key=lambda item: item[1], reverse=True)
        lines.append("## Mean Ranking")
        for idx, (name, value) in enumerate(ranking, start=1):
            lines.append(f"- {idx}. {name}: {value:.6f}")
        lines.append("")

    lines.append("## Best Policy Counts")
    best_counts = summary["best_policy_counts"]
    assert isinstance(best_counts, dict)
    for p in policies:
        lines.append(f"- {p}: {best_counts.get(p, 0)}")
    lines.append("")

    _ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_nonzero_gain(gain_maps_path: Path) -> bool:
    if not gain_maps_path.exists():
        return False
    with gain_maps_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if abs(float(row.get("gain", 0.0))) > 1e-12:
                return True
    return False


def _generate_experiment_report(
    experiment_root: Path,
    report_root: Path,
    skip_builtin_visualizers: bool,
) -> Optional[Dict[str, object]]:
    experiment_name = experiment_root.name
    try:
        rows = _load_prompt_seed_rows(experiment_root)
    except FileNotFoundError:
        print(f"[plot_results] Skip {experiment_name}: aggregates missing")
        return None

    if not rows:
        print(f"[plot_results] Skip {experiment_name}: no prompt_seed_aggregates")
        return None

    policies = _discover_policies(rows)
    if not policies:
        print(f"[plot_results] Skip {experiment_name}: no policy columns")
        return None

    exp_report_dir = report_root / experiment_name
    plots_dir = exp_report_dir / "plots"
    _ensure_dir(plots_dir)

    # Core plots
    _plot_gap_hist(rows, plots_dir / "heavy_minus_fast_hist.png", f"{experiment_name}: heavy-fast gap")
    _plot_fast_vs_heavy_scatter(rows, plots_dir / "fast_vs_heavy_scatter.png", f"{experiment_name}: fast vs heavy")
    _plot_policy_box(
        {p: _policy_deltas_vs_fast(rows, p) for p in policies},
        plots_dir / "policy_delta_vs_fast_box.png",
        f"{experiment_name}: policy improvement over all_fast",
        "score(policy) - score(all_fast)",
    )
    _plot_policy_box(
        {p: _policy_deltas_vs_heavy(rows, p) for p in policies},
        plots_dir / "policy_delta_vs_heavy_box.png",
        f"{experiment_name}: policy difference vs all_heavy",
        "score(policy) - score(all_heavy)",
    )
    _plot_policy_box(
        {p: _clip(_policy_recoveries(rows, p), -2.0, 2.0) for p in policies},
        plots_dir / "policy_recovery_box_clipped.png",
        f"{experiment_name}: recovery ratio (clipped to [-2, 2])",
        "recovery ratio (clipped)",
    )
    _plot_best_policy_counts(rows, policies, plots_dir / "best_policy_counts.png")
    _plot_pairwise_win_heatmap(rows, policies, plots_dir / "pairwise_win_rate_heatmap.png")
    _plot_oracle_coverage(rows, plots_dir / "oracle_coverage_heatmap.png")
    _plot_oracle_vs_prefix(rows, plots_dir / "oracle_vs_prefix_delta_scatter.png")

    # Optional: regenerate existing experiment visualizations into report folder.
    if not skip_builtin_visualizers:
        agg_dir = experiment_root / "aggregates"
        policy_jsonl = agg_dir / "policy_results.jsonl"
        gain_jsonl = agg_dir / "gain_maps.jsonl"

        if policy_jsonl.exists() and policy_jsonl.stat().st_size > 0:
            try:
                _generate_recovery_barplot_builtin(policy_jsonl, plots_dir / "recovery_barplot_builtin.png")
            except Exception as exc:
                print(f"[plot_results] {experiment_name}: skip builtin recovery barplot ({exc})")

        if _has_nonzero_gain(gain_jsonl):
            try:
                _generate_gain_heatmap_builtin(gain_jsonl, plots_dir / "gain_heatmap_builtin.png")
                _generate_best_vs_median_gain_plot_builtin(gain_jsonl, plots_dir / "best_vs_median_gain_builtin.png")
                _generate_oracle_position_histogram_builtin(gain_jsonl, plots_dir / "oracle_position_histogram_builtin.png")
            except Exception as exc:
                print(f"[plot_results] {experiment_name}: skip builtin gain plots ({exc})")

    # Summary artifacts
    summary = _summarize_experiment(rows, policies)
    summary_path = exp_report_dir / "summary.json"
    _ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_markdown_summary(experiment_name, summary, policies, exp_report_dir / "summary.md")

    print(
        f"[plot_results] {experiment_name}: "
        f"{summary['num_prompt_seed']} prompt-seed aggregates, "
        f"{len(policies)} policies, report -> {exp_report_dir}"
    )
    return {
        "experiment": experiment_name,
        "report_dir": str(exp_report_dir),
        "summary": summary,
    }


def _write_overview(report_root: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Keyframe-Budget Decision Overview")
    lines.append("")
    if not summaries:
        lines.append("- No experiment reports were generated.")
        lines.append("")
        (report_root / "overview.md").write_text("\n".join(lines), encoding="utf-8")
        return

    for entry in summaries:
        exp = str(entry["experiment"])
        summary = entry["summary"]
        assert isinstance(summary, dict)
        hvf = summary["heavy_vs_fast"]
        assert isinstance(hvf, dict)
        cov = summary["oracle_policy_coverage"]
        assert isinstance(cov, dict)
        lines.append(f"## {exp}")
        lines.append(
            f"- Prompt-seed count: {summary['num_prompt_seed']}; "
            f"heavy better in {hvf['heavy_better_count']} cases, worse in {hvf['heavy_worse_count']} cases."
        )
        lines.append(
            f"- Mean heavy-fast gap: {hvf['mean_gap']:.6f}; "
            f"median gap: {hvf['median_gap']:.6f}."
        )
        lines.append(
            f"- Oracle coverage: {cov['count_with_oracle_top_m']} / {summary['num_prompt_seed']}."
        )

        policy_stats = summary["policy_stats"]
        assert isinstance(policy_stats, dict)
        ranking: List[Tuple[str, float]] = []
        for name, stats in policy_stats.items():
            assert isinstance(stats, dict)
            mu = stats.get("delta_vs_fast_mean")
            if mu is not None:
                ranking.append((str(name), float(mu)))
        ranking = sorted(ranking, key=lambda item: item[1], reverse=True)
        if ranking:
            best_name, best_value = ranking[0]
            lines.append(f"- Best mean delta vs fast: {best_name} ({best_value:.6f}).")
        lines.append(f"- Detailed summary: `{exp}/summary.md`")
        lines.append("")

    _ensure_dir(report_root)
    (report_root / "overview.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decision-focused plots and summaries from keyframe-budget outputs."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/keyframe_budget",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="outputs/keyframe_budget/decision_report",
        help="Destination directory for generated report artifacts.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment names under output-root.",
    )
    parser.add_argument(
        "--skip-builtin-visualizers",
        action="store_true",
        help="Skip compatibility-style recovery/gain plots generated from jsonl aggregates.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_dir).resolve()
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    if not experiments:
        raise ValueError("No experiments provided.")

    _ensure_dir(report_root)
    generated: List[Mapping[str, object]] = []
    for name in experiments:
        exp_root = output_root / name
        if not exp_root.exists():
            print(f"[plot_results] Skip {name}: directory not found ({exp_root})")
            continue
        summary = _generate_experiment_report(
            experiment_root=exp_root,
            report_root=report_root,
            skip_builtin_visualizers=bool(args.skip_builtin_visualizers),
        )
        if summary is not None:
            generated.append(summary)

    _write_overview(report_root, generated)
    print(f"[plot_results] Done. Decision report root: {report_root}")
    print(f"[plot_results] Open: {report_root / 'overview.md'}")


if __name__ == "__main__":
    main()
