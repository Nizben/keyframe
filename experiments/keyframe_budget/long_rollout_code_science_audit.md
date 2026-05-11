# Long-Rollout Code and Scientific-Rigor Audit

Date: 2026-05-04

Scope: this audit covers the long-rollout keyframe-budget experiment path, including schedule construction, rollout generation, suffix materialization, VBench launch scripts, VBench aggregation, cheap proxy metrics, existing long-rollout outputs, and experiment documentation. The goal is to identify implementation bugs, suspicious assumptions, and scientific-validity risks before using the results as paper evidence.

## Executive Summary

The core generation path appears structurally correct: schedules are built as intended, `steps_per_chunk` is validated, the long-video pipeline reinitializes the sampler per chunk, and sampled rollout metadata confirms the expected NFE contracts (`all_fast=320`, `all_heavy=960`, `single_heavy_i=336`) with `step_match_ok=true`.

The highest-risk issues are downstream of generation:

- The VBench aggregation parser is too broad and likely mixes official aggregate scores with per-video scores. The evidence is direct: full-video summaries show `n_values=76` for most metrics despite 72 videos, and `background_consistency` shows `n_values=220`. This may not fully explain the `all_heavy < all_fast` result, but it is not clean enough for paper-grade claims.
- The recovery/oracle normalization assumes `score(all_heavy) > score(all_fast)`. VBench violates this for most full-video metrics, so any "recovery toward all_heavy" plot or interpretation is invalid for those metrics.
- Current VBench aggregation loses prompt/seed identity, so the strongest causal/keyframe signal cannot be tested cleanly as paired per-sample deltas.
- `all_heavy` being globally worse than `all_fast` is scientifically suspicious but not impossible. Given the parser and metric-orientation risks, it should be treated as an audit trigger, not as a conclusion.

Bottom line: I would trust the generated rollout architecture more than I would trust the current aggregate VBench interpretation. Before paper-facing claims, fix the VBench parser, preserve per-sample identities, and run paired analyses.

## Files Reviewed

Core experiment code:

- `experiments/keyframe_budget/schedules.py`
- `experiments/keyframe_budget/runner.py`
- `experiments/keyframe_budget/run_experiment.py`
- `experiments/keyframe_budget/oracle.py`
- `experiments/keyframe_budget/aggregate.py`
- `experiments/keyframe_budget/metrics.py`
- `experiments/keyframe_budget/boundaries.py`
- `experiments/keyframe_budget/visualize.py`

Long-video backend:

- `long_video/pipeline/causal_diffusion_inference.py`
- `long_video/configs/rolling_forcing_dmd.yaml`

Long-rollout scripts:

- `experiments/keyframe_budget/scripts/finish_long_rollout_suffix_materialization.slurm`
- `experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes.py`
- `experiments/keyframe_budget/scripts/prepare_long_rollout_suffix_manifest.py`
- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm`
- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_suffixes_36gpus.slurm`
- `experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py`
- `experiments/keyframe_budget/scripts/cheap_chunk_metric_proxy.py`
- related older/smoke scripts in `experiments/keyframe_budget/scripts/`

Experiment artifacts inspected:

- `outputs/long_keyframe_budget/long_rollout_discovery_p000_s0/rollouts/motion_001/0/*/rollout_meta.json`
- `outputs/long_keyframe_budget/long_rollout_discovery/aggregates/*`
- `outputs/long_keyframe_budget/vbench_analysis/*`
- `outputs/long_keyframe_budget/cheap_chunk_metrics/*`
- `experiments/keyframe_budget/long_rollout_experiment_sync.md`

## What Looks Correct

### Schedule Construction

`schedules.py` correctly implements the intended policy families:

- `all_fast`: every chunk uses `fast_steps`.
- `all_heavy`: every chunk uses `heavy_steps`.
- `single_heavy_i`: exactly one chunk uses `heavy_steps`; the others use `fast_steps`.
- `uniform_top_m`, `random_top_m`, `prefix_top_m`, and `oracle_top_m` are compute-matched through `assert_equal_total_steps` for mixed policies.

The sampled output metadata confirms this is not just theoretical. For `motion_001`, seed `0`:

- `all_fast`: 40 chunks, all step counts 8, `total_nfe=320`, `requested_total_steps=320`.
- `all_heavy`: 40 chunks, all step counts 24, `total_nfe=960`, `requested_total_steps=960`.
- `single_heavy_00`: first chunk 24, remaining chunks 8, `total_nfe=336`, `requested_total_steps=336`.
- `single_heavy_39`: last chunk 24, earlier chunks 8, `total_nfe=336`, `requested_total_steps=336`.

All sampled rollouts had `status=success`, `step_match_ok=true`, and `budget_match_ok=true`.

### Step Counts Are Actually Used

In `runner.py`, the rollout calls `pipeline.inference(..., steps_per_chunk=spec.steps_per_chunk, return_chunk_logs=True)`. In `long_video/pipeline/causal_diffusion_inference.py`, the pipeline checks `len(steps_per_chunk) == num_blocks` and uses:

```python
sampling_steps = int(steps_per_chunk[chunk_idx])
sample_scheduler = self._initialize_sample_scheduler(noise, sampling_steps=sampling_steps)
```

This is the key architectural requirement for the long-rollout experiment. I do not see evidence that the schedule is ignored.

### Deterministic Pairing Is Mostly Preserved

`runner.py` calls `set_seed(spec.seed)` immediately before sampling `sampled_noise`. This means different schedules for the same prompt/seed should use the same initial noise, so comparisons are paired rather than independent.

The VBench full and suffix Slurm scripts also enforce common prompt/seed sets across cohorts. This is scientifically important and is one of the stronger parts of the current setup.

### Suffix Materialization Is Mostly Safe and Re-runnable

The suffix materialization script uses `chunk_boundaries.json`, computes visible start/end indices from latent chunk boundaries, and writes one `suffix_from_XX.mp4` per chunk. The finishing Slurm script is re-runnable: it skips completed rollout dirs if the suffix manifest has 40 clips and all 40 suffix videos exist with nonzero size.

The finish script also restricts discovery to `long_rollout_discovery_p???_s?`, which prevents accidental traversal of unrelated output folders.

## High-Priority Findings

### 1. VBench Aggregation Parser Is Too Broad

Severity: high.

File: `experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py`

The parser recursively collects all numeric values underneath keys matching the metric name. This is defensive, but it is too permissive for final analysis. The current aggregate output strongly suggests double counting or schema mixing:

- Full videos have 72 prompt/seed videos per cohort.
- Most full metrics report `n_values=76`, not 72.
- `background_consistency` reports `n_values=220`.
- `dynamic_degree` reports only `n_values=4`, likely one aggregate per GPU shard.

This means different metrics are not being aggregated under the same statistical unit. Some metrics may be averaging per-video scores plus one shard aggregate; others may be averaging nested sub-scores; `dynamic_degree` may be averaging only shard-level outputs.

Risk:

- Cohort means may be slightly or substantially biased depending on VBench JSON schema.
- Variance/standard deviation is not interpretable.
- Claims like "`all_fast` beats `all_heavy` globally" are not paper-grade until parser correctness is verified against raw VBench JSON.

Required fix:

- Inspect raw VBench JSON files on Jean Zay for each metric.
- Implement metric-specific extraction of official per-video records where possible.
- Preserve `(prompt_id, seed, schedule, suffix_start, chunk_id)` in every record.
- Recompute summaries from per-video rows, not from recursively collected numeric leaves.

### 2. Recovery Normalization Assumes All-Heavy Is Better

Severity: high.

Files:

- `experiments/keyframe_budget/aggregate.py`
- `experiments/keyframe_budget/oracle.py`
- `experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py`
- `experiments/keyframe_budget/visualize.py`

The current normalization uses:

```python
(policy_score - all_fast_score) / (all_heavy_score - all_fast_score)
```

This is only meaningful when higher is better and `all_heavy_score > all_fast_score`. The current full-video VBench results violate that assumption for most metrics:

- `aesthetic_quality`: all_heavy - all_fast = `-0.016596`
- `background_consistency`: `-1.462924`
- `imaging_quality`: `-2.280725`
- `motion_smoothness`: `-0.005625`
- `subject_consistency`: `-0.028903`
- `overall_consistency`: `+0.002594`
- `dynamic_degree`: `0.0`

Risk:

- Recovery plots can invert meaning.
- Negative denominators make "better than all_fast" appear as negative recovery.
- Near-zero denominators explode ratios.
- Oracle gain normalization can exaggerate values when the all-fast/all-heavy gap is small.

Required fix:

- Stop using all-heavy-normalized recovery for VBench unless all-heavy is verified as a valid upper anchor for that metric.
- Report raw paired deltas versus `all_fast`.
- Separately report "delta versus all_heavy" only as a diagnostic, not as recovery.
- For oracle discovery, consider ranking by raw single-heavy gain and reporting denominator sign explicitly.

### 3. VBench Aggregation Loses Paired Sample Identity

Severity: high.

File: `experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py`

The VBench Slurm scripts create symlink names containing sample keys, but the aggregator only produces cohort/chunk means. It does not retain per-video records keyed by prompt and seed.

Risk:

- The strongest hypothesis is about non-uniform causal leverage per video/prompt/seed, but the analysis mostly averages globally across 72 samples.
- Keyframe effects may be prompt-specific and disappear in global means.
- We cannot compute paired confidence intervals, sign tests, per-prompt rank correlations, or prompt-level heterogeneity from the current CSVs.

Required fix:

- Parse sample IDs from VBench result rows or from symlink filenames.
- Generate a long-form CSV with one row per evaluated video:
  `split, metric, prompt_id, seed, schedule, heavy_idx, suffix_start, score`.
- Use paired deltas:
  `score(single_heavy_i, prompt, seed) - score(all_fast, prompt, seed)`.
- Analyze within-prompt/seed rank structure before global averaging.

### 4. `all_heavy < all_fast` Is Suspicious but Not Yet Diagnosed

Severity: high scientific-risk finding.

Current full-video VBench means show `all_heavy` below `all_fast` for 5 of 7 metrics. This does not match the naive scientific expectation that more denoising steps should improve quality.

Possible explanations:

- Real model behavior: this causal/distilled rollout regime may not be monotonic in step count, especially with recurrent cached context and a DMD/flow-matching distillation setup.
- Metric mismatch: VBench metrics may reward smoothness/static consistency or image priors that are not the same as subjective long-video quality.
- Parser bug: current aggregation may be mixing aggregate and per-video values.
- Evaluation artifact: VBench custom-input mode may handle long videos, suffixes, frame sampling, or metric-specific preprocessing differently than expected.
- Generation bug not seen in schedule code: less likely based on sampled metadata, but still possible without a full metadata sweep and visual spot-check.

Interpretation:

The user's theory that "this smells like a bug" is reasonable. I would not present `all_fast > all_heavy` as a scientific finding yet. I would present it only as an audit anomaly that forced a deeper parser/raw-video check.

## Medium-Priority Findings

### 5. Resume Logic Can Skip Incomplete VBench Chunks

Severity: medium.

Files:

- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm`
- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_suffixes_36gpus.slurm`

The resume guard skips a GPU chunk if any nonempty `*_eval_results.json` exists in the output directory:

```bash
find "${CUSTOM_OUT_DIR}" -maxdepth 1 -type f -name '*_eval_results.json' -size +0c
```

Risk:

- If VBench wrote a partial/corrupt/incomplete JSON before failing, reruns could skip it.
- The script does not verify that the result contains all expected videos.

Required fix:

- Store an expected manifest next to every output chunk.
- On resume, validate that the eval JSON contains the expected number of video results.
- If counts mismatch, rerun that chunk.

### 6. Suffix Windows Near the End Are Shorter

Severity: medium.

File: `experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes.py`

Suffixes are clipped to the remaining video:

```python
latent_end_for_suffix = min(latent_start + suffix_window_latent, num_output_frames_latent)
```

This is safe, but late suffix starts have shorter horizons. For example, `suffix_from_39` covers only the final chunk tail, not a full 32-latent-frame suffix.

Risk:

- Comparing suffix starts directly can conflate intervention position with suffix duration.
- Fairness is valid within a fixed `suffix_from_XX` cohort, but not across suffix-start cohorts unless duration is modeled.

Required fix:

- Add `num_frames` or `visible_duration_sec` to every suffix VBench record.
- Analyze fixed suffix starts separately.
- Avoid claiming that suffix start A is better than suffix start B unless durations are equal or controlled.

### 7. Cheap Proxy Experiment Uses an Ambiguous Oracle Target

Severity: medium.

File: `experiments/keyframe_budget/scripts/cheap_chunk_metric_proxy.py`

The cheap metrics compare all-fast motion features against oracle ranks derived from the in-repo proxy `score`, not VBench. That proxy score is:

```python
0.60 * temporal_diff + 0.35 * sharpness - 0.05 * brightness_std
```

Risk:

- The target is not the same as final VBench quality.
- It heavily rewards motion and sharpness; the "hard chunk" target may partly be a proxy self-fulfilling signal.
- `gain_norm` inherits the all-heavy denominator issue.

Current result:

- Best single cheap feature is shallow: `suffix_frame_l1` has mean within-sample Spearman about `0.216` and top-10 recall about `0.394`.
- LOPO rank-combination does not improve meaningfully: mean top-10 recall about `0.375`, mean Spearman about `0.097`.

Interpretation:

The cheap non-learned metrics provide weak signal. They are useful as diagnostic features, not as a validated proxy for hard chunks.

### 8. Long-Rollout Reproducibility Is Spread Across Scripts and Environment Overrides

Severity: medium.

Files:

- `experiments/keyframe_budget/configs/discovery_split.yaml`
- `experiments/keyframe_budget/scripts/keyframe_budget_core.slurm`
- `experiments/keyframe_budget/scripts/discovery_split_array.slurm`
- `long_video/configs/rolling_forcing_dmd.yaml`

The committed `discovery_split.yaml` is the old short-video/default path. The actual long-rollout setup depends on Slurm environment overrides such as `MODEL_CONFIG_PATH`, `CHECKPOINT_PATH`, `NUM_OUTPUT_FRAMES`, `OUTPUT_ROOT`, and `EXPERIMENT_NAME`.

Risk:

- Exact reproduction is fragile.
- A future run could silently use the short-video config if environment variables are missing.
- The long-video config file contains `generator_ckpt: ../checkpoints/chunkwise/causal_ode.pt`, while the experiment likely loads `checkpoints/chunkwise/longvideo.pt` through the runner checkpoint override.

Required fix:

- Commit a canonical `configs/long_rollout_discovery.yaml`.
- Commit a canonical `configs/long_rollout_policy_eval.yaml` when policy eval is implemented.
- Write the fully resolved config into each rollout directory or experiment root.

### 9. VBench Text-Conditioned Metrics Would Be Invalid in Current Script Pattern

Severity: medium.

The current VBench scripts run `custom_input` mode without per-video prompt files. That is acceptable for the current selected no-text metrics, but it would not be acceptable for text-alignment or prompt-consistency dimensions.

Risk:

- If future metrics are added casually, they may run without prompt conditioning and produce meaningless numbers.

Required fix:

- Explicitly gate allowed VBench metrics.
- If text metrics are needed, generate prompt files keyed to symlink sample names.

### 10. Dynamic Degree Is Saturated

Severity: medium scientific limitation.

In full-video VBench, `dynamic_degree` is `1.0` for `all_fast`, `all_heavy`, and every `single_heavy_i`. It provides no ranking signal for the current dataset.

Recommendation:

- Exclude it from keyframe claims.
- Keep it only as a sanity metric that the videos are classified as dynamic.

## Lower-Priority Findings and Cleanup Items

### 11. Documentation Contains a Stale Assumption

File: `experiments/keyframe_budget/long_rollout_experiment_sync.md`

The document still states that `all_heavy` being better than `all_fast` is expected and central to the gap story. Later sections correctly say VBench does not support that naive story. This should be revised to avoid internal contradiction.

### 12. Older Short-Video Experiment Files Can Be Confused with Long-Rollout Work

Files:

- `experiments/keyframe_budget/configs/discovery_split.yaml`
- `experiments/keyframe_budget/configs/policy_eval_100.yaml`
- `experiments/keyframe_budget/scripts/policy_eval_100*.slurm`

These are valid historically but not the current long-rollout target. They make it easy to confuse short-video policy eval with the missing long-rollout policy eval.

Recommendation:

- Add explicit comments to old configs.
- Add new long-rollout configs.

### 13. VBench Smoke Script Has a Variable Naming Mismatch

File: `experiments/keyframe_budget/scripts/vbench_smoke_long_rollout_preflight.slurm`

The smoke script exports `GPUS_PER_JOB`, while the newer full/suffix scripts use `GPUS_PER_TASK`. This may be harmless if the smoke calls the older 32-GPU script, but it is confusing and should be normalized.

### 14. Plots Can Encode Misleading Semantics

Files:

- `experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py`
- `experiments/keyframe_budget/visualize.py`

Plots label recovery `0=all_fast`, `1=all_heavy`, even when all-heavy is worse. This is visually dangerous.

Recommendation:

- Replace recovery plots with raw delta curves.
- Draw both all-fast and all-heavy baselines as score lines, not semantic anchors.

## Current Empirical Signals

### Full-Video VBench

From `outputs/long_keyframe_budget/vbench_analysis/vbench_cohort_summary.csv`:

| Metric | all_fast | all_heavy | heavy-fast | Single-heavy above all_fast |
| --- | ---: | ---: | ---: | ---: |
| imaging_quality | 46.010613 | 43.729888 | -2.280725 | 30/40 |
| overall_consistency | 0.146882 | 0.149476 | +0.002594 | 30/40 |
| aesthetic_quality | 0.473274 | 0.456677 | -0.016596 | 8/40 |
| subject_consistency | 0.725243 | 0.696340 | -0.028903 | 5/40 |
| motion_smoothness | 0.932596 | 0.926971 | -0.005625 | 9/40 |
| background_consistency | 285.471948 | 284.009024 | -1.462924 | 14/40 |
| dynamic_degree | 1.000000 | 1.000000 | 0.000000 | 0/40 |

The most consistent positive signal is not "all-heavy improves everything." It is "some sparse single-heavy placements outperform all-fast on some metrics." For example, `imaging_quality` has 30 of 40 single-heavy chunk placements above all-fast, with top chunks around indices 2-4 and 8.

### Sparse Suffix VBench

The suffix grid is useful but should be interpreted by fixed suffix start. Within a fixed suffix start, cohorts are fair. Across suffix starts, duration and position differ.

Observed qualitative signal so far:

- `aesthetic_quality`: later suffix starts often prefer `single_heavy_00`.
- `motion_smoothness`: later suffix starts often prefer `single_heavy_12`.
- `subject_consistency`: effects are smaller and scattered.
- `dynamic_degree`: mostly saturated/discrete and not useful.

This supports the idea that single early interventions can affect later suffixes, but the signal is metric-specific and not yet cleanly linked to per-sample oracle structure.

### Cheap Proxy Metrics

The non-learned cheap metrics show shallow but nonzero correlation with oracle ranks:

- `suffix_frame_l1`: mean within-sample Spearman about `0.216`, mean top-10 recall about `0.394`.
- `frame_l1`: mean within-sample Spearman about `0.197`, mean top-10 recall about `0.378`.
- LOPO rank-combination: mean top-10 recall about `0.375`, mean Spearman about `0.097`.

This is not strong enough to replace hard evaluation. It is enough to motivate more diagnostics.

## Scientific Fairness Assessment

### Strengths

- Prompt/seed pairing is preserved in generation.
- Discovery cohort has 24 prompts x 3 seeds = 72 paired samples.
- Full VBench scripts enforce common sample sets across cohorts.
- Sparse suffix VBench scripts enforce common sample sets across selected suffix cohorts.
- Baseline and intervention policies share the same prompt/seed/noise setup.
- Single-heavy schedules have an interpretable one-chunk perturbation.

### Weaknesses

- The analysis currently averages globally and loses paired identities.
- The all-heavy baseline is not a reliable upper anchor under VBench.
- The VBench parser may mix aggregation units.
- Some metrics are saturated or metric-specific in ways that weaken general claims.
- Suffix windows are not equal duration near the end.
- The cheap proxy target is the internal proxy oracle, not final human/VBench quality.

## Trust Level by Component

| Component | Trust Level | Rationale |
| --- | --- | --- |
| Schedule constructors | High | Simple code, validated by metadata. |
| Per-chunk step application | High | Pipeline uses `steps_per_chunk[chunk_idx]`; metadata confirms actual NFE. |
| Rollout metadata | Medium-high | Good contracts, but full sweep validation still recommended. |
| Suffix materialization | Medium-high | Uses explicit boundaries; end-window duration caveat. |
| VBench Slurm fairness | Medium-high | Common sample enforcement is good; resume completeness is weak. |
| VBench aggregation | Low-medium | Broad parser and lost sample identity are serious issues. |
| Recovery/oracle normalization for VBench | Low | Invalid when all-heavy is not better. |
| Cheap proxy conclusions | Medium-low | Signal exists but is weak and target is internal proxy. |

## Required Next Actions Before Paper-Facing Claims

1. Audit raw VBench JSON schema on Jean Zay for each metric.
2. Rewrite `aggregate_long_rollout_vbench.py` to emit one row per video/sample, not just cohort means.
3. Recompute full and suffix summaries with paired prompt/seed deltas.
4. Replace recovery plots with raw delta plots and paired confidence intervals.
5. Run a complete metadata sweep over all generated rollouts:
   - all `all_fast` should have `total_nfe=320`.
   - all `all_heavy` should have `total_nfe=960`.
   - all `single_heavy_i` should have `total_nfe=336`.
   - all should have `step_match_ok=true`.
6. Visually inspect a small but targeted sample where VBench says all-heavy is much worse than all-fast.
7. Add canonical long-rollout config YAMLs for discovery and policy eval.
8. Implement the long-rollout policy evaluation Slurm for oracle, uniform/periodic, random, and prefix once the VBench parsing is fixed.

## Recommended Paper-Safe Framing Right Now

Safe:

- "The long-rollout architecture successfully supports per-chunk compute interventions."
- "Sparse single-heavy interventions produce non-uniform VBench effects across chunk positions."
- "The strongest current signal is metric-dependent sparse placement, not monotonic improvement from more compute everywhere."
- "Cheap non-learned motion features show weak but nonzero alignment with hard chunks."

Not safe yet:

- "`all_fast` is scientifically better than `all_heavy`."
- "`all_heavy` is a valid oracle upper bound."
- "The normalized recovery curves prove keyframe budget recovery."
- "Cheap metrics can replace expensive evaluation."
- "The globally best chunk index is universal across prompts."

## Final Assessment

The experiment architecture is plausible and mostly implemented correctly at the generation level. The current results are not yet clean enough for final scientific claims because the analysis layer has concrete validity issues. The most important fix is to rebuild VBench aggregation around per-video, prompt/seed-paired records and remove all-heavy-normalized recovery from metrics where all-heavy is not a valid upper baseline.

After that fix, the right analysis is not "does all-heavy beat all-fast globally?" The right analysis is "for each prompt/seed, do a small number of single-heavy chunk placements produce reproducible positive deltas, and are those placements non-uniform, suffix-persistent, and predictable by cheap non-learned features?"
