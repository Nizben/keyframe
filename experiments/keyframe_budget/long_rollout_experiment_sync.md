# Long-Rollout Keyframe Budget Experiment Sync

This document is the current reference for the long-rollout keyframe-budget experiments. It records the experiment logic, what has already been run, what is currently running, and what signal we want to extract for the paper and the next meeting.

Last updated: 2026-04-27.

## Core Claim

The experiment tests whether compute has sparse causal value over time in long video generation.

The key hypothesis is:

> In long autoregressive rollouts, only a small subset of chunks has large downstream impact on video quality. Those chunks behave like keyframes or key moments: spending extra denoising compute there should improve later content, motion, and consistency more than spending the same compute elsewhere.

The important point is not just that `all_heavy` is better than `all_fast`. That is expected. The scientific question is whether individual chunks have non-uniform causal leverage, and whether a small set of high-leverage chunks can explain a meaningful fraction of the quality gap.

## Why Long Rollouts

Short clips are not ideal for this hypothesis because there is limited future horizon for an intervention to matter. The long-rollout setting is where the signal should live:

- A chunk can affect future motion, scene identity, object persistence, and coherence.
- Errors can compound over many future chunks.
- Heavy compute at a single chunk may have downstream effects beyond the local frames.
- Suffix evaluation can isolate downstream consequences from a chosen chunk onward.

This is why we are now treating the old short-video `outputs/keyframe_budget` experiments as secondary and focusing on:

- `outputs/long_keyframe_budget`
- `outputs/long_keyframe_budget_suffixes`

## Generation Architecture

The audited long-rollout discovery set uses the `long_video` backend.

Generation constants:

- backend: `long_video`
- config: `long_video/configs/rolling_forcing_dmd.yaml`
- checkpoint: `checkpoints/chunkwise/longvideo.pt`
- latent output frames: `120`
- visible output frames: `477`
- fps: `16`
- latent frames per chunk: `3`
- total chunks: `40`
- fast denoising steps: `8`
- heavy denoising steps: `24`
- fixed-budget target average: `12`
- fixed-budget heavy chunk count: `m = 10`

The visible frame count is `477`, not `480`, because of the WAN latent-to-visible frame mapping. This was checked during the generation audit and is consistent with the architecture.

## Discovery Design

For each prompt and seed, the discovery sweep contains:

- `all_fast`: all 40 chunks use 8 steps.
- `all_heavy`: all 40 chunks use 24 steps.
- `single_heavy_00` through `single_heavy_39`: exactly one chunk uses 24 steps, all others use 8 steps.

Scale:

- 24 prompts
- 3 seeds
- 72 prompt-seed samples
- 42 rollouts per prompt-seed sample
- 3024 full videos total

The discovery gain map is:

```text
gain_i = score(single_heavy_i) - score(all_fast)
gain_norm_i = gain_i / (score(all_heavy) - score(all_fast))
```

The chunk ranking induced by `gain_norm_i` is the empirical estimate of which chunks are keyframe-like for that prompt and seed.

## Suffix Design

Suffixes are materialized offline from existing full videos. They are not new generations.

For a rollout and chunk start `t`, `suffix_from_t` is cut from:

- `full.mp4`
- `chunk_boundaries.json`
- `rollout_meta.json`

The intended suffix horizon is fixed:

- `suffix_window_latent = 32`
- clipped near the end of the video when needed

This matters because it avoids giving early chunks much longer evaluation windows than late chunks. The suffix protocol asks:

> If chunk `i` is made heavy, does the downstream suffix from time `t` improve, and where is that effect strongest?

## What Has Been Completed

The long-rollout discovery generation is complete and audited.

Current confirmed state:

- `outputs/long_keyframe_budget` contains the 72 discovery shard folders.
- Each shard corresponds to one prompt-seed pair.
- Each shard is expected to contain 42 full rollouts.
- Sampled videos match the expected architecture: `832x480`, `16 fps`, `477` frames.
- The merged discovery aggregate exists under `outputs/long_keyframe_budget/long_rollout_discovery/aggregates`.

Suffix materialization was completed after repairing two missing source rollouts:

- `motion_017/0/single_heavy_27`
- `motion_024/1/single_heavy_25`

The repair was run with:

```bash
sbatch --array=2045,2967 experiments/keyframe_budget/scripts/finish_long_rollout_suffix_materialization.slurm
```

The repair logs showed both tasks completed and created 40 suffix clips each.

## VBench Infrastructure

Two VBench Slurm scripts exist:

- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm`
- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_suffixes_36gpus.slurm`

Both scripts:

- scan only `long_rollout_discovery_p???_s?` folders
- use strict prompt-seed fairness by default
- support `RESUME_SKIP_COMPLETED=1`
- support `COHORT_FILTER_REGEX`
- use 7 metrics x 9 cohort shards = 63 array tasks
- use 4 H100 GPUs per task
- cap concurrency at 9 tasks, i.e. up to 36 H100 GPUs

Both scripts have been patched so temporary VBench shard symlinks default to node-local scratch:

```bash
TMP_BASE="${SLURM_TMPDIR:-/tmp}/..."
```

This avoids rebuilding huge persistent symlink trees under `/lustre/.../VBench/temp_shards`.

## VBench Problems Found

The first full-video VBench run failed because discovery scanned too broadly and picked up non-discovery folders such as `long_rollout_stage0`. That created duplicate sample keys. The script now filters strictly to discovery shards.

The first suffix VBench run failed because strict fairness correctly detected missing suffix cohorts. The missing suffixes came from two missing source rollouts. Those were repaired.

The next VBench runs failed because of storage quota:

- full-video VBench completed `dynamic_degree` and part of `motion_smoothness`
- suffix VBench failed during `dynamic_degree`
- failures happened while writing VBench JSONs and while creating persistent temp symlinks

Interpretation:

- The experimental data are structurally sound.
- The main blocker was storage pressure, likely inode quota as much as byte quota.
- Suffix VBench is much larger than full-video VBench because the naive design has `42 policies x 40 suffix starts = 1680` suffix cohorts.

## Sparse Suffix Strategy

To keep scientific integrity while reducing output size, we switched to a fixed sparse grid instead of evaluating every suffix cohort.

The sparse grid keeps all 72 prompt-seed samples, but reduces the evaluated policy and suffix axes.

Policy grid:

- `all_fast`
- `all_heavy`
- `single_heavy_00`
- `single_heavy_04`
- `single_heavy_08`
- `single_heavy_12`
- `single_heavy_16`
- `single_heavy_20`
- `single_heavy_24`
- `single_heavy_28`
- `single_heavy_32`
- `single_heavy_36`
- `single_heavy_39`

Suffix-start grid:

- `suffix_from_00`
- `suffix_from_04`
- `suffix_from_08`
- `suffix_from_12`
- `suffix_from_16`
- `suffix_from_20`
- `suffix_from_24`
- `suffix_from_28`
- `suffix_from_32`
- `suffix_from_36`
- `suffix_from_39`

This reduces suffix cohorts from:

```text
42 policies x 40 suffix starts = 1680 cohorts
```

to:

```text
13 policies x 11 suffix starts = 143 cohorts
```

The key fairness property is preserved: every selected cohort is evaluated on the same 72 prompt-seed samples.

The sparse suffix smoke test passed:

- job: `252025`
- selected cohorts: `143`
- strict fairness: `72/72` samples for every selected cohort
- smoke cap: `4 / 72` samples evaluated
- temp shards used `/tmp`
- `dynamic_degree` completed with no quota error

## GPU Queue Status

The two main VBench jobs were submitted on 2026-04-26, but they did not start overnight. As of the morning of 2026-04-27, both are still pending in Slurm with reason `Priority`.

This means the meeting cannot rely on fresh VBench results from these jobs. The meeting story should use the already completed discovery audit, suffix materialization audit, strict-fairness checks, sparse-suffix smoke test, and cheap-metric proxy results. The queued VBench jobs remain the next validation step, not available evidence yet.

Pending sparse suffix VBench:

```text
JOBID: 252837
STATE: PD
REASON: Priority
ARRAY: 0-17,36-53%9
```

Pending full-video VBench resume:

```text
JOBID: 252841
STATE: PD
REASON: Priority
ARRAY: 10-62%9
```

## Queued VBench Jobs

Two main VBench jobs were submitted on 2026-04-26.

Sparse suffix VBench:

```bash
export SPARSE_SUFFIX_REGEX='^(all_fast|all_heavy|single_heavy_(00|04|08|12|16|20|24|28|32|36|39))__suffix_from_(00|04|08|12|16|20|24|28|32|36|39)$'

COHORT_FILTER_REGEX="$SPARSE_SUFFIX_REGEX" \
RUN_TAG=long_rollout_suffix_sparse_grid_4metrics_36gpus \
RESUME_SKIP_COMPLETED=1 \
sbatch --array=0-17,36-53%9 experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_suffixes_36gpus.slurm
```

Submitted job:

- `252837`

This evaluates four VBench metrics:

- `dynamic_degree`
- `motion_smoothness`
- `aesthetic_quality`
- `subject_consistency`

Full-video VBench resume:

```bash
RESUME_SKIP_COMPLETED=1 \
sbatch --array=10-62%9 experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm
```

Submitted job:

- `252841`

Rationale for starting at task `10`:

- full-video `dynamic_degree` tasks `0-8` already completed
- full-video `motion_smoothness` task `9` completed
- remaining full-video tasks are `10-62`

## Signal Available Now

The currently available signal is not final VBench evidence. It is a combination of dataset-level, protocol-level, and cheap-proxy evidence.

Available evidence:

- The long-rollout discovery set exists at full intended scale: 72 prompt-seed samples and 3024 full videos.
- The generation architecture was audited: sampled outputs match the expected long-video architecture, including 477 visible frames.
- The discovery design is exhaustive over single-heavy interventions, so it is capable of measuring per-chunk causal sensitivity.
- Strict suffix fairness now passes after repairing the two missing source rollouts.
- The sparse suffix smoke test passed: 143 sparse cohorts were selected, every selected cohort had 72/72 samples, and VBench completed on the capped smoke subset without quota failure.
- Cheap non-learned metrics show above-random alignment with hard-chunk rankings.
- Full-video VBench `dynamic_degree` completed all 9 cohort shards from the earlier run, so dynamic-degree scores should exist for the full-video discovery sweep. They still need to be aggregated into a table before being used as quantitative meeting evidence.
- Full-video VBench `motion_smoothness` only completed shard `0/9`; the other shards failed from quota, so it is not yet usable as a full sweep.
- Sparse suffix VBench only has the smoke-test `dynamic_degree` result so far; the real sparse suffix job stayed pending overnight.

The cleanest meeting claim is therefore:

> We have built and audited the long-rollout discovery dataset needed to test keyframe existence. Preliminary non-learned proxy evidence suggests chunk importance is not arbitrary noise. The queued VBench jobs are designed to test whether this signal appears in perceptual full-video and suffix metrics.

This is weaker than the final policy claim, but it is scientifically honest and still useful.

## Cheap Metric Proxy Experiment

We also tested whether cheap, non-learned metrics can predict hard chunks.

The current cheap metrics include:

- local frame L1
- local edge L1
- local optical-flow magnitude
- suffix-window frame L1
- suffix-window edge L1
- suffix-window optical-flow magnitude
- boundary discontinuity features
- acceleration features
- suffix-minus-local contrast features

Important constraint:

- no learned image encoder
- no neural semantic metric
- only cheap non-learned statistics from existing videos

The best simple proxy so far is `suffix_frame_l1`.

Observed cheap-metric signal:

- `suffix_frame_l1` mean within-sample Spearman around `0.216`
- `suffix_frame_l1` top-10 recall around `0.394`
- random top-10 baseline is `0.25`
- `frame_l1` mean within-sample Spearman around `0.197`
- `frame_l1` top-10 recall around `0.378`
- `edge_l1` top-10 recall around `0.392`
- leave-one-prompt-out rank-combination mean top-10 recall around `0.375`
- leave-one-prompt-out rank-combination mean Spearman around `0.097`

The rank-combination score did not clearly improve over the simple best metric. The useful conclusion is not that the cheap proxy solves keyframe detection, but that a shallow non-learned proxy already carries above-random information about hard chunks.

For the meeting, this supports:

> Chunk importance is not arbitrary noise; it leaves a measurable footprint in cheap video dynamics.

## VBench Status Update

The earlier state was partial: `dynamic_degree` had completed first, while other full-video and sparse-suffix metrics were delayed by queueing and quota issues. This is now superseded by the completed VBench aggregation below.

## What Signal We Want For The Meeting

The meeting goal is not to prove the full final policy story yet. The immediate goal is to show credible evidence that keyframe-like chunks exist.

The strongest near-term signals are:

1. Non-uniform single-heavy gain maps.

If one heavy chunk gives much larger gain than the median chunk for the same prompt and seed, that supports sparse causal importance.

2. Concentrated gain rankings.

If top chunks explain a disproportionate share of positive gain, this supports a keyframe-budget story.

3. Sparse suffix VBench effects.

If `single_heavy_i` improves suffix quality most when suffix windows include or follow chunk `i`, and if effects are larger for some `i` than others, that is direct downstream evidence. This is now available for the sparse suffix grid on four metrics.

4. Full-video VBench sweep.

If full-video VBench differs meaningfully across `single_heavy_i`, that shows the keyframe effect is visible at the final artifact level, not only in internal scores. This is now available for all seven full-video VBench metrics.

5. Cheap metric correlation.

If cheap suffix dynamics correlate above random with the hard-chunk ranking, that suggests there may be a practical proxy for deciding where compute should go.

## Completed VBench Aggregation

The queued VBench jobs eventually completed and were aggregated with:

```bash
python experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py
```

Aggregation output:

```text
outputs/long_keyframe_budget/vbench_analysis/
```

Generated files:

- `vbench_eval_records.csv`
- `vbench_cohort_summary.csv`
- `full_normalized_recovery.csv`
- `summary.json`
- plots under `outputs/long_keyframe_budget/vbench_analysis/plots/`

Aggregation status:

- records: `3464`
- cohort summaries: `866`
- warnings: `0`
- full-video VBench: `7` metrics x `42` cohorts
- sparse suffix VBench: `4` metrics x `143` cohorts

Full-video metric coverage:

- `dynamic_degree`: 42 cohorts
- `motion_smoothness`: 42 cohorts
- `overall_consistency`: 42 cohorts
- `imaging_quality`: 42 cohorts
- `aesthetic_quality`: 42 cohorts
- `subject_consistency`: 42 cohorts
- `background_consistency`: 42 cohorts

Sparse suffix metric coverage:

- `dynamic_degree`: 143 cohorts
- `motion_smoothness`: 143 cohorts
- `aesthetic_quality`: 143 cohorts
- `subject_consistency`: 143 cohorts

## VBench Findings

The VBench results are useful, but they do not support the naive story that `all_heavy` is uniformly better than `all_fast`.

Important caveat:

- For these VBench metrics, high-is-better is the safer convention.
- `all_heavy` is often worse than `all_fast`.
- Therefore, normalized recovery toward `all_heavy` is not always the right interpretation.
- The more useful first-pass read is single-heavy delta relative to `all_fast`.

Full-video findings:

- `dynamic_degree` is saturated: `all_fast`, `all_heavy`, and all `single_heavy_i` are `1.0`. It is not informative for this dataset.
- `imaging_quality` shows the clearest full-video single-heavy signal. Several early chunks improve over `all_fast`.
- `overall_consistency` also shows a non-uniform single-heavy signal.
- `background_consistency` has non-uniform effects, with a few chunks above `all_fast`.
- `aesthetic_quality`, `motion_smoothness`, and `subject_consistency` have smaller single-heavy effects.

Full-video examples, high-is-better deltas against `all_fast`:

- `imaging_quality`: best chunks are `3`, `2`, `8`, `1`, `4`; chunk `3` is about `+0.59`.
- `overall_consistency`: best chunks are `3`, `7`, `1`, `12`, `0`; chunk `3` is about `+0.00097`.
- `background_consistency`: best chunks are `2`, `3`, `12`, `8`, `9`; chunk `2` is about `+0.291`.
- `motion_smoothness`: best chunks are `13`, `12`, `16`, `22`, `26`, but effects are small.
- `subject_consistency`: best chunks are `3`, `15`, `27`, `28`, `26`, but effects are very small.

Full-video baseline behavior:

- `all_heavy - all_fast` is negative for `aesthetic_quality`, `background_consistency`, `imaging_quality`, `motion_smoothness`, and `subject_consistency`.
- `all_heavy - all_fast` is positive only for `overall_consistency`.
- `dynamic_degree` has no difference.

Sparse suffix findings:

- The sparse suffix run is complete for the intended four metrics.
- `dynamic_degree` is mostly saturated or near-discrete and is not very informative.
- `aesthetic_quality`, `motion_smoothness`, and `subject_consistency` show non-uniform suffix effects.
- The suffix signal is not mostly diagonal: the best heavy chunk for a suffix window is not usually the same as the suffix start.
- This suggests that global averaging over prompt/seed may be hiding prompt-specific causal structure.

Sparse suffix examples, high-is-better deltas against `all_fast`:

- `aesthetic_quality`: later suffixes often prefer `heavy_00`; some windows prefer `heavy_32`, `heavy_36`, or `heavy_12`.
- `motion_smoothness`: late suffixes often prefer `heavy_12`; deltas grow for later suffix starts.
- `subject_consistency`: improvements are small and scattered; `heavy_04`, `heavy_20`, `heavy_24`, `heavy_32`, `heavy_36`, and `heavy_39` appear in different windows.

Interpretation:

- The results support “sparse compute placement matters” more than “more compute everywhere helps.”
- The keyframe effect is metric-dependent.
- The keyframe effect is likely prompt/seed-dependent.
- Global chunk averages are probably too blunt to recover the strongest oracle signal.
- The next analysis should compare VBench deltas with the discovery oracle ranks per prompt/seed, not only through global curves.

## Next Experiments And Analyses

The next step is no longer to get VBench to run; it is to extract the right signal from the completed VBench outputs.

1. Per prompt/seed VBench-vs-oracle alignment.

The current aggregation averages over all 72 prompt-seed samples inside each cohort. That is useful for global sanity, but it likely washes out the keyframe signal. The next analysis should recover per-sample VBench scores and compare:

- discovery oracle rank of chunk `i`
- VBench delta of `single_heavy_i` relative to `all_fast`
- suffix VBench delta for sparse `suffix_from_t`

The target question is:

> Do chunks ranked important by the discovery score also produce better VBench outcomes for that prompt and seed?

2. Metric-specific keyframe maps.

The current results already show that different metrics prefer different chunks. We should generate per-metric maps:

- top chunks for `imaging_quality`
- top chunks for `overall_consistency`
- top chunks for `motion_smoothness`
- top chunks for suffix metrics

This will clarify whether “keyframes” are universal or metric-specific.

3. Long-rollout fixed-budget policy generation.

This is the next paper-level experiment after the discovery evidence is validated. It should generate equal-budget policies with 10 heavy chunks:

- `uniform_top_m`
- `random_top_m`
- `prefix_top_m`
- `oracle_top_m`

The objective is to test whether keyframe rankings are exploitable, not just detectable.

4. Cheap-proxy policy candidate.

If we want a lightweight additional baseline, we can define a non-learned proxy policy using the current best cheap signal:

- `cheap_suffix_frame_top_m`
- optionally a diversity-constrained variant to avoid selecting adjacent chunks only

This would test whether a cheap metric can approximate the oracle enough to be useful.

## What Is Still Missing

The final fixed-budget policy evaluation is not implemented for long rollouts yet.

The missing long-rollout policies are:

- `uniform_top_m`
- `random_top_m`
- `prefix_top_m`
- `oracle_top_m`

There are old policy-eval scripts for the short-video setting, but they are not the correct final long-rollout policy pipeline.

The final paper-level comparison should generate full videos under equal compute budget:

```text
40 chunks, 10 heavy chunks, 30 fast chunks
```

and compare:

- oracle sparse compute allocation
- uniform allocation
- random allocation
- prefix allocation

This phase is needed to claim that discovered keyframes can be exploited, not just that they exist.

## Current Interpretation

The experiment is now past the infrastructure/debugging phase and into analysis.

- The long-rollout discovery dataset exists.
- The suffix materialization is repaired.
- Strict fairness checks are working.
- Full-video VBench is complete for all seven metrics.
- Sparse suffix VBench is complete for the intended four metrics.
- VBench aggregation completed with no parser warnings.
- Cheap non-learned metrics show shallow but above-random alignment with hard chunks.

Current story:

> We have an exhaustive single-heavy long-rollout discovery set, repaired fair suffixes, completed full-video VBench, and completed sparse suffix VBench. The results show that global `all_heavy` is not reliably better than `all_fast`, but single-heavy interventions have non-uniform, metric-dependent effects. This supports sparse compute placement as the right object of study, but the strongest signal likely requires per prompt/seed oracle alignment rather than global averaging.
