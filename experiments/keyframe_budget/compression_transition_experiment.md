# Compression-Transition Hypothesis Experiment

## Status

This document records the rationale, design, implementation details, and results for the compression-transition hypothesis experiment.

Current implementation status:

- Target experiment family: AR-long keyframe-budget rollouts generated with `ar_diffusion.pt`.
- VBench source: completed AR-long full-video VBench results.
- Latent source: deterministic latent-only replay, because the original rollouts stored only latent shape metadata, not latent tensors.
- No DeltaTok training is performed.
- Latent extraction is complete for all 72 prompt-seed samples.
- Transition scoring, VBench merge, and analysis are complete.

Main result:

- There is a shallow but nonzero keyframe signal in cheap latent transition scores.
- The strongest predictor is latent transition acceleration `A_max`, not the PCA compression residual `R_max`.
- The paper-facing claim should be conservative: high-acceleration transitions are modestly enriched for useful heavy chunks.

## Rationale

This experiment is inspired by DeltaTok / DeltaWorld, arXiv `2604.04913`.

The relevant idea is not the full tokenizer training recipe. The relevant claim is that video generation has a strong transition-compression structure:

- Consecutive frames or latent features often differ in structured, low-dimensional ways.
- A strong prior is that much of the next frame is "no change" or predictable from the previous frame.
- Therefore, the important signal may live in the residual transition, not in the absolute frame representation.

Translated to our long AR keyframe-budget setting:

- Some chunks are genuinely high-information transitions.
- Extra denoising should help most on those chunks.
- Extra denoising may hurt on low-transition chunks by injecting unnecessary residual detail or latent jitter.
- All-heavy generation may degrade temporal metrics if it increases transition energy everywhere.

The core question:

> Is long AR video generation primarily a transition-compression problem, where heavy denoising is useful only when the next chunk is hard to predict from previous latent changes?

## Scope

First target: **AR-long only**.

Reasons:

- This is the corrected teacher-style setup using `ar_diffusion.pt`, not the invalid earlier `longvideo.pt` distilled setup.
- The generation grid is complete.
- Full-video VBench is complete for all seven metrics.
- The long rollout setting is where the keyframe signal is expected to be strongest.

Input rollout root:

```text
outputs/ar_teacher_long_keyframe_budget
```

Expected generation grid:

```text
72 prompt-seed samples
42 schedules per sample
3024 full videos total
```

Schedules per prompt-seed:

```text
all_fast
all_heavy
single_heavy_00
...
single_heavy_39
```

VBench source:

```text
outputs/ar_teacher_long_keyframe_budget/vbench_analysis/vbench_video_records.csv
```

VBench coverage:

```text
7 metrics x 42 cohorts x 72 samples
```

Metrics:

```text
dynamic_degree
motion_smoothness
overall_consistency
imaging_quality
aesthetic_quality
subject_consistency
background_consistency
```

## Why Latent Replay Is Needed

The original AR-long rollouts did call the pipeline with `return_latents=True`, but the latents were not written to disk.

The saved `rollout_meta.json` contains:

```text
latent_shape: [1, 120, 16, 60, 104]
chunk_logs
schedule_name
steps_per_chunk
checkpoint
prompt_id
seed
```

It does **not** contain the actual latent tensor.

Therefore, this experiment uses deterministic replay:

1. Read existing rollout metadata.
2. Reconstruct the same prompt, seed, checkpoint, config, backend, and step schedule.
3. Recreate the same initial Gaussian noise using `set_seed(seed)`.
4. Run the AR diffusion pipeline with:

```python
return_video=False
return_chunk_logs=True
steps_per_chunk=...
```

5. Save the final clean latent chunks before VAE decoding.

This avoids VAE decode and video writing, but still reruns the denoising process.

## Latent Representation

Each AR-long rollout has:

```text
120 latent frames
3 latent frames per causal chunk
40 chunks
latent tensor shape: [1, 120, 16, 60, 104]
```

The extraction script reshapes the latent tensor to:

```text
[40, 3, 16, 60, 104]
```

For chunk `c`:

```text
h[c] = final clean latent for chunk c
```

This is saved as one tensor per rollout. The default extraction mode is now lean because the full cache exceeded the Jean Zay disk quota.

```text
outputs/ar_teacher_long_keyframe_budget/compression_transition/latents/
  ar_teacher_long_pXXX_sY/
    <prompt_id>/
      <seed>/
        <schedule>/
          chunk_latents.pt
          chunk_latents_meta.json
```

The latent tensor keeps the replay dtype, normally `bfloat16`, and is converted to `float32` only during scoring.

Lean storage policy:

```text
all_fast:        save all 40 chunks
all_heavy:       save all 40 chunks
single_heavy_i: save only chunks i-1, i, i+1, clipped to valid chunk bounds
```

This preserves all quantities needed for the current compression-transition test:

```text
D, A, R: computed from full all_fast
C_F, C_H: computed from full all_fast/all_heavy
Q_i: computed from saved single_heavy_i chunk i versus all_fast chunk i
```

The scorer remains backward-compatible with older full single-heavy latent files. Each manifest row records `latent_save_mode` and `saved_chunk_indices_json` so the scorer can map a lean tensor row back to the original chunk index.

## Job 1: Latent Extraction

Script:

```text
experiments/keyframe_budget/scripts/extract_ar_long_chunk_latents.py
```

Slurm launcher:

```text
experiments/keyframe_budget/scripts/extract_ar_long_chunk_latents.slurm
```

Default Slurm resources:

```text
partition/account: hdy@h100
constraint: h100
gpus per task: 1
cpus per task: 32
time: 04:00:00
array: 0-71%12
```

Each array task corresponds to one prompt-seed sample:

```text
task_id = prompt_index * 3 + seed
```

Each full task extracts 42 schedules.

Smoke command:

```bash
SCHEDULES=all_fast,all_heavy,single_heavy_00,single_heavy_20,single_heavy_39 \
sbatch --array=0 experiments/keyframe_budget/scripts/extract_ar_long_chunk_latents.slurm
```

Full extraction command:

```bash
sbatch --array=0-71%12 experiments/keyframe_budget/scripts/extract_ar_long_chunk_latents.slurm
```

Expected manifest rows after full extraction:

```text
72 prompt-seed samples x 42 schedules = 3024 rows
```

Manifest shards:

```text
outputs/ar_teacher_long_keyframe_budget/compression_transition/manifests/manifest_task_*.parquet
```

Manifest columns:

```text
experiment_name
prompt_id
seed
schedule
run_type
heavy_idx
chunk_idx
latent_path
num_chunks
chunk_frames
checkpoint
config_path
source_rollout_meta
```

## Job 2: Transition Scoring

Script:

```text
experiments/keyframe_budget/scripts/compute_compression_transition_scores.py
```

Command:

```bash
sbatch experiments/keyframe_budget/scripts/compute_compression_transition_scores.slurm
```

The direct Python command was killed on the login/interactive node, likely due to resource limits. The production path is the CPU Slurm array scorer:

```text
experiments/keyframe_budget/scripts/compute_compression_transition_scores.slurm
```

This runs one task per prompt-seed sample plus one merge task.

Expected rows:

```text
72 prompt-seed samples x 40 single-heavy chunks = 2880 rows
```

### Chunk State

For scoring, each chunk latent is flattened:

```text
h[c] in R^(3 * 16 * 60 * 104)
```

The score still semantically corresponds to one causal chunk.

### Delta Magnitude

For transition `c-1 -> c`:

```text
d_c = h_F[c] - h_F[c-1]
```

```text
D_c = ||h_F[c] - h_F[c-1]||_2^2 / (||h_F[c-1]||_2^2 + eps)
```

Interpretation:

```text
large transition relative to previous latent state
```

### Delta Acceleration

For `c >= 2`:

```text
A_c = ||(h_F[c] - h_F[c-1]) - (h_F[c-1] - h_F[c-2])||_2^2
```

Interpretation:

```text
unexpected change in transition direction or velocity
```

### Online Compression Residual

For current delta `d_c`, build a low-rank basis from previous deltas:

```text
U_{c-1} = PCA(d_1, ..., d_{c-1})
```

Then:

```text
R_c = ||d_c - Proj_U(d_c)||_2^2 / (||d_c||_2^2 + eps)
```

Interpretation:

```text
how poorly the current transition is compressed by previous transition directions
```

Implementation detail:

- The script does not form a full covariance matrix over latent dimensions.
- It uses the small Gram/SVD formulation over the previous delta vectors.
- Default PCA rank:

```text
pca_rank = 8
```

### Heavy-Induced Residual

For single-heavy chunk `i`:

```text
Q_i = ||h_i[i] - h_F[i]||_2^2
```

Interpretation:

```text
how much heavy denoising perturbs the chunk latent relative to all-fast
```

### Offset Handling

A heavy chunk can affect incoming and outgoing transitions.

For each chunk `i`, the table stores:

```text
D_in,  D_out,  D_max
A_in,  A_out,  A_max
R_in,  R_out,  R_max
Q
```

Definitions:

```text
score_in(i)  = transition i-1 -> i
score_out(i) = transition i -> i+1
score_max(i) = max(score_in, score_out)
```

Initial expected primary predictor:

```text
R_max
```

Actual best predictor after analysis:

```text
A_max
```

## Job 3: Merge With VBench

Script:

```text
experiments/keyframe_budget/scripts/merge_compression_with_vbench.py
```

Command:

```bash
python experiments/keyframe_budget/scripts/merge_compression_with_vbench.py \
  --scores outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_scores.parquet \
  --vbench outputs/ar_teacher_long_keyframe_budget/vbench_analysis/vbench_video_records.csv \
  --out outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet
```

For each metric `m`:

```text
delta_{i,m} = VBench_m(single_heavy_i) - VBench_m(all_fast)
```

The table also stores:

```text
VBench_m(all_heavy) - VBench_m(all_fast)
```

### Temporal Gain

Temporal metrics:

```text
background_consistency
subject_consistency
motion_smoothness
overall_consistency
```

For each prompt-seed and metric:

```text
z_{i,m} = z-score over chunks i=0..39 of delta_{i,m}
```

Then:

```text
delta_temp_i = mean_m z_{i,m}
```

### Quality Gain

Quality metrics:

```text
imaging_quality
aesthetic_quality
```

Similarly:

```text
delta_qual_i = mean_m z_{i,m}
```

## Job 4: Analysis And Plots

Script:

```text
experiments/keyframe_budget/scripts/analyze_compression_hypothesis.py
```

Command:

```bash
python experiments/keyframe_budget/scripts/analyze_compression_hypothesis.py \
  --table outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet \
  --out_dir outputs/ar_teacher_long_keyframe_budget/compression_transition/analysis
```

Outputs:

```text
spearman_correlations.csv
topk_overlap.csv
auroc_bad_chunks.csv
allfast_vs_allheavy_delta_energy.csv
scatter_Rmax_vs_delta_temp.png
heatmap_transition_scores_vs_vbench.png
allfast_vs_allheavy_delta_energy.png
```

## Hypotheses

### H1: Useful Heavy Chunks Are High-Transition Chunks

Test:

```text
Spearman(D_max, delta_temp)
Spearman(A_max, delta_temp)
Spearman(R_max, delta_temp)
```

Also report `in` and `out` variants.

Expected result:

```text
R_max > A_max > D_max
```

Interpretation:

```text
The most useful heavy chunks are those whose transitions are least compressible from prior transitions.
```

### H2: Bad Heavy Chunks Are Low-Delta Perturbations

Test whether bad chunks have:

```text
low D_max
high Q
delta_temp < 0
```

Operational predictor:

```text
z(Q) - z(D_max)
```

Analysis output:

```text
auroc_bad_chunks.csv
```

Positive result:

```text
AUROC > 0.5 for predicting delta_temp < 0
```

Interpretation:

```text
Heavy denoising hurts when it injects latent perturbations into otherwise compressible transitions.
```

### H3: All-Heavy Destroys Temporal Compressibility

Compute:

```text
C_F = sum_c ||h_F[c] - h_F[c-1]||_2^2
C_H = sum_c ||h_H[c] - h_H[c-1]||_2^2
```

Compare against:

```text
VBench_m(all_heavy) - VBench_m(all_fast)
```

for temporal metrics.

Positive result:

```text
C_H > C_F
```

while temporal VBench drops.

Interpretation:

```text
All-heavy creates excessive transition energy / latent jitter.
```

### H4: Transition Scores Explain Temporal Metrics Better Than Quality Metrics

Compare correlations against:

```text
delta_temp
delta_qual
per-metric deltas
```

Positive result:

```text
R/A predictors are stronger for temporal metrics than for imaging/aesthetic quality.
```

## Win Conditions

Any of the following gives useful signal:

1. `R_max` or `A_max` predicts top useful chunks better than random.
2. Low `D_max` plus high `Q` predicts negative temporal gain.
3. `C_H > C_F` while all-heavy temporal VBench is worse than all-fast.
4. Transition scores explain temporal metrics better than quality metrics.

Strongest possible claim:

> Long AR video generation is a transition-compression problem. Heavy denoising helps only on high-information transitions and hurts when applied to compressible low-delta regions.

## Expected Counts

After latent extraction:

```text
manifest rows: 3024
latent files: 3024
```

With lean extraction, the file count is unchanged but most single-heavy files contain only 2-3 chunks instead of all 40 chunks. This reduces the dominant storage term by roughly an order of magnitude while keeping the testable hypotheses intact.

After transition scoring:

```text
transition rows: 2880
```

After merge:

```text
compression_vbench_table rows: 2880
```

After analysis:

```text
spearman rows: predictors x targets
topk rows: predictors x targets x k
energy rows: 72
```

## Results

Analysis date: `2026-05-10`.

Final analysis directory:

```text
outputs/ar_teacher_long_keyframe_budget/compression_transition/analysis
```

Generated files:

```text
spearman_correlations.csv
topk_overlap.csv
auroc_bad_chunks.csv
allfast_vs_allheavy_delta_energy.csv
scatter_Rmax_vs_delta_temp.png
heatmap_transition_scores_vs_vbench.png
allfast_vs_allheavy_delta_energy.png
```

### Extraction Summary

Latent extraction completed after switching to lean storage.

```text
manifest shards: 72 / 72
manifest rows: 3024
prompt-seed samples: 72
schedules per sample: 42
full latent rows: 144
lean single-heavy rows: 2880
failed final tasks: 0
```

The final manifest coverage is balanced:

```text
all_fast: 72
all_heavy: 72
single_heavy_00 ... single_heavy_39: 72 each
```

### Transition Score Summary

Transition scoring output:

```text
outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_scores.parquet
```

Validation:

```text
rows: 2880
prompt-seed samples: 72
rows per sample: 40
heavy_idx range: 0..39
```

Expected NaN pattern:

```text
D_in / D_out: boundary chunks only
A_in / A_out / R_in / R_out: early transition and boundary chunks
```

Core latent score ranges:

```text
Q mean: 10097.78
C_F mean: 6597307.41
C_H mean: 6454013.03
C_H - C_F mean: -143294.38
D_max mean: 0.5132
A_max mean: 491908.06
R_max mean: 0.9021
```

### Merge Summary

Merged table:

```text
outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet
```

Validation:

```text
rows: 2880
prompt-seed samples: 72
delta_temp rows: 2880
delta_qual rows: 2880
```

The aggregate target distributions are centered by construction because they are within-sample z-score aggregates:

```text
delta_temp mean: approximately 0
delta_temp std: 0.5903
delta_temp min/max: -4.3358 / 3.2454

delta_qual mean: approximately 0
delta_qual std: 0.7731
delta_qual min/max: -4.3156 / 4.5917
```

### H1 Results: Useful Heavy Chunks Are High-Transition Chunks

Main finding:

```text
A_max is the best cheap latent predictor for temporal gain.
R_max is not the best predictor.
```

Global Spearman correlations with `delta_temp`:

```text
A_max: +0.0916
A_out: +0.0828
A_in:  +0.0795
D_in:  +0.0785
D_max: +0.0688
D_out: +0.0655
R_max: -0.0167
Q:     -0.0050
```

Within-sample mean Spearman correlations with `delta_temp`:

```text
A_max: +0.0696
A_out: +0.0539
A_in:  +0.0531
D_in:  +0.0406
D_max: +0.0225
R_max: -0.0170
Q:     -0.0227
```

Top-k gain analysis for `delta_temp`:

```text
A_max top-5 mean delta_temp:  +0.125
random top-5 mean delta_temp: +0.025
oracle top-5 mean delta_temp: +0.836

A_max top-10 mean delta_temp:  +0.100
random top-10 mean delta_temp: -0.007
oracle top-10 mean delta_temp: +0.601
```

Top-k overlap with oracle useful chunks:

```text
A_max top-10 overlap: 0.308
random expectation for top-10 of 40: 0.250
```

Interpretation:

```text
H1 is weakly supported, but the support is for latent acceleration A_max rather than PCA compression residual R_max.
Useful heavy chunks are modestly enriched at high-acceleration transitions.
The effect is real enough to investigate, but too shallow to claim a strong automatic oracle from these scores alone.
```

### H2 Results: Bad Heavy Chunks Are Low-Delta, High-Perturbation Chunks

Tested score:

```text
z(Q) - z(D_max)
```

Target:

```text
delta_temp < 0
```

Results:

```text
AUROC z(Q)-z(D_max): 0.5456
AUROC Q_z:           0.5063
AUROC D_max_z:       0.4794
AUROC A_max_z:       0.4593
AUROC R_max_z:       0.5142
```

Low-D/high-Q subset:

```text
subset size: 70 chunks
subset mean delta_temp: +0.0199
overall negative fraction: 0.4774
subset negative fraction: 0.5143
```

Interpretation:

```text
H2 is only weakly supported.
The combined low-D/high-Q score is slightly above random for detecting negative temporal gain, but the effect is shallow.
The subset has a higher negative fraction than baseline, but its mean delta_temp is not meaningfully negative.
This is not strong enough for a central claim.
```

### H3 Results: All-Heavy Destroys Temporal Compressibility

Energy summary across 72 samples:

```text
mean C_F: 6597307.41
mean C_H: 6454013.03
mean C_H - C_F: -143294.38
fraction C_H > C_F: 0.625
```

All-heavy temporal VBench delta:

```text
mean temporal all-heavy delta: -0.0092
```

Correlation:

```text
Spearman(C_H - C_F, temporal all-heavy delta): +0.2532
```

Interpretation:

```text
H3 is not cleanly supported.
C_H exceeds C_F for 62.5% of samples, but the mean C_H - C_F is slightly negative because some samples have large negative energy differences.
All-heavy temporal VBench is slightly worse on average, but not in a way that clearly supports "all-heavy destroys temporal compressibility by increasing transition energy."
The positive correlation between energy increase and temporal all-heavy delta also argues against the simplest version of H3.
```

### H4 Results: Transition Scores Versus Temporal And Quality Metrics

For `delta_qual`, the best global Spearman correlations are also shallow:

```text
R_max: +0.0534
R_out: +0.0443
R_in:  +0.0442
A_max: +0.0423
A_in:  +0.0410
A_out: +0.0408
```

Metric-specific observations:

```text
imaging_quality_delta: A_max is the strongest positive predictor among transition scores, Spearman +0.0612.
aesthetic_quality_delta: D_out/D_in/D_max are negative predictors, strongest Spearman -0.0979 for D_out.
background_consistency_delta: R_max is weakly negative, Spearman -0.0578.
subject_consistency_delta and motion_smoothness_delta: correlations are near zero.
overall_consistency_delta: weakly negative relation with D_max and Q.
```

Interpretation:

```text
Transition scores do not explain temporal metrics much better than quality metrics.
The clearest signal remains A_max versus delta_temp, but the separation between temporal and quality behavior is not strong.
```

## Paper-Facing Takeaway

Conservative claim:

> In AR-long rollouts, useful heavy-denoising chunks are modestly enriched around high latent-acceleration transitions. This supports the idea that keyframes are tied to non-smooth transition events, but the current cheap compression residual is not sufficient as a strong standalone oracle.

What should not be claimed from the current results:

```text
Do not claim that PCA compression residual R_max is the dominant predictor.
Do not claim that all-heavy globally destroys temporal compressibility via increased transition energy.
Do not claim that low-D/high-Q robustly identifies harmful chunks.
```

More accurate framing:

```text
The compression-transition angle is directionally useful, but in this implementation the actionable signal is acceleration/change-of-transition rather than low-rank residual compressibility.
```

## Follow-Up Lane: A-Router Policy Rollouts

The correlation analysis above is diagnostic only. It asks whether single-heavy chunk gains are enriched at high `A_max` chunks. The next lane tests whether that signal survives when used as an actual multi-chunk routing policy.

### A-Router Hypothesis

```text
If high latent acceleration marks useful transition events, then heavy-denoising the top-k A_max chunks should beat random and periodic heavy chunks under the same compute budget.
```

This is the most important follow-up because it turns the weak single-heavy enrichment into a direct policy question:

```text
Does A_max remain useful when several selected chunks are made heavy in the same rollout?
```

### Policy Set

For each prompt-seed sample, generate the following policies for:

```text
k = 5
k = 10
```

Policies:

```text
periodic_k
random_k_r0
random_k_r1
random_k_r2
Amax_top_k
Amax_top_k_no_chunk0
oracle_top_k
```

Baselines:

```text
all_fast
all_heavy
```

The baselines are not regenerated in this lane. They are reused from the completed AR-long VBench baseline/discovery run:

```text
outputs/ar_teacher_long_keyframe_budget/vbench_analysis/vbench_video_records.csv
```

This avoids unnecessary generation and keeps comparisons paired to the same prompt-seed samples.

### Policy Definitions

For a fixed prompt-seed and fixed `k`:

```text
periodic_k:
  k uniformly spaced chunks over the 40-chunk rollout

random_k_r0/r1/r2:
  k random chunks, using deterministic stable seeds per prompt/seed/k/replicate

Amax_top_k:
  top-k chunks by A_max from compression_vbench_table.parquet

Amax_top_k_no_chunk0:
  top-k chunks by A_max after excluding chunk 0

oracle_top_k:
  top-k chunks by previous single-heavy delta_temp
```

Important scientific distinction:

```text
Amax_top_k is the tested deployable offline router.
oracle_top_k is not deployable; it is a label-derived upper bound.
```

### Compute Fairness

For each fixed `k`, all mixed policies have the same number of heavy chunks and therefore the same requested NFE budget:

```text
fast_steps = 8
heavy_steps = 24
total_chunks = 40
requested NFE = (40 - k) * 8 + k * 24
```

Therefore:

```text
k=5  requested NFE = 400
k=10 requested NFE = 480
```

Fair comparisons are made only within the same `k`.

The `all_fast` and `all_heavy` baselines have different compute budgets and are used only as reference anchors:

```text
all_fast requested NFE = 320
all_heavy requested NFE = 960
```

### Implementation

Generation script:

```text
experiments/keyframe_budget/scripts/generate_ar_long_router_policy.py
```

Generation Slurm launcher:

```text
experiments/keyframe_budget/scripts/ar_teacher_long_router_policy_array.slurm
```

VBench launcher:

```text
experiments/keyframe_budget/scripts/vbench_ar_teacher_long_router_policy.slurm
```

Analysis script:

```text
experiments/keyframe_budget/scripts/analyze_ar_long_router_policy.py
```

Output root:

```text
outputs/ar_teacher_long_router_policy
```

Generated rollout layout:

```text
outputs/ar_teacher_long_router_policy/
  ar_teacher_long_router_pXXX_sY/
    rollouts/
      <prompt_id>/
        <seed>/
          <policy_name>/
            full.mp4
            rollout_meta.json
            metrics.json
            chunk_boundaries.json
    router_policy_manifest.json
    router_policy_results.json
```

Each `router_policy_manifest.json` records:

```text
policy_name
selected_chunks
steps_per_chunk
total_nfe_requested
```

### Commands

Smoke generation:

```bash
sbatch --array=0 experiments/keyframe_budget/scripts/ar_teacher_long_router_policy_array.slurm
```

Full generation:

```bash
sbatch experiments/keyframe_budget/scripts/ar_teacher_long_router_policy_array.slurm
```

VBench on A100:

```bash
sbatch experiments/keyframe_budget/scripts/vbench_ar_teacher_long_router_policy.slurm
```

Aggregate router VBench:

```bash
python experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py \
  --full_root /lustre/fswork/projects/rech/hdy/ujc37rw/VBench/evaluation_results/keyframe_budget_ar_teacher_long_router_policy/ar_teacher_long_router_policy_full \
  --output_root outputs/ar_teacher_long_router_policy/vbench_analysis \
  --skip_suffix
```

Analyze router policy results:

```bash
python experiments/keyframe_budget/scripts/analyze_ar_long_router_policy.py
```

### Primary Evaluation

Primary target:

```text
delta_temp_raw = mean raw paired delta versus all_fast over:
background_consistency
subject_consistency
motion_smoothness
overall_consistency
```

The raw paired target is primary because it avoids overweighting the three random replicates in within-sample z-score normalization.

Secondary targets:

```text
delta_temp_z
delta_qual
vbench_mean_delta
per-metric deltas
metric recoveries versus all_fast/all_heavy
```

Primary comparisons:

```text
Amax_top_k > mean(random_k_r0, random_k_r1, random_k_r2)
Amax_top_k > periodic_k
```

Upper-bound comparison:

```text
Amax_top_k versus oracle_top_k
```

It is acceptable and expected if:

```text
Amax_top_k << oracle_top_k
```

The main win condition is:

```text
Amax_top_k beats random and periodic under the same k budget.
```

### Planned Output Files

Analysis output directory:

```text
outputs/ar_teacher_long_router_policy/analysis
```

Expected files:

```text
router_policy_by_sample.csv
router_policy_summary.csv
router_policy_pairwise_tests.csv
router_policy_chosen_chunks.csv
router_policy_plots/
```

### A-Router Results

Analysis date: `2026-05-11`.

Analysis output:

```text
outputs/ar_teacher_long_router_policy/analysis
```

Generated result files:

```text
router_policy_by_sample.csv
router_policy_summary.csv
router_policy_pairwise_tests.csv
router_policy_chosen_chunks.csv
router_policy_plots/delta_temp_raw_k05.png
router_policy_plots/delta_temp_raw_k10.png
router_policy_plots/vbench_mean_delta_k05.png
router_policy_plots/vbench_mean_delta_k10.png
```

#### Generation Summary

```text
generated prompt-seed samples: 72
expected prompt-seed samples: 72
policies per sample: 14
expected policy videos: 1008
generated policy videos: 1008
failed tasks: 0 observed in completed output/log checks
notes: all mixed policies are compute-matched within k
```

#### VBench Summary

```text
metrics: 7
router cohorts: 98 = 7 metrics x 14 policies
video records: 7056 = 98 cohorts x 72 prompt-seed samples
missing cohorts: 0
warnings: 0
```

VBench cohort coverage is balanced:

```text
each metric has 14 schedules
each metric/schedule cohort has 72 samples
```

Schedules evaluated:

```text
Amax_top_k05
Amax_top_k05_no_chunk0
Amax_top_k10
Amax_top_k10_no_chunk0
oracle_top_k05
oracle_top_k10
periodic_k05
periodic_k10
random_k05_r0
random_k05_r1
random_k05_r2
random_k10_r0
random_k10_r1
random_k10_r2
```

Manifest fairness checks:

```text
k=5 policies: 5 heavy chunks, requested NFE 400
k=10 policies: 10 heavy chunks, requested NFE 480
```

Chunk-0 note:

```text
Amax_top_k never selected chunk 0 in this run because A_max has no valid chunk-0 incoming/outgoing acceleration score.
Therefore Amax_top_k and Amax_top_k_no_chunk0 are effectively identical or near-identical.
```

#### k=5 Results

```text
Amax_top_k05 delta_temp_raw: +0.002067
random_k05 mean delta_temp_raw: -0.001044
periodic_k05 delta_temp_raw: -0.006660
oracle_top_k05 delta_temp_raw: +0.007675

Amax - random mean paired diff: +0.003111
Amax - random median paired diff: +0.001443
Amax win rate vs random: 0.5833
Amax - random approximate 95% CI: [-0.00228, +0.00850]

Amax - periodic mean paired diff: +0.008727
Amax - periodic median paired diff: +0.005381
Amax win rate vs periodic: 0.6111
Amax - periodic approximate 95% CI: [+0.00241, +0.01504]

Amax - oracle mean paired diff: -0.005607
Amax win rate vs oracle: 0.4444
```

Secondary `vbench_mean_delta`:

```text
Amax_top_k05: +0.158841
random_k05 mean: +0.037477
periodic_k05: +0.061614
oracle_top_k05: +0.129215

Amax - random mean paired diff: +0.121363
Amax - periodic mean paired diff: +0.097227
```

Interpretation:

```text
k=5 supports the A-router hypothesis directionally.
Amax beats both random and periodic on the primary temporal target.
The periodic comparison is clearer than the random comparison; the random comparison has positive mean and win rate but an approximate CI that includes zero.
Oracle remains better on delta_temp_raw, which is expected because it uses label-derived single-heavy VBench information.
```

#### k=10 Results

```text
Amax_top_k10 delta_temp_raw: +0.000345
random_k10 mean delta_temp_raw: -0.002825
periodic_k10 delta_temp_raw: -0.005138
oracle_top_k10 delta_temp_raw: +0.005609

Amax - random mean paired diff: +0.003170
Amax - random median paired diff: +0.002995
Amax win rate vs random: 0.5833
Amax - random approximate 95% CI: [-0.00121, +0.00755]

Amax - periodic mean paired diff: +0.005483
Amax - periodic median paired diff: +0.003704
Amax win rate vs periodic: 0.6111
Amax - periodic approximate 95% CI: [+0.000008, +0.010958]

Amax - oracle mean paired diff: -0.005264
Amax win rate vs oracle: 0.3333
```

Secondary `vbench_mean_delta`:

```text
Amax_top_k10: +0.124994
random_k10 mean: +0.023332
periodic_k10: +0.045907
oracle_top_k10: +0.132288

Amax - random mean paired diff: +0.101662
Amax - periodic mean paired diff: +0.079087
```

Interpretation:

```text
k=10 also supports the A-router hypothesis directionally.
Amax again beats random and periodic on the primary temporal target.
The effect is smaller than k=5 in absolute delta_temp_raw, and the oracle gap is larger in win-rate terms.
```

#### No-Chunk0 Ablation

```text
Amax_top_k05_no_chunk0 versus Amax_top_k05:
  mean delta_temp_raw difference Amax - no_chunk0: +0.000036
  win rate Amax over no_chunk0: 0.5139

Amax_top_k10_no_chunk0 versus Amax_top_k10:
  mean delta_temp_raw difference Amax - no_chunk0: -0.000015
  win rate Amax over no_chunk0: 0.4306

Interpretation:
  No meaningful difference.
  In practice Amax_top_k already excluded chunk 0 because A_max is unavailable/NaN at chunk 0.
```

#### Paper-Facing Router Takeaway

The router experiment provides the first actual policy-level support for the acceleration-keyframe story:

> Selecting heavy-denoising chunks by latent acceleration produces small but consistent temporal VBench gains over random and periodic chunk selection at matched compute budgets.

The scientifically careful version is:

```text
A_max is not a strong oracle, but it is better than chance as a routing signal.
The effect is modest and strongest as an enrichment/routing prior, not as a complete keyframe policy.
```

What this result supports:

```text
Keyframe usefulness is not uniformly distributed.
High latent-acceleration chunks contain actionable routing signal.
The weak single-heavy A_max enrichment survives multi-chunk rollouts.
```

What it does not support:

```text
A_max solves routing.
A_max reaches oracle performance.
The DeltaTokens-style PCA residual R is useful in this latent space.
All-heavy degradation is explained by simple transition-energy inflation.
```

#### Router Caveats

- Absolute temporal deltas are small.
- Amax versus random is directionally positive but the approximate 95% intervals include zero for the primary target.
- Amax versus periodic is more robust in this run.
- Oracle remains better for temporal delta, as expected.
- The primary target is raw paired temporal VBench delta; VBench is noisy and may not perfectly track visual quality.
- `Amax_top_k_no_chunk0` is not informative here because `Amax_top_k` already never selected chunk 0.
- This is still an offline router because it uses precomputed all-fast latent replay scores.

## Caveats

- The analysis uses VBench metrics, which are noisy and can disagree with human visual judgment.
- `delta_temp` and `delta_qual` are within-sample z-score aggregates, so their absolute scales are relative within each prompt-seed sample.
- The strongest correlations are small; this is an enrichment signal, not a reliable deterministic policy yet.
- The PCA residual uses final clean latents and an online low-rank basis over only 39 transitions. It may be too short/noisy to expose a strong DeltaTokens-style compressibility signal.
- The latent replay should be deterministic relative to the generation metadata, but it is still a second forward pass rather than latents captured during original video generation.
- Lean single-heavy storage preserves the exact quantities used here, but future analyses requiring the full single-heavy trajectory would need either full storage or another targeted extraction pass.

## Corrected Router V2 Batch

The first router-policy batch was useful as a policy diagnostic, but it omitted same-batch `all_fast` and `all_heavy` baselines. That made normalized recovery analysis underdetermined and forced comparisons against external AR-long baseline VBench results.

The corrected v2 batch fixes the bookkeeping:

```text
output root:
  outputs/ar_teacher_long_router_policy_v2

policies per prompt-seed:
  all_fast
  all_heavy
  Amax_top_k05
  Amax_top_k10
  periodic_k05
  periodic_k10
  random_k05_r00 ... random_k05_r19
  random_k10_r00 ... random_k10_r19
  single_heavy_oracle_k05
  single_heavy_oracle_k10
```

The old `oracle_top_k` label is renamed to `single_heavy_oracle_k` because the ranking comes from previous single-heavy VBench deltas. It is an offline diagnostic upper bound, not a true multi-heavy interaction oracle.

The corrected analysis should report:

```text
Amax - all_fast
Amax - random_mean
Amax - periodic
Amax recovery toward single_heavy_oracle
all_heavy - all_fast
```

Additional audit fixes:

```text
latent replay:
  use_ema is read from rollout metadata instead of hardcoded false.

H2 bad-chunk score:
  z(Q) - z(D_max) is normalized within each prompt-seed sample.

uncertainty:
  Spearman/top-k/H2/router pairwise reports include bootstrap confidence intervals.
```

Results space:

```text
Pending corrected v2 generation and VBench.
```

## Targeted Amax Ablation Lane

Motivation:

```text
The corrected router evidence suggests Amax helps, but only weakly.
The next question is why it helps:
  - transition irregularity alone,
  - fast-solver instability,
  - medium-heavy denoising being better than 24-step heavy,
  - or Amax implicitly targeting quality rather than temporal consistency.
```

New output root:

```text
outputs/ar_teacher_long_router_ablation
```

New S-instability score:

```text
S_i = mean_k ||x_i^{k+1} - x_i^k||_2 / (||x_i^k||_2 + eps)
```

Implementation detail:

```text
S is computed from an 8-step all-fast replay.
Only scalar per-step relative movements are logged.
Full denoising trajectories are not saved.
```

Important entrypoint distinction:

```text
runner.py normal rollouts intentionally do not emit S.
generate_ar_long_router_policy.py intentionally remains the corrected v2 router entrypoint.

The S ablation entrypoints are:
  extract_ar_long_solver_instability.py / compute_ar_long_solver_instability.py
  merge_ar_long_solver_instability.py
  merge_solver_instability_with_compression.py
  generate_ar_long_router_ablation.py
```

Ablation policies:

```text
all_fast
all_heavy
random_k05_h24_r00 ... random_k05_h24_r19
Amax_k05_h24
S_top_k05_h24
AplusS_top_k05_h24
AtimesS_top_k05_h24
Amax_k05_h12
Amax_k05_h16
Amax_k05_h20
global_mean_oracle_k05_h24
imaging_oracle_k05_h24
temporal_oracle_k05_h24
```

Total cohorts:

```text
32 cohorts
7 VBench metrics
224 VBench array tasks
```

Primary comparisons:

```text
AplusS_top_k05_h24 vs random mean
AplusS_top_k05_h24 vs Amax_k05_h24
S_top_k05_h24 vs Amax_k05_h24
Amax step-size curve: h12, h16, h20, h24
Amax_k05_h24 vs global/imaging/temporal single-heavy oracles
```

Win condition:

```text
AplusS_top_k05_h24 improves vbench_mean_delta over random
and improves delta_temp_raw more clearly than Amax_k05_h24.
```

Visual audit:

```text
The analysis writes visual_audit_candidates.csv with prompt-seeds where:
  Amax >> random
  Amax << random
  Amax >> temporal oracle
```

Launch readiness checks:

```bash
python experiments/keyframe_budget/scripts/check_ar_long_router_ablation_ready.py
```

Expected state before generation:

```text
72 S-preview shards
2880 merged S rows
2880 merged compression+S rows
columns: A_max, S_instability, AplusS, AtimesS
```

Expected state after ablation generation:

```text
72 router_ablation_manifest.json files
32 cohorts per manifest
all non-baseline policies select exactly 5 chunks
VBench slurm: 7 metrics x 32 cohorts = array 0-223
```
