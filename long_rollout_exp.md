# Long Rollout Experiments Reference

## 1) Why We Built This

The long-rollout experiment suite is designed to test a causal claim:

> Under a fixed denoising budget, only a sparse subset of chunks has high downstream impact on long-horizon video quality.

This is not a short-video benchmark. The objective is to measure long-range temporal effects under autoregressive generation, where early and mid-rollout choices can affect later content quality and coherence.

Core goals:

- Move from short rollouts to **long rollouts** (`num_output_frames=120` latent frames).
- Keep generation and evaluation **decoupled** (different environments, reproducible handoff).
- Enforce **fair policy comparisons** under controlled step budgets.
- Produce artifacts that support **full-video and suffix-window analyses** without regeneration.

---

## 2) Experimental Design Principles

### 2.1 Generation/Evaluation Separation

Generation produces assets and metadata only:

- `full.mp4`
- `rollout_meta.json`
- `chunk_boundaries.json`

Evaluation (VBench) consumes those outputs later in a separate environment.

This separation is intentional:

- avoids cross-env dependency coupling,
- keeps generation stable and repeatable,
- allows evaluation protocol iteration without rerunning expensive generation.

### 2.2 Fairness and Scientific Rigor

We enforce comparability through:

- fixed prompt and seed protocol,
- fixed model/config/checkpoint per run family,
- controlled schedule families with explicit budget accounting,
- deterministic schedule construction where applicable,
- stage-gated progression (sanity -> discovery -> policy-scale),
- strict cohort matching for evaluation (`strict` or `intersect` mode).

---

## 3) Long-Rollout Generation Stack

## 3.1 Backend Choice

Long rollouts use the long-video path:

- config: `long_video/configs/rolling_forcing_dmd.yaml`
- inference backend: `long_video/pipeline/causal_diffusion_inference.py`
- checkpoint: `checkpoints/chunkwise/longvideo.pt` (or HF snapshot equivalent)

Why: this backend is designed for longer temporal windows and causal cache behavior.

## 3.2 Policy/Schedule Families

Generation supports schedule families such as:

- `all_fast`
- `all_heavy`
- `single_heavy_i`
- policy comparators (`uniform_top_m`, `random_top_m`, `prefix_top_m`, optional `oracle_top_m`)

Each schedule is defined by `steps_per_chunk` and audited against actual executed steps.

### 3.2.1 Concrete schedule math (as implemented)

Schedule construction lives in `experiments/keyframe_budget/schedules.py`.

Key constants used in long-rollout runs:

- `fast_steps = 8`
- `heavy_steps = 24`
- `target_avg_steps = 12`
- `num_output_frames = 120`
- `num_frame_per_block = 3` (from `rolling_forcing_dmd.yaml`)
- `independent_first_frame = false` (from merged default config)

Chunk count is computed by `infer_num_chunks(...)` in `runner.py`:

- `total_chunks = num_output_frames / num_frame_per_block = 120 / 3 = 40`

Mixed-policy heavy count is computed by:

- `m = floor(T * (B - L) / (H - L))`
- with `T=40, B=12, L=8, H=24` -> `m = floor(40 * 4 / 16) = 10`

So, for mixed policies in this setup:

- heavy chunks per video: **10**
- fast chunks per video: **30**
- requested total steps: `(30 * 8) + (10 * 24) = 480`

### 3.2.2 What each policy does concretely

- `all_fast`: 40/40 chunks at `8` steps (`320` total).
- `all_heavy`: 40/40 chunks at `24` steps (`960` total).
- `single_heavy_i`: chunk `i` at `24`, all others at `8` (40 variants).
- `uniform_top_m`: picks `m=10` approximately evenly spaced chunk indices.
- `random_top_m`: picks `m=10` chunk indices via deterministic RNG seed.
- `prefix_top_m`: picks chunk indices `[0..m-1]` (front-loaded heavy compute).
- `oracle_top_m`: picks top-`m` chunk indices from precomputed ranked gains.

### 3.2.3 Oracle policy mechanics

Oracle ranking is generated from exhaustive single-heavy sweeps in `oracle.py`:

1. Run `all_fast`, `all_heavy`, and all `single_heavy_i`.
2. For each `i`, compute `gain_i = score(single_heavy_i) - score(all_fast)`.
3. Normalize by `(score(all_heavy) - score(all_fast))` to get `gain_norm_i`.
4. Sort descending by `gain_i` and assign ranks.
5. `oracle_top_m` selects top-ranked valid chunk indices from that ordering.

Important:

- `oracle_top_m` requires non-empty ranked indices and enough valid entries.
- In policy-eval-only runs, rankings can be loaded from `oracle_source`.

## 3.3 Runtime Stabilization

Two critical runtime adaptations are part of long-rollout support:

- Per-chunk denoising step control and chunk logs in `long_video` pipeline.
- KV-cache overflow-safe write/roll behavior in `wan/modules/causal_model.py`.

These are required for stable long autoregressive runs and trustworthy step-level accounting.

---

## 4) Metadata Contract (Generation -> Evaluation)

## 4.1 Output Layout

Expected rollout layout:

```text
outputs/long_keyframe_budget/<experiment>/rollouts/<prompt_id>/<seed>/<schedule_name>/
  full.mp4
  rollout_meta.json
  chunk_boundaries.json
```

## 4.2 `rollout_meta.json` Responsibilities

Contains reproducibility and audit fields, including:

- experiment/prompt/seed/schedule identity,
- backend/config/checkpoint,
- requested schedule (`steps_per_chunk`),
- actual step logs (`chunk_logs`),
- budget checks (`requested_total_steps`, `total_nfe`, `budget_match_ok`),
- decoded frame stats (`decoded_total_frames`, fps),
- status/error fields.

## 4.3 `chunk_boundaries.json` Responsibilities

Defines latent and visible boundary mapping per chunk.

Boundary mapping uses centralized utilities in:

- `experiments/keyframe_budget/boundaries.py`

This avoids duplicated latent->visible logic and keeps suffix extraction deterministic.

### 4.4 What chunk boundaries mean

Boundary construction is centralized in `experiments/keyframe_budget/boundaries.py`.

It records, for each chunk:

- `latent_start`, `latent_end`
- `visible_start`, `visible_end`

Latent-to-visible mapping uses WAN first-frame-aware conversion:

- latent index `0` -> visible `0`
- latent index `k>0` -> `1 + (k-1) * latent_to_visible_ratio`

For long-rollout defaults (`120` latent frames, `3` latent frames per chunk, ratio `4`):

- number of chunks is `40`
- each chunk advances by 3 latent frames
- visible boundaries are derived with the WAN mapping and stored explicitly

These boundaries are the contract used later for suffix-window extraction.

### 4.5 Suffixes: what is implemented now vs later

Implemented now:

- generation stores `suffix_window_latent` in metadata,
- generation stores chunk boundaries needed to cut suffix clips offline.

Not materialized during generation:

- suffix video files themselves.

So suffix evaluation is enabled by metadata contract, not by generation-time clip dumping.

---

## 5) Stage-Gated Execution Model

`run_experiment.py` supports stage-aware controls:

- `stage_name`
- `require_stage_gate`
- `stage_prerequisites`
- `run_baseline_only` (stage-0 style quick checks)

Each stage can emit a gate artifact under:

- `<output_root>/<experiment>/stage_gate/<stage_name>.json`

This is used to prevent scaling before sanity criteria are met.

### 5.1 Stage behavior implemented in code

Stage logic is enforced in `run_experiment.py`:

- `stage0`: requires `run_baseline_only=true` (quick backend sanity).
- `stage1`/`stage2`: require exhaustive mode and exactly one prompt x one seed.
- `stage3`: requires exhaustive mode and forbids policy-comparison mode.

Prerequisite enforcement:

- if `require_stage_gate=true`, each stage checks prerequisite stage artifacts.

Success criteria checks:

- if exhaustive is enabled, `gain_maps.jsonl` must be produced and non-empty.
- if policy comparison is enabled, `policy_results.jsonl` must be produced and non-empty.

### 5.2 Prompt x seed structure in practice

The framework is config-driven (`num_prompts`, `seeds`), but in long-rollout stage3 array execution we used shard-style units of:

- **1 prompt x 1 seed per shard task** (as seen in stage3 logs),
- then many shard tasks cover the full discovery set.

This keeps each task bounded while preserving global coverage via array manifests.

---

## 6) Metrics Strategy

## 6.1 Internal Metrics (Generation-Time)

Generation writes local proxy metrics for operational continuity and scoring plumbing.
These are useful for debugging and intermediate comparisons, but are not the final benchmark.

Tracking outputs:

- prompt/seed aggregate JSONs: `aggregates/prompt_seed_aggregates/*.json`
- oracle gain map stream: `aggregates/gain_maps.jsonl`
- policy result stream: `aggregates/policy_results.jsonl`
- plots: gain heatmap / recovery barplot / oracle position histogram / suffix gain curves

## 6.2 VBench Metrics (Evaluation-Time)

For long-rollout policy comparison, we currently focus on:

- `dynamic_degree`
- `motion_smoothness`
- `overall_consistency`
- `imaging_quality`
- `aesthetic_quality`
- `subject_consistency`
- `background_consistency`

Rationale:

- these are supported in `custom_input` mode,
- they jointly cover motion, visual quality, temporal consistency, and subject/background persistence,
- they avoid unsupported semantic dimensions that require extra prompt/category metadata in VBench custom mode.

---

## 7) VBench Evaluation Protocol for Long Rollouts

## 7.1 Cohort-Aware Evaluation (Do Not Mix Policies)

Evaluation groups videos into cohorts derived from rollout structure.
A cohort corresponds to a specific policy/suffix identity, and is evaluated separately.

This avoids signal dilution from mixing policy outputs in one pooled score.

Concrete cohorting in `vbench_overnight_long_rollout_32gpus.slurm`:

- discovers videos recursively under long-rollout output root,
- parses `rollouts/<prompt>/<seed>/<schedule>/<video>.mp4` when available,
- builds `sample_key = <prompt>::<seed>`,
- builds cohort tag from schedule + filename stem,
- evaluates each cohort separately.

Because generation writes `full.mp4`, cohort tags typically look like:

- `<schedule_name>__full`

## 7.2 Fair Sample Matching

Evaluation supports:

- `FAIR_MATCH_MODE=strict`: require complete matched sets across cohorts.
- `FAIR_MATCH_MODE=intersect`: use only common sample keys across cohorts.

Default is strict for maximum comparability.

Per-cohort shard manifests are written before evaluation so each metric has auditable input sets.

## 7.3 Scripts

Primary overnight evaluator:

- `experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_32gpus.slurm`

Preflight smoke (same logic, capped sample/cohort counts):

- `experiments/keyframe_budget/scripts/vbench_smoke_long_rollout_preflight.slurm`

The smoke run is mandatory before overnight when code or environment changed.

---

## 8) Recommended Operational Flow

1. **Generation completed** and outputs verified under `outputs/long_keyframe_budget`.
2. Run **VBench smoke preflight** on all target metrics (capped sample set).
3. Review logs for:
   - cohort discovery count,
   - common sample count,
   - no worker crashes,
   - result JSON emission for each metric.
4. If clean, run **overnight VBench**.
5. Aggregate and compare per-cohort/per-policy metric outputs.

### 8.1 Smoke vs overnight (implemented knobs)

The long-rollout VBench scripts support smoke caps while reusing the same logic path:

- `SMOKE_MAX_COMMON_SAMPLES`
- `SMOKE_MAX_COHORTS`

Smoke should keep these low (e.g., 3 samples / 8 cohorts) and overnight should leave them at `0` (disabled).

---

## 9) Failure Modes to Watch

- Wrong output root (short-run folder instead of long-rollout folder).
- Policy mixing due to flat directory evaluation.
- Unequal cohort sample counts causing unfair comparison.
- Silent worker failure masked by background process handling.
- KV-cache edge-case failures at long horizons.

These are specifically addressed by the current scripts and runtime patches.

---

## 10) Going Forward

This reference is the baseline protocol for long-rollout experiments.
Any changes to:

- rollout backend,
- cache logic,
- schedule semantics,
- boundary mapping,
- VBench cohort/fairness policy

should be treated as experiment design changes and documented before execution.

---

## 11) Example Run Card (Stage3, Single Prompt x Seed)

This section gives a concrete reference for one stage3 shard unit.

### 11.1 Unit definition

One stage3 shard task runs one `(prompt_id, seed)` pair with:

- `run_exhaustive_single_heavy = true`
- `run_policy_comparison = false`
- long-rollout settings (`num_output_frames=120`, backend `long_video`)

Given `total_chunks = 40`, exhaustive outputs are:

- `all_fast` (1 rollout)
- `all_heavy` (1 rollout)
- `single_heavy_00 ... single_heavy_39` (40 rollouts)

Total rollouts for this single prompt-seed unit: **42**.

### 11.2 Per-rollout file landing

For each schedule, files land under:

```text
outputs/long_keyframe_budget/<experiment_name>/rollouts/<prompt_id>/<seed>/<schedule_name>/
  full.mp4
  rollout_meta.json
  metrics.json
  chunk_boundaries.json
```

Example schedule paths for one unit:

- `.../rollouts/<prompt_id>/<seed>/all_fast/`
- `.../rollouts/<prompt_id>/<seed>/all_heavy/`
- `.../rollouts/<prompt_id>/<seed>/single_heavy_00/`
- `.../rollouts/<prompt_id>/<seed>/single_heavy_01/`
- ...
- `.../rollouts/<prompt_id>/<seed>/single_heavy_39/`

### 11.3 Aggregate/result tracking for this unit

After rollouts finish, tracking artifacts for the experiment are updated:

- prompt-seed aggregate:
  - `aggregates/prompt_seed_aggregates/<prompt_id>_seed<seed>.json`
- gain map rows:
  - `aggregates/gain_maps.jsonl`
- policy rows (if policy mode enabled):
  - `aggregates/policy_results.jsonl`

For stage3 discovery (`run_policy_comparison=false`), the critical outputs are:

- per-schedule rollout metadata and media files,
- prompt-seed aggregate with oracle gain records,
- `gain_maps.jsonl` for downstream ranking/analysis.

### 11.4 How this maps to VBench cohorts

Because generation names schedules explicitly and stores one `full.mp4` per schedule,
VBench cohorting naturally resolves to schedule-specific groups such as:

- `all_fast__full`
- `all_heavy__full`
- `single_heavy_00__full`
- ...
- `single_heavy_39__full`

This is the core reason policy identity is preserved cleanly into evaluation.

