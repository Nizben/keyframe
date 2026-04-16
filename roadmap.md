# v3.5 Keyframe-Budget Validation on Top of Causal Forcing

This document is a **Cursor-ready implementation handoff** for building the v3.5 keyframe-budget validation experiments **on top of the existing Causal Forcing codebase**.

The main correction relative to the previous direction is simple:

> **Do not build a new autoregressive video generator from scratch.**
> Use the existing **chunk-wise autoregressive video diffusion generator** from **Causal Forcing**, and implement a flexible **per-chunk denoising-budget schedule** plus an experiment harness around it.

---

# 1. High-level objective

We want to validate the following hypothesis behind v3.5:

1. In autoregressive video generation, **not all chunks are equally important**.
2. Some chunks behave like **budget-sensitive keyframes**: spending more denoising budget on them yields larger improvements than spending the same extra budget elsewhere.
3. At fixed total compute, **non-uniform allocation of denoising steps across chunks** can outperform uniform or random allocation.

This is **not** a new training framework yet.
This stage is an **inference-time validation study**.

---

# 2. Core implementation decision

## 2.1 What we are NOT building

We are **not** implementing:
- a new AR video model,
- a new training pipeline,
- a new chunked generator from scratch,
- a hybrid system that swaps unrelated fast/heavy checkpoints inside the same rollout.

## 2.2 What we ARE building

We are building an **experiment framework** on top of **Causal Forcing** that:
- runs the existing **chunk-wise AR diffusion generator**,
- allows specifying a different **number of denoising steps for each chunk**,
- runs controlled rollout comparisons,
- logs metrics and metadata,
- computes chunk importance maps,
- compares allocation policies under fixed compute budgets.

So the right mental model is:

> **Causal Forcing provides the AR generator.**
> **We add chunk-level budget scheduling and evaluation around it.**

---

# 3. Base framework and model choice

## 3.1 Use Causal Forcing, chunk-wise

Target codebase: **Causal Forcing**

We want the **chunk-wise** setting, not the frame-wise one.

Reason:
- our hypothesis is chunk-level,
- chunk-wise is more stable,
- chunk-level scheduling maps naturally to the generator,
- chunk-wise aligns better with the v3.5 routing formulation.

## 3.2 Which checkpoint to start from

Start from the **chunk-wise AR diffusion checkpoint**, not the final distilled few-step model.

Use:
- `chunkwise/ar_diffusion.pt`

Do **not** start from:
- `chunkwise/causal_forcing.pt`

Reason:
- for the keyframe-budget study, we need a generator whose output quality still depends meaningfully on the **number of denoising steps**,
- that is exactly the AR diffusion stage,
- the final distilled model is not the cleanest place to validate a step-allocation hypothesis.

## 3.3 Important constraint

Inside a single rollout:
- keep the **same model**,
- keep the **same causal history**,
- keep the **same prompt**,
- keep the **same seed / initial randomness**,
- only vary the **per-chunk denoising budget**.

Do **not** swap different checkpoints mid-rollout.
That would break the clean autoregressive comparison.

---

# 4. What needs to be implemented

There are three main implementation blocks.

## 4.1 Block A — flexible per-chunk budget scheduling

We need to modify the chunk-wise inference loop so that each chunk can use a different number of denoising steps.

Instead of one global parameter like:

```python
num_inference_steps = S
```

we want:

```python
steps_per_chunk = [s_1, s_2, ..., s_T]
```

where `T` is the number of generated chunks.

Then the rollout should do:
- chunk 1: use `s_1` denoising steps,
- append generated chunk to AR history/cache,
- chunk 2: use `s_2`,
- append,
- etc.

This is the single most important code change.

## 4.2 Block B — experiment harness

We need a reusable experiment runner that can:
- generate videos under many chunk schedules,
- keep prompt/seed/control fixed,
- log outputs in a structured way,
- compute metrics,
- aggregate results.

## 4.3 Block C — analysis/oracle builder

We need tooling that:
- runs exhaustive single-heavy-chunk sweeps,
- estimates per-chunk gain maps,
- compares budget-allocation policies,
- saves everything in machine-readable form.

---

# 5. Core experiment object

The central object for the whole framework should be something like:

```python
RolloutSpec(
    prompt_id: str,
    prompt_text: str,
    seed: int,
    checkpoint: str,
    schedule_name: str,
    steps_per_chunk: List[int],
    num_chunks: int,
    output_dir: str,
)
```

And the central runner:

```python
def run_rollout(spec: RolloutSpec) -> RolloutResult:
    ...
```

The point is to make **schedule** a first-class object.

---

# 6. Minimal code architecture to add

Suggested new structure inside the Causal Forcing repo:

```text
experiments/
  keyframe_budget/
    configs/
      smoke_test.yaml
      exhaustive_sweep.yaml
      policy_eval.yaml
    prompts/
      motion_rich_24.json
      motion_rich_100.json
    schedules.py
    runner.py
    oracle.py
    metrics.py
    aggregate.py
    visualize.py

outputs/
  keyframe_budget/
    <exp_name>/
      rollouts/
      metrics/
      aggregates/
      plots/
```

You do not need to reorganize the whole repository. The point is to isolate the v3.5 experiment layer.

---

# 7. Step scheduling design

## 7.1 Schedule types

Implement the following schedule constructors.

### A. All-fast

```python
[L] * T
```

### B. All-heavy

```python
[H] * T
```

### C. Single-heavy-at-i

```python
[L, ..., L]
with position i replaced by H
```

### D. Uniform-top-m

Choose `m` chunk indices approximately equally spaced.

### E. Random-top-m

Choose `m` chunk indices uniformly at random.
Use a reproducible RNG seed.

### F. Prefix-top-m

First `m` chunks heavy.

### G. Oracle-top-m

Choose the `m` chunks with the largest gains from the exhaustive sweep.

---

# 8. Fixed compute matching

Policies must be compared at equal compute.

Let:
- `L` = fast denoising budget,
- `H` = heavy denoising budget,
- `T` = number of chunks,
- `B` = target average budget per chunk.

Then choose:

```python
m = floor(T * (B - L) / (H - L))
```

This gives the number of heavy chunks for the mixed policies.

Example:
- `L = 8`
- `H = 24`
- `B = 12`
- `T = 27`

Then:

```python
m = floor(27 * (12 - 8) / (24 - 8)) = floor(27 * 4 / 16) = 6
```

So any mixed policy should use exactly `6` heavy chunks and `21` fast chunks.

This is critical.
Without compute matching, the comparison is not valid.

---

# 9. Recommended first hyperparameters

Use these first:

```python
FAST_STEPS = 8
HEAVY_STEPS = 24
TARGET_AVG_STEPS = 12
```

Then later maybe add:

```python
HEAVY_STEPS in {16, 24, 32}
TARGET_AVG_STEPS in {10, 12, 14}
```

But do not overcomplicate the first implementation.

---

# 10. The key experiment: exhaustive single-heavy sweep

This is the most important first study.

For each prompt and seed:

1. generate `V_fast` with `all-fast`,
2. generate `V_heavy` with `all-heavy`,
3. for every chunk `i` from `1` to `T`, generate `V_i` with `single-heavy-at-i`.

So if there are `T` chunks, this gives:
- 1 fast video,
- 1 heavy video,
- `T` single-intervention videos.

This lets us estimate the chunk gain map.

## 10.1 Gain definitions

For a global score `Score(.)`, define:

```python
g_i = Score(V_i) - Score(V_fast)
```

Also define normalized recovery:

```python
g_i_norm = g_i / (Score(V_heavy) - Score(V_fast) + eps)
```

This tells us how much of the full heavy-vs-fast gap is recovered by spending extra budget only at chunk `i`.

## 10.2 Suffix gain

This is very important.
Do not only score the full video.
Also score the suffix from chunk `i` onward.

Define:

```python
g_i_suffix = Score(V_i[i:]) - Score(V_fast[i:])
```

This is closer to the actual v3.5 claim:
- chunk `i` is important if improving it improves the **causal future**.

If full-video scores are hard to make suffix-aware in the beginning, approximate suffix analysis with:
- per-frame image metrics on frames after the chunk,
- embedding similarity curves,
- temporal consistency or motion quality over the suffix.

But the goal is to make suffix-level evaluation explicit.

---

# 11. Policy comparison experiment

After the exhaustive sweep, compare the following policies at fixed compute:

- all-fast
- all-heavy
- uniform-top-m
- random-top-m
- prefix-top-m
- oracle-top-m

This produces the clean result:

> At the same compute budget, does allocating heavy denoising to the right chunks beat naive allocation?

This is the main empirical validation for the v3.5 budget-allocation story.

---

# 12. Prompt splits

Do not start from generic prompt sets with almost no motion.
Use motion-rich prompts.

Suggested rollout plan:

## 12.1 Smoke test

- 8 prompts
- 1 seed each

Goal:
- verify the code,
- verify logging,
- verify schedules are actually applied,
- verify gains are not flat.

## 12.2 Discovery split

- 24 motion-rich prompts
- 3 seeds each

Goal:
- run exhaustive single-heavy sweeps,
- build gain maps,
- validate that chunk importance is non-uniform.

## 12.3 Large policy split

- 100 motion-rich prompts
- 3 seeds each

Goal:
- compare all-fast, all-heavy, uniform, random, prefix, oracle.

---

# 13. Metrics to compute

Use both benchmark-style metrics and experiment-specific metrics.

## 13.1 Benchmark-style metrics

Compute at least:
- VBench main scores,
- Dynamic Degree,
- Instruction Following,
- VisionReward if available.

These align the study with the existing AR video evaluation ecosystem.

## 13.2 Experiment-specific metrics

### A. Recovery ratio

For any policy `P`:

```python
recovery(P) = (Score(P) - Score(all_fast)) / (Score(all_heavy) - Score(all_fast) + eps)
```

This is one of the most useful summary numbers.

### B. Gain concentration

Given the single-heavy gains `{g_i}` for a video, compute:
- max gain,
- mean gain,
- median gain,
- top-3 gain mass,
- Gini coefficient over positive gains,
- rank gap: best gain minus median gain.

If keyframes exist, the distribution should be non-uniform and concentrated.

### C. Suffix sensitivity

For each chunk, log the downstream gain on the suffix.

### D. Position bias

Aggregate gain by chunk position index.
Check whether the best chunks systematically occur:
- early,
- middle,
- late,
- or are prompt-dependent.

---

# 14. Output and logging schema

For every rollout, save:

```json
{
  "experiment_name": "...",
  "prompt_id": "...",
  "prompt_text": "...",
  "seed": 0,
  "checkpoint": "chunkwise/ar_diffusion.pt",
  "schedule_name": "single_heavy_at_07",
  "steps_per_chunk": [8, 8, 8, 24, 8, ...],
  "num_chunks": 27,
  "fast_steps": 8,
  "heavy_steps": 24,
  "target_avg_steps": 12,
  "wall_clock_sec": 0.0,
  "total_nfe": 0,
  "video_path": "...",
  "thumbnail_dir": "...",
  "metrics_path": "...",
  "status": "success"
}
```

For every evaluated video, save a separate metrics JSON.

For every prompt/seed pair, save an aggregate file containing:
- all schedule results,
- gains relative to fast,
- oracle chunk ranking.

---

# 15. Recommended folder structure for outputs

```text
outputs/keyframe_budget/<exp_name>/
  rollouts/
    <prompt_id>/<seed>/<schedule_name>/
      video.mp4
      frames/
      thumbs/
      rollout_meta.json
      metrics.json
  aggregates/
    gain_maps.jsonl
    policy_results.jsonl
  plots/
    gain_heatmap.png
    recovery_barplot.png
    suffix_gain_curves.png
```

---

# 16. What to instrument in the inference loop

The exact repo files may differ slightly, but conceptually the chunk-wise inference loop should expose something like:

```python
for chunk_idx in range(num_chunks):
    steps = steps_per_chunk[chunk_idx]
    chunk = sample_next_chunk(
        model=model,
        history=history,
        prompt=prompt,
        num_inference_steps=steps,
        ...
    )
    history = update_history(history, chunk)
```

The main implementation goal is to pass **chunk-specific `num_inference_steps`** into the sampling call.

Also log at runtime:
- `chunk_idx`,
- requested step count,
- actual step count,
- chunk runtime,
- any warnings/errors.

---

# 17. Pseudocode for the core runner

```python
@dataclass
class RolloutSpec:
    prompt_id: str
    prompt_text: str
    seed: int
    checkpoint: str
    steps_per_chunk: list[int]
    schedule_name: str
    output_dir: str


def run_rollout(spec: RolloutSpec):
    set_seed(spec.seed)
    model = load_model(spec.checkpoint)
    history = init_history()
    chunk_outputs = []
    per_chunk_logs = []

    for chunk_idx, num_steps in enumerate(spec.steps_per_chunk):
        t0 = time.time()

        chunk = sample_next_chunk(
            model=model,
            history=history,
            prompt=spec.prompt_text,
            num_inference_steps=num_steps,
        )

        dt = time.time() - t0
        history = append_chunk_to_history(history, chunk)
        chunk_outputs.append(chunk)
        per_chunk_logs.append({
            "chunk_idx": chunk_idx,
            "num_steps": num_steps,
            "runtime_sec": dt,
        })

    video = decode_and_save_video(chunk_outputs, spec.output_dir)
    meta = save_rollout_meta(spec, per_chunk_logs, video)
    return meta
```

---

# 18. Pseudocode for exhaustive single-heavy sweep

```python
def exhaustive_single_heavy_sweep(prompt, seed, T, L, H):
    results = []

    fast_schedule = [L] * T
    heavy_schedule = [H] * T

    results.append(run_rollout(make_spec(prompt, seed, "all_fast", fast_schedule)))
    results.append(run_rollout(make_spec(prompt, seed, "all_heavy", heavy_schedule)))

    for i in range(T):
        schedule = [L] * T
        schedule[i] = H
        results.append(run_rollout(make_spec(prompt, seed, f"single_heavy_{i:02d}", schedule)))

    return results
```

---

# 19. Pseudocode for oracle construction

After exhaustive sweep for one prompt/seed pair:

```python
def build_oracle_from_single_heavy_results(result_dict):
    fast_score = result_dict["all_fast"]["score"]
    heavy_score = result_dict["all_heavy"]["score"]

    gains = []
    for schedule_name, item in result_dict.items():
        if not schedule_name.startswith("single_heavy_"):
            continue
        chunk_idx = parse_chunk_idx(schedule_name)
        gain = item["score"] - fast_score
        gain_norm = gain / (heavy_score - fast_score + 1e-8)
        gains.append({
            "chunk_idx": chunk_idx,
            "gain": gain,
            "gain_norm": gain_norm,
        })

    gains = sorted(gains, key=lambda x: x["gain"], reverse=True)
    return gains
```

The oracle top-`m` schedule is then simply the `m` highest-gain chunk indices.

---

# 20. What to save now for future router training

Even if the current stage is only validation, save cheap chunk-level features from the **all-fast** rollout.
These will later become router inputs.

For each chunk, if possible save:
- chunk index `k`,
- normalized position `k / T`,
- motion energy across the chunk boundary,
- latent difference from previous chunk,
- image-embedding difference from previous chunk,
- first-step residual norm,
- optional cheap defect proxy,
- chunk runtime.

Also save the oracle labels from the exhaustive sweep:
- raw gain,
- normalized gain,
- rank among chunks,
- binary top-`m` membership.

This turns the validation code into the data engine for the future router.

---

# 21. Plots to generate

Generate these plots automatically.

## 21.1 Gain heatmap

Rows = videos (prompt+seed)  
Columns = chunk index  
Value = `g_i_norm`

This is the clearest visual evidence for non-uniform chunk importance.

## 21.2 Recovery plot

Bar chart comparing:
- all-fast
- uniform-top-m
- random-top-m
- prefix-top-m
- oracle-top-m
- all-heavy

using normalized recovery.

## 21.3 Best-vs-median gain plot

For each video, compare:
- best chunk gain,
- median chunk gain.

## 21.4 Position histogram

Histogram of oracle chunk positions.

## 21.5 Suffix gain curves

For each intervention chunk index, plot downstream suffix gain.

---

# 22. Minimal implementation order

The correct implementation order is:

## Step 1
Make the chunk-wise Causal Forcing inference loop accept:

```python
steps_per_chunk: List[int]
```

## Step 2
Implement schedule constructors:
- all-fast
- all-heavy
- single-heavy-at-i
- uniform-top-m
- random-top-m
- prefix-top-m

## Step 3
Implement experiment runner for prompt/seed batches.

## Step 4
Save videos, metadata, and metric JSONs.

## Step 5
Implement exhaustive sweep aggregator and oracle builder.

## Step 6
Implement policy comparison scripts.

## Step 7
Implement plots and summary tables.

Do not start by building a router.
First validate the existence of key budget-sensitive chunks.

---

# 23. First experiment config to run

Use this first:

```yaml
experiment_name: smoke_test
checkpoint: chunkwise/ar_diffusion.pt
fast_steps: 8
heavy_steps: 24
target_avg_steps: 12
num_prompts: 8
seeds: [0]
run_exhaustive_single_heavy: true
run_policy_comparison: false
prompt_set: motion_rich_8.json
```

Goal:
- verify the full loop works,
- verify schedules are respected,
- verify gain maps are not flat.

Then move to:

```yaml
experiment_name: discovery_split
checkpoint: chunkwise/ar_diffusion.pt
fast_steps: 8
heavy_steps: 24
target_avg_steps: 12
num_prompts: 24
seeds: [0, 1, 2]
run_exhaustive_single_heavy: true
run_policy_comparison: true
prompt_set: motion_rich_24.json
```

And later:

```yaml
experiment_name: policy_eval_100
checkpoint: chunkwise/ar_diffusion.pt
fast_steps: 8
heavy_steps: 24
target_avg_steps: 12
num_prompts: 100
seeds: [0, 1, 2]
run_exhaustive_single_heavy: false
run_policy_comparison: true
prompt_set: motion_rich_100.json
oracle_source: discovery_split
```

---

# 24. Failure modes to watch for

## 24.1 Schedules not actually changing anything

Possible cause:
- the code still uses one global denoising-step variable internally.

Action:
- add explicit per-chunk logging,
- assert the requested step count matches the actual one used.

## 24.2 Seed mismatch across schedules

Possible cause:
- random state reset differs between runs,
- per-schedule generation is not controlled.

Action:
- strictly fix seeds,
- log seeds and RNG state if needed.

## 24.3 Not compute-matching policies

Possible cause:
- comparing policies with different total NFEs.

Action:
- explicitly compute expected total steps before each run,
- assert equality when comparing mixed policies.

## 24.4 Using different checkpoints inside one rollout

Do not do this.
This would contaminate the AR comparison.

## 24.5 Relying only on full-video metrics

This misses the causal leverage story.
Try to include suffix-aware analysis from the beginning.

---

# 25. Final conceptual summary

The framework to implement is:

1. **Use Causal Forcing as the existing AR chunk-wise video generator.**
2. **Modify inference so each chunk can have its own denoising step budget.**
3. **Run exhaustive single-heavy-chunk interventions to estimate chunk importance.**
4. **Build oracle chunk rankings from these gains.**
5. **Compare oracle allocation against uniform/random/prefix allocation at fixed compute.**
6. **Save all data in a form that can later bootstrap router training.**

That is the correct implementation pivot.

You are **not** rebuilding the generator.
You are building an **experiment and analysis framework around an existing causal generator**.

---

# 26. Immediate next task for Cursor

The very next concrete coding task should be:

> Find the chunk-wise inference path in the Causal Forcing repo and make it accept a list `steps_per_chunk`, then ensure chunk `k` is sampled with `steps_per_chunk[k]` before updating the AR history.

After that, implement the exhaustive single-heavy sweep runner.

One last simple note: this code is meant to be run on the JeanZay HPC cluster (on H100 gpus) to which i am connected through sshfs, so keep that in mind when running commands especially relating to env and env variables.
