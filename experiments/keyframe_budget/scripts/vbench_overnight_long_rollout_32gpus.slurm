#!/bin/bash
#SBATCH --job-name=kfb_vbench_long
#SBATCH -A hdy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
conda activate "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"
FULL_INFO_JSON="${FULL_INFO_JSON:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench/vbench/VBench_full_info.json}"

# Long-rollout experiment root. Override if your completed run lives elsewhere.
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget}"
RUN_TAG="${RUN_TAG:-long_rollout_eval}"
MATERIALIZE_SUFFIXES="${MATERIALIZE_SUFFIXES:-1}"
SUFFIX_OUTPUT_ROOT="${SUFFIX_OUTPUT_ROOT:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget_suffixes}"
SUFFIX_OVERWRITE="${SUFFIX_OVERWRITE:-0}"

# Fairness controls:
# - strict: evaluate only samples available in all cohorts; fail if any cohort misses samples.
# - intersect: evaluate common samples only, even if some cohorts are incomplete.
FAIR_MATCH_MODE="${FAIR_MATCH_MODE:-strict}"

GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"
VBENCH_MODE="${VBENCH_MODE:-custom_input}"
# Optional smoke caps (0 means disabled).
SMOKE_MAX_COMMON_SAMPLES="${SMOKE_MAX_COMMON_SAMPLES:-0}"
SMOKE_MAX_COHORTS="${SMOKE_MAX_COHORTS:-0}"

METRICS=(
  "dynamic_degree"
  "motion_smoothness"
  "overall_consistency"
  "imaging_quality"
  "aesthetic_quality"
  "subject_consistency"
  "background_consistency"
)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#METRICS[@]}" ]; then
  echo "[vbench-long] Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi
CURRENT_METRIC="${METRICS[$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "$INPUT_VIDEO_DIR" ]; then
  echo "[vbench-long] INPUT_VIDEO_DIR not found: $INPUT_VIDEO_DIR"
  exit 1
fi
if [[ "$INPUT_VIDEO_DIR" != *long_rollout* ]] && [[ "$INPUT_VIDEO_DIR" != *keyframe_budget* ]]; then
  echo "[vbench-long] Refusing unknown input root: $INPUT_VIDEO_DIR"
  echo "[vbench-long] Point INPUT_VIDEO_DIR to your long-rollout output tree."
  exit 1
fi
if [ ! -f "$FULL_INFO_JSON" ]; then
  echo "[vbench-long] Missing VBench full info json: $FULL_INFO_JSON"
  exit 1
fi
if [ "$VBENCH_MODE" != "custom_input" ]; then
  echo "[vbench-long] This script is configured for custom_input mode only."
  exit 1
fi

if [ "$MATERIALIZE_SUFFIXES" = "1" ]; then
  SUFFIX_MARKER_DIR="${SUFFIX_OUTPUT_ROOT}/_suffix_materialization/${RUN_TAG}"
  SUFFIX_DONE_FILE="${SUFFIX_MARKER_DIR}/ready.ok"
  SUFFIX_FAIL_FILE="${SUFFIX_MARKER_DIR}/ready.fail"
  mkdir -p "$SUFFIX_MARKER_DIR"

  if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    if [ ! -f "$SUFFIX_DONE_FILE" ] || [ "$SUFFIX_OVERWRITE" = "1" ]; then
      rm -f "$SUFFIX_FAIL_FILE" "$SUFFIX_DONE_FILE"
      CMD=(
        python "${ROOT_DIR}/experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes.py"
        --input_root "$INPUT_VIDEO_DIR"
        --output_root "$SUFFIX_OUTPUT_ROOT"
      )
      if [ "$SUFFIX_OVERWRITE" = "1" ]; then
        CMD+=(--overwrite)
      fi
      if "${CMD[@]}"; then
        touch "$SUFFIX_DONE_FILE"
      else
        touch "$SUFFIX_FAIL_FILE"
        echo "[vbench-long] Suffix materialization failed."
        exit 1
      fi
    fi
  else
    MAX_WAIT_SECONDS=7200
    WAITED=0
    while [ ! -f "$SUFFIX_DONE_FILE" ] && [ ! -f "$SUFFIX_FAIL_FILE" ]; do
      sleep 15
      WAITED=$((WAITED + 15))
      if [ "$WAITED" -ge "$MAX_WAIT_SECONDS" ]; then
        echo "[vbench-long] Timeout waiting for suffix materialization marker."
        exit 1
      fi
    done
    if [ -f "$SUFFIX_FAIL_FILE" ]; then
      echo "[vbench-long] Suffix materialization failed in array task 0."
      exit 1
    fi
  fi

  INPUT_VIDEO_DIR="$SUFFIX_OUTPUT_ROOT"
fi

mkdir -p "$ROOT_DIR/logs/keyframe_budget"
cd "$VBENCH_DIR"

TMP_BASE="${TMP_BASE:-$VBENCH_DIR/temp_shards/keyframe_budget_long_rollout}"
DISCOVERY_JSON="${DISCOVERY_JSON:-${SLURM_TMPDIR:-/tmp}/kfb_vbench_discovery_${SLURM_JOB_ID:-manual}.json}"
METRIC_WORK_ROOT="${TMP_BASE}/${RUN_TAG}/${CURRENT_METRIC}"
OUT_ROOT="${OUT_ROOT:-$VBENCH_DIR/evaluation_results/keyframe_budget_long_rollout/${RUN_TAG}/${CURRENT_METRIC}}"

echo "=========================================================="
echo " Metric:            $CURRENT_METRIC"
echo " Input video root:  $INPUT_VIDEO_DIR"
echo " Fairness mode:     $FAIR_MATCH_MODE"
echo " GPUs per job:      $GPUS_PER_JOB"
echo " VBench mode:       $VBENCH_MODE"
echo " Max common sample: $SMOKE_MAX_COMMON_SAMPLES"
echo " Max cohorts:       $SMOKE_MAX_COHORTS"
echo " Discovery json:    $DISCOVERY_JSON"
echo "=========================================================="

python - <<'PY' "$INPUT_VIDEO_DIR" "$DISCOVERY_JSON" "$FAIR_MATCH_MODE" "$SMOKE_MAX_COMMON_SAMPLES" "$SMOKE_MAX_COHORTS"
import json
import os
import sys
from collections import defaultdict

input_root, out_json, fairness_mode, max_common_samples, max_cohorts = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
if fairness_mode not in {"strict", "intersect"}:
    raise ValueError(f"Unknown FAIR_MATCH_MODE={fairness_mode}")
if max_common_samples < 0:
    raise ValueError(f"SMOKE_MAX_COMMON_SAMPLES must be >= 0, got {max_common_samples}")
if max_cohorts < 0:
    raise ValueError(f"SMOKE_MAX_COHORTS must be >= 0, got {max_cohorts}")

cohort_to_samples = defaultdict(dict)
unparsed = []

for dirpath, _, filenames in os.walk(input_root):
    for name in filenames:
        if not name.endswith(".mp4"):
            continue
        full_path = os.path.join(dirpath, name)
        rel = os.path.relpath(full_path, input_root)
        parts = rel.split(os.sep)
        stem = os.path.splitext(name)[0]

        # Expected layout:
        # .../rollouts/<prompt_id>/<seed>/<schedule_name>/<video_name>.mp4
        if "rollouts" in parts:
            ridx = parts.index("rollouts")
            if len(parts) > ridx + 4:
                prompt_id = parts[ridx + 1]
                seed = parts[ridx + 2]
                schedule = parts[ridx + 3]
                sample_key = f"{prompt_id}::{seed}"
                cohort_tag = f"{schedule}__{stem}"
                if sample_key in cohort_to_samples[cohort_tag]:
                    raise RuntimeError(
                        f"Duplicate sample '{sample_key}' in cohort '{cohort_tag}'. "
                        f"Existing={cohort_to_samples[cohort_tag][sample_key]} New={full_path}"
                    )
                cohort_to_samples[cohort_tag][sample_key] = full_path
                continue

        # Fallback: group by parent dir + stem.
        parent = os.path.basename(os.path.dirname(full_path)) or "root"
        sample_key = rel
        cohort_tag = f"{parent}__{stem}"
        cohort_to_samples[cohort_tag][sample_key] = full_path
        unparsed.append(rel)

if not cohort_to_samples:
    raise RuntimeError(f"No .mp4 videos found under {input_root}")

sample_sets = [set(samples.keys()) for samples in cohort_to_samples.values()]
common_samples = set.intersection(*sample_sets)
all_counts = {k: len(v) for k, v in sorted(cohort_to_samples.items())}
max_count = max(all_counts.values())

if not common_samples:
    raise RuntimeError("No common sample keys across discovered cohorts.")

if fairness_mode == "strict":
    bad = {k: c for k, c in all_counts.items() if c != max_count}
    if bad:
        raise RuntimeError(
            "Strict fairness failed. Cohorts must have identical sample counts. "
            f"counts={all_counts}"
        )
    for cohort_tag, samples in cohort_to_samples.items():
        missing = sorted(set(common_samples) - set(samples.keys()))
        if missing:
            raise RuntimeError(f"Cohort {cohort_tag} missing common samples: {missing[:5]}")

cohort_items = sorted(cohort_to_samples.items())
if max_cohorts > 0:
    cohort_items = cohort_items[:max_cohorts]
common_samples_sorted = sorted(common_samples)
if max_common_samples > 0:
    common_samples_sorted = common_samples_sorted[:max_common_samples]

payload = {
    "input_root": input_root,
    "fairness_mode": fairness_mode,
    "cohorts": {
        cohort_tag: {
            "sample_count": len(samples),
            "samples": dict(sorted(samples.items())),
        }
        for cohort_tag, samples in cohort_items
    },
    "common_samples": common_samples_sorted,
    "all_common_samples_count": len(common_samples),
    "unparsed_rel_paths_preview": sorted(unparsed)[:25],
}

os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)

print(f"[vbench-long] Discovered cohorts: {len(payload['cohorts'])}")
print(f"[vbench-long] Common sample count: {len(payload['common_samples'])}")
if unparsed:
    print(f"[vbench-long] WARNING: used fallback parsing for {len(unparsed)} files.")
print(f"[vbench-long] Wrote discovery: {out_json}")
PY

PIDS=()
FAIL=0

while IFS= read -r COHORT_TAG; do
  [ -z "$COHORT_TAG" ] && continue
  COHORT_WORK="${METRIC_WORK_ROOT}/${COHORT_TAG}"
  SHARD_ROOT="${COHORT_WORK}/shards"
  OUT_BASE="${OUT_ROOT}/${COHORT_TAG}"
  mkdir -p "$SHARD_ROOT" "$OUT_BASE"

  python - <<'PY' "$DISCOVERY_JSON" "$COHORT_TAG" "$SHARD_ROOT" "$GPUS_PER_JOB"
import json
import math
import os
import shutil
import sys

discovery_json, cohort_tag, shard_root, num_chunks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
if num_chunks <= 0:
    raise ValueError(f"num_chunks must be > 0, got {num_chunks}")

with open(discovery_json, "r", encoding="utf-8") as f:
    payload = json.load(f)

cohort = payload["cohorts"][cohort_tag]["samples"]
common = set(payload["common_samples"])
pairs = [(sample_key, path) for sample_key, path in sorted(cohort.items()) if sample_key in common]
if not pairs:
    raise RuntimeError(f"Cohort {cohort_tag} has no common samples to evaluate.")

if os.path.exists(shard_root):
    shutil.rmtree(shard_root)
os.makedirs(shard_root, exist_ok=True)

manifest = os.path.join(shard_root, "video_manifest.tsv")
with open(manifest, "w", encoding="utf-8") as f:
    f.write("sample_key\tvideo_path\n")
    for sample_key, path in pairs:
        f.write(f"{sample_key}\t{path}\n")

chunk_size = math.ceil(len(pairs) / num_chunks)
for i in range(num_chunks):
    chunk_dir = os.path.join(shard_root, f"chunk_{i}")
    os.makedirs(chunk_dir, exist_ok=True)
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(pairs))
    for idx, (sample_key, src_path) in enumerate(pairs[start:end], start=start):
        safe_key = sample_key.replace("::", "__")
        dst_name = f"{idx:06d}_{safe_key}.mp4"
        dst_path = os.path.join(chunk_dir, dst_name)
        if os.path.lexists(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)

print(f"[vbench-long] Cohort={cohort_tag} prepared {len(pairs)} videos at {shard_root}")
print(f"[vbench-long] Cohort manifest={manifest}")
PY

  echo "[vbench-long] Cohort ${COHORT_TAG} ready for ${CURRENT_METRIC}"

  for GPU_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    TARGET_DIR="${SHARD_ROOT}/chunk_${GPU_ID}"
    CUSTOM_OUT_DIR="${OUT_BASE}/chunk_${GPU_ID}"
    mkdir -p "$CUSTOM_OUT_DIR"

    DELAY=$((GPU_ID * STAGGER_SECONDS))
    UNIQUE_PORT=$((29500 + SLURM_ARRAY_TASK_ID * 100 + GPU_ID))

    (
      echo "[vbench-long] ${COHORT_TAG} GPU ${GPU_ID} sleep ${DELAY}s"
      sleep "$DELAY"

      if [ -z "$(ls -A "$TARGET_DIR")" ]; then
        echo "[vbench-long] ${COHORT_TAG} GPU ${GPU_ID} chunk empty, skip."
        exit 0
      fi

      CUDA_VISIBLE_DEVICES="$GPU_ID" MASTER_PORT="$UNIQUE_PORT" python evaluate.py \
        --videos_path "$TARGET_DIR" \
        --full_json_dir "$FULL_INFO_JSON" \
        --dimension "$CURRENT_METRIC" \
        --mode "$VBENCH_MODE" \
        --output_path "$CUSTOM_OUT_DIR"

      echo "[vbench-long] ${COHORT_TAG} GPU ${GPU_ID} done."
    ) &
    PIDS+=("$!")
  done

  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAIL=1
    fi
  done
  PIDS=()
done < <(python - <<'PY' "$DISCOVERY_JSON"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
for cohort in sorted(payload["cohorts"].keys()):
    print(cohort)
PY
)

if [ "$FAIL" -ne 0 ]; then
  echo "[vbench-long] Failures detected for metric ${CURRENT_METRIC}"
  exit 1
fi

echo "[vbench-long] Completed metric ${CURRENT_METRIC} across all cohorts."
