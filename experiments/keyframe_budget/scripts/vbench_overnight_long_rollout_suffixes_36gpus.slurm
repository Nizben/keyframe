#!/bin/bash
#SBATCH --job-name=kfb_vbench_suffix36
#SBATCH -A hdy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --array=0-62%9
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# VBench overnight evaluation for materialized long-rollout suffix clips.
#
# Parallelization model:
# - 7 VBench metrics x 9 cohort shards = 63 Slurm array tasks.
# - each task gets 4 H100 GPUs and evaluates its assigned metric/cohort shard.
# - array concurrency is capped at 9 tasks, so the job can use up to 36 H100 GPUs.
#
# Input is expected to be the suffix tree:
#   outputs/long_keyframe_budget_suffixes/long_rollout_discovery_p*/rollouts/.../suffixes/*.mp4
#
# This script intentionally does not materialize suffixes. Finish that first with:
#   sbatch experiments/keyframe_budget/scripts/finish_long_rollout_suffix_materialization.slurm

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
conda activate "${CONDA_ENV_NAME}"

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"
FULL_INFO_JSON="${FULL_INFO_JSON:-${VBENCH_DIR}/vbench/VBench_full_info.json}"

INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-${ROOT_DIR}/outputs/long_keyframe_budget_suffixes}"
RUN_TAG="${RUN_TAG:-long_rollout_suffixes_36gpus}"
FAIR_MATCH_MODE="${FAIR_MATCH_MODE:-strict}"
VBENCH_MODE="${VBENCH_MODE:-custom_input}"
RESUME_SKIP_COMPLETED="${RESUME_SKIP_COMPLETED:-1}"

METRIC_SHARDS="${METRIC_SHARDS:-9}"
GPUS_PER_TASK="${GPUS_PER_TASK:-4}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"

# Optional smoke/debug caps. Leave at 0 for the overnight run.
SMOKE_MAX_COMMON_SAMPLES="${SMOKE_MAX_COMMON_SAMPLES:-0}"
SMOKE_MAX_COHORTS="${SMOKE_MAX_COHORTS:-0}"

# Optional cohort regex, e.g. '.*__(full|suffix_from_00|suffix_from_16)$'.
# Empty means evaluate every discovered suffix cohort.
COHORT_FILTER_REGEX="${COHORT_FILTER_REGEX:-}"

METRICS=(
  "dynamic_degree"
  "motion_smoothness"
  "overall_consistency"
  "imaging_quality"
  "aesthetic_quality"
  "subject_consistency"
  "background_consistency"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
METRIC_COUNT="${#METRICS[@]}"
TOTAL_TASKS=$((METRIC_COUNT * METRIC_SHARDS))
if [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "[vbench-suffix36] Invalid task id ${TASK_ID}; expected < ${TOTAL_TASKS}."
  exit 1
fi

METRIC_INDEX=$((TASK_ID / METRIC_SHARDS))
COHORT_SHARD_INDEX=$((TASK_ID % METRIC_SHARDS))
CURRENT_METRIC="${METRICS[$METRIC_INDEX]}"

if [ ! -d "${INPUT_VIDEO_DIR}" ]; then
  echo "[vbench-suffix36] INPUT_VIDEO_DIR not found: ${INPUT_VIDEO_DIR}"
  exit 1
fi
if [ ! -f "${FULL_INFO_JSON}" ]; then
  echo "[vbench-suffix36] Missing VBench full info json: ${FULL_INFO_JSON}"
  exit 1
fi
if [ "${VBENCH_MODE}" != "custom_input" ]; then
  echo "[vbench-suffix36] This script is configured for VBench custom_input mode only."
  exit 1
fi
if [ "${FAIR_MATCH_MODE}" != "strict" ] && [ "${FAIR_MATCH_MODE}" != "intersect" ]; then
  echo "[vbench-suffix36] FAIR_MATCH_MODE must be strict or intersect, got ${FAIR_MATCH_MODE}."
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs/keyframe_budget"
cd "${VBENCH_DIR}"

TMP_BASE="${TMP_BASE:-${SLURM_TMPDIR:-/tmp}/keyframe_budget_long_rollout_suffix36}"
TASK_WORK_ROOT="${TMP_BASE}/${RUN_TAG}/${CURRENT_METRIC}/cohort_shard_${COHORT_SHARD_INDEX}"
DISCOVERY_JSON="${SLURM_TMPDIR:-/tmp}/kfb_suffix36_discovery_${SLURM_JOB_ID:-manual}_${TASK_ID}.json"
COHORT_LIST_PATH="${SLURM_TMPDIR:-/tmp}/kfb_suffix36_cohorts_${SLURM_JOB_ID:-manual}_${TASK_ID}.txt"
OUT_ROOT="${OUT_ROOT:-${VBENCH_DIR}/evaluation_results/keyframe_budget_long_rollout_suffix36/${RUN_TAG}/${CURRENT_METRIC}}"

echo "=========================================================="
echo " Metric:             ${CURRENT_METRIC}"
echo " Metric index:       ${METRIC_INDEX}/${METRIC_COUNT}"
echo " Cohort shard:       ${COHORT_SHARD_INDEX}/${METRIC_SHARDS}"
echo " Input video root:   ${INPUT_VIDEO_DIR}"
echo " Output root:        ${OUT_ROOT}"
echo " Fairness mode:      ${FAIR_MATCH_MODE}"
echo " GPUs per task:      ${GPUS_PER_TASK}"
echo " VBench mode:        ${VBENCH_MODE}"
echo " Resume skip done:   ${RESUME_SKIP_COMPLETED}"
echo " Cohort filter:      ${COHORT_FILTER_REGEX:-<none>}"
echo " Discovery json:     ${DISCOVERY_JSON}"
echo "=========================================================="

python - <<'PY' \
  "${INPUT_VIDEO_DIR}" \
  "${DISCOVERY_JSON}" \
  "${COHORT_LIST_PATH}" \
  "${FAIR_MATCH_MODE}" \
  "${SMOKE_MAX_COMMON_SAMPLES}" \
  "${SMOKE_MAX_COHORTS}" \
  "${COHORT_FILTER_REGEX}" \
  "${METRIC_SHARDS}" \
  "${COHORT_SHARD_INDEX}"
import json
import os
import re
import sys
from collections import defaultdict

(
    input_root,
    discovery_json,
    cohort_list_path,
    fairness_mode,
    max_common_samples,
    max_cohorts,
    cohort_filter_regex,
    metric_shards,
    cohort_shard_index,
) = sys.argv[1:10]
max_common_samples = int(max_common_samples)
max_cohorts = int(max_cohorts)
metric_shards = int(metric_shards)
cohort_shard_index = int(cohort_shard_index)

if max_common_samples < 0 or max_cohorts < 0:
    raise ValueError("Smoke caps must be >= 0.")
if metric_shards <= 0:
    raise ValueError(f"METRIC_SHARDS must be > 0, got {metric_shards}.")
if not 0 <= cohort_shard_index < metric_shards:
    raise ValueError(f"Invalid cohort shard index: {cohort_shard_index}/{metric_shards}.")

cohort_pattern = re.compile(cohort_filter_regex) if cohort_filter_regex else None
cohort_to_samples = defaultdict(dict)
unparsed = []
shard_pattern = re.compile(r"^long_rollout_discovery_p\d{3}_s\d+$")

for shard_name in sorted(os.listdir(input_root)):
    if not shard_pattern.match(shard_name):
        continue
    rollout_root = os.path.join(input_root, shard_name, "rollouts")
    if not os.path.isdir(rollout_root):
        unparsed.append(os.path.join(shard_name, "rollouts"))
        continue
    for dirpath, _, filenames in os.walk(rollout_root):
        for name in filenames:
            if not name.endswith(".mp4"):
                continue
            full_path = os.path.join(dirpath, name)
            rel = os.path.relpath(full_path, input_root)
            parts = rel.split(os.sep)
            stem = os.path.splitext(name)[0]

            # Expected suffix layout:
            # long_rollout_discovery_pXXX_sY/rollouts/<prompt>/<seed>/<schedule>/suffixes/suffix_from_XX.mp4
            if len(parts) == 7 and parts[1] == "rollouts" and parts[5] == "suffixes":
                prompt_id = parts[2]
                seed = parts[3]
                schedule = parts[4]
                sample_key = f"{prompt_id}::{seed}"
                cohort_tag = f"{schedule}__{stem}"
                if cohort_pattern and not cohort_pattern.search(cohort_tag):
                    continue
                if sample_key in cohort_to_samples[cohort_tag]:
                    raise RuntimeError(
                        f"Duplicate sample '{sample_key}' in cohort '{cohort_tag}'. "
                        f"Existing={cohort_to_samples[cohort_tag][sample_key]} New={full_path}"
                    )
                cohort_to_samples[cohort_tag][sample_key] = full_path
                continue

            unparsed.append(rel)

if not cohort_to_samples:
    raise RuntimeError(f"No suffix .mp4 videos found under {input_root}")

sample_sets = [set(samples.keys()) for samples in cohort_to_samples.values()]
common_samples = set.intersection(*sample_sets)
all_counts = {k: len(v) for k, v in sorted(cohort_to_samples.items())}
max_count = max(all_counts.values())

if not common_samples:
    raise RuntimeError("No common sample keys across discovered suffix cohorts.")

if fairness_mode == "strict":
    bad = {k: c for k, c in all_counts.items() if c != max_count}
    if bad:
        preview = dict(list(bad.items())[:20])
        raise RuntimeError(
            "Strict fairness failed. Cohorts must have identical sample counts. "
            f"bad_count={len(bad)} bad_preview={preview}"
        )
    for cohort_tag, samples in cohort_to_samples.items():
        missing = sorted(set(common_samples) - set(samples.keys()))
        if missing:
            raise RuntimeError(f"Cohort {cohort_tag} missing common samples: {missing[:5]}")

cohort_items = sorted(cohort_to_samples.items())
if max_cohorts > 0:
    cohort_items = cohort_items[:max_cohorts]

cohort_items = [
    item for idx, item in enumerate(cohort_items)
    if idx % metric_shards == cohort_shard_index
]
if not cohort_items:
    raise RuntimeError(
        f"No cohorts assigned to shard {cohort_shard_index}/{metric_shards}; "
        "reduce METRIC_SHARDS or check cohort filters."
    )

common_samples_sorted = sorted(common_samples)
if max_common_samples > 0:
    common_samples_sorted = common_samples_sorted[:max_common_samples]

payload = {
    "input_root": input_root,
    "fairness_mode": fairness_mode,
    "metric_shards": metric_shards,
    "cohort_shard_index": cohort_shard_index,
    "cohort_filter_regex": cohort_filter_regex,
    "cohorts": {
        cohort_tag: {
            "sample_count": len(samples),
            "samples": dict(sorted(samples.items())),
        }
        for cohort_tag, samples in cohort_items
    },
    "common_samples": common_samples_sorted,
    "all_common_samples_count": len(common_samples),
    "all_discovered_cohort_count": len(cohort_to_samples),
    "assigned_cohort_count": len(cohort_items),
    "all_counts_min": min(all_counts.values()),
    "all_counts_max": max(all_counts.values()),
    "unparsed_rel_paths_preview": sorted(unparsed)[:25],
}

os.makedirs(os.path.dirname(discovery_json), exist_ok=True)
with open(discovery_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)

with open(cohort_list_path, "w", encoding="utf-8") as f:
    for cohort_tag in payload["cohorts"]:
        f.write(cohort_tag)
        f.write("\n")

print(f"[vbench-suffix36] Discovered cohorts: {len(cohort_to_samples)}")
print(f"[vbench-suffix36] Assigned cohorts: {len(cohort_items)}")
print(f"[vbench-suffix36] Common samples: {len(common_samples_sorted)} / {len(common_samples)}")
print(f"[vbench-suffix36] Per-cohort sample count min/max: {payload['all_counts_min']}/{payload['all_counts_max']}")
if unparsed:
    print(f"[vbench-suffix36] WARNING: ignored {len(unparsed)} unparsed mp4 paths.")
print(f"[vbench-suffix36] Wrote discovery: {discovery_json}")
print(f"[vbench-suffix36] Wrote cohort list: {cohort_list_path}")
PY

PIDS=()
FAIL=0

while IFS= read -r COHORT_TAG; do
  [ -z "${COHORT_TAG}" ] && continue
  COHORT_WORK="${TASK_WORK_ROOT}/${COHORT_TAG}"
  SHARD_ROOT="${COHORT_WORK}/shards"
  OUT_BASE="${OUT_ROOT}/${COHORT_TAG}"
  mkdir -p "${SHARD_ROOT}" "${OUT_BASE}"

  python - <<'PY' "${DISCOVERY_JSON}" "${COHORT_TAG}" "${SHARD_ROOT}" "${GPUS_PER_TASK}"
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

print(f"[vbench-suffix36] Cohort={cohort_tag} prepared {len(pairs)} videos at {shard_root}")
print(f"[vbench-suffix36] Cohort manifest={manifest}")
PY

  echo "[vbench-suffix36] Cohort ${COHORT_TAG} ready for ${CURRENT_METRIC}"

  PIDS=()
  for GPU_ID in $(seq 0 $((GPUS_PER_TASK - 1))); do
    TARGET_DIR="${SHARD_ROOT}/chunk_${GPU_ID}"
    CUSTOM_OUT_DIR="${OUT_BASE}/chunk_${GPU_ID}"
    mkdir -p "${CUSTOM_OUT_DIR}"

    DELAY=$((GPU_ID * STAGGER_SECONDS))
    UNIQUE_PORT=$((29500 + TASK_ID * 10 + GPU_ID))

    (
      echo "[vbench-suffix36] ${CURRENT_METRIC} ${COHORT_TAG} GPU ${GPU_ID} sleep ${DELAY}s"
      sleep "${DELAY}"

      if [ -z "$(ls -A "${TARGET_DIR}")" ]; then
        echo "[vbench-suffix36] ${CURRENT_METRIC} ${COHORT_TAG} GPU ${GPU_ID} chunk empty, skip."
        exit 0
      fi
      if [ "${RESUME_SKIP_COMPLETED}" = "1" ] && find "${CUSTOM_OUT_DIR}" -maxdepth 1 -type f -name '*_eval_results.json' -size +0c | grep -q .; then
        echo "[vbench-suffix36] ${CURRENT_METRIC} ${COHORT_TAG} GPU ${GPU_ID} existing eval result found, skip."
        exit 0
      fi

      CUDA_VISIBLE_DEVICES="${GPU_ID}" MASTER_PORT="${UNIQUE_PORT}" python evaluate.py \
        --videos_path "${TARGET_DIR}" \
        --full_json_dir "${FULL_INFO_JSON}" \
        --dimension "${CURRENT_METRIC}" \
        --mode "${VBENCH_MODE}" \
        --output_path "${CUSTOM_OUT_DIR}"

      echo "[vbench-suffix36] ${CURRENT_METRIC} ${COHORT_TAG} GPU ${GPU_ID} done."
    ) &
    PIDS+=("$!")
  done

  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      FAIL=1
    fi
  done

  if [ "${FAIL}" -ne 0 ]; then
    echo "[vbench-suffix36] Failure detected in cohort ${COHORT_TAG} for metric ${CURRENT_METRIC}"
    exit 1
  fi
done < "${COHORT_LIST_PATH}"

echo "[vbench-suffix36] Completed metric ${CURRENT_METRIC}, cohort shard ${COHORT_SHARD_INDEX}/${METRIC_SHARDS}."
