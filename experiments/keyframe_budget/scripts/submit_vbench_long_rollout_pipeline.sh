#!/bin/bash
set -euo pipefail

# Two-phase launch:
#   1) Parallel suffix materialization (CPU array jobs)
#   2) VBench evaluation (GPU array jobs), dependency afterok

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
RUN_TAG="${RUN_TAG:-long_rollout_eval}"
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget}"
SUFFIX_OUTPUT_ROOT="${SUFFIX_OUTPUT_ROOT:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget_suffixes}"
MANIFEST_DIR="${MANIFEST_DIR:-$SUFFIX_OUTPUT_ROOT/_suffix_materialization/$RUN_TAG}"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_DIR/rollout_manifest.txt}"
MAX_SUFFIX_TASKS="${MAX_SUFFIX_TASKS:-32}"
ARRAY_CHUNK_SIZE="${ARRAY_CHUNK_SIZE:-200}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-0}"
SUFFIX_OVERWRITE="${SUFFIX_OVERWRITE:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
SUBMIT_TIMEOUT_SEC="${SUBMIT_TIMEOUT_SEC:-120}"
MAX_SUBMIT_ATTEMPTS="${MAX_SUBMIT_ATTEMPTS:-12}"
ALLOW_FULL_SUBMIT="${ALLOW_FULL_SUBMIT:-0}"
FULL_SUBMIT_THRESHOLD="${FULL_SUBMIT_THRESHOLD:-1000}"

cd "$ROOT_DIR"
mkdir -p "$MANIFEST_DIR"

python "experiments/keyframe_budget/scripts/prepare_long_rollout_suffix_manifest.py" \
  --input_root "$INPUT_VIDEO_DIR" \
  --manifest_path "$MANIFEST_PATH" \
  --max_rollouts "$MAX_ROLLOUTS"

ROLLOUT_COUNT=$(python - <<'PY' "$MANIFEST_PATH"
import sys
path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
)

if [ "$ROLLOUT_COUNT" -le 0 ]; then
  echo "[submit-vbench-pipeline] No rollout entries in manifest."
  exit 1
fi

if [ "$MAX_ROLLOUTS" -eq 0 ] && [ "$ROLLOUT_COUNT" -gt "$FULL_SUBMIT_THRESHOLD" ] && [ "$ALLOW_FULL_SUBMIT" != "1" ]; then
  echo "[submit-vbench-pipeline] SAFETY STOP: manifest has ${ROLLOUT_COUNT} rollouts (> ${FULL_SUBMIT_THRESHOLD})."
  echo "[submit-vbench-pipeline] Refusing large submission unless ALLOW_FULL_SUBMIT=1."
  echo "[submit-vbench-pipeline] For smoke, set MAX_ROLLOUTS=200 (or similar)."
  exit 1
fi

ARRAY_MAX=$((ROLLOUT_COUNT - 1))
if [ "$ARRAY_CHUNK_SIZE" -le 0 ]; then
  echo "[submit-vbench-pipeline] ARRAY_CHUNK_SIZE must be > 0, got $ARRAY_CHUNK_SIZE"
  exit 1
fi
echo "[submit-vbench-pipeline] rollout_count=$ROLLOUT_COUNT max_suffix_tasks=$MAX_SUFFIX_TASKS chunk_size=$ARRAY_CHUNK_SIZE"

submit_with_retry() {
  local cmd="$1"
  local max_attempts="$MAX_SUBMIT_ATTEMPTS"
  local sleep_sec=20
  local attempt=1
  local out=""
  local err_file="/tmp/kfb_submit_err_$$.log"
  local out_file="/tmp/kfb_submit_out_$$.log"
  while [ "$attempt" -le "$max_attempts" ]; do
    echo "[submit-vbench-pipeline] sbatch attempt ${attempt}/${max_attempts}"
    rm -f "$err_file" "$out_file"

    # Portable timeout wrapper (avoids depending on `timeout` binary).
    bash -lc "$cmd" >"$out_file" 2>"$err_file" &
    pid=$!
    elapsed=0
    while kill -0 "$pid" >/dev/null 2>&1; do
      sleep 1
      elapsed=$((elapsed + 1))
      if [ "$elapsed" -ge "$SUBMIT_TIMEOUT_SEC" ]; then
        kill "$pid" >/dev/null 2>&1 || true
        sleep 1
        kill -9 "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
        rc=124
        break
      fi
    done
    if [ "${rc:-0}" != "124" ]; then
      wait "$pid"
      rc=$?
    fi

    if [ "$rc" -eq 0 ]; then
      out=$(cat "$out_file" || true)
      rm -f "$err_file" "$out_file"
      if [ -z "$out" ]; then
        echo "[submit-vbench-pipeline] sbatch returned empty output (attempt ${attempt}/${max_attempts}), retrying in ${sleep_sec}s..."
        attempt=$((attempt + 1))
        sleep "$sleep_sec"
        sleep_sec=$((sleep_sec + 10))
        continue
      fi
      printf "%s\n" "$out"
      return 0
    fi

    err_msg=$(cat "$err_file" || true)
    if [ "$rc" -eq 124 ]; then
      echo "[submit-vbench-pipeline] sbatch timed out after ${SUBMIT_TIMEOUT_SEC}s (attempt ${attempt}/${max_attempts}), retrying in ${sleep_sec}s..."
      sleep "$sleep_sec"
      attempt=$((attempt + 1))
      sleep_sec=$((sleep_sec + 10))
      continue
    fi
    if echo "$err_msg" | grep -q "Resource temporarily unavailable\\|temporarily unable to accept job"; then
      echo "[submit-vbench-pipeline] sbatch overloaded (attempt ${attempt}/${max_attempts}), retrying in ${sleep_sec}s..."
      sleep "$sleep_sec"
      attempt=$((attempt + 1))
      sleep_sec=$((sleep_sec + 10))
    else
      echo "$err_msg"
      rm -f "$err_file" "$out_file"
      return 1
    fi
  done
  echo "[submit-vbench-pipeline] sbatch retry budget exhausted."
  cat "$err_file" || true
  rm -f "$err_file" "$out_file"
  return 1
}

SUFFIX_JOBS=()
START=0
while [ "$START" -lt "$ROLLOUT_COUNT" ]; do
  END=$((START + ARRAY_CHUNK_SIZE - 1))
  if [ "$END" -gt "$ARRAY_MAX" ]; then
    END="$ARRAY_MAX"
  fi
  LOCAL_COUNT=$((END - START + 1))
  ARRAY_SPEC="0-$((LOCAL_COUNT - 1))%${MAX_SUFFIX_TASKS}"
  echo "[submit-vbench-pipeline] submitting suffix chunk start=$START end=$END array=$ARRAY_SPEC"
  CMD="sbatch --parsable --array=\"$ARRAY_SPEC\" --export=ALL,ROOT_DIR=\"$ROOT_DIR\",INPUT_VIDEO_DIR=\"$INPUT_VIDEO_DIR\",SUFFIX_OUTPUT_ROOT=\"$SUFFIX_OUTPUT_ROOT\",MANIFEST_PATH=\"$MANIFEST_PATH\",MANIFEST_OFFSET=\"$START\",SUFFIX_OVERWRITE=\"$SUFFIX_OVERWRITE\",CONDA_ENV_NAME=\"$CONDA_ENV_NAME\" \"experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes_array.slurm\""
  JOB_ID=$(submit_with_retry "$CMD")
  SUFFIX_JOBS+=("$JOB_ID")
  START=$((END + 1))
done

DEP_IDS=$(IFS=:; echo "${SUFFIX_JOBS[*]}")
VBENCH_CMD="sbatch --parsable --dependency=afterok:${DEP_IDS} --export=ALL,RUN_TAG=\"$RUN_TAG\",MATERIALIZE_SUFFIXES=0,INPUT_VIDEO_DIR=\"$SUFFIX_OUTPUT_ROOT\",CONDA_ENV_NAME=\"$CONDA_ENV_NAME\" \"experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_32gpus.slurm\""
VBENCH_JOB=$(submit_with_retry "$VBENCH_CMD")

echo "[submit-vbench-pipeline] ROOT_DIR=$ROOT_DIR"
echo "[submit-vbench-pipeline] MANIFEST_PATH=$MANIFEST_PATH"
echo "[submit-vbench-pipeline] SUFFIX_JOBS=${SUFFIX_JOBS[*]}"
echo "[submit-vbench-pipeline] VBENCH_JOB=$VBENCH_JOB"
