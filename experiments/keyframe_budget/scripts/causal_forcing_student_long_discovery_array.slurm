#!/bin/bash
#SBATCH --job-name=kfb_cfs_arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/keyframe_budget/%x-%A_%a.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%A_%a.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100
#SBATCH --array=0-71%12

set -euo pipefail

module purge
module load arch/${JEANZAY_ARCH:-h100}
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc
conda activate deepforcing

export http_proxy=${http_proxy:-http://10.10.2.1:8123}
export https_proxy=${https_proxy:-http://10.10.2.1:8123}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

resolve_snapshot_dir() {
  local repo_dir="$1"
  local snapshots_dir="$repo_dir/snapshots"
  if [ ! -d "$snapshots_dir" ]; then
    return 1
  fi
  find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DEVICE="${DEVICE:-cuda:0}"
cd "${ROOT_DIR}"

mkdir -p logs/keyframe_budget checkpoints/chunkwise wan_models

export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
HF_HUB_DIR="$HF_HOME/hub"
WAN_REPO_DIR=${WAN_REPO_DIR:-$HF_HUB_DIR/models--Wan-AI--Wan2.1-T2V-1.3B}
CF_REPO_DIR=${CF_REPO_DIR:-$HF_HUB_DIR/models--zhuhz22--Causal-Forcing}
WAN_SNAPSHOT_DIR=${WAN_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$WAN_REPO_DIR" || true)}
CF_SNAPSHOT_DIR=${CF_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$CF_REPO_DIR" || true)}

if [ -n "${WAN_SNAPSHOT_DIR:-}" ] && [ ! -e "wan_models/Wan2.1-T2V-1.3B" ]; then
  ln -s "$WAN_SNAPSHOT_DIR" "wan_models/Wan2.1-T2V-1.3B"
fi

if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/causal_forcing.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/causal_forcing.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/causal_forcing.pt" "checkpoints/chunkwise/causal_forcing.pt"
fi
if [ ! -e "checkpoints/chunkwise/causal_forcing.pt" ]; then
  echo "[student-keyframe] Missing checkpoints/chunkwise/causal_forcing.pt"
  echo "[student-keyframe] Checked snapshot: ${CF_SNAPSHOT_DIR:-<none>}/chunkwise/causal_forcing.pt"
  exit 1
fi

PROMPT_SOURCE="${ROOT_DIR}/experiments/keyframe_budget/prompts/motion_rich_24.json"
PROMPT_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
TASK_SEED="${SEED_INDEX}"
PROMPT_TAG=$(printf "%03d" "${PROMPT_INDEX}")

TASK_PROMPT_FILE="${SLURM_TMPDIR:-/tmp}/kfb_causal_forcing_student_prompt_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.json"
export PROMPT_SOURCE PROMPT_INDEX TASK_PROMPT_FILE
python3 - <<'PY'
import json
import os

src = os.environ["PROMPT_SOURCE"]
idx = int(os.environ["PROMPT_INDEX"])
dst = os.environ["TASK_PROMPT_FILE"]
with open(src, "r", encoding="utf-8") as f:
    payload = json.load(f)
if not isinstance(payload, list):
    raise TypeError(f"Expected prompt list, got {type(payload).__name__}")
if idx < 0 or idx >= len(payload):
    raise IndexError(f"Prompt index {idx} out of bounds for {len(payload)} prompts.")
item = payload[idx]
if isinstance(item, str):
    prompt = {"prompt_id": f"prompt_{idx:04d}", "prompt_text": item}
elif isinstance(item, dict):
    prompt = item
else:
    raise TypeError(f"Unsupported prompt item type: {type(item).__name__}")
with open(dst, "w", encoding="utf-8") as f:
    json.dump([prompt], f, indent=2, ensure_ascii=True)
PY

BASE_CONFIG="${ROOT_DIR}/experiments/keyframe_budget/configs/causal_forcing_student_long_discovery.yaml"
TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/kfb_causal_forcing_student_${SLURM_JOB_ID:-manual}_${TASK_ID}.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_forcing_student_long_keyframe_budget}"
SINGLE_HEAVY_INDICES="${SINGLE_HEAVY_INDICES:-}"
export BASE_CONFIG TMP_CONFIG OUTPUT_ROOT TASK_PROMPT_FILE TASK_SEED PROMPT_TAG SINGLE_HEAVY_INDICES
python3 - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["BASE_CONFIG"])
cfg["experiment_name"] = f"causal_forcing_student_long_p{os.environ['PROMPT_TAG']}_s{os.environ['TASK_SEED']}"
cfg["output_root"] = os.environ["OUTPUT_ROOT"]
cfg["prompt_set"] = os.environ["TASK_PROMPT_FILE"]
cfg["num_prompts"] = 1
cfg["seeds"] = [int(os.environ["TASK_SEED"])]
if os.environ.get("SINGLE_HEAVY_INDICES"):
    cfg["single_heavy_indices"] = [
        int(x.strip()) for x in os.environ["SINGLE_HEAVY_INDICES"].split(",") if x.strip()
    ]
OmegaConf.save(cfg, os.environ["TMP_CONFIG"])
print("[student-keyframe] Effective config:")
print(OmegaConf.to_yaml(cfg))
PY

if [ "${FORCE:-0}" = "1" ]; then
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_student_experiment \
    --config_path "${TMP_CONFIG}" \
    --device "${DEVICE}" \
    --force
else
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_student_experiment \
    --config_path "${TMP_CONFIG}" \
    --device "${DEVICE}"
fi
