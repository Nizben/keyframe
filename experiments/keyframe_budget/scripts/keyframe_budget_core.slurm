#!/bin/bash
#SBATCH --job-name=kfb_v35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/keyframe_budget/%x-%j.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%j.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100

set -euo pipefail

module purge
module load arch/h100
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

  local latest
  latest=$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  if [ -n "$latest" ]; then
    printf "%s\n" "$latest"
    return 0
  fi

  return 1
}

is_long_video_config() {
  local cfg_path="$1"
  python - <<'PY' "$cfg_path"
import os
import sys
from omegaconf import OmegaConf

cfg_path = sys.argv[1]
try:
    cfg = OmegaConf.load(cfg_path)
except Exception:
    print("0")
    raise SystemExit(0)

backend = str(cfg.get("backend", "") or "").lower()
model_config_path = str(cfg.get("model_config_path", "") or "").lower()
checkpoint = str(cfg.get("checkpoint", "") or "").lower()
if (
    backend == "long_video"
    or "long_video" in model_config_path
    or checkpoint.endswith("/longvideo.pt")
    or checkpoint.endswith("longvideo.pt")
):
    print("1")
else:
    print("0")
PY
}

ROOT_DIR=${ROOT_DIR:-$WORK/Causal-Forcing}
CONFIG_PATH=${EXP_CONFIG:-experiments/keyframe_budget/configs/smoke_test.yaml}
TMP_CONFIG=${SLURM_TMPDIR:-/tmp}/kfb_${SLURM_JOB_ID:-manual}.yaml
DEVICE=${DEVICE:-cuda:0}

cd "$ROOT_DIR"

mkdir -p logs/keyframe_budget
mkdir -p outputs/keyframe_budget
mkdir -p checkpoints/chunkwise
mkdir -p wan_models

export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
HF_HUB_DIR="$HF_HOME/hub"
WAN_REPO_DIR=${WAN_REPO_DIR:-$HF_HUB_DIR/models--Wan-AI--Wan2.1-T2V-1.3B}
CF_REPO_DIR=${CF_REPO_DIR:-$HF_HUB_DIR/models--zhuhz22--Causal-Forcing}

WAN_SNAPSHOT_DIR=${WAN_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$WAN_REPO_DIR" || true)}
CF_SNAPSHOT_DIR=${CF_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$CF_REPO_DIR" || true)}

if [ -n "${WAN_SNAPSHOT_DIR:-}" ] && [ ! -e "wan_models/Wan2.1-T2V-1.3B" ]; then
  ln -s "$WAN_SNAPSHOT_DIR" "wan_models/Wan2.1-T2V-1.3B"
fi

if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/ar_diffusion.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" "checkpoints/chunkwise/ar_diffusion.pt"
fi
if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/longvideo.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/longvideo.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/longvideo.pt" "checkpoints/chunkwise/longvideo.pt"
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
LONG_VIDEO_FLAG="$(is_long_video_config "$CONFIG_PATH")"
if [ -z "$CHECKPOINT_PATH" ] && [ "$LONG_VIDEO_FLAG" = "1" ] && [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/longvideo.pt" ]; then
  CHECKPOINT_PATH="${CF_SNAPSHOT_DIR}/chunkwise/longvideo.pt"
fi
if [ -z "$CHECKPOINT_PATH" ] && [ "$LONG_VIDEO_FLAG" != "1" ] && [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  CHECKPOINT_PATH="${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt"
fi

export ROOT_DIR CONFIG_PATH TMP_CONFIG CHECKPOINT_PATH
python - <<'PY'
import os
from omegaconf import OmegaConf

def parse_bool(value: str):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{value}'.")

cfg = OmegaConf.load(os.environ["CONFIG_PATH"])
root_dir = os.environ["ROOT_DIR"]

if os.environ.get("CHECKPOINT_PATH"):
    cfg["checkpoint"] = os.environ["CHECKPOINT_PATH"]
if os.environ.get("EXPERIMENT_NAME"):
    cfg["experiment_name"] = os.environ["EXPERIMENT_NAME"]
if os.environ.get("OUTPUT_ROOT"):
    cfg["output_root"] = os.environ["OUTPUT_ROOT"]
if os.environ.get("PROMPT_SET"):
    cfg["prompt_set"] = os.environ["PROMPT_SET"]
if os.environ.get("MODEL_CONFIG_PATH"):
    cfg["model_config_path"] = os.environ["MODEL_CONFIG_PATH"]
if os.environ.get("ORACLE_SOURCE"):
    cfg["oracle_source"] = os.environ["ORACLE_SOURCE"]
if os.environ.get("DTYPE"):
    cfg["dtype"] = os.environ["DTYPE"]
if os.environ.get("SINGLE_HEAVY_INDICES"):
    cfg["single_heavy_indices"] = [
        int(x.strip()) for x in os.environ["SINGLE_HEAVY_INDICES"].split(",") if x.strip()
    ]
if os.environ.get("EXPECTED_CHECKPOINT_BASENAME"):
    cfg["expected_checkpoint_basename"] = os.environ["EXPECTED_CHECKPOINT_BASENAME"]
if os.environ.get("FORBIDDEN_BACKEND"):
    cfg["forbidden_backend"] = os.environ["FORBIDDEN_BACKEND"]

for env_key, cfg_key in {
    "NUM_PROMPTS": "num_prompts",
    "FAST_STEPS": "fast_steps",
    "HEAVY_STEPS": "heavy_steps",
    "TARGET_AVG_STEPS": "target_avg_steps",
    "NUM_OUTPUT_FRAMES": "num_output_frames",
}.items():
    if os.getenv(env_key):
        cfg[cfg_key] = int(os.getenv(env_key))

if os.getenv("SEEDS"):
    cfg["seeds"] = [int(x.strip()) for x in os.getenv("SEEDS").split(",") if x.strip()]

if os.getenv("RUN_EXHAUSTIVE") is not None:
    cfg["run_exhaustive_single_heavy"] = parse_bool(os.getenv("RUN_EXHAUSTIVE"))
if os.getenv("RUN_POLICY") is not None:
    cfg["run_policy_comparison"] = parse_bool(os.getenv("RUN_POLICY"))
if os.getenv("USE_EMA") is not None:
    cfg["use_ema"] = parse_bool(os.getenv("USE_EMA"))
if os.getenv("DEBUG_CACHE_LOGS") is not None:
    cfg["debug_cache_logs"] = parse_bool(os.getenv("DEBUG_CACHE_LOGS"))
if os.getenv("SAVE_CHUNK_BOUNDARIES") is not None:
    cfg["save_chunk_boundaries"] = parse_bool(os.getenv("SAVE_CHUNK_BOUNDARIES"))

for key in ("checkpoint", "model_config_path", "prompt_set", "output_root"):
    value = cfg.get(key, "")
    if not value:
        raise ValueError(f"Missing required config key: {key}")
    if not os.path.isabs(value):
        cfg[key] = os.path.join(root_dir, value)

for key in ("checkpoint", "model_config_path", "prompt_set"):
    if not os.path.exists(cfg[key]):
        raise FileNotFoundError(f"{key} does not exist: {cfg[key]}")

OmegaConf.save(cfg, os.environ["TMP_CONFIG"])
print("[keyframe-budget] Effective config:")
print(OmegaConf.to_yaml(cfg))
PY

if [ "${FORCE:-0}" = "1" ]; then
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE" \
    --force
else
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE"
fi
