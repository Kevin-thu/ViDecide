#!/usr/bin/env bash
# Launch Wan2.2-TI2V-5B training with a config file.
#
#   bash scripts/train.sh sft       CONFIG GPUS LOGDIR [key=value ...]   # supervised fine-tuning (configs/sft)
#   bash scripts/train.sh posttrain CONFIG GPUS LOGDIR [key=value ...]   # CFT and its baselines (configs/cft)
#
# Examples:
#   bash scripts/train.sh sft configs/sft/sokoban_k1.yaml 0,1 outputs/sft/sokoban_k1
#   bash scripts/train.sh sft configs/sft/sokoban_k1_stage2.yaml 0,1 outputs/sft/sokoban_k1_stage2 \
#       resume_from=outputs/sft/sokoban_k1/checkpoint_model_023750/model.pt
#   bash scripts/train.sh posttrain configs/cft/sokoban.yaml 0,1,2 outputs/cft/sokoban \
#       metadata_path=outputs/cft_data/sokoban/metadata_cft.json \
#       sft_checkpoint=outputs/sft/sokoban_k1_stage2/checkpoint_model_047500/model.pt
#
# The number of GPUs times batch_size must equal total_batch_size of the config (the paper's batch sizes).
# Set WANDB=1 to log to Weights & Biases (configure wandb_* in the config); the default is off.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 4 ]]; then
  sed -n '2,17p' "$0" >&2; exit 2
fi
stage="$1"; config="$2"; gpus="$3"; logdir="$4"; shift 4
case "$stage" in
  sft) entry=scripts/sft/train_sft.py ;;
  posttrain) entry=scripts/posttrain/train_posttrain.py ;;
  *) echo "unknown stage '$stage' (sft or posttrain)" >&2; exit 2 ;;
esac
world=$(echo "$gpus" | tr ',' '\n' | grep -c .)

python - "$config" "$world" "$@" <<'PY'
import sys
sys.path.insert(0, "src")
from omegaconf import OmegaConf
from videcide.config import load_config

config, world = load_config(sys.argv[1]), int(sys.argv[2])
overrides = [arg for arg in sys.argv[3:] if "=" in arg and not arg.startswith("-")]
if overrides:
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
batch, total = int(config.get("batch_size", 1)), int(config.total_batch_size)
if world * batch != total:
    sys.exit(f"{world} GPU(s) x batch_size {batch} != total_batch_size {total} in {sys.argv[1]}: use {total // batch} GPU(s)")
PY

wandb_flag=(--disable-wandb)
[[ "${WANDB:-0}" == 1 ]] && wandb_flag=()
mkdir -p "$logdir"
export CUDA_VISIBLE_DEVICES="$gpus"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
set +e
torchrun --standalone --nproc_per_node="$world" "$entry" --config-path "$config" --logdir "$logdir" \
  "${wandb_flag[@]}" "$@" 2>&1 | tee -a "$logdir/train.log"
exit "${PIPESTATUS[0]}"
