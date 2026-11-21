#!/usr/bin/env bash
# Quick start: solve one puzzle of the Sokoban test set (3 boxes, 39 moves) with the released CFT model, write the video
# (it ends at the model's own EOS frame) and print the result of the strict rule verifier.
#
#   bash run_example.sh                      # Sokoban test example
#   EXAMPLE=flowfree_test_00005 MODEL=checkpoints/videcide/flowfree_cft bash run_example.sh   # FlowFree test, 8x8
#
# Needs one GPU, checkpoints/Wan2.2-TI2V-5B and checkpoints/Wan2.2-TI2V-5B-Diffusers/vae (see README),
# and the released model in $MODEL. Set HF_MODEL_REPO to download a missing model from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"

EXAMPLE="${EXAMPLE:-sokoban_test_00756}"
MODEL="${MODEL:-checkpoints/videcide/sokoban_cft}"

if [[ ! -f "$MODEL/model.safetensors" ]]; then
  if [[ -z "${HF_MODEL_REPO:-}" ]]; then
    echo "No model in $MODEL. Download it first (see README, Models and Data) or set HF_MODEL_REPO." >&2
    exit 1
  fi
  name="$(basename "$MODEL")"
  huggingface-cli download "$HF_MODEL_REPO" --include "$name/*" --local-dir "$(dirname "$MODEL")"
fi

python demo.py --model "$MODEL" --example "$EXAMPLE" --out outputs/demo
