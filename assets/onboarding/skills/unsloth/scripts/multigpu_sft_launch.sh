#!/usr/bin/env bash
# 🦥 Unsloth Multi-GPU Distributed SFT Launcher Script using `uv`
#
# Usage:
#   chmod +x multigpu_sft_launch.sh
#   ./multigpu_sft_launch.sh 2 "unsloth/Qwen2.5-7B-Instruct" "outputs/multigpu_sft"

NUM_GPUS=${1:-2}
MODEL_NAME=${2:-"unsloth/Qwen2.5-7B-Instruct"}
OUTPUT_DIR=${3:-"outputs/multigpu_sft"}

echo "🦥 Launching Unsloth SFT across $NUM_GPUS GPUs using torchrun & uv..."

uv run torchrun \
    --nproc_per_node="$NUM_GPUS" \
    scripts/finetune_sft.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2 \
    --grad_accum 4 \
    --max_steps 100

echo "✅ Distributed training finished!"
