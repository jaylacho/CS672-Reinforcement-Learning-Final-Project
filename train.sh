#!/bin/bash
# Wandb configuration
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="bfdd0d769aba148308af46bcd279a787071a7790"
export CUDA_VISIBLE_DEVICES=1

# MineDojo headless mode (required for training)
export MINEDOJO_HEADLESS=1

# Run training
python train.py \
    --algorithm dpo \

echo ""
echo "Training completed!"
