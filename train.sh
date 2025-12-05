#!/bin/bash
# Wandb configuration
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="bfdd0d769aba148308af46bcd279a787071a7790"
export CUDA_VISIBLE_DEVICES=6

# MineDojo headless mode (required for training)
export MINEDOJO_HEADLESS=1

# # Run training
# python train.py \
#     --algorithm ppo \

# echo ""
# echo "Training completed!"


ROOT_SAVE_DIR="/home/jeehye/RL-GPT/checkpoint"

python train.py \
    --algorithm ppo \
    --save-path "${ROOT_SAVE_DIR}" \
    --exp-name ppo-with-noise \
    --noise-start-std 0.0 \
    --noise-end-std 0.0 \
    --noise-decay-epochs 0