#!/bin/bash

# Example script to run rollout on an entire dataset
# Modify the paths below according to your setup

# Model checkpoint path
CKPT_PATH="/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt"

# Dataset directory
# DATASET_DIR="dataset_example/irom_1126_base2"
DATASET_DIR="dataset_example/irom_play"

# Optional: Model index (if not specified, will auto-assign based on models.txt)
MODEL_INDEX=0

NUM_SAMPLES=6

# Optional: limit number of trajectories (useful for debugging)
# MAX_TRAJECTORIES=5

python scripts/rollout_replay_dataset_sample.py \
    --ckpt_path ${CKPT_PATH} \
    --dataset_dir ${DATASET_DIR} \
    --start_idx 0 \
    --seed 0 \
    --model_index ${MODEL_INDEX} \
    --num_samples ${NUM_SAMPLES}
# Output will be saved to:
# /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/video_{model_index}/
# Videos named as: 0.mp4, 1.mp4, etc.
# Model registry at: /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/models.txt
