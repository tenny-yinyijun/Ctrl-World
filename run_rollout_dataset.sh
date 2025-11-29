#!/bin/bash

# Example script to run rollout on an entire dataset
# Modify the paths below according to your setup

# Model checkpoint path
CKPT_PATH="path/to/your/checkpoint.pt"

# Dataset directory
DATASET_DIR="dataset_example/irom_1126_base2"

# Optional: Model index (if not specified, will auto-assign based on models.txt)
# MODEL_INDEX=0

# Optional: limit number of trajectories (useful for debugging)
# MAX_TRAJECTORIES=5

python scripts/rollout_replay_dataset.py \
    --ckpt_path ${CKPT_PATH} \
    --dataset_dir ${DATASET_DIR} \
    --start_idx 0 \
    --downsampled \
    # --model_index ${MODEL_INDEX} \  # Uncomment to specify model index
    # --max_trajectories ${MAX_TRAJECTORIES}  # Uncomment to limit trajectories

# Output will be saved to:
# /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/video_{model_index}/
# Videos named as: 0.mp4, 1.mp4, etc.
# Model registry at: /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/models.txt
