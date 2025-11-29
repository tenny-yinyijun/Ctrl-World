#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=100G                                      ## Memory per node
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --job-name=rollout                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/tom-project/video_models/Ctrl-World

conda activate ctrl-world

# Model checkpoint path
CKPT_PATH="/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt"

# Dataset directory
DATASET_DIR="dataset_example/irom_1126_base2"
# DATASET_DIR="dataset_example/irom_1126_play"
# DATASET_DIR="dataset_example/irom_play"

# Optional: Model index (if not specified, will auto-assign based on models.txt)
# MODEL_INDEX=0

# Optional: limit number of trajectories (useful for debugging)
# MAX_TRAJECTORIES=5

python scripts/rollout_replay_dataset.py \
    --ckpt_path ${CKPT_PATH} \
    --dataset_dir ${DATASET_DIR} \
    --start_idx 0 \
    --model_index 1 \
    --gripper_annotation
    # --model_index ${MODEL_INDEX} \  # Uncomment to specify model index
    # --max_trajectories ${MAX_TRAJECTORIES}  # Uncomment to limit trajectories

# Output will be saved to:
# /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/video_{model_index}/
# Videos named as: 0.mp4, 1.mp4, etc.
# Model registry at: /n/fs/tom-project/video_models/Ctrl-World/dataset_eval/{dataset_name}/models.txt
