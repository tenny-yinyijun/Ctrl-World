#!/bin/bash
#SBATCH --job-name=rollout_parallel
#SBATCH --output=logs/rollout_parallel_%A_%a.out
#SBATCH --error=logs/rollout_parallel_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=8:00:00
#SBATCH --exclude=neu[301,306,309,311]

# Create logs directory if it doesn't exist
mkdir -p logs

conda activate ctrl-world

# Required arguments
CKPT_PATH=${1:?"Error: CKPT_PATH is required. Usage: sbatch $0 CKPT_PATH DATASET_DIR [OPTIONS]"}
DATASET_DIR=${2:?"Error: DATASET_DIR is required. Usage: sbatch $0 CKPT_PATH DATASET_DIR [OPTIONS]"}

# Optional arguments (passed through)
shift 2
OPTIONAL_ARGS="$@"

# Number of GPUs (set by array size)
NUM_GPUS=8
GPU_ID=$SLURM_ARRAY_TASK_ID

echo "=================================================="
echo "Parallel rollout processing - GPU $GPU_ID/$NUM_GPUS"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Checkpoint: $CKPT_PATH"
echo "Dataset: $DATASET_DIR"
echo "Optional args: $OPTIONAL_ARGS"
echo "=================================================="

# Run the script
python3 scripts/rollout_replay_dataset_sample_parallel.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --gpu_id $GPU_ID \
    --num_gpus $NUM_GPUS \
    $OPTIONAL_ARGS

echo "=================================================="
echo "GPU $GPU_ID processing completed"
echo "=================================================="


# sbatch run_rollout_parallel_array.sh \
#       "/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt" \
#       "dataset_example/irom_1126_all2" \
#       --start_idx 0 \
#       --seed 0 \
#       --model_index 0 \
#       --num_samples 6     

# sbatch run_rollout_parallel_array.sh \
#       "/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt" \
#       "dataset_example/irom_play" \
#       --start_idx 0 \
#       --seed 0 \
#       --model_index 0 \
#       --num_samples 6           

# sbatch run_rollout_parallel_array.sh \
#       "/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt" \
#       "dataset_example/irom_1126_play" \
#       --start_idx 0 \
#       --seed 0 \
#       --model_index 0 \
#       --num_samples 6                                                    