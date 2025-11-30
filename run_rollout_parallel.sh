#!/bin/bash
#SBATCH --job-name=rollout_parallel
#SBATCH --output=logs/rollout_parallel_%j.out
#SBATCH --error=logs/rollout_parallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Required arguments
CKPT_PATH=${1:?"Error: CKPT_PATH is required. Usage: sbatch $0 CKPT_PATH DATASET_DIR [OPTIONS]"}
DATASET_DIR=${2:?"Error: DATASET_DIR is required. Usage: sbatch $0 CKPT_PATH DATASET_DIR [OPTIONS]"}

# Optional arguments (passed through)
shift 2
OPTIONAL_ARGS="$@"

# Number of GPUs
NUM_GPUS=8

echo "=================================================="
echo "Starting parallel rollout processing"
echo "=================================================="
echo "Checkpoint: $CKPT_PATH"
echo "Dataset: $DATASET_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Optional args: $OPTIONAL_ARGS"
echo "=================================================="

# Launch parallel processes
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching GPU $GPU_ID..."

    srun --ntasks=1 --exclusive --gpus=1 \
        python3 scripts/rollout_replay_dataset_sample_parallel.py \
        --ckpt_path "$CKPT_PATH" \
        --dataset_dir "$DATASET_DIR" \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        $OPTIONAL_ARGS &
done

# Wait for all background jobs to complete
wait

echo "=================================================="
echo "All parallel processes completed"
echo "=================================================="
