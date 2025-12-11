#!/bin/bash

# Batch SLURM job submission script for evaluation inference
# Submits one job per model-dataset pair

# Usage:
#   bash eval/submit_batch_jobs.sh

# ============================================
# Configuration
# ============================================

# List of model aliases from model_registry.json
MODEL_ALIASES=(
    "base_model"
    "1201-demo-v0-ckpt45000"
    "1211-humanplay-ckpt32500"
    "1201-play400-v0-ckpt90000"
    "1211-play4000-v0-ckpt95000"
)

# List of dataset names (under dataset_example/)
DATASET_NAMES=(
    # "eval_random"
    "eval_v0_clean_success"
    "eval_v0_dyn"
    # "droid_validation"
    # Add more datasets here
)

# Path to model registry
REGISTRY_PATH="/n/fs/tom-project/video_models/Ctrl-World/evaluation_inf_results/model_registry.json"

# Output base directory
OUTPUT_BASE_DIR="evaluation_inf_results"

################# DON"T CHANGE #################

# Starting frame index
START_IDX=0

# Whether data is already downsampled to 5Hz (set to "true" or "false")
DOWNSAMPLED="false"

# Whether to annotate gripper values on frames (set to "true" or "false")
GRIPPER_ANNOTATION="false"

# Optional: limit number of trajectories (set to empty string or number)
MAX_TRAJECTORIES=""  # e.g., "5" for debugging, "" for all

# Skip if output directory already exists (set to "true" or "false")
SKIP_EXISTING="true"

# ============================================
# Job Submission
# ============================================

echo "=========================================="
echo "Batch SLURM Job Submission"
echo "=========================================="
echo "Models: ${MODEL_ALIASES[@]}"
echo "Datasets: ${DATASET_NAMES[@]}"
echo "Total jobs: $((${#MODEL_ALIASES[@]} * ${#DATASET_NAMES[@]}))"
echo "=========================================="

# Create slurm_outputs directory structure
mkdir -p slurm_outputs

SUBMITTED_COUNT=0
SKIPPED_COUNT=0

# Loop through all model-dataset pairs
for MODEL_ALIAS in "${MODEL_ALIASES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        # Create job name
        JOB_NAME="eval_${MODEL_ALIAS}_${DATASET_NAME}"

        # Check if output already exists
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATASET_NAME}/${MODEL_ALIAS}"
        if [ "${SKIP_EXISTING}" = "true" ] && [ -d "${OUTPUT_DIR}" ]; then
            echo "SKIP: ${JOB_NAME} (output exists: ${OUTPUT_DIR})"
            ((SKIPPED_COUNT++))
            continue
        fi

        # Create job-specific output directory
        mkdir -p "slurm_outputs/${JOB_NAME}"

        # Submit job with environment variables
        echo "SUBMIT: ${JOB_NAME}"
        sbatch \
            --job-name="${JOB_NAME}" \
            --export=ALL,MODEL_ALIAS="${MODEL_ALIAS}",DATASET_NAME="${DATASET_NAME}",REGISTRY_PATH="${REGISTRY_PATH}",OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}",START_IDX="${START_IDX}",DOWNSAMPLED="${DOWNSAMPLED}",GRIPPER_ANNOTATION="${GRIPPER_ANNOTATION}",MAX_TRAJECTORIES="${MAX_TRAJECTORIES}" \
            eval/slurm_eval_job.sh

        ((SUBMITTED_COUNT++))

        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo "Jobs submitted: ${SUBMITTED_COUNT}"
echo "Jobs skipped: ${SKIPPED_COUNT}"
echo "Total: $((SUBMITTED_COUNT + SKIPPED_COUNT))"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check outputs in: slurm_outputs/"
echo "Results will be in: ${OUTPUT_BASE_DIR}/"
