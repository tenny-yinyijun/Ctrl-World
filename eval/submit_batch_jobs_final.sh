#!/bin/bash

# Batch SLURM job submission script for v2 dataset evaluation
# Runs multiple models on a single v2 dataset with multiple test cases
#
# V2 dataset structure:
#   dataset_root/
#     manifest.json           # Contains test case metadata
#     test_case_1/
#       annotation/
#       metainfo/
#       videos/
#     test_case_2/
#       ...
#
# Usage:
#   bash eval/submit_batch_jobs_ginal.sh

# ============================================
# Configuration
# ============================================

# List of model aliases from model_registry.json
MODEL_ALIASES=(
    # "v2_demo_ailab"
    # "v2_play_robot_ailab"
    # "v2_play_robot_ailab_2"
    # "v2_play_robot_curr_ailab"
    # "v2_play_robot_curr_ailab_2"
    "v1_play_robot_ailab"
    "v1_play_robot_ailab_2"
)

# Path to v2 dataset (contains manifest.json and test case subdirectories)
DATASET_ROOT="/n/fs/iromdata/projects/v1_test_combined"

# Path to model registry
REGISTRY_PATH="/n/fs/iromdata/video_models/Ctrl-World/assets/model_registry.json"

# Output base directory
OUTPUT_BASE_DIR="/n/fs/iromdata/video_models/Ctrl-World/1228_eval"

################# DON'T CHANGE #################

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
# Validate dataset structure
# ============================================

if [ ! -f "${DATASET_ROOT}/manifest.json" ]; then
    echo "ERROR: ${DATASET_ROOT}/manifest.json not found"
    echo "This script is for datasets with manifest.json"
    echo "For old-style datasets, use submit_batch_jobs.sh instead"
    exit 1
fi

# Extract dataset name from path
DATASET_NAME=$(basename "${DATASET_ROOT}")

# Read test cases from manifest.json
# This uses jq to parse the JSON - install with: module load jq or sudo yum install jq
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required to parse manifest.json"
    echo "Install with: module load jq"
    exit 1
fi

# Get list of test cases from manifest
TEST_CASES=($(jq -r '.test_cases | keys[]' "${DATASET_ROOT}/manifest.json"))

if [ ${#TEST_CASES[@]} -eq 0 ]; then
    echo "ERROR: No test cases found in manifest.json"
    exit 1
fi

# ============================================
# Job Submission
# ============================================

echo "=========================================="
echo "Batch SLURM Job Submission"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Test Cases: ${TEST_CASES[@]}"
echo "Models: ${MODEL_ALIASES[@]}"
echo "Total jobs: $((${#MODEL_ALIASES[@]} * ${#TEST_CASES[@]}))"
echo "=========================================="

# Create slurm_outputs directory structure
mkdir -p slurm_outputs

SUBMITTED_COUNT=0
SKIPPED_COUNT=0

# Loop through all model-testcase pairs
for MODEL_ALIAS in "${MODEL_ALIASES[@]}"; do
    for TEST_CASE in "${TEST_CASES[@]}"; do
        # Create job name
        JOB_NAME="evalf_${DATASET_NAME}_${TEST_CASE}_${MODEL_ALIAS}"

        # Check if test case directory exists
        TEST_CASE_DIR="${DATASET_ROOT}/${TEST_CASE}"
        if [ ! -d "${TEST_CASE_DIR}" ]; then
            echo "WARNING: Test case directory not found: ${TEST_CASE_DIR}"
            echo "SKIP: ${JOB_NAME}"
            ((SKIPPED_COUNT++))
            continue
        fi

        # Check if output already exists
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATASET_NAME}/${TEST_CASE}/${MODEL_ALIAS}"
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
            --export=ALL,MODEL_ALIAS="${MODEL_ALIAS}",DATASET_NAME="${DATASET_NAME}",TEST_CASE="${TEST_CASE}",TEST_CASE_DIR="${TEST_CASE_DIR}",REGISTRY_PATH="${REGISTRY_PATH}",OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}",START_IDX="${START_IDX}",DOWNSAMPLED="${DOWNSAMPLED}",GRIPPER_ANNOTATION="${GRIPPER_ANNOTATION}",MAX_TRAJECTORIES="${MAX_TRAJECTORIES}" \
            eval/slurm_eval_job_final.sh

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
echo "Results will be in: ${OUTPUT_BASE_DIR}/${DATASET_NAME}/"
echo ""
