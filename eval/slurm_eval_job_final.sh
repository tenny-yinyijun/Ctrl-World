#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=40G                                      ## Memory per node
#SBATCH --time=5:00:00                                 ## Walltime
#SBATCH --job-name=eval_v2_inf                          ## Job Name (will be overridden by submit script)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,303,306,309,311]

source ~/.bashrc

cd /n/fs/iromdata/video_models/Ctrl-World

conda activate ctrl-world

# These variables will be set by the submit_batch_jobs_v2.sh script:
# MODEL_ALIAS - Model alias from model_registry.json
# DATASET_NAME - Name of the v2 dataset (e.g., "v2_test")
# TEST_CASE - Test case name (e.g., "deformable", "miss")
# TEST_CASE_DIR - Full path to test case directory
# REGISTRY_PATH - Path to model_registry.json
# OUTPUT_BASE_DIR - Base output directory
# START_IDX - Starting frame index
# DOWNSAMPLED - Whether data is downsampled
# GRIPPER_ANNOTATION - Whether to annotate gripper values

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "Model: ${MODEL_ALIAS}"
echo "Dataset: ${DATASET_NAME}"
echo "Test Case: ${TEST_CASE}"
echo "Test Case Dir: ${TEST_CASE_DIR}"
echo "Registry: ${REGISTRY_PATH}"
echo "Output Base Dir: ${OUTPUT_BASE_DIR}"
echo "=========================================="

# Build command with required arguments
CMD="python eval/rollout_single_model.py \
    --model_alias ${MODEL_ALIAS} \
    --dataset_dir ${TEST_CASE_DIR} \
    --registry_path ${REGISTRY_PATH} \
    --output_base_dir ${OUTPUT_BASE_DIR}/${DATASET_NAME}/${TEST_CASE} \
    --start_idx ${START_IDX}"

# Add optional flags
if [ "${DOWNSAMPLED}" = "true" ]; then
    CMD="${CMD} --downsampled"
fi

if [ "${GRIPPER_ANNOTATION}" = "true" ]; then
    CMD="${CMD} --gripper_annotation"
fi

if [ -n "${MAX_TRAJECTORIES}" ] && [ "${MAX_TRAJECTORIES}" != "None" ]; then
    CMD="${CMD} --max_trajectories ${MAX_TRAJECTORIES}"
fi

echo "Executing: ${CMD}"
echo "=========================================="

# Run the inference
eval ${CMD}

EXIT_CODE=$?

echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "Results saved to: ${OUTPUT_BASE_DIR}/${DATASET_NAME}/${TEST_CASE}/${MODEL_ALIAS}"
echo "=========================================="

exit ${EXIT_CODE}
