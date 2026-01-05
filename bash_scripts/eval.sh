#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=20G                                      ## Memory per node
#SBATCH --time=1:00:00                                  ## Walltime
#SBATCH --job-name=evald                              ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/iromdata/video_models/Ctrl-World

conda activate ctrl-world

# export environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/process_irom_data.py \
    --irom_data_path /n/fs/iromdata/irom_droid_data/eval/v1/sanity \
    --output_path /n/fs/iromdata/world_model_test/v1/0104_sanity \
    --dataset_type demo \
    --test

# python scripts/process_irom_data.py \
#     --irom_data_path /n/fs/iromdata/irom_droid_data/play_data/v0_2025_12_28_1-2200 \
#     --output_path /n/fs/iromdata/world_model_data/play/auto/v0_2025_12_28_2000 \
#     --dataset_type play

    