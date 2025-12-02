#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:8                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=8                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=200G                                      ## Memory per node
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --job-name=ctrl                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/tom-project/video_models/Ctrl-World

conda activate ctrl-world

# export environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# play data
# irom_play+irom_1126_play

# demonstration
# v0_1201_100

# accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example/demo --dataset_meta_info_path dataset_meta_info --dataset_names v0_1201_100 --config droid_irom_finetune --tag "1201"


accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names irom_1130_play_v0 --config droid_irom_finetune --tag "1201"
