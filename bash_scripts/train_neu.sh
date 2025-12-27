#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:8                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=8                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=200G                                      ## Memory per node
#SBATCH --time=72:00:00                                 ## Walltime
#SBATCH --job-name=ctrl                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/iromdata/video_models/Ctrl-World

# source .venv/bin/activate
conda activate ctrl-world

# export environment variables
# export WANDB_MODE=offline

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# first update relevant hyperparameters/variables in droid_irom_finetune_neu.py. Then run:

accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /n/fs/iromdata/world_model_data/demo \
  --dataset_names v1_1222_50 \
  --config droid_irom_finetune_neu \
  --tag "1223-demo-neu"