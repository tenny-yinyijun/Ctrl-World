#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4        
#SBATCH --mem=300G                                      ## Memory
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --job-name=cw                                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu

source ~/.bashrc

cd /scratch/gpfs/AM43/yy4041/Ctrl-World

source .venv/bin/activate
# or: conda activate ctrl-world

# export environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# first update relevant hyperparameters/variables in droid_irom_finetune.py. Then run:

accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /scratch/gpfs/AM43/yy4041/data/robot-play \
  --dataset_names v2_2025_12_17_1300 \
  --config droid_irom_finetune \
  --tag "1220-ailab"