#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G                                      ## Memory
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --job-name=cw-curriculum                        ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu

source ~/.bashrc

cd /scratch/gpfs/AM43/yy4041/Ctrl-World

source .venv/bin/activate
# or: conda activate ctrl-world

# export environment variables
export WANDB_MODE=offline

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Curriculum Learning Training
# Uses droid_irom_finetune_curriculum.py config (curriculum learning enabled)
# Default settings:
#   - Linear schedule from easy to uniform sampling
#   - Initial distribution: [0.6, 0.3, 0.1, 0.0, 0.0] (60% easiest)
#   - Final distribution: [0.2, 0.2, 0.2, 0.2, 0.2] (uniform)
#   - Active curriculum: steps 1k-80k (79k steps)
#   - Warmup: 1000 steps (uniform sampling at start)
#   - Stabilization: 20,000 steps (uniform sampling at end for model stabilization)
#   - Update interval: 2000 steps
#
# Timeline: [Warmup 1k] → [Curriculum 1k-80k] → [Stabilization 80k-100k]
#
# To customize: Edit droid_irom_finetune_curriculum.py (see example strategies)
# To disable curriculum: Use regular train.sh with droid_irom_finetune config

accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /scratch/gpfs/AM43/yy4041/data/robot-play \
  --dataset_names v2_2025_12_17_1300 \
  --config droid_irom_finetune_curriculum \
  --tag "curriculum-linear-100k"
