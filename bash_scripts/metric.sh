#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=50G                                      ## Memory per node
#SBATCH --time=5:00:00                                  ## Walltime
#SBATCH --job-name=metric                              ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/iromdata/video_models/Ctrl-World

conda activate ctrl-world

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python metric/compute_experiment_metrics.py --results_dir /n/fs/iromdata/video_models/Ctrl-World/1227_eval/v2_test_combined

