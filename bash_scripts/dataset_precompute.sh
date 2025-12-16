#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=40                               ## CPU cores per task
#SBATCH --mem=50G                                      ## Memory per node
#SBATCH --time=24:00:00                                 ## Walltime
#SBATCH --job-name=data                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/tom-project/video_models/Ctrl-World

conda activate ctrl-world

accelerate launch --num_processes 40 dataset_example/preprocess_video.py --droid_hf_path /n/fs/iromdata/DROID/droid_1.0.1 --droid_output_path /n/fs/iromdata/DROID_processed