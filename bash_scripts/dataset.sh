#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=100G                                      ## Memory per node
#SBATCH --time=4:00:00                                 ## Walltime
#SBATCH --job-name=rollout                   ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu[301,306,309,311]

source ~/.bashrc

cd /n/fs/tom-project/video_models/Ctrl-World

conda activate ctrl-world

# demonstration format
# python dataset_example/extract_latent_irom.py \
#     --irom_data_path /n/fs/iromdata/irom_droid_data/demo/2025-12-01_clean_demo_v0-100 \
#     --output_path dataset_example/demo/v0_1201_100 \
#     --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid

# python dataset_example/extract_latent_irom.py \
#     --irom_data_path /n/fs/iromdata/irom_droid_data/held-out/v0_dyn \
#     --output_path dataset_example/eval_v0_dyn \
#     --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid \
#     --skip_start_frames 20

# play data format
python dataset_example/extract_latent_irom_play.py \
    --irom_data_path /n/fs/iromdata/irom_droid_data/play_data/v2_2025_12_12_1600 \
    --output_path /n/fs/iromdata/world_model_data/play/auto/v2_2025_12_12_1600 \
    --svd_path stable-video-diffusion-img2vid