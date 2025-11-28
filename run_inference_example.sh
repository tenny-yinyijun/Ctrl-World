#!/bin/bash

# Example script to run inference on the new IROM dataset
# Update the paths below according to your setup

# Path to your trained checkpoint
# CKPT_PATH="/n/fs/tom-project/video_models/Ctrl-World/checkpoints/Ctrl-World/checkpoint-10000.pt"
CKPT_PATH="/n/fs/tom-project/video_models/Ctrl-World/model_ckpt/droid_irom_textcondtrue/checkpoint-2000.pt"
# CKPT_PATH="/n/fs/tom-project/video_models/Ctrl-World/model_ckpt/droid_irom_textcondfalse/checkpoint-8000.pt"


# Paths to model components (should match training config)
SVD_MODEL_PATH="/n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="/n/fs/tom-project/video_models/Ctrl-World/clip-vit-base-patch32"

# Dataset paths are already configured in droid_irom_highres_withtextcond.py
# The config uses: /n/fs/iromdata/video_model_training/lab/droid_inference_config.py

# Run inference
python scripts/rollout_replay_traj.py \
    --ckpt_path "${CKPT_PATH}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --task_type replay \
    --task_name ckpt2000

# The results will be saved to:
# synthetic_traj/Rollouts_replay/video/
