#!/bin/bash

# download clip model
git clone https://huggingface.co/openai/clip-vit-base-patch32

# download stable video diffusion weights
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
sudo apt-get install git-lfs -y
git lfs install
cd stable-video-diffusion-img2vid
git lfs pull
cd ..

# download droid checkpoint from ctrl-world
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/yjguo/Ctrl-World/resolve/main/checkpoint-10000.pt
cd ..