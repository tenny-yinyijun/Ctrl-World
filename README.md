# Ctrl-World (Adapted)

Original README: [readme.md](readme.md)

## Environment Setup

First create the environment with either uv or conda:
```bash
## Option 1 (recommended): uv

# Install 
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

### Option 2: conda
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt
```

Next install the required dependencies:

```bash
# stable-video-diffusion-img2vid
# 
```

## Data Processing

TODO add play data and combine 

```bash
# pre-compute latent
python dataset_example/extract_latent_irom.py \
    --irom_data_path /path/to/raw/data \
    --output_path /path/to/latent/dataset \
    --svd_path /path/to/stable-video-diffusion-img2vid

# pre-extract samples
python dataset_meta_info/create_meta_info.py --droid_output_path /path/to/latent/dataset
```

## Training

```bash
accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example/... --dataset_names dataset_name --config droid_irom_finetune --tag "test"
```

See [train.sh](bash_scripts/train.sh) for an example

## Inference

## Evaluation


## Utilities