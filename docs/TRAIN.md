# Training Instructions

## Data Processing

If the provided dataset is already processed, ignore this step. 
Otherwise if the data is in raw demo/play data format:

```bash
# Convert (example: play)
python scripts/process_irom_data.py \
    --irom_data_path /path/to/raw/data \
    --output_path /path/to/latent/dataset \
    --dataset_type play \
    --distributed
```

Example slurm script: [bash_scripts/examples/dataset.sh](../bash_scripts/examples/dataset.sh)

## Launch Training

```bash
# setup wandb
echo 'export WANDB_API_KEY=your-wandb-api-key' >> ~/.bashrc && source ~/.bashrc

# install ffmpeg (if needed)
sudo apt update
sudo apt install ffmpeg
```

Then run the actual training command:
```bash
accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /path/to/datasets \
  --dataset_names v0_1208-2200 \
  --config droid_irom_finetune \
  --tag "MMDD-test"
```
See [bash_scripts/examples/train.sh](../bash_scripts/examples/train.sh) for an example. To view metrics (for wandb offline mode):

```bash
wandb sync /path/to/wandb/run
```

## Ablations

Curriculum:

Training on subset of data: