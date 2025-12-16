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
uv pip install torch==2.7.1+cu126 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit

### Option 2: conda
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt
```

Next install the required dependencies:

```bash
cd Ctrl-World
bash bash_scripts/setup/download_models.sh
```

## Data Processing (Optional)

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

## Training

See [train.sh](bash_scripts/train.sh) for an example:

```bash
accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /path/to/datasets \
  --dataset_names v0_1208-2200 \
  --config droid_irom_finetune \
  --tag "MMDD-test"
```

## Evaluation

### Running Batch Inference + Metric Computation
```bash
# First update assets/model_registry.json and submit_batch_jobs.sh, then run:
bash eval/submit_batch_jobs.sh

# After all generations are completed, compute the metrics:
python eval/metric/compute_metrics.py \
    --results_dir /path/to/results \
    --output_dir /path/to/outputs \
    --metrics mse ssim lpips \
    --num_workers 4
```

<!-- ### Visualization

```bash
# Visualizing average metric scores per-dataset
python visualize_average.py \
    --input /n/fs/tom-project/video_models/Ctrl-World/results/test/metrics/per_video_metrics.json \
    --view average \
    --metrics lpips \
    --model-order base_model 1201-demo-v0-ckpt45000 1211-humanplay-ckpt30000 1201-play400-v0-ckpt90000 1211-play4000-v0-ckpt95000

python visualize_average.py     --input /n/fs/tom-project/video_models/Ctrl-World/eval/metric_sanity/outputs/per_video_metrics.json     --view view2     --metrics psnr ssim lpips mse   --model-order base_model 1201-demo-v0-ckpt45000 1211-humanplay-ckpt70000 1201-play400-v0-ckpt90000 1211-play4000-v0-ckpt95000 1211-play4000-v0-ckptbig


python eval/metrics/visualize_average.py --result_dir /n/fs/worldmodeliw/ctrlworld/evaluation_inf_results

# Visualizing rank between two models of single dataset
python visualize_rank.py \
    --input /n/fs/tom-project/video_models/Ctrl-World/eval/metric_sanity/outputs/per_video_metrics.json \
    --dataset v0_dyn \
    --model1 1201-play400-v0-ckpt90000 \
    --model2 1201-demo-v0-ckpt45000  \
    --view view2 \
    --output ./plots/rank_comparison_new.png

python visualize_rank.py \
    --input /n/fs/tom-project/video_models/Ctrl-World/eval/metric_sanity/outputs/per_video_metrics.json \
    --dataset v0_dyn_2 \
    --model1 1211-play4000-v0-ckpt95000 \
    --model2 1201-demo-v0-ckpt45000  \
    --view view2 \
    --output ./plots/rank_comparison_new.png

``` -->