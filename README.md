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

Next install the required dependencies (TODO):

```bash
# https://huggingface.co/openai/clip-vit-base-patch32
# https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
# https://huggingface.co/yjguo/Ctrl-World
```

## Data Processing

Combine (TODO)

```bash
# original for ctrl-world
accelerate launch dataset_example/extract_latent.py --droid_hf_path /n/fs/iromdata/DROID/droid_1.0.1 --droid_output_path /n/fs/iromdata/droid_processed --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid

# pre-compute latent
python dataset_example/extract_latent_irom.py \
    --irom_data_path /path/to/raw/data \
    --output_path /path/to/latent/dataset \
    --svd_path /path/to/stable-video-diffusion-img2vid

# pre-extract samples
python dataset_meta_info/create_meta_info.py --droid_output_path /path/to/latent/dataset

### test
python dataset_example/extract_latent_irom.py \
    --irom_data_path /n/fs/iromdata/irom_droid_data/sanity/v0_dyn \
    --output_path /n/fs/iromdata/world_model_data/sanity/v0_dyn \
    --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid

python dataset_example/extract_latent_irom.py \
    --irom_data_path /n/fs/iromdata/irom_droid_data/sanity/v2_dyn \
    --output_path /n/fs/iromdata/world_model_data/sanity/v2_dyn \
    --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid

```

## Training

```bash
accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example/... --dataset_names dataset_name --config droid_irom_finetune --tag "test"
```

See [train.sh](bash_scripts/train.sh) for an example


## Evaluation

### Running Batch Inference + Metric Computation
```bash
# First update model_registry.json and submit_batch_jobs.sh, then run:
bash eval/submit_batch_jobs.sh

# After all generations are completed, compute the metrics:
python eval/metric/compute_metrics.py \
    --results_dir /n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results \
    --output_dir /n/fs/tom-project/video_models/Ctrl-World/eval/metric_sanity/outputs \
    --metrics mse ssim lpips \
    --num_workers 4

# Aggregate the metrics (TODO combine)
python metric/aggregate_metrics.py \
    --input /n/fs/worldmodeliw/ctrlworld/evaluation_inf_results/metrics/per_video_metrics.json
```

### Visualization

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

```