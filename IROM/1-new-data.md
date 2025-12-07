# Shortcuts

## 1. Converting new dataset

See [dataset.sh](bash_scripts/dataset.sh) for slurm job

```bash
# For demonstrations
python dataset_example/extract_latent_irom.py \
    --irom_data_path /n/fs/iromdata/irom_droid_data/demo/2025-12-01_clean_demo_v0-100 \
    --output_path dataset_example/demo/v0_1201_100 \
    --svd_path /n/fs/tom-project/video_models/Ctrl-World/stable-video-diffusion-img2vid

# For play data

```

```bash
# Compute meta info
python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/demo/v0_1201_100 --dataset_name v0_1201_100

```

## 2. Launch Training

```bash
# Full fine-tine
python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/demo/v0_1201_100 --dataset_name v0_1201_100

# Lora
 TODO
```

## 3. Evaluation: Inference


```bash
# Launch single eval
python eval/rollout_single_model.py \
  --model_alias base_model \
  --dataset_dir dataset_example/held-out/eval_v0_dyn \
  --registry_path model_registry.json \
  --output_base_dir results/test

# Launch batch eval
bash eval/submit_batch_jobs.sh
```

## 4. Evaluation: Metric Computation


## Utilities

### Data visualizers

```bash
# Calculate Raw Time
python util_scripts/calculate_time.py dataset_example/irom_1126_play

# Play dataset visualizer: refer to /n/fs/iromdata/irom_droid_data/conversion_utils/visualize_all_videos.py

```

### Analysis
