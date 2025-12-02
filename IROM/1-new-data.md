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
```

## 3. Evaluation

Currently support 4 sets of evaluation: clean demonstration, noisy demonstration, policy roll-out (good), policy roll-out (bad). You can find these 

```bash

```