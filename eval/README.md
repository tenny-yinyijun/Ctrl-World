# Evaluation Inference Pipeline

This directory contains scripts for running batch inference with multiple models on multiple datasets, with separate view video outputs for easy metric computation.

## Directory Structure

```
evaluation_inf_results/
├── {dataset_name}/               # e.g., eval_random, irom_1126_base2
│   ├── groundtruth/              # Shared groundtruth (generated once per dataset)
│   │   ├── {trajectory_id}/      # e.g., 0, 1, 2...
│   │   │   ├── view0.mp4
│   │   │   ├── view1.mp4
│   │   │   └── view2.mp4
│   ├── {model_alias}/            # e.g., base_model, lora_kitchen_v1
│   │   ├── {trajectory_id}/      # e.g., 0, 1, 2...
│   │   │   ├── view0.mp4
│   │   │   ├── view1.mp4
│   │   │   └── view2.mp4
│   │   └── run_metadata.json     # Run info (timestamp, args, stats)
```

## Setup

### 1. Create Model Registry

First, create a `model_registry.json` file at the project root with your models:

```json
{
  "base_model": {
    "alias": "base_model",
    "mode": "full",
    "checkpoint_path": "/path/to/checkpoint.pt",
    "description": "Base model trained on DROID dataset"
  },
  "lora_kitchen_v1": {
    "alias": "lora_kitchen_v1",
    "mode": "lora",
    "checkpoint_path": "/path/to/lora_adapters/",
    "base_checkpoint_path": "/path/to/base.pt",
    "lora_target_modules": ["to_k", "to_v", "to_q", "to_out.0"],
    "description": "LoRA fine-tuned on kitchen tasks"
  }
}
```

## Usage

### Single Model Inference

Run inference for a single model on a single dataset:

```bash
python eval/rollout_single_model.py \
  --model_alias base_model \
  --dataset_dir dataset_example/eval_random \
  --registry_path model_registry.json \
  --output_base_dir evaluation_inf_results \
  --downsampled
```

**Arguments:**
- `--model_alias`: Model alias from model_registry.json (required)
- `--dataset_dir`: Path to dataset directory (required)
- `--registry_path`: Path to model_registry.json (default: `model_registry.json`)
- `--output_base_dir`: Base directory for results (default: `evaluation_inf_results`)
- `--start_idx`: Starting frame index (default: 0)
- `--max_trajectories`: Limit number of trajectories for debugging
- `--gripper_annotation`: Annotate gripper values on frames
- `--downsampled`: Input is already downsampled to 5Hz

### Batch Inference

Run inference for multiple models on multiple datasets:

```bash
python eval/batch_inference.py \
  --model_aliases base_model lora_kitchen_v1 lora_wipe_v2 \
  --dataset_dirs dataset_example/eval_random dataset_example/eval_kitchen \
  --registry_path model_registry.json \
  --output_base_dir evaluation_inf_results \
  --downsampled
```

**Arguments:**
- `--model_aliases`: List of model aliases (required)
- `--dataset_dirs`: List of dataset directories (required)
- `--registry_path`: Path to model_registry.json (default: `model_registry.json`)
- `--output_base_dir`: Base directory for results (default: `evaluation_inf_results`)
- `--start_idx`: Starting frame index (default: 0)
- `--max_trajectories`: Limit trajectories per dataset
- `--gripper_annotation`: Annotate gripper values
- `--downsampled`: Input is downsampled to 5Hz
- `--no_skip_existing`: Re-run even if output exists

**Note:** By default, batch inference skips model-dataset combinations that already have output directories. Use `--no_skip_existing` to override.

### SLURM Batch Submission

For running on a SLURM cluster, use the batch job submission script to submit one job per model-dataset pair:

**Step 1:** Edit `eval/submit_batch_jobs.sh` to configure your models and datasets:

```bash
# List of model aliases from model_registry.json
MODEL_ALIASES=(
    "base_model"
    "lora_model_v1"
    "lora_model_v2"
)

# List of dataset names (under dataset_example/)
DATASET_NAMES=(
    "eval_random"
    "eval_kitchen"
)

# Additional configuration
DOWNSAMPLED="true"
GRIPPER_ANNOTATION="false"
MAX_TRAJECTORIES=""  # Empty for all, or set a number
SKIP_EXISTING="true"  # Skip if output exists
```

**Step 2:** Submit all jobs:

```bash
bash eval/submit_batch_jobs.sh
```

This will submit one SLURM job for each model-dataset pair. Jobs are named `eval_{model}_{dataset}` for easy identification.

**Monitor jobs:**
```bash
# View all your jobs
squeue -u $USER

# View specific evaluation jobs
squeue -u $USER --name=eval_*

# Check job output logs
tail -f slurm_outputs/eval_base_model_eval_random/out_log_*.out
```

**SLURM configuration:**
- 1 GPU per job
- 8 CPU cores
- 100GB memory
- 48 hour time limit
- Logs saved to `slurm_outputs/{job_name}/`

You can modify these settings in `eval/slurm_eval_job.sh`.

## Examples

### Example 1: Test single model

```bash
# Test inference on a small number of trajectories
python eval/rollout_single_model.py \
  --model_alias base_model \
  --dataset_dir dataset_example/eval_random \
  --max_trajectories 5 \
  --downsampled
```

### Example 2: Full evaluation of all models

```bash
# Evaluate all models in registry on all test datasets
python eval/batch_inference.py \
  --model_aliases base_model lora_v1 lora_v2 lora_v3 \
  --dataset_dirs dataset_example/eval_random dataset_example/eval_kitchen \
  --downsampled
```

### Example 3: Debug a specific model

```bash
# Run with gripper annotation for debugging
python eval/rollout_single_model.py \
  --model_alias lora_kitchen_v1 \
  --dataset_dir dataset_example/eval_kitchen \
  --gripper_annotation \
  --max_trajectories 3 \
  --downsampled
```

## Output Files

After running inference, you'll have:

1. **Prediction videos**: `evaluation_inf_results/{dataset_name}/{model_alias}/{traj_id}/view{0,1,2}.mp4`
2. **Groundtruth videos**: `evaluation_inf_results/{dataset_name}/groundtruth/{traj_id}/view{0,1,2}.mp4`
3. **Run metadata**: `evaluation_inf_results/{dataset_name}/{model_alias}/run_metadata.json`

The metadata file contains:
- Model configuration
- Processing statistics (success/failure counts)
- Arguments used for the run
- Timestamp

## Computing Metrics

With videos separated by view, you can easily compute per-view metrics:

```python
import cv2
import numpy as np

def compute_mse(pred_path, gt_path):
    """Compute MSE between prediction and groundtruth videos."""
    pred = cv2.VideoCapture(pred_path)
    gt = cv2.VideoCapture(gt_path)

    mse_values = []
    while True:
        ret_pred, frame_pred = pred.read()
        ret_gt, frame_gt = gt.read()

        if not ret_pred or not ret_gt:
            break

        mse = np.mean((frame_pred.astype(float) - frame_gt.astype(float)) ** 2)
        mse_values.append(mse)

    return np.mean(mse_values)

# Compute per-view MSE
for view_idx in range(3):
    pred_path = f"evaluation_inf_results/eval_random/base_model/0/view{view_idx}.mp4"
    gt_path = f"evaluation_inf_results/eval_random/groundtruth/0/view{view_idx}.mp4"
    mse = compute_mse(pred_path, gt_path)
    print(f"View {view_idx} MSE: {mse:.2f}")
```

## Model Registry Format

The `model_registry.json` file should be at the project root and contain:

### For full fine-tuned models:
```json
{
  "model_alias": {
    "alias": "model_alias",
    "mode": "full",
    "checkpoint_path": "/absolute/path/to/checkpoint.pt",
    "description": "Optional description",
    "svd_model_path": "/path/to/svd/model",  # Optional
    "clip_model_path": "/path/to/clip/model"  # Optional
  }
}
```

### For LoRA models:
```json
{
  "model_alias": {
    "alias": "model_alias",
    "mode": "lora",
    "checkpoint_path": "/absolute/path/to/lora_adapters/",
    "base_checkpoint_path": "/absolute/path/to/base.pt",
    "description": "Optional description",
    "lora_target_modules": ["to_k", "to_v", "to_q", "to_out.0"],  # Optional - uses config default if not specified
    "svd_model_path": "/path/to/svd/model",  # Optional
    "clip_model_path": "/path/to/clip/model"  # Optional
  }
}
```

**Required fields for LoRA models:**
- `alias`, `mode`, `checkpoint_path`, `base_checkpoint_path`

**Optional fields:**
- `lora_target_modules`: Only specify if you need to override the default from `droid_inference_config_lora.py`
- `svd_model_path`, `clip_model_path`: Override default model paths if needed
- `description`, `created_date`: For documentation purposes

## Troubleshooting

### Issue: "Model not found in registry"
**Solution:** Check that the model alias exists in `model_registry.json` and the path is correct.

### Issue: "Checkpoint file not found"
**Solution:** Verify the `checkpoint_path` in model_registry.json points to an existing file/directory.

### Issue: "Out of memory"
**Solution:** Reduce `--max_trajectories` to process fewer trajectories at once.

### Issue: Videos look incorrect
**Solution:**
- Verify `--downsampled` flag matches your dataset format
- Check if gripper annotation is interfering (try without `--gripper_annotation`)
- Inspect `run_metadata.json` for any warnings or errors

## Notes

- **Groundtruth videos are shared**: They're saved once per dataset, not per model, to save disk space.
- **Skip existing outputs**: By default, batch inference skips existing directories. This allows resuming interrupted runs.
- **No modifications to existing code**: All scripts in `eval/` are standalone and import from existing code without modifying it.
