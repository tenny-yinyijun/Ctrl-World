# Evaluation Instructions

All model checkpoints should be updated in [assets/model_registry.json](../assets/model_registry.json) before starting evaluation.


## 1. Running Batch Generation

First, specify the models and test dataset in submit_batch_jobs_final.sh. Then, run:
```bash
bash eval/submit_batch_jobs_final.sh
```
This will submit one job per model + eval_item.

## 2. Computing Metrics

After obtaining results, compute metrics:
```bash
# After all generations are completed, compute the metrics:
python eval/metric/compute_metrics.py \
    --results_dir /path/to/results \
    --output_dir /path/to/outputs \
    --metrics mse ssim lpips \
    --num_workers 4
```
