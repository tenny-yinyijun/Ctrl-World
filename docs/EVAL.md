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
# compute and store per-video metrics
python metric/compute_experiment_metrics.py --results_dir /path/to/results

# visualize the computed metrics
python metric/analyze_metrics.py \
    --metrics_file /n/fs/iromdata/video_models/Ctrl-World/1227_eval/v2_test_combined/metrics/summary_statistics.json \
    --table \
    --metric mse
```
