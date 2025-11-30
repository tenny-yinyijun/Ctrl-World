import argparse
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import warnings
from metric_utils import VideoMetric


def load_models_mapping(dataset_path):
    """
    Load models.txt to get model_id to checkpoint mapping.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        dict mapping model_id (str) to checkpoint path
    """
    models_file = Path(dataset_path) / 'models.txt'

    if not models_file.exists():
        warnings.warn(f"models.txt not found in {dataset_path}")
        return {}

    models = {}
    with open(models_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            model_id, checkpoint_path = line.split(':', 1)
            models[model_id.strip()] = checkpoint_path.strip()

    return models


def get_video_pairs(dataset_path, model_id):
    """
    Find all ground truth and prediction video pairs for a given dataset and model.

    Args:
        dataset_path: Path to dataset (e.g., dataset_eval_samples/irom_1126_base2)
        model_id: Model ID (e.g., '0')

    Returns:
        List of tuples (gt_path, pred_path, metadata_dict)
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    video_dir = dataset_path / f'video_{model_id}'

    if not video_dir.exists():
        warnings.warn(f"video_{model_id} not found in {dataset_path}")
        return []

    pairs = []

    # Iterate through sample directories (0, 1, 2, ...)
    sample_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()],
                        key=lambda x: int(x.name))

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        gt_dir = sample_dir / 'groundtruth'
        pred_dir = sample_dir / 'prediction'

        if not gt_dir.exists() or not pred_dir.exists():
            warnings.warn(f"Missing groundtruth or prediction in {sample_dir}")
            continue

        # Find all view2 videos in ground truth
        gt_videos = sorted(gt_dir.glob('*_view2.mp4'),
                          key=lambda x: int(x.stem.split('_')[0]))

        for gt_video in gt_videos:
            # Extract frame_id from filename (e.g., "0_view2.mp4" -> "0")
            frame_id = gt_video.stem.split('_')[0]
            pred_video = pred_dir / f'{frame_id}_view2.mp4'

            if not pred_video.exists():
                warnings.warn(f"Prediction not found: {pred_video}")
                continue

            metadata = {
                'dataset_name': dataset_name,
                'model_id': model_id,
                'traj_id': f'video_{model_id}',
                'sample_id': sample_id,
                'frame_id': frame_id
            }

            pairs.append((str(gt_video), str(pred_video), metadata))

    return pairs


def compute_metrics_for_dataset(dataset_path, metric_names, device='cuda'):
    """
    Compute metrics for all models in a dataset.

    Args:
        dataset_path: Path to dataset directory
        metric_names: List of metric names to compute
        device: Device to use for computation

    Returns:
        dict: Results organized by model_id and metric_name
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    # Load model mappings
    models = load_models_mapping(dataset_path)

    if not models:
        # If no models.txt, try to find video_* directories
        video_dirs = sorted([d for d in dataset_path.iterdir()
                           if d.is_dir() and d.name.startswith('video_')])
        models = {d.name.replace('video_', ''): 'unknown' for d in video_dirs}

    print(f"\nFound {len(models)} model(s) in {dataset_name}:")
    for model_id, checkpoint in models.items():
        print(f"  Model {model_id}: {checkpoint}")

    # Initialize metric calculator
    metric_calculator = VideoMetric(device=device)

    results = {}

    for model_id in models.keys():
        print(f"\n{'='*70}")
        print(f"Processing model {model_id} in {dataset_name}")
        print('='*70)

        # Get all video pairs for this model
        pairs = get_video_pairs(dataset_path, model_id)

        if not pairs:
            print(f"No video pairs found for model {model_id}")
            continue

        print(f"Found {len(pairs)} video pairs")

        # Initialize results for this model
        model_results = {metric: {} for metric in metric_names}

        # Compute metrics for each pair
        for gt_path, pred_path, metadata in tqdm(pairs, desc=f"Computing metrics"):
            try:
                # Load videos
                gt_video = metric_calculator.load_video(gt_path)
                pred_video = metric_calculator.load_video(pred_path)

                # Check dimensions
                metric_calculator.check_video_dimensions(
                    gt_video, pred_video, gt_path, pred_path
                )

                # Compute metrics
                metrics = metric_calculator.compute_metrics(
                    gt_video, pred_video, metric_names
                )

                # Create unique identifier for this sample
                sample_key = f"{metadata['traj_id']}/{metadata['sample_id']}/{metadata['frame_id']}"

                # Store results for each metric
                for metric_name, metric_value in metrics.items():
                    model_results[metric_name][sample_key] = float(metric_value)

            except Exception as e:
                print(f"\nError processing {gt_path}: {e}")
                continue

        # Store results for this model
        results[model_id] = model_results

        # Print summary
        print(f"\nProcessed {len(pairs)} pairs for model {model_id}")
        for metric_name in metric_names:
            values = list(model_results[metric_name].values())
            if values:
                print(f"  {metric_name.upper()}: mean={np.mean(values):.4f}, "
                      f"std={np.std(values):.4f}, "
                      f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    return results


def save_results(results, dataset_name, output_dir):
    """
    Save metric results to JSON files.

    Args:
        results: dict organized as {model_id: {metric_name: {sample_key: value}}}
        dataset_name: Name of the dataset
        output_dir: Base output directory
    """
    output_dir = Path(output_dir)

    for model_id, model_results in results.items():
        model_dir = output_dir / dataset_name / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, metric_values in model_results.items():
            output_file = model_dir / f'{metric_name}.json'

            with open(output_file, 'w') as f:
                json.dump(metric_values, f, indent=2)

            print(f"Saved {len(metric_values)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute video quality metrics for evaluation samples'
    )
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                       help='Paths to datasets (e.g., dataset_eval_samples/irom_1126_base2)')
    parser.add_argument('--output_dir', type=str,
                       default='dataset_eval_metrics',
                       help='Output directory for metric results (default: dataset_eval_metrics)')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['lpips', 'mse', 'ssim'],
                       help='Metrics to compute (default: lpips mse ssim)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    print(f"Computing metrics: {args.metrics}")
    print(f"Using device: {args.device}")

    # Process each dataset
    for dataset_path in args.dataset_paths:
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset_path}")
        print('='*70)

        dataset_name = Path(dataset_path).name

        try:
            results = compute_metrics_for_dataset(
                dataset_path,
                args.metrics,
                args.device
            )

            if results:
                save_results(results, dataset_name, args.output_dir)
            else:
                print(f"No results to save for {dataset_name}")

        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("Done!")
    print(f"Results saved to: {args.output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()


# Example usage:
# python metric/compute_metric_scores.py \
#     --dataset_paths dataset_eval_samples/irom_1126_base2 dataset_eval_samples/irom_1126_play \
#     --metrics lpips mse ssim \
#     --device cuda