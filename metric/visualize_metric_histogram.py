import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_metric_data(dataset_path):
    """
    Load all metric data from a dataset directory.

    Args:
        dataset_path: Path to dataset directory (e.g., dataset_eval_metrics/irom_1126_all2)

    Returns:
        dict: {metric_name: [list of values]}
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Collect all metric values across all models
    metric_data = defaultdict(list)

    # Find all model directories (e.g., "0/", "1/", etc.)
    model_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()],
                       key=lambda x: x.name)

    if not model_dirs:
        raise ValueError(f"No model directories found in {dataset_path}")

    print(f"Found {len(model_dirs)} model directory(ies): {[d.name for d in model_dirs]}")

    # Load all metric files
    for model_dir in model_dirs:
        metric_files = sorted(model_dir.glob('*.json'))

        for metric_file in metric_files:
            metric_name = metric_file.stem  # e.g., "mse", "lpips", "ssim"

            with open(metric_file, 'r') as f:
                data = json.load(f)
                values = list(data.values())
                metric_data[metric_name].extend(values)
                print(f"  Loaded {len(values)} values from {metric_file.relative_to(dataset_path)}")

    return dict(metric_data)


def normalize_to_range(values, target_min=-1, target_max=1):
    """
    Normalize values to target range [target_min, target_max].

    Args:
        values: numpy array of values
        target_min: minimum of target range
        target_max: maximum of target range

    Returns:
        numpy array of normalized values
    """
    values = np.array(values)
    data_min = values.min()
    data_max = values.max()

    if data_max == data_min:
        # All values are the same, return middle of target range
        return np.full_like(values, (target_min + target_max) / 2)

    # Normalize to [0, 1] then scale to [target_min, target_max]
    normalized = (values - data_min) / (data_max - data_min)
    scaled = normalized * (target_max - target_min) + target_min

    return scaled


def plot_histogram(values, title, xlabel, output_path, n_bins=100):
    """
    Create and save a histogram plot.

    Args:
        values: array of values to plot
        title: plot title
        xlabel: x-axis label
        output_path: path to save the figure
        n_bins: number of bins (default: 100)
    """
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with 100 bins spanning [min, max]
    counts, bins, patches = ax.hist(values, bins=n_bins, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Occurrence Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (f'n = {len(values)}\n'
                 f'mean = {np.mean(values):.4f}\n'
                 f'std = {np.std(values):.4f}\n'
                 f'min = {np.min(values):.4f}\n'
                 f'max = {np.max(values):.4f}')

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def visualize_metrics(dataset_name, dataset_metrics_dir='dataset_eval_metrics', n_bins=100):
    """
    Create histogram visualizations for all metrics in a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'irom_1126_all2')
        dataset_metrics_dir: Base directory containing metric data
        n_bins: Number of bins for histograms (default: 100)
    """
    dataset_path = Path(dataset_metrics_dir) / dataset_name

    print(f"Loading metric data from: {dataset_path}")
    metric_data = load_metric_data(dataset_path)

    if not metric_data:
        print("No metric data found!")
        return

    print(f"\nFound {len(metric_data)} metric(s): {list(metric_data.keys())}")
    print(f"\nGenerating histograms with {n_bins} bins...")

    # Create visualizations for each metric
    for metric_name, values in metric_data.items():
        print(f"\nProcessing metric: {metric_name.upper()}")
        print(f"  Total values: {len(values)}")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")

        # 1. Unnormalized histogram
        output_path_unnorm = dataset_path / f"{metric_name}_histogram.png"
        plot_histogram(
            values=values,
            title=f"{metric_name.upper()} Distribution - {dataset_name}",
            xlabel=f"{metric_name.upper()} Value",
            output_path=output_path_unnorm,
            n_bins=n_bins
        )

        # 2. Normalized histogram (to [-1, 1])
        normalized_values = normalize_to_range(values, target_min=-1, target_max=1)
        output_path_norm = dataset_path / f"{metric_name}_histogram_normalized.png"
        plot_histogram(
            values=normalized_values,
            title=f"{metric_name.upper()} Distribution (Normalized to [-1, 1]) - {dataset_name}",
            xlabel=f"Normalized {metric_name.upper()} Value",
            output_path=output_path_norm,
            n_bins=n_bins
        )

        print(f"  Normalized range: [{np.min(normalized_values):.4f}, {np.max(normalized_values):.4f}]")

    print(f"\nAll histograms saved to: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize metric distributions as histograms'
    )
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of dataset under dataset_eval_metrics/ (e.g., irom_1126_all2)')
    parser.add_argument('--dataset_metrics_dir', type=str,
                       default='dataset_eval_metrics',
                       help='Base directory containing metric data (default: dataset_eval_metrics)')
    parser.add_argument('--n_bins', type=int, default=100,
                       help='Number of bins for histograms (default: 100)')

    args = parser.parse_args()

    visualize_metrics(
        dataset_name=args.dataset_name,
        dataset_metrics_dir=args.dataset_metrics_dir,
        n_bins=args.n_bins
    )

    print("\nDone!")


if __name__ == '__main__':
    main()


# Example usage:
# python metric/visualize_metric_histogram.py --dataset_name irom_1126_all2
# python metric/visualize_metric_histogram.py --dataset_name irom_1126_base2 --n_bins 50
