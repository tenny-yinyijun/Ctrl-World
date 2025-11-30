import argparse
from pathlib import Path
import numpy as np
import pickle
import json
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import warnings


def load_tsne_embeddings(embedding_file):
    """
    Load precomputed t-SNE embeddings from pickle file.

    Args:
        embedding_file: Path to pickle file created by compute_tsne_embeddings.py

    Returns:
        tuple: (embeddings, labels, metadata)
    """
    print(f"Loading t-SNE embeddings from: {embedding_file}")
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    labels = data['labels']
    metadata = data.get('metadata', {})

    print(f"Loaded {data['num_samples']} samples")
    print(f"Embedding shape: {embeddings.shape}")
    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    return embeddings, labels, metadata


def load_metric_scores(metric_dir, dataset_name, model_id, metric_name):
    """
    Load metric scores from JSON file.

    Args:
        metric_dir: Base metrics directory (e.g., 'dataset_eval_metrics')
        dataset_name: Name of the dataset
        model_id: Model ID
        metric_name: Name of the metric (e.g., 'lpips', 'mse', 'ssim')

    Returns:
        dict mapping sample_key to metric value, or None if not found
    """
    metric_file = Path(metric_dir) / dataset_name / model_id / f'{metric_name}.json'

    if not metric_file.exists():
        warnings.warn(f"Metric file not found: {metric_file}")
        return None

    with open(metric_file, 'r') as f:
        scores = json.load(f)

    print(f"Loaded {len(scores)} metric scores from {metric_file}")
    return scores


def get_available_metrics(metric_dir, labels):
    """
    Automatically detect available metrics for the datasets in labels.

    Args:
        metric_dir: Base metrics directory
        labels: List of label dicts

    Returns:
        dict: {(dataset_name, model_id): [metric_names]}
    """
    available = {}
    metric_dir = Path(metric_dir)

    if not metric_dir.exists():
        return available

    # Get unique dataset and model combinations
    unique_combos = set((label['dataset_name'], label['traj_id'].replace('video_', ''))
                       for label in labels)

    for dataset_name, model_id in unique_combos:
        model_dir = metric_dir / dataset_name / model_id

        if not model_dir.exists():
            continue

        # Find all JSON files in this directory
        metric_files = list(model_dir.glob('*.json'))
        metric_names = [f.stem for f in metric_files]

        if metric_names:
            available[(dataset_name, model_id)] = metric_names

    return available


def compute_metric_opacity(metric_scores, labels, metric_name, alpha_range=(0.2, 1.0), power=2.5):
    """
    Compute opacity values based on metric scores.
    Worse performance = higher opacity (darker), better = lower opacity (lighter).
    Uses power transformation to emphasize the long tail (worst performers).

    Args:
        metric_scores: dict mapping sample_key to metric value
        labels: list of label dicts for each sample
        metric_name: name of the metric (for determining if higher is better/worse)
        alpha_range: (min_alpha, max_alpha) for opacity range
        power: exponent for power transformation (higher = more emphasis on tail)

    Returns:
        tuple: (list of opacity values (0-1) for each sample,
                dict with normalization info for colorbar)
    """
    if metric_scores is None:
        return [1.0] * len(labels), None

    # Create sample keys for each label
    scores = []
    for label in labels:
        model_id = label['traj_id'].replace('video_', '')
        sample_key = f"{label['traj_id']}/{label['sample_id']}/{label['frame_id']}"

        if sample_key in metric_scores:
            scores.append(metric_scores[sample_key])
        else:
            scores.append(None)

    # Handle missing scores
    valid_scores = [s for s in scores if s is not None]

    if not valid_scores:
        warnings.warn(f"No valid metric scores found for {metric_name}")
        return [1.0] * len(labels), None

    # Normalize scores to [0, 1]
    # For LPIPS and MSE: higher is worse, so map high values to high opacity
    # For SSIM: higher is better, so map low values to high opacity
    scores_array = np.array(scores, dtype=float)
    valid_mask = ~np.isnan(scores_array)

    if valid_mask.sum() == 0:
        return [1.0] * len(labels), None

    min_score = np.min(scores_array[valid_mask])
    max_score = np.max(scores_array[valid_mask])

    if max_score == min_score:
        return [np.mean(alpha_range)] * len(labels), None

    # Normalize to [0, 1]
    normalized = (scores_array - min_score) / (max_score - min_score)

    # For SSIM (higher is better), invert the normalization
    if metric_name.lower() == 'ssim':
        normalized = 1.0 - normalized

    # Apply power transformation to emphasize the tail (worst performers)
    # This makes average/good samples very light, and only bad samples dark
    normalized_power = normalized ** power

    # Map to alpha range
    # worse performance (higher normalized value) -> higher opacity
    alpha_values = alpha_range[0] + normalized_power * (alpha_range[1] - alpha_range[0])

    # Handle missing scores with default opacity
    alpha_values[~valid_mask] = np.mean(alpha_range)

    # Store normalization info for colorbar
    norm_info = {
        'min_score': min_score,
        'max_score': max_score,
        'power': power,
        'alpha_range': alpha_range,
        'metric_name': metric_name
    }

    return alpha_values.tolist(), norm_info


def add_metric_colorbar(cbar_ax, norm_info):
    """
    Add a colorbar showing the mapping from metric values to opacity.

    Args:
        cbar_ax: matplotlib axis for the colorbar
        norm_info: dict with normalization info from compute_metric_opacity
    """
    min_score = norm_info['min_score']
    max_score = norm_info['max_score']
    power = norm_info['power']
    alpha_range = norm_info['alpha_range']
    metric_name = norm_info['metric_name']

    # Create a gradient showing opacity mapping
    n_steps = 100

    # For SSIM (higher is better), we want to show values from high to low (top to bottom)
    # For LPIPS/MSE (lower is better), we want to show values from low to high (top to bottom)
    is_higher_better = metric_name.lower() == 'ssim'

    # Create metric values from best to worst (top to bottom in colorbar)
    if is_higher_better:
        # SSIM: high values are good (top), low values are bad (bottom)
        metric_values = np.linspace(max_score, min_score, n_steps)
    else:
        # LPIPS/MSE: low values are good (top), high values are bad (bottom)
        metric_values = np.linspace(min_score, max_score, n_steps)

    # Compute normalized values (0 = best, 1 = worst)
    normalized = (metric_values - min_score) / (max_score - min_score)
    if is_higher_better:
        normalized = 1.0 - normalized

    # Apply power transformation
    normalized_power = normalized ** power

    # Compute opacity values
    alpha_values = alpha_range[0] + normalized_power * (alpha_range[1] - alpha_range[0])

    # Draw rectangles with varying opacity (gray color)
    for i in range(n_steps):
        rect = Rectangle((0, i/n_steps), 1, 1/n_steps,
                        facecolor=(0.3, 0.3, 0.3, alpha_values[i]),
                        edgecolor='none')
        cbar_ax.add_patch(rect)

    # Add border
    cbar_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1.5))

    # Set axis limits and remove ticks
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_aspect('auto')

    # Add metric value labels at key points
    # Show values at 0%, 25%, 50%, 75%, 100% of the range
    label_positions = [0, 0.25, 0.5, 0.75, 1.0]

    yticks = []
    yticklabels = []
    for pos in label_positions:
        idx = int(pos * (n_steps - 1))
        yticks.append(pos)
        yticklabels.append(f'{metric_values[idx]:.4f}')

    cbar_ax.set_yticks(yticks)
    cbar_ax.set_yticklabels(yticklabels, fontsize=9)
    cbar_ax.yaxis.tick_right()

    # Add title to colorbar
    better_label = 'Better' if not is_higher_better else 'Better'
    worse_label = 'Worse' if not is_higher_better else 'Worse'

    cbar_ax.text(0.5, 1.05, metric_name.upper(), transform=cbar_ax.transAxes,
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    cbar_ax.text(0.5, 1.02, '(darker = worse)', transform=cbar_ax.transAxes,
                ha='center', va='bottom', fontsize=8, style='italic')


def plot_tsne_by_dataset(embedded, labels, title, output_path):
    """
    Create t-SNE visualization colored by dataset.

    Args:
        embedded: np.array of shape [N, 2] with t-SNE coordinates
        labels: list of dicts with metadata
        title: plot title
        output_path: where to save the plot
    """
    print(f"\nCreating t-SNE visualization for {len(labels)} samples...")

    # Color by dataset
    unique_datasets = sorted(set(label['dataset_name'] for label in labels))
    print(f"Datasets: {unique_datasets}")

    # Assign colors
    if len(unique_datasets) <= 3:
        custom_colors = ["#ffa586", "#83ed5f", "#bd91ff"]
    elif len(unique_datasets) <= 10:
        custom_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    else:
        custom_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_datasets)))

    dataset_colors = {
        dataset: custom_colors[i]
        for i, dataset in enumerate(unique_datasets)
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Count samples per dataset for legend
    dataset_counts = {
        dataset: sum(1 for label in labels if label['dataset_name'] == dataset)
        for dataset in unique_datasets
    }

    # Plot each sample
    for i, label in enumerate(labels):
        x, y = embedded[i]
        dataset = label['dataset_name']
        color = dataset_colors[dataset]

        # Plot the point
        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black', linewidth=1.0, c=[color])

        # Add label with trajectory, sample, and frame ID
        label_text = f"{label['sample_id']}-{label['frame_id']}"
        # label_text = f"{label['traj_id'][-1]}-{label['sample_id']}-{label['frame_id']}"

        ax.annotate(label_text, (x, y), fontsize=5, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                            alpha=0.4, edgecolor='black', linewidth=0.5))

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=dataset_colors[dataset], markersize=12,
                  label=f'{dataset} (n={dataset_counts[dataset]})',
                  markeredgecolor='black', markeredgewidth=1.5)
        for dataset in unique_datasets
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    full_title = f'{title}\n(Wrist Camera - View 2)'
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text box with info
    total_samples = len(labels)
    num_datasets = len(set(label['dataset_name'] for label in labels))
    unique_trajs = len(set((label['dataset_name'], label['traj_id']) for label in labels))

    info_text = (f'Total samples: {total_samples}\n'
                f'Unique trajectories: {unique_trajs}\n'
                f'Datasets: {num_datasets}')

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_tsne_by_dataset_with_metric(embedded, labels, metric_scores, metric_name,
                                     title, output_path, power=2.5):
    """
    Create t-SNE visualization colored by dataset with opacity by metric.

    Args:
        embedded: np.array of shape [N, 2] with t-SNE coordinates
        labels: list of dicts with metadata
        metric_scores: dict mapping sample_key to metric value
        metric_name: name of the metric
        title: plot title
        output_path: where to save the plot
        power: power transformation exponent for opacity mapping
    """
    print(f"\nCreating t-SNE visualization for {len(labels)} samples...")
    print(f"Using metric: {metric_name}")

    # Color by dataset
    unique_datasets = sorted(set(label['dataset_name'] for label in labels))
    print(f"Datasets: {unique_datasets}")

    # Assign colors
    if len(unique_datasets) <= 3:
        custom_colors = ["#ffa586", "#83ed5f", "#bd91ff"]
    elif len(unique_datasets) <= 10:
        custom_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    else:
        custom_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_datasets)))

    dataset_colors = {
        dataset: custom_colors[i]
        for i, dataset in enumerate(unique_datasets)
    }

    # Compute opacity values based on metric (uniform across all datasets)
    alpha_values, norm_info = compute_metric_opacity(metric_scores, labels, metric_name, power=power)
    print(f"Using power transformation: {power} (higher = more emphasis on tail)")
    print(f"Opacity range computed uniformly across ALL datasets: [{norm_info['min_score']:.4f}, {norm_info['max_score']:.4f}]")

    # Identify top 10 worst performers per dataset
    is_higher_better = metric_name.lower() == 'ssim'
    worst_samples_per_dataset = set()

    print(f"\nIdentifying top 10 worst performers per dataset:")
    for dataset in unique_datasets:
        # Get all samples for this dataset with valid scores
        dataset_samples = []
        for i, label in enumerate(labels):
            if label['dataset_name'] == dataset:
                sample_key = f"{label['traj_id']}/{label['sample_id']}/{label['frame_id']}"
                if sample_key in metric_scores and metric_scores[sample_key] is not None:
                    dataset_samples.append((i, label, metric_scores[sample_key]))

        if not dataset_samples:
            continue

        # Sort by metric score
        if is_higher_better:
            # For SSIM, lower is worse
            dataset_samples.sort(key=lambda x: x[2])
        else:
            # For LPIPS/MSE, higher is worse
            dataset_samples.sort(key=lambda x: x[2], reverse=True)

        # Take top 10 worst
        top_10_worst = dataset_samples[:min(10, len(dataset_samples))]
        print(f"  {dataset}: {len(top_10_worst)} worst samples")
        for idx, label, score in top_10_worst:
            worst_samples_per_dataset.add(idx)
            print(f"    - {label['sample_id']}-{label['frame_id']}: {metric_name}={score:.4f}")

    print(f"\nTotal samples with red borders: {len(worst_samples_per_dataset)}")

    # Create plot with space for colorbar
    fig = plt.figure(figsize=(18, 12))
    # Main plot takes most of the space, colorbar on the right
    ax = plt.subplot2grid((1, 20), (0, 0), colspan=18)
    cbar_ax = plt.subplot2grid((1, 20), (0, 19), colspan=1)

    # Count samples per dataset for legend
    dataset_counts = {
        dataset: sum(1 for label in labels if label['dataset_name'] == dataset)
        for dataset in unique_datasets
    }

    # Plot each sample with opacity based on performance
    for i, label in enumerate(labels):
        x, y = embedded[i]
        dataset = label['dataset_name']
        color = dataset_colors[dataset]
        alpha = alpha_values[i]

        # Convert color to RGBA and apply alpha
        if isinstance(color, str):
            rgba = to_rgba(color)
        else:
            rgba = tuple(color)
        rgba_with_alpha = (*rgba[:3], alpha)

        # Determine border color: red for worst 10 per dataset, black otherwise
        is_worst = i in worst_samples_per_dataset
        edge_color = 'red' if is_worst else 'black'
        edge_width = 2.0 if is_worst else 1.0

        # Plot the point (don't specify alpha parameter - use embedded alpha in color)
        ax.scatter(x, y, s=100, edgecolors=edge_color, linewidth=edge_width,
                  c=[rgba_with_alpha])

        # Add label with trajectory, sample, and frame ID
        label_text = f"{label['sample_id']}-{label['frame_id']}"
        # label_text = f"{label['traj_id'][-1]}-{label['sample_id']}-{label['frame_id']}"

        # Adjust text box alpha based on metric
        box_alpha = min(alpha * 0.6, 0.9)
        box_edge_color = 'red' if is_worst else 'black'
        box_edge_width = 1.0 if is_worst else 0.5
        ax.annotate(label_text, (x, y), fontsize=5, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                            alpha=box_alpha, edgecolor=box_edge_color, linewidth=box_edge_width))

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=dataset_colors[dataset], markersize=12,
                  label=f'{dataset} (n={dataset_counts[dataset]})',
                  markeredgecolor='black', markeredgewidth=1.5)
        for dataset in unique_datasets
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    full_title = f'{title}\n(Wrist Camera - View 2, Opacity by {metric_name.upper()})'
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text box with info
    total_samples = len(labels)
    num_datasets = len(set(label['dataset_name'] for label in labels))
    unique_trajs = len(set((label['dataset_name'], label['traj_id']) for label in labels))

    # Compute metric statistics
    valid_scores = [v for v in metric_scores.values() if v is not None]
    if valid_scores:
        metric_stats = (f'{metric_name.upper()}: '
                       f'mean={np.mean(valid_scores):.4f}, '
                       f'std={np.std(valid_scores):.4f}\n'
                       f'min={np.min(valid_scores):.4f}, '
                       f'max={np.max(valid_scores):.4f}')
    else:
        metric_stats = f'{metric_name.upper()}: No scores available'

    info_text = (f'Total samples: {total_samples}\n'
                f'Unique trajectories: {unique_trajs}\n'
                f'Datasets: {num_datasets}\n'
                f'{metric_stats}\n'
                f'Darker = worse performance\n'
                f'Red border = top 10 worst per dataset')

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add colorbar showing metric value to opacity mapping
    if norm_info is not None:
        add_metric_colorbar(cbar_ax, norm_info)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize t-SNE from precomputed embeddings (fast visualization without recomputing t-SNE)'
    )
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to pickle file with precomputed t-SNE embeddings')
    parser.add_argument('--output_dir', type=str, default='metric/tsne_plots',
                       help='Output directory for plots (default: metric/tsne_plots)')
    parser.add_argument('--output_prefix', type=str, default=None,
                       help='Prefix for output files (default: derived from embedding file)')
    parser.add_argument('--use_metric', action='store_true',
                       help='Create plots with opacity based on performance metrics')
    parser.add_argument('--metric_dir', type=str, default='dataset_eval_metrics',
                       help='Directory containing metric scores (default: dataset_eval_metrics)')
    parser.add_argument('--power', type=float, default=2.5,
                       help='Power transformation exponent (default: 2.5, higher = more emphasis on worst performers)')

    args = parser.parse_args()

    # Load precomputed t-SNE embeddings
    embedded, labels, metadata = load_tsne_embeddings(args.embedding_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output prefix
    if args.output_prefix is None:
        embedding_file_name = Path(args.embedding_file).stem
        output_prefix = embedding_file_name.replace('_tsne_embeddings', '')
    else:
        output_prefix = args.output_prefix

    print(f"\n{'='*70}")
    print("Creating t-SNE visualizations from precomputed embeddings...")
    print('='*70)

    # Create basic visualization colored by dataset
    output_file = output_dir / f'{output_prefix}_tsne_dataset.png'
    plot_tsne_by_dataset(
        embedded,
        labels,
        't-SNE: Evaluation Samples (CLIP Features)',
        output_file
    )

    # Create metric-based visualizations if requested
    if args.use_metric:
        print(f"\n{'='*70}")
        print("Creating metric-based visualizations...")
        print('='*70)

        # Automatically detect available metrics
        available_metrics = get_available_metrics(args.metric_dir, labels)

        if not available_metrics:
            print(f"Warning: No metrics found in {args.metric_dir}")
            print("Skipping metric-based visualizations.")
        else:
            print(f"\nFound metrics for {len(available_metrics)} dataset/model combinations:")
            for (dataset_name, model_id), metric_names in available_metrics.items():
                print(f"  {dataset_name} (model {model_id}): {metric_names}")

            # Get all unique metric names across all datasets
            all_metric_names = set()
            for metric_names in available_metrics.values():
                all_metric_names.update(metric_names)

            print(f"\nCreating plots for metrics: {sorted(all_metric_names)}")

            # Create a plot for each metric
            for metric_name in sorted(all_metric_names):
                print(f"\n--- Processing metric: {metric_name} ---")

                # Load metric scores for all datasets
                # Combine scores from all datasets/models
                combined_scores = {}
                for (dataset_name, model_id), metric_names in available_metrics.items():
                    if metric_name in metric_names:
                        scores = load_metric_scores(
                            args.metric_dir, dataset_name, model_id, metric_name
                        )
                        if scores:
                            combined_scores.update(scores)

                if not combined_scores:
                    print(f"Warning: No scores found for {metric_name}, skipping")
                    continue

                # Create visualization
                output_file_metric = output_dir / f'{output_prefix}_tsne_{metric_name}.png'
                plot_tsne_by_dataset_with_metric(
                    embedded,
                    labels,
                    combined_scores,
                    metric_name,
                    't-SNE: Evaluation Samples (CLIP Features)',
                    output_file_metric,
                    power=args.power
                )

    print(f"\n{'='*70}")
    print("Done!")
    print(f"Plot(s) saved to: {output_dir}/")
    print('='*70)


if __name__ == '__main__':
    main()


# Example usage:
# Basic visualization (just colors by dataset):
# python metric/visualize_tsne_precomputed.py \
#     --embedding_file dataset_eval_tsne/1126_all_data_tsne_embeddings.pkl \
#     --output_prefix 1126_all_data
#
# With metric-based opacity (emphasizes worst performers via power transformation):
# python metric/visualize_tsne_precomputed.py \
#     --embedding_file dataset_eval_tsne/1126_all_data_tsne_embeddings.pkl \
#     --output_prefix 1126_all_data \
#     --use_metric \
#     --metric_dir dataset_eval_metrics \
#     --power 2.5
#
# Change visualization parameters (e.g., different power for opacity):
# python metric/visualize_tsne_precomputed.py \
#     --embedding_file dataset_eval_tsne/1126_all_data_tsne_embeddings.pkl \
#     --output_prefix 1126_all_data_v2 \
#     --use_metric \
#     --power 3.0
