import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_latent_videos(dataset_path, num_samples=16):
    """
    Load all latent videos from the dataset, organized by trajectory.

    Args:
        dataset_path: Path to the dataset (e.g., dataset_example/droid_irom)
        num_samples: Number of frames to uniformly sample from each video

    Returns:
        trajectories: dict mapping traj_id -> dict of {view_id -> sampled_tensor}
    """
    latent_videos_path = Path(dataset_path) / "latent_videos"

    if not latent_videos_path.exists():
        raise ValueError(f"Latent videos path not found: {latent_videos_path}")

    trajectories = {}

    # Process both train and val
    for split in ["train", "val"]:
        split_path = latent_videos_path / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping...")
            continue

        # Iterate through trajectory directories
        for traj_dir in tqdm(sorted(split_path.iterdir()), desc=f"Loading {split}"):
            if not traj_dir.is_dir():
                continue

            traj_id = traj_dir.name
            trajectories[traj_id] = {}

            # Load each view (0, 1, 2)
            for view_id in [0, 1, 2]:
                view_path = traj_dir / f"{view_id}.pt"
                if not view_path.exists():
                    continue

                # Load the latent tensor [T, C, H, W]
                latent = torch.load(view_path, map_location='cpu')

                # Uniformly sample frames
                T = latent.shape[0]
                indices = np.linspace(0, T - 1, num_samples, dtype=int)
                sampled = latent[indices]

                trajectories[traj_id][view_id] = sampled

    return trajectories


def prepare_trajectory_features(trajectories_dict, view_ids=[0, 1, 2]):
    """
    Prepare features for t-SNE by combining views within each trajectory.

    Args:
        trajectories_dict: dict mapping dataset_name -> (traj_id -> dict of {view_id -> tensor})
        view_ids: which views to include when computing trajectory features

    Returns:
        features: np.array of shape [N, D] where N is number of trajectories
        labels: list of (dataset_name, traj_id) tuples
    """
    features_list = []
    labels = []

    for dataset_name, trajectories in trajectories_dict.items():
        for traj_id, views in sorted(trajectories.items()):
            # Check if all required views are present
            if not all(vid in views for vid in view_ids):
                print(f"Warning: {dataset_name}/{traj_id} missing some views, skipping...")
                continue

            # Concatenate the views to create a single feature vector for the trajectory
            view_features = []
            for view_id in sorted(view_ids):
                # Flatten each view [T, C, H, W] -> [T*C*H*W]
                flat = views[view_id].flatten().numpy()
                view_features.append(flat)

            # Concatenate all views
            traj_feature = np.concatenate(view_features)
            features_list.append(traj_feature)
            labels.append((dataset_name, traj_id))

    features = np.stack(features_list, axis=0)
    return features, labels


def plot_tsne(features, labels, title, output_path, view_description):
    """
    Compute t-SNE and create visualization.

    Args:
        features: np.array of shape [N, D]
        labels: list of (dataset_name, traj_id) tuples
        title: plot title
        output_path: where to save the plot
        view_description: description of which views were used
    """
    print(f"\nComputing t-SNE for {title}...")
    print(f"Number of trajectories: {len(labels)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Views used: {view_description}")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embedded = tsne.fit_transform(features)

    # Get unique datasets and assign colors
    unique_datasets = sorted(set(dataset_name for dataset_name, _ in labels))

    # Use a colormap to generate distinct colors
    if len(unique_datasets) <= 10:
        # Use tab10 for up to 10 datasets
        cmap = plt.cm.tab10
    else:
        # Use tab20 for more datasets
        cmap = plt.cm.tab20

    dataset_colors = {dataset: cmap(i / max(len(unique_datasets) - 1, 1))
                     for i, dataset in enumerate(unique_datasets)}

    # Create plot
    _, ax = plt.subplots(figsize=(16, 12))

    # Count trajectories per dataset for legend
    dataset_counts = {dataset: sum(1 for d, _ in labels if d == dataset)
                     for dataset in unique_datasets}

    # Plot each trajectory
    for i, (dataset_name, traj_id) in enumerate(labels):
        x, y = embedded[i]
        color = dataset_colors[dataset_name]

        # Plot the point
        ax.scatter(x, y, s=150, alpha=0.6, edgecolors='black', linewidth=1.5, c=[color])

        # Add trajectory ID label
        ax.annotate(traj_id, (x, y), fontsize=8, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                            alpha=0.5, edgecolor='black', linewidth=0.5))

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=dataset_colors[dataset], markersize=12,
                                 label=f'{dataset} (n={dataset_counts[dataset]})',
                                 markeredgecolor='black', markeredgewidth=1.5)
                      for dataset in unique_datasets]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    ax.set_title(f'{title}\n({view_description})', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text box with info
    total_trajs = len(labels)
    info_text = f'Total trajectories: {total_trajs}\nDatasets: {len(unique_datasets)}\nFeature dim: {features.shape[1]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE visualizations of trajectory latent features')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True,
                       help='Dataset names (e.g., droid_irom droid_irom_lowres)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to uniformly sample from each video (default: 16)')
    parser.add_argument('--output_dir', type=str, default='visualization/outputs',
                       help='Output directory for plots (default: visualization/outputs)')
    parser.add_argument('--output_prefix', type=str, default=None,
                       help='Prefix for output files (default: auto-generated from dataset names)')

    args = parser.parse_args()

    # Determine output prefix
    if args.output_prefix is None:
        if len(args.dataset_names) == 1:
            output_prefix = args.dataset_names[0]
        else:
            output_prefix = '_vs_'.join(args.dataset_names)
    else:
        output_prefix = args.output_prefix

    print(f"Loading {len(args.dataset_names)} dataset(s)...")
    print("="*70)

    # Load all datasets
    all_trajectories = {}
    for dataset_name in args.dataset_names:
        dataset_path = Path('dataset_example') / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}, skipping...")
            continue

        print(f"\nDataset: {dataset_name}")
        print(f"Path: {dataset_path}")

        # Load all latent videos organized by trajectory
        trajectories = load_latent_videos(dataset_path, num_samples=args.num_frames)
        all_trajectories[dataset_name] = trajectories

        print(f"Loaded {len(trajectories)} trajectories")

        # Count views per trajectory
        view_counts = {0: 0, 1: 0, 2: 0}
        for views in trajectories.values():
            for view_id in views.keys():
                view_counts[view_id] += 1

        print(f"  View 0 (Left View): {view_counts[0]} videos")
        print(f"  View 1 (Right View): {view_counts[1]} videos")
        print(f"  View 2 (Wrist View): {view_counts[2]} videos")

    if not all_trajectories:
        print("Error: No valid datasets loaded!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Version 1: All views (0, 1, 2) combined
    print("\n" + "="*70)
    print("Creating t-SNE visualization with all views (0, 1, 2) combined...")
    print("="*70)
    features_all, labels_all = prepare_trajectory_features(all_trajectories, view_ids=[0, 1, 2])
    plot_tsne(
        features_all,
        labels_all,
        't-SNE: Trajectory Comparison',
        output_dir / f'{output_prefix}_tsne_all_views.png',
        'Features from Left + Right + Wrist views'
    )

    # Version 2: Only views 0 and 1 combined
    print("\n" + "="*70)
    print("Creating t-SNE visualization with views 0 and 1 combined...")
    print("="*70)
    features_01, labels_01 = prepare_trajectory_features(all_trajectories, view_ids=[0, 1])
    plot_tsne(
        features_01,
        labels_01,
        't-SNE: Trajectory Comparison',
        output_dir / f'{output_prefix}_tsne_lr_views.png',
        'Features from Left + Right views only'
    )

    print("\n" + "="*70)
    print("Done! Visualizations saved to:")
    print(f"  - {output_dir / f'{output_prefix}_tsne_all_views.png'}")
    print(f"  - {output_dir / f'{output_prefix}_tsne_lr_views.png'}")
    print("="*70)


if __name__ == '__main__':
    main()
