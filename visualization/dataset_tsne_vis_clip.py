import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(model_name='openai/clip-vit-base-patch32', device='cuda'):
    """
    Load CLIP model and processor.

    Args:
        model_name: HuggingFace model name for CLIP
        device: Device to load model on

    Returns:
        model, processor
    """
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def extract_clip_features_from_video(video_path, model, processor, num_samples=16, device='cuda'):
    """
    Extract CLIP features from a video by sampling frames.

    Args:
        video_path: Path to video file
        model: CLIP model
        processor: CLIP processor
        num_samples: Number of frames to uniformly sample
        device: Device for computation

    Returns:
        features: Tensor of shape [num_samples, feature_dim], or None if video is too short
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        cap.release()
        return None

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if video has enough frames (also catches 0-frame videos)
    if total_frames < num_samples:
        print(f"Warning: Video {video_path} has only {total_frames} frames (need at least {num_samples}), skipping...")
        cap.release()
        return None

    # Calculate frame indices to sample
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")

    cap.release()

    if not frames:
        print(f"Warning: No frames extracted from {video_path}")
        return None

    # Check if we extracted the expected number of frames
    if len(frames) != num_samples:
        print(f"Warning: Video {video_path} only extracted {len(frames)}/{num_samples} frames, skipping...")
        return None

    # Process frames with CLIP
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return features.cpu()


def load_videos_with_clip_features(dataset_path, model, processor, num_samples=16, device='cuda'):
    """
    Load all videos from the dataset and extract CLIP features, organized by trajectory.

    Args:
        dataset_path: Path to the dataset (e.g., dataset_example/droid_irom)
        model: CLIP model
        processor: CLIP processor
        num_samples: Number of frames to uniformly sample from each video
        device: Device for computation

    Returns:
        trajectories: dict mapping traj_id -> dict of {view_id -> features_tensor}
    """
    videos_path = Path(dataset_path) / "videos"

    if not videos_path.exists():
        raise ValueError(f"Videos path not found: {videos_path}")

    trajectories = {}

    # Process both train and val
    for split in ["train", "val"]:
        split_path = videos_path / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping...")
            continue

        # Iterate through trajectory directories
        traj_dirs = sorted(split_path.iterdir())
        for traj_dir in tqdm(traj_dirs, desc=f"Processing {split} videos"):
            if not traj_dir.is_dir():
                continue

            traj_id = traj_dir.name
            trajectories[traj_id] = {}

            # Load each view (0, 1, 2)
            for view_id in [0, 1, 2]:
                view_path = traj_dir / f"{view_id}.mp4"
                if not view_path.exists():
                    continue

                try:
                    # Extract CLIP features from video
                    features = extract_clip_features_from_video(
                        view_path, model, processor, num_samples, device
                    )
                    # Skip if video was too short or had other issues
                    if features is None:
                        continue
                    trajectories[traj_id][view_id] = features
                except Exception as e:
                    print(f"Error processing {view_path}: {e}")
                    continue

    # Remove trajectories that have no valid views
    trajectories = {traj_id: views for traj_id, views in trajectories.items() if views}

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
                # Flatten each view [T, D] -> [T*D]
                flat = views[view_id].flatten().numpy()
                view_features.append(flat)

            # Concatenate all views
            traj_feature = np.concatenate(view_features)
            features_list.append(traj_feature)
            labels.append((dataset_name, traj_id))

    if not features_list:
        raise ValueError("No valid trajectories found after filtering!")

    # Check for shape consistency before stacking
    shapes = [f.shape for f in features_list]
    if len(set(shapes)) > 1:
        print(f"Error: Feature shape mismatch detected!")
        print(f"Found {len(set(shapes))} different shapes:")
        for i, (label, shape) in enumerate(zip(labels, shapes)):
            print(f"  {label}: {shape}")
        raise ValueError("All trajectory features must have the same shape for t-SNE visualization")

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
    parser = argparse.ArgumentParser(description='Create t-SNE visualizations using CLIP features from raw videos')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True,
                       help='Dataset names (e.g., droid_irom droid_irom_lowres)')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames to uniformly sample from each video (default: 16)')
    parser.add_argument('--output_dir', type=str, default='visualization/outputs',
                       help='Output directory for plots (default: visualization/outputs)')
    parser.add_argument('--output_prefix', type=str, default=None,
                       help='Prefix for output files (default: auto-generated from dataset names)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name (default: openai/clip-vit-base-patch32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")

    # Load CLIP model
    model, processor = load_clip_model(args.clip_model, device)

    # Determine output prefix
    if args.output_prefix is None:
        if len(args.dataset_names) == 1:
            output_prefix = args.dataset_names[0]
        else:
            output_prefix = '_vs_'.join(args.dataset_names)
    else:
        output_prefix = args.output_prefix

    print(f"\nLoading {len(args.dataset_names)} dataset(s)...")
    print("="*70)

    # Load all datasets and extract CLIP features
    all_trajectories = {}
    for dataset_name in args.dataset_names:
        dataset_path = Path('dataset_example') / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}, skipping...")
            continue

        print(f"\nDataset: {dataset_name}")
        print(f"Path: {dataset_path}")

        # Load all videos and extract CLIP features
        trajectories = load_videos_with_clip_features(
            dataset_path, model, processor, num_samples=args.num_frames, device=device
        )
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
        't-SNE: Trajectory Comparison (CLIP Features)',
        output_dir / f'{output_prefix}_tsne_clip_all_views.png',
        'CLIP features from Left + Right + Wrist views'
    )

    # Version 2: Only views 0 and 1 combined
    print("\n" + "="*70)
    print("Creating t-SNE visualization with views 0 and 1 combined...")
    print("="*70)
    features_01, labels_01 = prepare_trajectory_features(all_trajectories, view_ids=[0, 1])
    plot_tsne(
        features_01,
        labels_01,
        't-SNE: Trajectory Comparison (CLIP Features)',
        output_dir / f'{output_prefix}_tsne_clip_lr_views.png',
        'CLIP features from Left + Right views only'
    )

    print("\n" + "="*70)
    print("Done! Visualizations saved to:")
    print(f"  - {output_dir / f'{output_prefix}_tsne_clip_all_views.png'}")
    print(f"  - {output_dir / f'{output_prefix}_tsne_clip_lr_views.png'}")
    print("="*70)


if __name__ == '__main__':
    main()
