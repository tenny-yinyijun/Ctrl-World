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


def extract_clip_features_from_frames(frames, model, processor, device='cuda'):
    """
    Extract CLIP features from a list of frames.

    Args:
        frames: List of numpy arrays (RGB images)
        model: CLIP model
        processor: CLIP processor
        device: Device for computation

    Returns:
        features: Tensor of shape [num_frames, feature_dim]
    """
    if not frames:
        return None

    # Process frames with CLIP
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return features.cpu()


def generate_non_overlapping_samples(total_frames, num_samples, horizon_length, frame_spacing):
    """
    Generate non-overlapping sample indices from a video.

    Args:
        total_frames: Total number of frames in the video
        num_samples: Maximum number of samples to generate (k)
        horizon_length: Number of frames per sample (x)
        frame_spacing: Spacing between frames within a sample (y)

    Returns:
        List of lists, where each inner list contains frame indices for one sample
        Returns empty list if not enough frames
    """
    # Calculate the span of frames needed for one sample
    # For example: horizon_length=4, frame_spacing=2 -> indices [0, 2, 4, 6] spans 7 frames
    sample_span = (horizon_length - 1) * frame_spacing + 1

    # Calculate maximum possible non-overlapping samples
    max_possible_samples = total_frames // sample_span

    if max_possible_samples == 0:
        return []

    # Actual number of samples is min of requested and possible
    actual_num_samples = min(num_samples, max_possible_samples)

    # Generate all possible starting positions
    possible_starts = []
    for i in range(total_frames - sample_span + 1):
        # Check if we can fit a full sample starting at position i
        end_idx = i + (horizon_length - 1) * frame_spacing
        if end_idx < total_frames:
            possible_starts.append(i)

    if len(possible_starts) < actual_num_samples:
        actual_num_samples = len(possible_starts)

    # Randomly select starting positions ensuring non-overlap
    selected_samples = []
    available_starts = possible_starts.copy()

    for _ in range(actual_num_samples):
        if not available_starts:
            break

        # Randomly select a start position
        start_idx = np.random.choice(available_starts)

        # Generate frame indices for this sample
        frame_indices = [start_idx + i * frame_spacing for i in range(horizon_length)]
        selected_samples.append(frame_indices)

        # Remove overlapping positions from available starts
        # A position overlaps if it's within [start_idx, start_idx + sample_span - 1]
        available_starts = [
            s for s in available_starts
            if s >= start_idx + sample_span or s + sample_span - 1 < start_idx
        ]

    return selected_samples


def extract_samples_from_video(video_path, model, processor, num_samples=3,
                               horizon_length=4, frame_spacing=2, device='cuda'):
    """
    Extract multiple non-overlapping samples from a video and get CLIP features.

    Args:
        video_path: Path to video file
        model: CLIP model
        processor: CLIP processor
        num_samples: Maximum number of samples to extract (k)
        horizon_length: Number of frames per sample (x)
        frame_spacing: Spacing between frames within a sample (y)
        device: Device for computation

    Returns:
        List of feature tensors, one per sample. Each tensor is [horizon_length, feature_dim]
        Returns empty list if video cannot be processed
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        cap.release()
        return []

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate non-overlapping sample indices
    sample_indices_list = generate_non_overlapping_samples(
        total_frames, num_samples, horizon_length, frame_spacing
    )

    if not sample_indices_list:
        sample_span = (horizon_length - 1) * frame_spacing + 1
        print(f"Warning: Video {video_path} has only {total_frames} frames "
              f"(need at least {sample_span} for one sample), skipping...")
        cap.release()
        return []

    # Extract frames for each sample
    all_sample_features = []

    for sample_idx, frame_indices in enumerate(sample_indices_list):
        frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                break

        # Only process if we got all required frames
        if len(frames) == horizon_length:
            features = extract_clip_features_from_frames(frames, model, processor, device)
            if features is not None:
                all_sample_features.append(features)
        else:
            print(f"Warning: Sample {sample_idx} from {video_path} incomplete "
                  f"({len(frames)}/{horizon_length} frames), skipping...")

    cap.release()
    return all_sample_features


def load_wrist_videos_with_clip_samples(dataset_path, model, processor, num_samples=3,
                                        horizon_length=4, frame_spacing=2, device='cuda',
                                        random_seed=42):
    """
    Load wrist camera videos (view 2) from the dataset and extract CLIP features from short-horizon samples.

    Args:
        dataset_path: Path to the dataset (e.g., dataset_example/droid_irom)
        model: CLIP model
        processor: CLIP processor
        num_samples: Maximum number of samples per video (k)
        horizon_length: Number of frames per sample (x)
        frame_spacing: Spacing between frames within a sample (y)
        device: Device for computation
        random_seed: Random seed for reproducibility

    Returns:
        samples: list of dicts, each containing:
            - 'dataset_name': name of the dataset
            - 'traj_id': trajectory ID
            - 'sample_id': sample index within this trajectory
            - 'features': tensor of shape [horizon_length, feature_dim]
    """
    np.random.seed(random_seed)
    videos_path = Path(dataset_path) / "videos"

    if not videos_path.exists():
        raise ValueError(f"Videos path not found: {videos_path}")

    all_samples = []

    # Process both train and val
    for split in ["train", "val"]:
        split_path = videos_path / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping...")
            continue

        # Iterate through trajectory directories
        traj_dirs = sorted(split_path.iterdir())
        for traj_dir in tqdm(traj_dirs, desc=f"Processing {split} right videos"):
            if not traj_dir.is_dir():
                continue

            traj_id = traj_dir.name

            # Load only wrist camera (view 2)
            view_path = traj_dir / "1.mp4"
            if not view_path.exists():
                continue

            try:
                # Extract samples from this video
                sample_features_list = extract_samples_from_video(
                    view_path, model, processor, num_samples,
                    horizon_length, frame_spacing, device
                )

                # Store each sample with metadata
                for sample_idx, features in enumerate(sample_features_list):
                    all_samples.append({
                        'traj_id': traj_id,
                        'sample_id': sample_idx,
                        'features': features
                    })

            except Exception as e:
                print(f"Error processing {view_path}: {e}")
                continue

    return all_samples


def prepare_sample_features(samples_dict):
    """
    Prepare features for t-SNE from wrist camera samples.

    Args:
        samples_dict: dict mapping dataset_name -> list of sample dicts

    Returns:
        features: np.array of shape [N, D] where N is number of samples
        labels: list of (dataset_name, traj_id, sample_id) tuples
    """
    features_list = []
    labels = []

    for dataset_name, samples in samples_dict.items():
        for sample in samples:
            # Flatten the features [T, D] -> [T*D]
            flat_features = sample['features'].flatten().numpy()
            features_list.append(flat_features)
            labels.append((dataset_name, sample['traj_id'], sample['sample_id']))

    if not features_list:
        raise ValueError("No valid samples found!")

    # Check for shape consistency
    shapes = [f.shape for f in features_list]
    if len(set(shapes)) > 1:
        print(f"Error: Feature shape mismatch detected!")
        print(f"Found {len(set(shapes))} different shapes:")
        for i, (label, shape) in enumerate(zip(labels[:10], shapes[:10])):
            print(f"  {label}: {shape}")
        raise ValueError("All sample features must have the same shape for t-SNE visualization")

    features = np.stack(features_list, axis=0)
    return features, labels


def plot_tsne(features, labels, title, output_path, horizon_info):
    """
    Compute t-SNE and create visualization.

    Args:
        features: np.array of shape [N, D]
        labels: list of (dataset_name, traj_id, sample_id) tuples
        title: plot title
        output_path: where to save the plot
        horizon_info: description of horizon parameters
    """
    print(f"\nComputing t-SNE for {title}...")
    print(f"Number of samples: {len(labels)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Horizon: {horizon_info}")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embedded = tsne.fit_transform(features)

    # Get unique datasets and assign colors
    unique_datasets = sorted(set(dataset_name for dataset_name, _, _ in labels))

    # Use a colormap to generate distinct colors
    if len(unique_datasets) <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20

    dataset_colors = {dataset: cmap(i / max(len(unique_datasets) - 1, 1))
                     for i, dataset in enumerate(unique_datasets)}

    # Create plot
    _, ax = plt.subplots(figsize=(16, 12))

    # Count samples per dataset for legend
    dataset_counts = {dataset: sum(1 for d, _, _ in labels if d == dataset)
                     for dataset in unique_datasets}

    # Plot each sample
    for i, (dataset_name, traj_id, sample_id) in enumerate(labels):
        x, y = embedded[i]
        color = dataset_colors[dataset_name]

        # Plot the point
        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black', linewidth=1.0, c=[color])

        # Add label with trajectory and sample ID
        label_text = f"{traj_id}-{sample_id}"
        ax.annotate(label_text, (x, y), fontsize=6, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                            alpha=0.4, edgecolor='black', linewidth=0.5))

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=dataset_colors[dataset], markersize=12,
                                 label=f'{dataset} (n={dataset_counts[dataset]})',
                                 markeredgecolor='black', markeredgewidth=1.5)
                      for dataset in unique_datasets]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    full_title = f'{title}\n(Wrist Camera Only)\n{horizon_info}'
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text box with info
    total_samples = len(labels)
    unique_trajs = len(set((d, t) for d, t, _ in labels))
    info_text = f'Total samples: {total_samples}\nUnique trajectories: {unique_trajs}\nDatasets: {len(unique_datasets)}\nFeature dim: {features.shape[1]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create t-SNE visualizations using CLIP features from short-horizon wrist camera samples'
    )
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True,
                       help='Dataset names (e.g., droid_irom droid_irom_lowres)')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Maximum number of samples per video (k, default: 3)')
    parser.add_argument('--horizon_length', type=int, default=4,
                       help='Number of frames per sample (x, default: 4)')
    parser.add_argument('--frame_spacing', type=int, default=2,
                       help='Spacing between frames within a sample (y, default: 2)')
    parser.add_argument('--output_dir', type=str, default='visualization/outputs',
                       help='Output directory for plots (default: visualization/outputs)')
    parser.add_argument('--output_prefix', type=str, default=None,
                       help='Prefix for output files (default: auto-generated from dataset names)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name (default: openai/clip-vit-base-patch32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')

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

    # Create horizon description
    horizon_info = (f"Samples: k≤{args.num_samples}, Horizon: {args.horizon_length} frames, "
                   f"Spacing: {args.frame_spacing}")
    print(f"\n{horizon_info}")
    print("="*70)

    print(f"\nLoading {len(args.dataset_names)} dataset(s)...")
    print("="*70)

    # Load all datasets and extract CLIP features from wrist camera samples
    all_samples = {}
    for dataset_name in args.dataset_names:
        dataset_path = Path('dataset_example') / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}, skipping...")
            continue

        print(f"\nDataset: {dataset_name}")
        print(f"Path: {dataset_path}")

        # Load samples from wrist camera videos only
        samples = load_wrist_videos_with_clip_samples(
            dataset_path, model, processor,
            num_samples=args.num_samples,
            horizon_length=args.horizon_length,
            frame_spacing=args.frame_spacing,
            device=device,
            random_seed=args.random_seed
        )
        all_samples[dataset_name] = samples

        print(f"Extracted {len(samples)} wrist camera samples")

        # Count unique trajectories
        unique_trajs = len(set(s['traj_id'] for s in samples))
        print(f"  From {unique_trajs} unique trajectories")

    if not all_samples:
        print("Error: No valid datasets loaded!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create suffix for filename
    suffix = f"_h{args.horizon_length}_s{args.frame_spacing}_k{args.num_samples}"

    # Create t-SNE visualization for wrist camera
    print("\n" + "="*70)
    print("Creating t-SNE visualization for wrist camera samples...")
    print("="*70)
    try:
        features, labels = prepare_sample_features(all_samples)
        output_file = output_dir / f'{output_prefix}_tsne_clip_shorthor_left{suffix}.png'
        plot_tsne(
            features,
            labels,
            't-SNE: Short-Horizon Sample Comparison (CLIP Features)',
            output_file,
            horizon_info
        )
        print("\n" + "="*70)
        print("Done! Visualization saved to:")
        print(f"  - {output_file}")
        print("="*70)
    except ValueError as e:
        print(f"Error creating visualization: {e}")


if __name__ == '__main__':
    main()
