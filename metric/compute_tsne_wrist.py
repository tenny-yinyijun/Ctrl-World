import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import cv2
from transformers import CLIPProcessor, CLIPModel
import pickle


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


def extract_frames_from_video(video_path):
    """
    Extract all frames from a video file.

    Args:
        video_path: Path to video file

    Returns:
        List of frames (RGB numpy arrays), or empty list if error
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        cap.release()
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def process_dataset_samples(dataset_path, model, processor, device='cuda'):
    """
    Process all samples from a dataset and extract CLIP features from wrist camera.

    Args:
        dataset_path: Path to dataset (e.g., dataset_eval_samples/irom_1126_base2)
        model: CLIP model
        processor: CLIP processor
        device: Device for computation

    Returns:
        List of dicts, each containing:
            - 'dataset_name': name of the dataset
            - 'traj_id': trajectory ID (e.g., 'video_0')
            - 'sample_id': sample ID within trajectory
            - 'frame_id': frame ID within sample
            - 'type': 'groundtruth'
            - 'features': flattened CLIP features [num_frames * feature_dim]
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    all_samples = []

    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")

    # Iterate through trajectory directories (video_0, video_1, ...)
    traj_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('video_')])
    
    for traj_dir in traj_dirs:
        traj_id = traj_dir.name

        # Iterate through sample directories (0, 1, 2, ...)
        sample_dirs = sorted([d for d in traj_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name))

        for sample_dir in tqdm(sample_dirs, desc=f"Processing {sample_dirs}"):
            sample_id = sample_dir.name

            # Process only groundtruth (predictions not used for feature analysis)
            for sample_type in ['groundtruth']:
                type_dir = sample_dir / sample_type

                if not type_dir.exists():
                    continue

                # Find all view2 videos in this directory
                view2_videos = sorted(type_dir.glob('*_view2.mp4'), key=lambda x: int(x.stem.split('_')[0]))

                for video_path in view2_videos:
                    # Extract frame_id from filename (e.g., "0_view2.mp4" -> "0")
                    frame_id = video_path.stem.split('_')[0]

                    try:
                        # Extract all frames from the video
                        frames = extract_frames_from_video(video_path)

                        if not frames:
                            print(f"Warning: No frames extracted from {video_path}")
                            continue

                        # Compute CLIP features for all frames
                        features = extract_clip_features_from_frames(frames, model, processor, device)

                        if features is None:
                            print(f"Warning: Failed to compute features for {video_path}")
                            continue

                        # Flatten features [num_frames, feature_dim] -> [num_frames * feature_dim]
                        flat_features = features.flatten()

                        # Store sample with metadata
                        all_samples.append({
                            'dataset_name': dataset_name,
                            'traj_id': traj_id,
                            'sample_id': sample_id,
                            'frame_id': frame_id,
                            'type': sample_type,
                            'features': flat_features.numpy(),
                            'num_frames': len(frames),
                            'feature_dim': features.shape[1]
                        })

                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        continue

    return all_samples


def save_results(samples, output_path):
    """
    Save computed features and metadata to disk.

    Args:
        samples: List of sample dicts with features and metadata
        output_path: Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    data = {
        'samples': samples,
        'num_samples': len(samples)
    }

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved {len(samples)} samples to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    datasets = set(s['dataset_name'] for s in samples)
    for dataset in sorted(datasets):
        dataset_samples = [s for s in samples if s['dataset_name'] == dataset]
        print(f"  {dataset}: {len(dataset_samples)} groundtruth samples")

        # Count unique trajectories
        unique_trajs = len(set(s['traj_id'] for s in dataset_samples))
        print(f"    - unique trajectories: {unique_trajs}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute CLIP features for evaluation samples (wrist camera only)'
    )
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                       help='Paths to datasets (e.g., dataset_eval_samples/irom_1126_base2)')
    parser.add_argument('--output_dir', type=str,
                       default='/n/fs/tom-project/video_models/Ctrl-World/dataset_eval_tsne',
                       help='Output directory for computed features')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (default: auto-generated from dataset names)')
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

    # Process all datasets
    all_samples = []
    for dataset_path in args.dataset_paths:
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset_path}")
        print('='*70)

        samples = process_dataset_samples(dataset_path, model, processor, device)
        all_samples.extend(samples)

        print(f"Extracted {len(samples)} samples from {dataset_path}")

    if not all_samples:
        print("Error: No valid samples found!")
        return

    # Determine output filename
    if args.output_name is None:
        dataset_names = [Path(p).name for p in args.dataset_paths]
        if len(dataset_names) == 1:
            output_name = f"{dataset_names[0]}_clip_features.pkl"
        else:
            output_name = f"{'_'.join(dataset_names)}_clip_features.pkl"
    else:
        output_name = args.output_name
        if not output_name.endswith('.pkl'):
            output_name += '.pkl'

    output_path = Path(args.output_dir) / output_name

    # Save results
    save_results(all_samples, output_path)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"Results saved to: {output_path}")
    print('='*70)


if __name__ == '__main__':
    main()


# Example usage:
# python metric/compute_tsne_wrist.py \
#     --dataset_paths dataset_eval_samples/irom_1126_base2 dataset_eval_samples/irom_1126_play
