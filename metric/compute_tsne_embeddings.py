import argparse
from pathlib import Path
import numpy as np
import pickle
from sklearn.manifold import TSNE


def load_features(feature_file):
    """
    Load computed CLIP features from pickle file.

    Args:
        feature_file: Path to pickle file created by compute_tsne_wrist.py

    Returns:
        data dict with 'samples' and 'num_samples'
    """
    print(f"Loading features from: {feature_file}")
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['num_samples']} samples")
    return data


def filter_samples(samples, include_datasets=None, include_types=None):
    """
    Filter samples based on dataset names.

    Args:
        samples: List of sample dicts
        include_datasets: List of dataset names to include (None = all)
        include_types: Deprecated (all samples are groundtruth)

    Returns:
        Filtered list of samples
    """
    filtered = samples

    if include_datasets is not None:
        filtered = [s for s in filtered if s['dataset_name'] in include_datasets]

    if include_types is not None:
        filtered = [s for s in filtered if s['type'] in include_types]

    print(f"After filtering: {len(filtered)} samples")
    return filtered


def prepare_features_for_tsne(samples):
    """
    Prepare features and labels for t-SNE visualization.

    Args:
        samples: List of sample dicts with features and metadata

    Returns:
        features: np.array of shape [N, D]
        labels: list of dicts with metadata for each sample
    """
    features_list = []
    labels = []

    for sample in samples:
        features_list.append(sample['features'])
        labels.append({
            'dataset_name': sample['dataset_name'],
            'traj_id': sample['traj_id'],
            'sample_id': sample['sample_id'],
            'frame_id': sample['frame_id'],
            'type': sample['type']
        })

    if not features_list:
        raise ValueError("No samples to compute t-SNE for!")

    features = np.stack(features_list, axis=0)
    return features, labels


def compute_tsne_embeddings(features, perplexity=None, random_state=42):
    """
    Compute t-SNE embeddings from features.

    Args:
        features: np.array of shape [N, D]
        perplexity: t-SNE perplexity parameter (default: min(30, N-1))
        random_state: Random seed for reproducibility

    Returns:
        embedded: np.array of shape [N, 2] with t-SNE coordinates
    """
    n_samples = len(features)
    if perplexity is None:
        perplexity = min(30, n_samples - 1)

    print(f"\nComputing t-SNE for {n_samples} samples...")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Perplexity: {perplexity}")

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embedded = tsne.fit_transform(features)

    print(f"t-SNE computation complete. Embedded shape: {embedded.shape}")
    return embedded


def save_tsne_embeddings(output_file, embedded, labels, metadata=None):
    """
    Save t-SNE embeddings and labels to pickle file.

    Args:
        output_file: Path to output pickle file
        embedded: np.array of shape [N, 2] with t-SNE coordinates
        labels: list of dicts with metadata for each sample
        metadata: optional dict with additional metadata (e.g., perplexity, random_state)
    """
    data = {
        'embeddings': embedded,
        'labels': labels,
        'num_samples': len(labels),
        'metadata': metadata or {}
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved t-SNE embeddings to: {output_file}")
    print(f"  Number of samples: {len(labels)}")
    print(f"  Embedding shape: {embedded.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute t-SNE embeddings from CLIP features and save for later visualization'
    )
    parser.add_argument('--feature_file', type=str, required=True,
                       help='Path to pickle file with computed CLIP features')
    parser.add_argument('--output_dir', type=str, default='dataset_eval_tsne',
                       help='Output directory for t-SNE embeddings (default: dataset_eval_tsne)')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (default: <feature_file_stem>_tsne_embeddings.pkl)')
    parser.add_argument('--include_datasets', type=str, nargs='+', default=None,
                       help='Only include these datasets (default: all)')
    parser.add_argument('--perplexity', type=int, default=None,
                       help='t-SNE perplexity parameter (default: min(30, N-1))')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Load features
    data = load_features(args.feature_file)
    samples = data['samples']

    # Filter samples by dataset if specified
    if args.include_datasets:
        samples = filter_samples(samples, args.include_datasets, None)

    # Prepare features for t-SNE
    features, labels = prepare_features_for_tsne(samples)

    # Compute t-SNE embeddings
    embedded = compute_tsne_embeddings(
        features,
        perplexity=args.perplexity,
        random_state=args.random_state
    )

    # Determine output filename
    if args.output_name is None:
        feature_file_stem = Path(args.feature_file).stem
        output_name = f"{feature_file_stem}_tsne_embeddings.pkl"
    else:
        output_name = args.output_name

    output_file = Path(args.output_dir) / output_name

    # Save embeddings with metadata
    metadata = {
        'feature_file': str(args.feature_file),
        'perplexity': args.perplexity or min(30, len(features) - 1),
        'random_state': args.random_state,
        'feature_dim': features.shape[1],
        'included_datasets': args.include_datasets
    }

    save_tsne_embeddings(output_file, embedded, labels, metadata)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"t-SNE embeddings saved to: {output_file}")
    print(f"You can now visualize with: python metric/visualize_tsne_precomputed.py --embedding_file {output_file}")
    print('='*70)


if __name__ == '__main__':
    main()


# Example usage:
# python metric/compute_tsne_embeddings.py \
#     --feature_file dataset_eval_tsne/1126_all_data.pkl \
#     --output_dir dataset_eval_tsne
#
# With custom parameters:
# python metric/compute_tsne_embeddings.py \
#     --feature_file dataset_eval_tsne/1126_all_data.pkl \
#     --output_dir dataset_eval_tsne \
#     --output_name custom_tsne.pkl \
#     --perplexity 50 \
#     --random_state 123
