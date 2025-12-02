#!/usr/bin/env python3
"""
Script to normalize actions in a dataset using statistics from action_norm_stat.json.
Similar to the normalize_bound method in dataset/dataset_droid_exp33.py.

Usage:
    python normalize_dataset_actions.py --dataset_path detector_data/irom_1126_all2
    python normalize_dataset_actions.py --dataset_path detector_data/irom_1126_all2 --output_dir detector_data/irom_1126_all2_normalized
"""

import json
import os
import argparse
import numpy as np
from tqdm import tqdm


def normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize data to [-1, 1] range using min-max normalization.

    Args:
        data: Input data to normalize
        data_min: Minimum bounds (p01 percentile)
        data_max: Maximum bounds (p99 percentile)
        clip_min: Minimum clip value (default: -1)
        clip_max: Maximum clip value (default: 1)
        eps: Small epsilon to avoid division by zero (default: 1e-8)

    Returns:
        Normalized data clipped to [clip_min, clip_max]
    """
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


def denormalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Denormalize data from [-1, 1] range back to original scale.

    Args:
        data: Normalized data to denormalize
        data_min: Minimum bounds (p01 percentile)
        data_max: Maximum bounds (p99 percentile)
        clip_min: Minimum clip value (default: -1)
        clip_max: Maximum clip value (default: 1)
        eps: Small epsilon for numerical stability (default: 1e-8)

    Returns:
        Denormalized data in original scale
    """
    clip_range = clip_max - clip_min
    rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
    return rdata


def load_normalization_stats(stats_path: str):
    """
    Load normalization statistics from JSON file.

    Args:
        stats_path: Path to the action_norm_stat.json file

    Returns:
        Tuple of (state_p01, state_p99) as numpy arrays
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    state_p01 = np.array(stats['state_01'])
    state_p99 = np.array(stats['state_99'])

    return state_p01, state_p99


def normalize_actions_in_file(action_file_path: str, state_p01: np.ndarray, state_p99: np.ndarray):
    """
    Load, normalize, and return actions from a single action file.

    Args:
        action_file_path: Path to the action JSON file
        state_p01: Minimum bounds for normalization
        state_p99: Maximum bounds for normalization

    Returns:
        Dictionary with normalized actions
    """
    with open(action_file_path, 'r') as f:
        action_data = json.load(f)

    # Convert actions to numpy array
    actions = np.array(action_data['actions'])

    # Normalize actions
    normalized_actions = normalize_bound(actions, state_p01, state_p99)

    # Create output dictionary
    output_data = {
        'length': action_data['length'],
        'actions': normalized_actions.tolist()
    }

    return output_data


def normalize_dataset(
    dataset_path: str,
    stats_path: str = "detector_data/action_norm_stat.json",
    output_dir: str = None,
    in_place: bool = False
):
    """
    Normalize all actions in a dataset.

    Args:
        dataset_path: Path to the dataset directory (e.g., detector_data/irom_1126_all2)
        stats_path: Path to the normalization statistics file
        output_dir: Optional output directory for normalized actions. If None, creates {dataset_path}_normalized
        in_place: If True, overwrites original action files (use with caution!)
    """
    # Load normalization statistics
    print(f"Loading normalization statistics from {stats_path}...")
    state_p01, state_p99 = load_normalization_stats(stats_path)
    print(f"  state_p01: {state_p01}")
    print(f"  state_p99: {state_p99}")

    # Setup input and output paths
    actions_dir = os.path.join(dataset_path, "actions")

    if not os.path.exists(actions_dir):
        raise ValueError(f"Actions directory not found: {actions_dir}")

    if in_place:
        output_actions_dir = actions_dir
        print(f"\n⚠️  WARNING: Normalizing actions in-place! Original files will be overwritten.")
    else:
        if output_dir is None:
            output_dir = f"{dataset_path}_normalized"
        output_actions_dir = os.path.join(output_dir, "actions")
        os.makedirs(output_actions_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

    # Get all action files
    action_files = sorted([f for f in os.listdir(actions_dir) if f.endswith('.json')])
    print(f"\nFound {len(action_files)} action files to normalize")

    # Process each action file
    for action_file in tqdm(action_files, desc="Normalizing actions"):
        input_path = os.path.join(actions_dir, action_file)
        output_path = os.path.join(output_actions_dir, action_file)

        # Normalize actions
        normalized_data = normalize_actions_in_file(input_path, state_p01, state_p99)

        # Save normalized actions
        with open(output_path, 'w') as f:
            json.dump(normalized_data, f, indent=2)

    print(f"\n✓ Successfully normalized {len(action_files)} action files")
    print(f"  Output location: {output_actions_dir}")

    # Copy annotation.json if exists and not in-place
    if not in_place:
        annotation_path = os.path.join(dataset_path, "annotation.json")
        if os.path.exists(annotation_path):
            output_annotation_path = os.path.join(output_dir, "annotation.json")
            import shutil
            shutil.copy2(annotation_path, output_annotation_path)
            print(f"  Copied annotation.json to output directory")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize actions in a dataset using action_norm_stat.json"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., detector_data/irom_1126_all2)"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="detector_data/action_norm_stat.json",
        help="Path to the normalization statistics file (default: detector_data/action_norm_stat.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for normalized actions (default: {dataset_path}_normalized)"
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Normalize actions in-place (overwrites original files - use with caution!)"
    )

    args = parser.parse_args()

    # Run normalization
    normalize_dataset(
        dataset_path=args.dataset_path,
        stats_path=args.stats_path,
        output_dir=args.output_dir,
        in_place=args.in_place
    )


if __name__ == "__main__":
    main()
