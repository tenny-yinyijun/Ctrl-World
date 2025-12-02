#!/usr/bin/env python3
"""
Calculate the total time/duration of a dataset based on raw_length in annotation JSONs.
Data is collected at 15Hz.

Usage:
    python util_scripts/calculate_time.py dataset_example/irom_1126_play
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def calculate_dataset_time(dataset_path: str) -> Dict[str, any]:
    """
    Calculate total frames and duration for a dataset.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Dictionary containing statistics for train, val, and total
    """
    dataset_dir = Path(dataset_path)
    annotation_dir = dataset_dir / "annotation"

    if not annotation_dir.exists():
        raise ValueError(f"Annotation directory not found: {annotation_dir}")

    results = {
        'train': {'count': 0, 'total_frames': 0},
        'val': {'count': 0, 'total_frames': 0},
        'total': {'count': 0, 'total_frames': 0}
    }

    # Process train and val splits
    for split in ['train', 'val']:
        split_dir = annotation_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found, skipping...")
            continue

        json_files = list(split_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    raw_length = data.get('raw_length', 0)
                    results[split]['total_frames'] += raw_length
                    results[split]['count'] += 1
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    # Calculate totals
    results['total']['count'] = results['train']['count'] + results['val']['count']
    results['total']['total_frames'] = results['train']['total_frames'] + results['val']['total_frames']

    # Calculate durations (15 frames = 1 second)
    for key in results:
        frames = results[key]['total_frames']
        results[key]['duration_seconds'] = frames / 15.0
        results[key]['duration_minutes'] = frames / 15.0 / 60.0
        results[key]['duration_hours'] = frames / 15.0 / 3600.0

    return results


def print_results(results: Dict[str, any], dataset_path: str):
    """Print formatted results."""
    print("=" * 80)
    print(f"Dataset Time Calculation for: {dataset_path}")
    print("=" * 80)
    print(f"Data collection rate: 15 Hz (15 frames = 1 second)\n")

    for split in ['train', 'val', 'total']:
        if results[split]['count'] == 0:
            continue

        print(f"{split.upper()} Split:")
        print(f"  Number of trajectories: {results[split]['count']}")
        print(f"  Total frames (raw_length): {results[split]['total_frames']:,}")
        print(f"  Duration: {results[split]['duration_seconds']:.2f} seconds")
        print(f"            {results[split]['duration_minutes']:.2f} minutes")
        print(f"            {results[split]['duration_hours']:.4f} hours")

        if results[split]['count'] > 0:
            avg_frames = results[split]['total_frames'] / results[split]['count']
            avg_seconds = avg_frames / 15.0
            print(f"  Average trajectory length: {avg_frames:.2f} frames ({avg_seconds:.2f} seconds)")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate dataset duration from annotation JSON files'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to the dataset directory (e.g., dataset_example/irom_1130_play_v0)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    try:
        results = calculate_dataset_time(args.dataset_path)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_results(results, args.dataset_path)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
