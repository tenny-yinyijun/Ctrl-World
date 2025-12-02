#!/usr/bin/env python3
"""
Script to generate annotation.json for detector data.

Usage:
    python generate_detector_annotation.py <folder_path>

Example:
    python generate_detector_annotation.py detector_data/irom_1126_all2
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List


def count_view2_images(images_folder: Path, trajectory_id: str) -> int:
    """Count the number of view2 images for a given trajectory."""
    trajectory_path = images_folder / trajectory_id
    if not trajectory_path.exists():
        return 0

    view2_images = list(trajectory_path.glob("*_view2.png"))
    return len(view2_images)


def load_metrics(metrics_folder: Path) -> tuple[Dict[str, float], Dict[str, float]]:
    """Load LPIPS and MSE metrics from JSON files."""
    lpips_path = metrics_folder / "lpips_view2.json"
    mse_path = metrics_folder / "mse_view2.json"

    with open(lpips_path, 'r') as f:
        lpips_data = json.load(f)

    with open(mse_path, 'r') as f:
        mse_data = json.load(f)

    return lpips_data, mse_data


def generate_annotation(folder_path: str) -> Dict:
    """Generate annotation data for the given folder."""
    base_path = Path(folder_path)
    images_folder = base_path / "images"
    actions_folder = base_path / "actions"
    metrics_folder = base_path / "metrics"

    # Load metrics
    print("Loading metrics...")
    lpips_data, mse_data = load_metrics(metrics_folder)

    # Get all trajectory IDs from images folder
    trajectory_ids = sorted(
        [d.name for d in images_folder.iterdir() if d.is_dir()],
        key=lambda x: int(x)
    )

    print(f"Found {len(trajectory_ids)} trajectories: {trajectory_ids}")

    # Build annotation structure
    trajectories = []

    for traj_id in trajectory_ids:
        print(f"Processing trajectory {traj_id}...")

        # Count samples
        num_samples = count_view2_images(images_folder, traj_id)

        # Build action path
        action_path = f"actions/{traj_id}.json"

        # Build samples list
        samples = []
        for sample_id in range(num_samples):
            # Calculate start_frame
            start_frame = 3 * sample_id

            # Get metrics
            metric_key = f"{traj_id}/{sample_id}_view2"
            lpips_score = lpips_data.get(metric_key)
            mse_score = mse_data.get(metric_key)

            # Build image path
            image_path = f"images/{traj_id}/{sample_id}_view2.png"

            sample_obj = {
                "sample_id": sample_id,
                "start_frame": start_frame,
                "lpips_score": lpips_score,
                "mse_score": mse_score,
                "image_path": image_path
            }
            samples.append(sample_obj)

        trajectory_obj = {
            "trajectory_id": int(traj_id),
            "num_samples": num_samples,
            "action_path": action_path,
            "samples": samples
        }
        trajectories.append(trajectory_obj)

    annotation = {
        "trajectories": trajectories
    }

    return annotation


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation.json for detector data"
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the detector data folder (e.g., detector_data/irom_1126_all2)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="annotation.json",
        help="Output filename (default: annotation.json)"
    )

    args = parser.parse_args()

    # Generate annotation
    annotation = generate_annotation(args.folder_path)

    # Write to file
    output_path = Path(args.folder_path) / args.output
    print(f"\nWriting annotation to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=2)

    print(f"Done! Generated annotation for {len(annotation['trajectories'])} trajectories.")

    # Print summary
    total_samples = sum(t['num_samples'] for t in annotation['trajectories'])
    print(f"Total samples across all trajectories: {total_samples}")


if __name__ == "__main__":
    main()
