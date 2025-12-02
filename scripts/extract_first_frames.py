#!/usr/bin/env python3
"""
Extract first frames from groundtruth videos in a dataset.

Usage:
    python extract_first_frames.py <dataset_path>

Example:
    python extract_first_frames.py detector_data/irom_1126_all2
"""

import os
import sys
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video and save as PNG."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read first frame from {video_path}")
        return False

    # Save the frame as PNG
    cv2.imwrite(str(output_path), frame)
    return True


def process_dataset(dataset_path):
    """Process the entire dataset and extract first frames."""
    # Resolve to absolute path (handles both relative and absolute paths)
    dataset_path = Path(dataset_path).resolve()
    videos_dir = dataset_path / "videos"
    images_dir = dataset_path / "images"

    if not videos_dir.exists():
        print(f"Error: videos directory not found at {videos_dir}")
        return

    print(f"Processing dataset at: {dataset_path}")

    # Create images directory
    images_dir.mkdir(exist_ok=True)
    print(f"Created images directory at {images_dir}")

    # Get all numbered subdirectories in videos/
    video_subdirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()],
                          key=lambda x: int(x.name))

    total_videos = 0
    for subdir in video_subdirs:
        groundtruth_dir = subdir / "groundtruth"
        if not groundtruth_dir.exists():
            print(f"Warning: No groundtruth directory found in {subdir}")
            continue

        # Count videos for this subdir
        video_files = list(groundtruth_dir.glob("*.mp4"))
        total_videos += len(video_files)

    print(f"Found {len(video_subdirs)} subdirectories with {total_videos} groundtruth videos")

    # Process each subdirectory
    processed = 0
    with tqdm(total=total_videos, desc="Extracting frames") as pbar:
        for subdir in video_subdirs:
            groundtruth_dir = subdir / "groundtruth"
            if not groundtruth_dir.exists():
                continue

            # Create corresponding images subdirectory
            images_subdir = images_dir / subdir.name
            images_subdir.mkdir(exist_ok=True)

            # Process all groundtruth videos
            for video_file in sorted(groundtruth_dir.glob("*.mp4")):
                # Output filename: change .mp4 to .png
                output_filename = video_file.stem + ".png"
                output_path = images_subdir / output_filename

                # Extract first frame
                if extract_first_frame(video_file, output_path):
                    processed += 1

                pbar.update(1)

    print(f"\nSuccessfully processed {processed}/{total_videos} videos")
    print(f"First frames saved to {images_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_first_frames.py <dataset_path>")
        print("Example: python extract_first_frames.py detector_data/irom_1126_all2")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Check if path exists (works with both relative and absolute paths)
    if not Path(dataset_path).exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    process_dataset(dataset_path)
