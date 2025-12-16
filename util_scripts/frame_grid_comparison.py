#!/usr/bin/env python3
"""
Frame grid comparison script.
Creates a summary image with N rows (one per folder) and M columns (uniformly sampled frames).
No labels or text - just a clean grid of frames.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

# ============================================================================
# CONFIGURATION: Modify these parameters
# ============================================================================
FOLDER_PATHS = [
    # '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/groundtruth/6',
    # '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/1201-demo-v0-ckpt45000/6',
    # '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/1201-play400-v0-ckpt90000/6',
    
    # '/n/fs/worldmodeliw/ctrlworld/evaluation_inf_results_1/eval_v0_dyn/groundtruth/15',
    # '/n/fs/worldmodeliw/ctrlworld/evaluation_inf_results_1/eval_v0_dyn/1201-play400-v0-ckpt90000/15',
    # '/n/fs/worldmodeliw/ctrlworld/evaluation_inf_results_1/eval_v0_dyn/1201-demo-v0-ckpt45000/15',

    '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/groundtruth/10',
    '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/1201-demo-v0-ckpt45000/10',
    '/n/fs/tom-project/video_models/Ctrl-World/evaluation_sanity_results_old/v0_dyn/1201-play400-v0-ckpt90000/10'
    
    
]

VIEW_NAME = 'view2'  # e.g., 'view0', 'view1', 'view2'
NUM_FRAMES = 10  # Number of frames to sample from each video
OUTPUT_PATH = 'frame_comparison_2.png'  # Output image path
# ============================================================================


def get_video_path(folder: str, view_name: str) -> str:
    """Find the video file for a given view in a folder."""
    folder_path = Path(folder)

    # Try common video extensions and naming patterns
    possible_names = [
        f'{view_name}.mp4',
        f'{view_name}.avi',
        f'*_{view_name}.mp4',
    ]

    for name in possible_names:
        if '*' in name:
            matches = list(folder_path.glob(name))
            if matches:
                return str(matches[0])
        else:
            video_path = folder_path / name
            if video_path.exists():
                return str(video_path)

    raise FileNotFoundError(f"No video found for {view_name} in {folder}")


def sample_frames_uniformly(video_path: str, num_samples: int) -> List[np.ndarray]:
    """
    Sample frames uniformly from a video.

    Args:
        video_path: Path to the video file
        num_samples: Number of frames to sample

    Returns:
        List of sampled frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame indices to sample uniformly
    if num_samples == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # If we can't read the frame, use a black frame
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    cap.release()
    return frames


def create_frame_grid(folders: List[str], view_name: str,
                     num_frames: int, output_path: str):
    """
    Create a grid image from uniformly sampled frames.

    Args:
        folders: List of folder paths
        view_name: Name of the view to extract (e.g., 'view2')
        num_frames: Number of frames to sample from each video
        output_path: Path to save the output image
    """
    if len(folders) == 0:
        raise ValueError("Expected at least one folder")

    print(f"Creating frame grid comparison:")
    print(f"  View: {view_name}")
    print(f"  Frames per video: {num_frames}")
    print(f"  Output: {output_path}")

    # Collect frames from all videos
    all_frames = []  # List of lists: all_frames[folder_idx][frame_idx]
    num_rows = len(folders)

    for i, folder in enumerate(folders):
        print(f"\nProcessing folder {i+1}/{num_rows}")
        print(f"  Path: {folder}")

        # Find video file
        video_path = get_video_path(folder, view_name)
        print(f"  Video: {video_path}")

        # Sample frames
        frames = sample_frames_uniformly(video_path, num_frames)
        print(f"  Sampled {len(frames)} frames")
        all_frames.append(frames)

    # Get frame dimensions (assume all frames have the same size)
    frame_height, frame_width = all_frames[0][0].shape[:2]

    # Create the output grid
    # Grid dimensions: num_rows x num_frames
    grid_height = frame_height * num_rows
    grid_width = frame_width * num_frames

    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Fill in the frames
    for row_idx in range(num_rows):
        y_start = row_idx * frame_height
        y_end = y_start + frame_height

        for col_idx in range(num_frames):
            frame = all_frames[row_idx][col_idx]
            x_start = col_idx * frame_width
            x_end = x_start + frame_width

            grid[y_start:y_end, x_start:x_end] = frame

    # Save the output
    cv2.imwrite(output_path, grid)
    print(f"\n✓ Grid image saved to: {output_path}")
    print(f"  Dimensions: {grid_width}x{grid_height}")


def main():
    """Main function."""
    create_frame_grid(
        folders=FOLDER_PATHS,
        view_name=VIEW_NAME,
        num_frames=NUM_FRAMES,
        output_path=OUTPUT_PATH
    )


if __name__ == '__main__':
    main()
