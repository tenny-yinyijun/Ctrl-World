#!/usr/bin/env python3
"""
Video comparison script for evaluation results.
Creates a combined video showing ground truth and model predictions side-by-side
for easy visual comparison across multiple trajectories.
Usage:
python util_scripts/result_comparison.py evaluation_sanity_results/v0_dyn --output my_comparison.mp4
python util_scripts/result_comparison.py evaluation_sanity_results/v0_dyn --max-trajectories 5
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# CONFIGURATION: Modify this list to select which models to display
# ============================================================================
MODELS_TO_DISPLAY = [
    'groundtruth',
    '1201-demo-v0-ckpt45000',
    '1211-humanplay-ckpt70000',
    '1201-play400-v0-ckpt90000',
    '1211-play4000-v0-ckpt95000',
    # 'base_model'
]
# ============================================================================


def get_video_properties(video_path: str) -> Tuple[int, int, int, float]:
    """Get video properties: width, height, frame_count, fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return width, height, frame_count, fps


def read_video_frames(video_path: str, target_frames: int = None) -> List[np.ndarray]:
    """Read all frames from a video, optionally padding to target_frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Pad with last frame if needed
    if target_frames and len(frames) < target_frames:
        last_frame = frames[-1] if frames else np.zeros((192, 320, 3), dtype=np.uint8)
        while len(frames) < target_frames:
            frames.append(last_frame.copy())

    return frames


def add_label_to_frame(frame: np.ndarray, label: str, position: str = 'top') -> np.ndarray:
    """Add a label text to a frame."""
    labeled_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Position
    if position == 'top':
        text_x = 10
        text_y = text_height + 10
        rect_y1 = 0
        rect_y2 = text_height + baseline + 15
    else:  # bottom
        text_x = 10
        text_y = frame.shape[0] - 10
        rect_y1 = frame.shape[0] - text_height - baseline - 15
        rect_y2 = frame.shape[0]

    # Draw background rectangle
    cv2.rectangle(labeled_frame, (0, rect_y1), (text_width + 20, rect_y2), bg_color, -1)

    # Draw text
    cv2.putText(labeled_frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return labeled_frame


def create_grid_video(video_paths_dict: Dict[str, Dict[str, str]], output_path: str,
                      models: List[str], views: List[str], fps: float = 4.0):
    """
    Create a grid video from multiple video sources.

    Args:
        video_paths_dict: Dict mapping model -> view -> video_path
        output_path: Output path for the video
        models: List of model names (e.g., ['groundtruth', 'model1', 'model2'])
        views: List of view names (e.g., ['view0', 'view1', 'view2'])
        fps: Frames per second

    Layout: Each column = one model (top to bottom: view0, view1, view2)
            Each row = one view (left to right: models in order)
    """
    # Get max frame count across all videos
    max_frames = 0
    frame_width, frame_height = None, None

    for model in models:
        if model not in video_paths_dict:
            continue
        for view in views:
            if view in video_paths_dict[model]:
                video_path = video_paths_dict[model][view]
                if os.path.exists(video_path):
                    width, height, frame_count, _ = get_video_properties(video_path)
                    max_frames = max(max_frames, frame_count)
                    if frame_width is None:
                        frame_width, frame_height = width, height

    if max_frames == 0:
        print(f"No valid videos found for this trajectory")
        return False

    # Read all video frames
    # Structure: video_frames[model][view] = list of frames
    video_frames = {}
    for model in models:
        if model not in video_paths_dict:
            continue
        video_frames[model] = {}
        for view in views:
            if view in video_paths_dict[model]:
                video_path = video_paths_dict[model][view]
                if os.path.exists(video_path):
                    print(f"  Reading {model}/{view}...")
                    frames = read_video_frames(video_path, max_frames)
                    video_frames[model][view] = frames

    if not video_frames:
        return False

    # Grid layout: rows = views (3), cols = models
    rows = len(views)
    cols = len([m for m in models if m in video_frames])

    # Create grid video
    grid_width = frame_width * cols
    grid_height = frame_height * rows

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))

    print(f"  Creating grid video with {rows} views x {cols} models, {max_frames} frames...")

    for frame_idx in tqdm(range(max_frames), desc="  Writing frames", leave=False):
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        col_idx = 0
        for model in models:
            if model not in video_frames:
                continue

            for row_idx, view in enumerate(views):
                if view not in video_frames[model]:
                    # Fill with black frame if view is missing
                    continue

                frame = video_frames[model][view][frame_idx]

                # Add label only to the top row (view0)
                if row_idx == 0:
                    labeled_frame = add_label_to_frame(frame, model, 'top')
                else:
                    labeled_frame = frame

                y_start = row_idx * frame_height
                y_end = (row_idx + 1) * frame_height
                x_start = col_idx * frame_width
                x_end = (col_idx + 1) * frame_width

                grid_frame[y_start:y_end, x_start:x_end] = labeled_frame

            col_idx += 1

        out.write(grid_frame)

    out.release()
    return True


def discover_models(results_folder: Path) -> List[str]:
    """Discover all model folders in the results directory."""
    models = []
    for item in sorted(results_folder.iterdir()):
        if item.is_dir() and item.name != 'groundtruth':
            models.append(item.name)
    return models


def discover_trajectories(results_folder: Path) -> List[int]:
    """Discover all trajectory indices from the groundtruth folder."""
    groundtruth_folder = results_folder / 'groundtruth'
    if not groundtruth_folder.exists():
        raise ValueError(f"Groundtruth folder not found: {groundtruth_folder}")

    trajectories = []
    for item in sorted(groundtruth_folder.iterdir()):
        if item.is_dir() and item.name.isdigit():
            trajectories.append(int(item.name))

    return sorted(trajectories)


def main():
    parser = argparse.ArgumentParser(description='Create comparison video from evaluation results')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--max-trajectories', type=int, default=None,
                       help='Maximum number of trajectories to process (for debugging)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: <results_folder>_comparison.mp4)')
    parser.add_argument('--fps', type=float, default=4.0,
                       help='Output video FPS (default: 4.0)')

    args = parser.parse_args()

    results_folder = Path(args.results_folder)
    if not results_folder.exists():
        print(f"Error: Results folder not found: {results_folder}")
        sys.exit(1)

    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"{results_folder.name}_comparison.mp4"

    print(f"Processing results from: {results_folder}")
    print(f"Output will be saved to: {output_path}")

    # Use configured models to display
    models_to_use = MODELS_TO_DISPLAY
    views_to_use = ['view0', 'view1', 'view2']

    # Discover trajectories
    trajectories = discover_trajectories(results_folder)

    if args.max_trajectories:
        trajectories = trajectories[:args.max_trajectories]

    print(f"Models to display: {models_to_use}")
    print(f"Views to use: {views_to_use}")
    print(f"Found {len(trajectories)} trajectories: {trajectories}")

    # Create temporary folder for trajectory videos
    temp_folder = Path('temp_trajectory_videos')
    temp_folder.mkdir(exist_ok=True)

    trajectory_videos = []

    # Process each trajectory
    for traj_idx in trajectories:
        print(f"\nProcessing trajectory {traj_idx}...")

        # Collect video paths for this trajectory
        # Structure: video_paths_dict[model][view] = path
        video_paths_dict = {}

        for model in models_to_use:
            video_paths_dict[model] = {}

            # Determine the base folder (groundtruth has special path)
            if model == 'groundtruth':
                model_folder = results_folder / 'groundtruth' / str(traj_idx)
            else:
                model_folder = results_folder / model / str(traj_idx)

            # Collect all view videos
            for view in views_to_use:
                video_path = model_folder / f'{view}.mp4'
                if video_path.exists():
                    video_paths_dict[model][view] = str(video_path)

        # Create grid video for this trajectory
        traj_output = temp_folder / f'trajectory_{traj_idx:03d}.mp4'
        success = create_grid_video(video_paths_dict, str(traj_output),
                                   models_to_use, views_to_use, args.fps)

        if success:
            trajectory_videos.append(str(traj_output))

    # Concatenate all trajectory videos
    if trajectory_videos:
        print(f"\nConcatenating {len(trajectory_videos)} trajectory videos...")

        # Read first video to get properties
        first_cap = cv2.VideoCapture(trajectory_videos[0])
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_cap.release()

        # Create temporary output video
        temp_output = f"{output_path}.temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, args.fps, (width, height))

        for traj_video in tqdm(trajectory_videos, desc="Concatenating"):
            cap = cv2.VideoCapture(traj_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()

        out.release()

        # Re-encode with H.264 for better compatibility
        print("\nRe-encoding to H.264 for better compatibility...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_output,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            output_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"Final video saved to: {output_path}")

            # Remove temporary file
            os.remove(temp_output)
        except subprocess.CalledProcessError as e:
            print(f"Error during re-encoding: {e.stderr.decode()}")
            print(f"Temporary video saved to: {temp_output}")
            sys.exit(1)

        # Clean up temporary files
        print("Cleaning up temporary files...")
        for traj_video in trajectory_videos:
            os.remove(traj_video)
        temp_folder.rmdir()
    else:
        print("No trajectory videos were created!")
        sys.exit(1)

    print("Done!")


if __name__ == '__main__':
    main()
