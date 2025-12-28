#!/usr/bin/env python3
"""
Utility functions for evaluation inference pipeline.
Handles video separation, directory management, and metadata generation.
"""

import os
import json
import numpy as np
import mediapy
from datetime import datetime
from typing import List, Dict, Tuple


def separate_views_from_concatenated(videos_cat: np.ndarray, num_views: int = 3) -> List[np.ndarray]:
    """
    Separate concatenated multi-view video into individual view videos.

    Args:
        videos_cat: Concatenated video array of shape (num_frames, height, width, 3)
                   where height contains all views stacked vertically
        num_views: Number of views (default: 3)

    Returns:
        List of video arrays, one per view
    """
    num_frames, total_height, width, channels = videos_cat.shape
    view_height = total_height // num_views

    views = []
    for i in range(num_views):
        start_h = i * view_height
        end_h = (i + 1) * view_height
        view_video = videos_cat[:, start_h:end_h, :, :]
        views.append(view_video)

    return views


def separate_prediction_and_groundtruth(videos_cat: np.ndarray, num_views: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Separate concatenated video that has both groundtruth and prediction.

    Layout:
        Row 1: [GT_view0][GT_view1][GT_view2]
        Row 2: [Pred_v0][Pred_v1][Pred_v2]

    Args:
        videos_cat: Concatenated video of shape (num_frames, height, width, 3)
                   where height has [groundtruth on top | prediction on bottom]
                   and width has [view0 | view1 | view2] side by side
        num_views: Number of views (default: 3)

    Returns:
        Tuple of (groundtruth_views, prediction_views), each a list of num_views videos
    """
    num_frames, total_height, total_width, channels = videos_cat.shape
    half_height = total_height // 2
    view_width = total_width // num_views

    gt_views = []
    pred_views = []

    # Split width into views, then split each view's height into GT/pred
    for i in range(num_views):
        start_w = i * view_width
        end_w = (i + 1) * view_width

        # Extract this view (both GT and pred)
        view = videos_cat[:, :, start_w:end_w, :]

        # Split into GT (top half) and prediction (bottom half)
        gt_view = view[:, :half_height, :, :]
        pred_view = view[:, half_height:, :, :]

        gt_views.append(gt_view)
        pred_views.append(pred_view)

    return gt_views, pred_views


def setup_output_directories(dataset_name: str, model_alias: str,
                            base_dir: str = "evaluation_inf_results") -> Dict[str, str]:
    """
    Create output directory structure for a model-dataset combination.

    Args:
        dataset_name: Name of the dataset (e.g., 'eval_random')
        model_alias: Alias of the model (e.g., 'base_model')
        base_dir: Base directory for evaluation results

    Returns:
        Dictionary with paths: 'dataset_dir', 'model_dir', 'groundtruth_dir'
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    model_dir = os.path.join(dataset_dir, model_alias)
    groundtruth_dir = os.path.join(dataset_dir, "groundtruth")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    return {
        'dataset_dir': dataset_dir,
        'model_dir': model_dir,
        'groundtruth_dir': groundtruth_dir
    }


def save_separate_views(videos_cat: np.ndarray, output_base_dir: str,
                       trajectory_id: str, fps: int = 4,
                       save_groundtruth: bool = True,
                       groundtruth_dir: str = None) -> Dict[str, List[str]]:
    """
    Save videos as separate view files.

    Args:
        videos_cat: Concatenated video array (num_frames, height, width, 3)
        output_base_dir: Base directory for saving prediction videos
        trajectory_id: Trajectory ID (e.g., '0', '1', '2')
        fps: Frames per second for video
        save_groundtruth: Whether to save groundtruth videos
        groundtruth_dir: Directory to save groundtruth (if save_groundtruth=True)

    Returns:
        Dictionary with 'prediction_paths' and 'groundtruth_paths' lists
    """
    # Separate groundtruth and prediction
    gt_views, pred_views = separate_prediction_and_groundtruth(videos_cat, num_views=3)

    # Save prediction views
    pred_traj_dir = os.path.join(output_base_dir, str(trajectory_id))
    os.makedirs(pred_traj_dir, exist_ok=True)

    prediction_paths = []
    for view_idx, view_video in enumerate(pred_views):
        video_path = os.path.join(pred_traj_dir, f"view{view_idx}.mp4")
        mediapy.write_video(video_path, view_video, fps=fps)
        prediction_paths.append(video_path)

    # Save groundtruth views (if enabled and directory provided)
    groundtruth_paths = []
    if save_groundtruth and groundtruth_dir is not None:
        gt_traj_dir = os.path.join(groundtruth_dir, str(trajectory_id))
        os.makedirs(gt_traj_dir, exist_ok=True)

        for view_idx, view_video in enumerate(gt_views):
            video_path = os.path.join(gt_traj_dir, f"view{view_idx}.mp4")
            # Only save if doesn't exist (groundtruth is shared across models)
            if not os.path.exists(video_path):
                mediapy.write_video(video_path, view_video, fps=fps)
            groundtruth_paths.append(video_path)

    return {
        'prediction_paths': prediction_paths,
        'groundtruth_paths': groundtruth_paths
    }


def save_run_metadata(model_dir: str, metadata: Dict) -> str:
    """
    Save metadata about a model's run on a dataset.

    Args:
        model_dir: Directory where the model's outputs are stored
        metadata: Dictionary containing run metadata

    Returns:
        Path to the saved metadata file
    """
    metadata_path = os.path.join(model_dir, "run_metadata.json")

    # Add timestamp if not present
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().isoformat()

    with open(metadata_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)

    return metadata_path


def load_model_registry(registry_path: str = "model_registry.json") -> Dict:
    """
    Load the model registry JSON file.

    Args:
        registry_path: Path to model_registry.json

    Returns:
        Dictionary of model configurations
    """
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Model registry not found at {registry_path}")

    with open(registry_path, 'r') as f:
        return json.load(f)


def get_model_config(model_alias: str, registry_path: str = "model_registry.json") -> Dict:
    """
    Get configuration for a specific model from the registry.

    Args:
        model_alias: Alias of the model to retrieve
        registry_path: Path to model_registry.json

    Returns:
        Dictionary containing model configuration
    """
    registry = load_model_registry(registry_path)

    if model_alias not in registry:
        raise ValueError(f"Model '{model_alias}' not found in registry. Available models: {list(registry.keys())}")

    return registry[model_alias]


def check_groundtruth_exists(groundtruth_dir: str, trajectory_id: str, num_views: int = 3) -> bool:
    """
    Check if groundtruth videos already exist for a trajectory.

    Args:
        groundtruth_dir: Directory containing groundtruth videos
        trajectory_id: Trajectory ID to check
        num_views: Number of views to check for

    Returns:
        True if all groundtruth views exist, False otherwise
    """
    traj_dir = os.path.join(groundtruth_dir, str(trajectory_id))
    if not os.path.exists(traj_dir):
        return False

    for view_idx in range(num_views):
        video_path = os.path.join(traj_dir, f"view{view_idx}.mp4")
        if not os.path.exists(video_path):
            return False

    return True


def is_v2_dataset(dataset_path: str) -> bool:
    """
    Check if a dataset follows the v2 format with test cases.

    V2 datasets have:
    - A manifest.json file at the root
    - Multiple test case subdirectories

    Args:
        dataset_path: Path to dataset directory

    Returns:
        True if dataset is v2 format, False otherwise
    """
    manifest_path = os.path.join(dataset_path, "manifest.json")
    return os.path.exists(manifest_path)


def load_dataset_manifest(dataset_path: str) -> Dict:
    """
    Load manifest.json from a v2 dataset.

    Args:
        dataset_path: Path to v2 dataset directory

    Returns:
        Dictionary containing manifest data

    Raises:
        FileNotFoundError: If manifest.json doesn't exist
    """
    manifest_path = os.path.join(dataset_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, 'r') as f:
        return json.load(f)


def get_test_cases(dataset_path: str) -> List[str]:
    """
    Get list of test case names from a v2 dataset's manifest.

    Args:
        dataset_path: Path to v2 dataset directory

    Returns:
        List of test case names (e.g., ['deformable', 'miss'])

    Raises:
        FileNotFoundError: If manifest.json doesn't exist
    """
    manifest = load_dataset_manifest(dataset_path)
    return list(manifest.get('test_cases', {}).keys())


def get_test_case_path(dataset_path: str, test_case: str) -> str:
    """
    Get full path to a test case subdirectory.

    Args:
        dataset_path: Path to v2 dataset directory
        test_case: Name of test case

    Returns:
        Full path to test case directory
    """
    return os.path.join(dataset_path, test_case)
