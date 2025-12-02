#!/usr/bin/env python3
"""
Single model inference script for evaluation.
Processes all trajectories in a dataset and saves separate view videos.
"""

import os
import sys
import json
import glob
from argparse import ArgumentParser
from tqdm.auto import tqdm

# Add parent directory to path to import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing functions from scripts/rollout_replay_dataset.py
from scripts.rollout_replay_dataset import agent, process_single_trajectory

# Import evaluation utilities
from eval.utils import (
    setup_output_directories,
    save_separate_views,
    save_run_metadata,
    get_model_config,
    check_groundtruth_exists
)


def run_inference(model_alias: str, dataset_dir: str, registry_path: str = "model_registry.json",
                 output_base_dir: str = "evaluation_inf_results",
                 start_idx: int = 0, max_trajectories: int = None,
                 gripper_annotation: bool = False, downsampled: bool = False):
    """
    Run inference for a single model on a dataset.

    Args:
        model_alias: Alias of the model from model_registry.json
        dataset_dir: Path to dataset directory
        registry_path: Path to model_registry.json
        output_base_dir: Base directory for evaluation results
        start_idx: Starting frame index for each trajectory
        max_trajectories: Maximum number of trajectories to process
        gripper_annotation: Whether to annotate gripper values on frames
        downsampled: Whether input is already downsampled to 5Hz
    """
    # Load model configuration from registry
    print(f"\nLoading model configuration for '{model_alias}'...")
    model_config = get_model_config(model_alias, registry_path)
    print(f"Model mode: {model_config['mode']}")
    print(f"Checkpoint: {model_config['checkpoint_path']}")

    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))

    # Setup output directories
    print(f"\nSetting up output directories...")
    dirs = setup_output_directories(dataset_name, model_alias, output_base_dir)
    print(f"Model output: {dirs['model_dir']}")
    print(f"Groundtruth output: {dirs['groundtruth_dir']}")

    # Import config based on model mode
    if model_config['mode'] == 'lora':
        from droid_inference_config_lora import wm_args
    else:
        from droid_inference_config import wm_args

    # Setup arguments
    args = wm_args(task_type='replay')

    # Set checkpoint path and related parameters
    args.ckpt_path = model_config['checkpoint_path']
    args.val_model_path = model_config['checkpoint_path']
    args.val_dataset_dir = dataset_dir
    args.start_idx = start_idx
    args.downsampled = downsampled
    args.gripper_annotation = gripper_annotation

    # Set LoRA-specific parameters
    if model_config['mode'] == 'lora':
        args.use_lora = True
        if 'base_checkpoint_path' in model_config:
            args.base_ckpt_path = model_config['base_checkpoint_path']
        if 'lora_target_modules' in model_config:
            args.lora_target_modules = model_config['lora_target_modules']

    # Override SVD and CLIP paths if specified in config
    if 'svd_model_path' in model_config and model_config['svd_model_path']:
        args.svd_model_path = model_config['svd_model_path']
    if 'clip_model_path' in model_config and model_config['clip_model_path']:
        args.clip_model_path = model_config['clip_model_path']

    # Find all trajectories in the dataset
    annotation_files = []
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, 'annotation', split)
        if os.path.exists(split_dir):
            split_files = glob.glob(os.path.join(split_dir, '*.json'))
            for f in split_files:
                traj_id = os.path.basename(f).replace('.json', '')
                annotation_files.append((split, traj_id))

    # Sort by trajectory ID
    annotation_files.sort(key=lambda x: int(x[1]))

    # Limit trajectories if specified
    if max_trajectories is not None:
        annotation_files = annotation_files[:max_trajectories]

    print(f"\nFound {len(annotation_files)} trajectories to process")
    print(f"Trajectory IDs: {[traj_id for _, traj_id in annotation_files]}")

    # Create agent
    print(f"\nInitializing model...")
    Agent = agent(args)

    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames

    # Process each trajectory
    successful_count = 0
    failed_count = 0
    failed_trajectories = []

    for split, val_id_i in tqdm(annotation_files, desc="Processing trajectories"):
        try:
            # Get trajectory length and calculate interact_num
            annotation_path = f"{dataset_dir}/annotation/{split}/{val_id_i}.json"
            with open(annotation_path) as f:
                anno = json.load(f)
                if 'observation.state.cartesian_position' in anno:
                    total_length = len(anno['observation.state.cartesian_position'])
                elif 'action' in anno:
                    total_length = len(anno['action'])
                else:
                    total_length = anno["video_length"]

            # Calculate interact_num
            downsample_factor = 1 if args.downsampled else 3
            actual_skip = args.skip_step * downsample_factor
            available_frames = (total_length - args.start_idx) // actual_skip
            history_frames = 8
            interact_num = max(1, (available_frames - history_frames) // (pred_step - 1) - 1)

            print(f"\nTrajectory {val_id_i} ({split}): total_length={total_length}, "
                  f"available_frames={available_frames}, interact_num={interact_num}")

            # Run inference on single trajectory
            video_cat, instruction = process_single_trajectory(
                Agent, val_id_i, args.start_idx, interact_num, pred_step, args
            )

            # Check if groundtruth already exists for this trajectory
            save_gt = not check_groundtruth_exists(dirs['groundtruth_dir'], val_id_i)

            # Save separate view videos
            saved_paths = save_separate_views(
                video_cat,
                output_base_dir=dirs['model_dir'],
                trajectory_id=val_id_i,
                fps=4,
                save_groundtruth=save_gt,
                groundtruth_dir=dirs['groundtruth_dir']
            )

            print(f"Saved prediction videos: {saved_paths['prediction_paths']}")
            if save_gt:
                print(f"Saved groundtruth videos: {saved_paths['groundtruth_paths']}")
            else:
                print(f"Groundtruth already exists, skipped saving")

            successful_count += 1

        except Exception as e:
            print(f"ERROR processing trajectory {val_id_i}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            failed_trajectories.append(val_id_i)
            continue

    # Save run metadata
    metadata = {
        'model_alias': model_alias,
        'model_config': model_config,
        'dataset_name': dataset_name,
        'dataset_dir': dataset_dir,
        'total_trajectories': len(annotation_files),
        'successful': successful_count,
        'failed': failed_count,
        'failed_trajectories': failed_trajectories,
        'args': {
            'start_idx': start_idx,
            'downsampled': downsampled,
            'gripper_annotation': gripper_annotation,
            'max_trajectories': max_trajectories
        }
    }
    metadata_path = save_run_metadata(dirs['model_dir'], metadata)

    # Print summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful_count}/{len(annotation_files)}")
    print(f"Failed: {failed_count}/{len(annotation_files)}")
    if failed_trajectories:
        print(f"Failed trajectories: {failed_trajectories}")
    print(f"\nPrediction videos saved to: {dirs['model_dir']}")
    print(f"Groundtruth videos saved to: {dirs['groundtruth_dir']}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference for a single model on a dataset")
    parser.add_argument('--model_alias', type=str, required=True,
                       help='Model alias from model_registry.json')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--registry_path', type=str, default='model_registry.json',
                       help='Path to model_registry.json')
    parser.add_argument('--output_base_dir', type=str, default='evaluation_inf_results',
                       help='Base directory for evaluation results')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting frame index for each trajectory')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to process')
    parser.add_argument('--gripper_annotation', action='store_true',
                       help='Annotate gripper values on frames')
    parser.add_argument('--downsampled', action='store_true',
                       help='Input is already downsampled to 5Hz')

    args = parser.parse_args()

    run_inference(
        model_alias=args.model_alias,
        dataset_dir=args.dataset_dir,
        registry_path=args.registry_path,
        output_base_dir=args.output_base_dir,
        start_idx=args.start_idx,
        max_trajectories=args.max_trajectories,
        gripper_annotation=args.gripper_annotation,
        downsampled=args.downsampled
    )
