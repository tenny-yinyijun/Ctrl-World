#!/usr/bin/env python3
"""
Batch inference script for evaluating multiple models on multiple datasets.
"""

import os
import sys
from argparse import ArgumentParser
from typing import List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.rollout_single_model import run_inference
from eval.utils import load_model_registry


def batch_inference(model_aliases: List[str],
                   dataset_dirs: List[str],
                   registry_path: str = "model_registry.json",
                   output_base_dir: str = "evaluation_inf_results",
                   start_idx: int = 0,
                   max_trajectories: Optional[int] = None,
                   gripper_annotation: bool = False,
                   downsampled: bool = False,
                   skip_existing: bool = True):
    """
    Run batch inference for multiple models on multiple datasets.

    Args:
        model_aliases: List of model aliases to evaluate
        dataset_dirs: List of dataset directories to evaluate on
        registry_path: Path to model_registry.json
        output_base_dir: Base directory for evaluation results
        start_idx: Starting frame index for each trajectory
        max_trajectories: Maximum number of trajectories per dataset
        gripper_annotation: Whether to annotate gripper values on frames
        downsampled: Whether input is already downsampled to 5Hz
        skip_existing: Skip if output directory already exists
    """
    # Load model registry to validate
    print("Loading model registry...")
    registry = load_model_registry(registry_path)
    print(f"Found {len(registry)} models in registry: {list(registry.keys())}")

    # Validate model aliases
    for alias in model_aliases:
        if alias not in registry:
            raise ValueError(f"Model '{alias}' not found in registry")

    # Validate dataset directories
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

    total_runs = len(model_aliases) * len(dataset_dirs)
    print(f"\n{'='*80}")
    print(f"BATCH INFERENCE")
    print(f"{'='*80}")
    print(f"Models to evaluate: {model_aliases}")
    print(f"Datasets to evaluate: {[os.path.basename(d) for d in dataset_dirs]}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*80}\n")

    completed_runs = 0
    skipped_runs = 0
    failed_runs = 0

    for model_idx, model_alias in enumerate(model_aliases, 1):
        for dataset_idx, dataset_dir in enumerate(dataset_dirs, 1):
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            run_name = f"[{completed_runs + skipped_runs + failed_runs + 1}/{total_runs}] {model_alias} on {dataset_name}"

            print(f"\n{'='*80}")
            print(f"Starting: {run_name}")
            print(f"{'='*80}")

            # Check if output directory already exists
            model_output_dir = os.path.join(output_base_dir, dataset_name, model_alias)
            if skip_existing and os.path.exists(model_output_dir):
                print(f"SKIPPED: Output directory already exists: {model_output_dir}")
                skipped_runs += 1
                continue

            try:
                run_inference(
                    model_alias=model_alias,
                    dataset_dir=dataset_dir,
                    registry_path=registry_path,
                    output_base_dir=output_base_dir,
                    start_idx=start_idx,
                    max_trajectories=max_trajectories,
                    gripper_annotation=gripper_annotation,
                    downsampled=downsampled
                )
                completed_runs += 1
                print(f"COMPLETED: {run_name}")

            except Exception as e:
                print(f"FAILED: {run_name}")
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_runs += 1
                continue

    # Print final summary
    print(f"\n{'='*80}")
    print(f"BATCH INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed_runs}")
    print(f"Skipped: {skipped_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Batch inference for multiple models and datasets")

    # Model and dataset selection
    parser.add_argument('--model_aliases', type=str, nargs='+', required=True,
                       help='List of model aliases from model_registry.json')
    parser.add_argument('--dataset_dirs', type=str, nargs='+', required=True,
                       help='List of dataset directories')

    # Configuration
    parser.add_argument('--registry_path', type=str, default='model_registry.json',
                       help='Path to model_registry.json')
    parser.add_argument('--output_base_dir', type=str, default='evaluation_inf_results',
                       help='Base directory for evaluation results')

    # Processing options
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting frame index for each trajectory')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to process per dataset')
    parser.add_argument('--gripper_annotation', action='store_true',
                       help='Annotate gripper values on frames')
    parser.add_argument('--downsampled', action='store_true',
                       help='Input is already downsampled to 5Hz')
    parser.add_argument('--no_skip_existing', action='store_true',
                       help='Do not skip existing output directories')

    args = parser.parse_args()

    batch_inference(
        model_aliases=args.model_aliases,
        dataset_dirs=args.dataset_dirs,
        registry_path=args.registry_path,
        output_base_dir=args.output_base_dir,
        start_idx=args.start_idx,
        max_trajectories=args.max_trajectories,
        gripper_annotation=args.gripper_annotation,
        downsampled=args.downsampled,
        skip_existing=not args.no_skip_existing
    )
