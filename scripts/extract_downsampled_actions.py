import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_downsampled_actions(dataset_path, downsample_factor=3):
    """
    Extract and downsample actions from a dataset.

    Args:
        dataset_path: Path to the dataset (e.g., 'dataset_example/irom_1126_all2')
        downsample_factor: Factor to downsample by (default: 3)

    Output:
        Saves downsampled actions to detector_data/{dataset_name}/actions/{trajectory_id}.json
        Each file contains:
            - "length": number of downsampled actions
            - "actions": downsampled action array
    """
    # Get dataset name from path
    dataset_name = os.path.basename(dataset_path.rstrip('/'))

    # Create output directory
    output_dir = f'detector_data/{dataset_name}/actions'
    os.makedirs(output_dir, exist_ok=True)

    # Find all annotation files (ignore train/test split)
    annotations_dir = os.path.join(dataset_path, 'annotation')

    # Get all JSON files from both train and val subdirectories
    annotation_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(annotations_dir, split)
        if os.path.exists(split_dir):
            json_files = list(Path(split_dir).glob('*.json'))
            annotation_files.extend(json_files)

    print(f"Found {len(annotation_files)} annotation files in {dataset_path}")

    # Process each annotation file
    for ann_file in tqdm(annotation_files, desc="Processing trajectories"):
        try:
            # Load annotation
            with open(ann_file, 'r') as f:
                label = json.load(f)

            # Extract cartesian and gripper positions
            cartesian_pose = np.array(label['observation.state.cartesian_position'])
            gripper_pose = np.array(label['observation.state.gripper_position'])

            # Add dimension to gripper_pose and concatenate
            gripper_pose = gripper_pose[..., np.newaxis]
            actions = np.concatenate((cartesian_pose, gripper_pose), axis=-1)

            # Downsample actions
            downsampled_actions = actions[::downsample_factor]

            # Get trajectory ID from filename
            traj_id = ann_file.stem  # filename without extension

            # Prepare output data
            output_data = {
                'length': len(downsampled_actions),
                'actions': downsampled_actions.tolist()
            }

            # Save to output file
            output_file = os.path.join(output_dir, f'{traj_id}.json')
            with open(output_file, 'w') as f:
                json.dump(output_data, f)

        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue

    print(f"Saved {len(annotation_files)} downsampled action files to {output_dir}")
    print(f"Sanity check example: 154 original frames → {len(range(0, 154, downsample_factor))} downsampled frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract downsampled actions from dataset')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset (e.g., dataset_example/irom_1126_all2)')
    parser.add_argument('--downsample_factor', type=int, default=3, help='Downsample factor (default: 3)')

    args = parser.parse_args()

    extract_downsampled_actions(args.dataset_path, args.downsample_factor)
