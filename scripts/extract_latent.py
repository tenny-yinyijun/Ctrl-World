import mediapy
import os
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
import torch
import numpy as np
import json
import h5py
from torch.utils.data import Dataset
from accelerate import Accelerator
from pathlib import Path


class EncodeLatentDatasetIROM(Dataset):
    def __init__(self, irom_data_path, new_path, svd_path, device, dataset_type='demo',
                 size=(192, 320), rgb_skip=3, skip_start_frames=0):
        """
        Dataset for encoding IROM trajectory data to latent space.

        Args:
            irom_data_path: Path to IROM data directory
            new_path: Output path for processed data
            svd_path: Path to SVD model
            device: Device to run VAE on
            dataset_type: Type of dataset - 'demo' or 'play'
            size: Target video size (height, width)
            rgb_skip: Frame skip for downsampling video
            skip_start_frames: Number of frames to skip at the start to remove idle frames (demo only)
        """
        self.irom_data_path = irom_data_path
        self.new_path = new_path
        self.dataset_type = dataset_type
        self.size = size
        self.skip = rgb_skip
        self.skip_start_frames = skip_start_frames if dataset_type == 'demo' else 0
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        # Collect all episode directories
        self.episodes = []
        irom_dir = Path(irom_data_path)

        if dataset_type == 'demo':
            self._collect_demo_episodes(irom_dir)
        elif dataset_type == 'play':
            self._collect_play_episodes(irom_dir)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'demo' or 'play'")

        print(f"Found {len(self.episodes)} episodes in {irom_data_path} (type: {dataset_type})")

    def _collect_demo_episodes(self, irom_dir):
        """Collect demo data episodes (with metadata files)."""
        for episode_dir in sorted(irom_dir.iterdir()):
            if episode_dir.is_dir():
                traj_file = episode_dir / "trajectory.h5"
                metadata_files = list(episode_dir.glob("metadata_*.json"))
                if traj_file.exists() and len(metadata_files) > 0:
                    self.episodes.append({
                        'dir': episode_dir,
                        'traj_file': traj_file,
                        'metadata_file': metadata_files[0]
                    })

    def _collect_play_episodes(self, irom_dir):
        """Collect play data episodes (with video files)."""
        for episode_dir in sorted(irom_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
            if episode_dir.is_dir():
                traj_file = episode_dir / "trajectory.h5"
                left_base_cam = episode_dir / "left_base_cam.mp4"
                right_base_cam = episode_dir / "right_base_cam.mp4"
                wrist_cam = episode_dir / "wrist_cam.mp4"

                if traj_file.exists() and left_base_cam.exists() and right_base_cam.exists() and wrist_cam.exists():
                    self.episodes.append({
                        'dir': episode_dir,
                        'traj_file': traj_file,
                        'left_base_cam': left_base_cam,
                        'right_base_cam': right_base_cam,
                        'wrist_cam': wrist_cam
                    })

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]

        if self.dataset_type == 'play':
            print(f"Processing episode {idx}: {episode['dir'].name}")

        traj_id = idx  # Use index as trajectory ID
        data_type = 'val' if traj_id % 10 == 9 else 'train'  # 10% val split

        try:
            if self.dataset_type == 'demo':
                self._process_demo_episode(episode, traj_id, data_type)
            else:  # play
                self._process_play_episode(episode, traj_id, data_type)

            if self.dataset_type == 'play':
                print(f"Successfully processed episode {traj_id}")
        except Exception as e:
            print(f"Error processing trajectory {traj_id} ({episode['dir'].name}): {e}")
            import traceback
            traceback.print_exc()
            return 0

        return 0

    def _process_demo_episode(self, episode, traj_id, data_type):
        """Process a demo data episode."""
        # Load metadata
        with open(episode['metadata_file'], 'r') as f:
            metadata = json.load(f)

        instruction = metadata.get('current_task', 'No task description')
        success = metadata.get('success', False)

        # Load trajectory data from HDF5
        with h5py.File(episode['traj_file'], 'r') as f:
            obs_car = f['observation/robot_state/cartesian_position'][:]
            obs_joint = f['observation/robot_state/joint_positions'][:]
            obs_gripper = f['observation/robot_state/gripper_position'][:]
            action_car = f['action/cartesian_position'][:]
            action_joint = f['action/joint_position'][:]
            action_gripper = f['action/gripper_position'][:]
            action_joint_vel = f['action/joint_velocity'][:]

        # Get video paths from metadata
        wrist_cam_serial = metadata.get('wrist_cam_serial', '10501775')
        ext1_cam_serial = metadata.get('ext1_cam_serial', '31177322')
        ext2_cam_serial = metadata.get('ext2_cam_serial', '38872458')

        video_paths = [
            str(episode['dir'] / 'recordings' / 'MP4' / f'{ext2_cam_serial}.mp4'),  # exterior_1_left
            str(episode['dir'] / 'recordings' / 'MP4' / f'{ext1_cam_serial}.mp4'),  # exterior_2_left
            str(episode['dir'] / 'recordings' / 'MP4' / f'{wrist_cam_serial}.mp4')  # wrist_left
        ]

        traj_info = {
            'success': success,
            'observation.state.cartesian_position': obs_car.tolist(),
            'observation.state.joint_position': obs_joint.tolist(),
            'observation.state.gripper_position': obs_gripper.tolist(),
            'action.cartesian_position': action_car.tolist(),
            'action.joint_position': action_joint.tolist(),
            'action.gripper_position': action_gripper.tolist(),
            'action.joint_velocity': action_joint_vel.tolist(),
        }

        self.process_traj(
            video_paths,
            traj_info,
            instruction,
            self.new_path,
            traj_id=traj_id,
            data_type=data_type,
            size=self.size,
            rgb_skip=self.skip,
            skip_start_frames=self.skip_start_frames,
            device=self.vae.device,
            dataset_type='demo'
        )

    def _process_play_episode(self, episode, traj_id, data_type):
        """Process a play data episode."""
        instruction = "Play data demonstration"
        success = True

        # Load trajectory data from HDF5
        with h5py.File(episode['traj_file'], 'r') as f:
            cartesian_position = f['data/cartesian_position'][:]
            joint_position = f['data/joint_position'][:]
            gripper_position = f['data/gripper_position'][:].flatten()
            action = f['data/action'][:]

        # Video paths for play data format (note: order is reversed - right before left)
        video_paths = [
            str(episode['right_base_cam']),  # index 0
            str(episode['left_base_cam']),   # index 1
            str(episode['wrist_cam'])        # index 2
        ]

        traj_info = {
            'success': success,
            'observation.state.cartesian_position': cartesian_position.tolist(),
            'observation.state.joint_position': joint_position.tolist(),
            'observation.state.gripper_position': gripper_position.tolist(),
            'action': action.tolist(),
        }

        self.process_traj(
            video_paths,
            traj_info,
            instruction,
            self.new_path,
            traj_id=traj_id,
            data_type=data_type,
            size=self.size,
            rgb_skip=self.skip,
            skip_start_frames=0,
            device=self.vae.device,
            dataset_type='play'
        )

    def process_traj(self, video_paths, traj_info, instruction, save_root, traj_id=0,
                     data_type='val', size=(192, 320), rgb_skip=3, skip_start_frames=0,
                     device='cuda', dataset_type='demo'):
        """
        Process a single trajectory: resize videos, encode to latents, save annotations.

        Args:
            skip_start_frames: Number of frames to skip at the start to remove idle frames
            dataset_type: Type of dataset - 'demo' or 'play'
        """
        for video_id, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            # Load and resize video and save
            video = mediapy.read_video(video_path)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
            frames = frames[skip_start_frames:]  # Skip idle frames at the start
            frames = frames[::rgb_skip]  # Skip frames to save memory
            x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=5)

            # Save SVD latent
            x = x.to(device)
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")

        # Record cartesian aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_gripper = np.array(traj_info['observation.state.gripper_position'])

        # Ensure gripper_position has the right shape
        if cartesian_gripper.ndim == 1:
            cartesian_gripper = cartesian_gripper[:, None]

        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper), axis=-1)[skip_start_frames::rgb_skip].tolist()

        # Build info dict based on dataset type
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": frames.shape[0],
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"}
            ],
            'states': cartesian_states,
            'observation.state.cartesian_position': traj_info['observation.state.cartesian_position'][skip_start_frames:],
            'observation.state.joint_position': traj_info['observation.state.joint_position'][skip_start_frames:],
            'observation.state.gripper_position': traj_info['observation.state.gripper_position'][skip_start_frames:],
        }

        # Add dataset-specific fields
        if dataset_type == 'demo':
            info.update({
                'action.cartesian_position': traj_info['action.cartesian_position'][skip_start_frames:],
                'action.joint_position': traj_info['action.joint_position'][skip_start_frames:],
                'action.gripper_position': traj_info['action.gripper_position'][skip_start_frames:],
                'action.joint_velocity': traj_info['action.joint_velocity'][skip_start_frames:],
            })
        else:  # play
            info['action'] = traj_info['action'][skip_start_frames:]

        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset_type', type=str, required=True, choices=['demo', 'play'],
                        help='Type of dataset: demo or play')
    parser.add_argument('--irom_data_path', type=str, required=True,
                        help='Path to IROM dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for processed data')
    parser.add_argument('--svd_path', type=str, default='stable-video-diffusion-img2vid',
                        help='Path to SVD model')
    parser.add_argument('--skip_start_frames', type=int, default=0,
                        help='Number of frames to skip at the start to remove idle frames (demo only)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: only process 5 samples')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDatasetIROM(
        irom_data_path=args.irom_data_path,
        new_path=args.output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        dataset_type=args.dataset_type,
        size=(192, 320),
        rgb_skip=3,  # Downsample to 5Hz (assuming 15Hz input)
        skip_start_frames=args.skip_start_frames,
    )

    tmp_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)

    for idx, _ in enumerate(tmp_data_loader):
        if idx % 10 == 0 and accelerator.is_main_process:
            print(f"Progress: {idx}/{len(dataset)} samples")
        if idx == 5 and args.debug:
            print(f"Debug mode: stopping after {idx} samples")
            break

    if accelerator.is_main_process:
        total_processed = min(idx + 1 if args.debug and idx == 5 else len(dataset), len(dataset))
        print(f"Processing complete! Processed {total_processed} samples total")

# Example usage:
# Demo data:
# accelerate launch scripts/extract_latent.py --dataset_type demo --irom_data_path /n/fs/iromdata/irom_droid_data/2025-11-18 --output_path dataset_example/irom_processed --skip_start_frames 0
#
# Play data:
# accelerate launch scripts/extract_latent.py --dataset_type play --irom_data_path /n/fs/iromdata/irom_droid_data/play_data/2025_11_23_2 --output_path dataset_example/irom_play_processed
