#!/usr/bin/env python3
"""
Parallel version of rollout_replay_dataset_sample.py
Supports multi-GPU parallel processing where each GPU processes a subset of trajectories.
"""

import numpy as np
import torch
import einops
from accelerate import Accelerator
import os
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import mediapy
import sys
import glob
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld


class agent():
    def __init__(self, args):
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load ctrl-world model
        self.model = CrtlWorld(args)

        # Load checkpoint - handle both LoRA and full checkpoints
        print(f"Loading checkpoint from {args.val_model_path}")
        if hasattr(args, 'use_lora') and args.use_lora:
            # LoRA mode: check if loading LoRA-only adapters or full checkpoint
            if os.path.isdir(args.val_model_path):
                # Loading LoRA-only adapters from directory
                print("Loading LoRA adapters from directory...")
                from peft import PeftModel
                self.model.unet = PeftModel.from_pretrained(
                    self.model.unet,
                    args.val_model_path,
                    is_trainable=False
                )
                print("LoRA adapters loaded successfully")

                # Check if action encoder weights exist and load them
                action_encoder_path = os.path.join(args.val_model_path, "action_encoder.pt")
                if os.path.exists(action_encoder_path):
                    print(f"Loading action encoder from {action_encoder_path}")
                    action_encoder_state = torch.load(action_encoder_path, map_location='cpu')
                    self.model.action_encoder.load_state_dict(action_encoder_state)
                    print("Action encoder loaded successfully")
                else:
                    print("No action_encoder.pt found in LoRA directory - using initialized weights")
            else:
                # Loading full checkpoint (.pt file)
                state_dict = torch.load(args.val_model_path, map_location='cpu')
                if any('lora' in k for k in state_dict.keys()):
                    print("Loading full checkpoint with LoRA weights...")
                    self.model.load_state_dict(state_dict, strict=True)
                else:
                    print("Warning: Loading non-LoRA checkpoint into LoRA model...")
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            # Full fine-tuning mode: standard loading
            state_dict = torch.load(args.val_model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("World model loaded successfully")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{args.val_dataset_dir}/annotation/train/{id}.json"

        # Try train first, then val
        if not os.path.exists(annotation_path):
            annotation_path = f"{args.val_dataset_dir}/annotation/val/{id}.json"

        with open(annotation_path) as f:
            anno = json.load(f)
            # Handle new dataset format
            if 'observation.state.cartesian_position' in anno:
                # New dataset format
                length = len(anno['observation.state.cartesian_position'])
            elif 'action' in anno:
                length = len(anno['action'])
            else:
                length = anno["video_length"]

        # Determine the actual skip step based on whether data is already downsampled
        # If not downsampled, data is at 15Hz and needs to be downsampled by factor of 3 to get 5Hz
        downsample_factor = 1 if args.downsampled else 3
        actual_skip = skip * downsample_factor

        frames_ids = np.arange(start_idx, start_idx + num_frames * actual_skip, actual_skip)

        # downsample action
        # length = length // actual_skip
        print(f"Original trajectory length: {length}")
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print(f"Ground truth frames ids (downsampled={args.downsampled}, downsample_factor={downsample_factor}): {frames_ids}")
        length = len(frames_ids)
        print(f"Total frames to process: {length}")
        print(f"skip: {skip}, actual_skip: {actual_skip}")
        # get action and joint pos - handle both old and new dataset formats
        instruction = ""

        if 'observation.state.cartesian_position' in anno:
            car_action = np.array(anno['observation.state.cartesian_position'])
            # obs_state_cart is 6-dim (xyz + rotation), need to add gripper state
            # Add a column of zeros for gripper (or use obs_state_jointpos[-1] if available)
            if car_action.shape[1] == 6:
                gripper_state = np.zeros((len(car_action), 1))
                if 'observation.state.gripper_position' in anno:
                    # Use gripper_position as gripper
                    gripper_state = np.array(anno['observation.state.gripper_position'])[:,None]
                car_action = np.concatenate([car_action, gripper_state], axis=1)
        else:
            car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]

        if 'observation.state.joint_position' in anno:
            joint_pos = np.array(anno['observation.state.joint_position'])
            # If joint_pos is 7-dim, add a gripper column to make it 8-dim
            if joint_pos.shape[1] == 7:
                gripper_state = joint_pos[:, -1:]  # Use last dim as gripper
                joint_pos = np.concatenate([joint_pos[:, :-1], gripper_state, gripper_state], axis=1)
        else:
            joint_pos = np.array(anno['joints'])
        joint_pos = joint_pos[frames_ids]

        # get videos - handle both old and new dataset formats
        video_dict = []
        video_latent = []

        if 'videos' in anno:
            # Old dataset format
            video_paths = [f"{val_dataset_dir}/{anno['videos'][i]['video_path']}" for i in range(len(anno['videos']))]
        else:
            # New dataset format - video is in videos_three_view_vertical/{id}.mp4
            video_paths = [f"{val_dataset_dir}/videos_three_view_vertical/{id}.mp4"]

        for video_path in video_paths:
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            # true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0, 3, 1, 2).to(device) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)

            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        args = self.args
        image_cond = video_latent_cond

        # Save gripper values before normalization (for annotation)
        gripper_values = action_cond[:, -1].copy() if hasattr(args, 'gripper_annotation') and args.gripper_annotation else None

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)

        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)  # (B, 8, 4, 32,32)

        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0)  # (bsz, 8,32,32)
        decoded_video = []
        bsz, frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, true_video.shape[0], args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video, dim=0)
        true_video = true_video.reshape(bsz, frame_num, *true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)  #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # Annotate gripper values on predicted frames if enabled
        if gripper_values is not None:
            # Extract gripper values for the predicted frames (skip history frames)
            pred_gripper_values = gripper_values[args.num_history:]
            # Annotate each view's frames
            for view_idx in range(bsz):
                for frame_idx in range(frame_num):
                    if frame_idx < len(pred_gripper_values):
                        gripper_val = pred_gripper_values[frame_idx]
                        # Make a contiguous copy of the frame for OpenCV
                        frame = np.ascontiguousarray(videos[view_idx, frame_idx])
                        text = f"G={gripper_val:.3f}"
                        # Position: top-left corner with some padding
                        position = (10, 30)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        color = (255, 255, 255)  # White text
                        thickness = 2
                        # Add black background for better readability
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        cv2.rectangle(frame, (position[0] - 5, position[1] - text_height - 5),
                                    (position[0] + text_width + 5, position[1] + baseline + 5),
                                    (0, 0, 0), -1)
                        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                        # Assign the annotated frame back
                        videos[view_idx, frame_idx] = frame

        # Return separate GT and pred videos (not concatenated)
        return true_video, videos, latents  # true_video: (3, pred_step, H, W, 3), videos: (3, pred_step, H, W, 3)


def process_single_sample(Agent, val_id_i, start_pos, sample_id, pred_step, args):
    """Process a single sample (one interaction) from a trajectory.

    Args:
        Agent: The agent object
        val_id_i: Trajectory ID
        start_pos: Starting frame position for this sample
        sample_id: Sample ID (chronologically ordered)
        pred_step: Number of frames to predict
        args: Arguments

    Returns:
        true_video: Ground truth video (3, pred_step, H, W, 3)
        pred_video: Predicted video (3, pred_step, H, W, 3)
        instruction: Text instruction (if any)
    """
    num_history = args.num_history
    num_frames = args.num_frames

    print(f"\n--- Processing sample {sample_id} starting at frame {start_pos} ---")

    # Load ground truth trajectory from beginning to sample end
    # We need frames from 0 to start_pos + pred_step for building history and the sample itself
    total_frames_needed = start_pos + pred_step
    eef_gt, joint_pos_gt, _, video_latents, instruction = Agent.get_traj_info(
        val_id_i, start_idx=0, steps=total_frames_needed
    )

    # Build history buffer from ground truth
    his_cond = []
    his_eef = []

    # Get the latent at start_pos - 1 (or 0 if start_pos is 0)
    current_frame_idx = max(0, start_pos - 1)
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
    current_latent = torch.cat([v[current_frame_idx] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)

    # Build history using indices [0, 0, -8, -6, -4, -2] relative to current position
    # For start_pos < 8, we need to pad with the first frame
    history_idx = [0, 0, -8, -6, -4, -2]
    his_pose_list = []
    his_cond_list = []

    for idx in history_idx:
        if idx == 0:
            # Always use frame 0
            frame_idx = 0
        else:
            # Negative index relative to current position
            frame_idx = max(0, start_pos + idx)  # Clamp to 0 if goes negative

        his_pose_list.append(eef_gt[frame_idx:frame_idx+1])  # (1, 7)
        his_cond_list.append(torch.cat([v[frame_idx] for v in video_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)

    his_pose = np.concatenate(his_pose_list, axis=0)  # (6, 7)
    his_cond_input = torch.cat(his_cond_list, dim=0).unsqueeze(0)  # (1, 6, 4, 72, 40)

    # Get the action for this sample
    cartesian_pose = eef_gt[start_pos:start_pos + pred_step]  # (pred_step, 7)
    action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)  # (6 + pred_step, 7)

    # Get ground truth video latents for this sample
    video_latent_true = [v[start_pos:start_pos + pred_step] for v in video_latents]

    print(f"Cartesian action (first): {cartesian_pose[0]}")
    print(f"Cartesian action (last): {cartesian_pose[-1]}")

    # Assertions
    assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
    assert action_cond.shape == (int(num_history + num_frames), 7), f"Expected action_cond shape ({int(num_history + num_frames)}, 7), got {action_cond.shape}"
    assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 4, 72, 40), got {his_cond_input.shape}"

    # Forward world model
    true_video, pred_video, predicted_latents = Agent.forward_wm(
        action_cond, video_latent_true, current_latent,
        his_cond=his_cond_input,
        text=instruction if Agent.args.text_cond else None
    )

    return true_video, pred_video, instruction


if __name__ == "__main__":
    from droid_inference_config import wm_args
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory (e.g., dataset_example/irom_1126_base2)')
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='replay')
    parser.add_argument('--downsampled', action='store_true', help='If set, assumes input is already downsampled to 5Hz. Otherwise, assumes 15Hz input and downsamples by factor of 3.')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index for each trajectory')
    parser.add_argument('--max_trajectories', type=int, default=None, help='Maximum number of trajectories to process (for debugging)')
    parser.add_argument('--model_index', type=int, default=None, help='Model index for video directory naming (e.g., 0 for video_0/). If not provided, will auto-assign based on models.txt')
    parser.add_argument('--gripper_annotation', action='store_true', help='If set, annotates gripper state values on video frames')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to take per trajectory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible sampling')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for parallel processing (processes trajectories where traj_idx %% num_gpus == gpu_id)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs for parallel processing')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    args.__dict__['gripper_annotation'] = args_new.gripper_annotation
    args.__dict__['num_samples'] = args_new.num_samples
    args.__dict__['gpu_id'] = args_new.gpu_id
    args.__dict__['num_gpus'] = args_new.num_gpus

    # Set random seed if provided
    if args_new.seed is not None:
        np.random.seed(args_new.seed)
        print(f"Random seed set to: {args_new.seed}")
    else:
        # set seed to default
        np.random.seed(0)

    # Set val_dataset_dir to the specified dataset directory
    args.val_dataset_dir = args.dataset_dir

    # Extract dataset name from dataset_dir path
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    # Setup evaluation directory structure
    eval_base_dir = "/n/fs/tom-project/video_models/Ctrl-World/dataset_eval_samples"
    dataset_eval_dir = os.path.join(eval_base_dir, dataset_name)
    os.makedirs(dataset_eval_dir, exist_ok=True)

    models_txt_path = os.path.join(dataset_eval_dir, "models.txt")

    # Determine model index and update models.txt (only GPU 0 should do this)
    if args.model_index is None:
        # Auto-assign model index based on models.txt
        if os.path.exists(models_txt_path):
            with open(models_txt_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        args.model_index = int(last_line.split(':')[0]) + 1
                    else:
                        args.model_index = 0
                else:
                    args.model_index = 0
        else:
            args.model_index = 0

    # Update models.txt with this checkpoint (only GPU 0 should do this to avoid race conditions)
    if args.gpu_id is None or args.gpu_id == 0:
        with open(models_txt_path, 'a') as f:
            f.write(f"{args.model_index}: {args.ckpt_path}\n")

    # Create video output directory
    video_dir = os.path.join(dataset_eval_dir, f"video_{args.model_index}")
    os.makedirs(video_dir, exist_ok=True)

    print(f"\nEvaluation setup:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model index: {args.model_index}")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Output directory: {video_dir}")
    print(f"  Models registry: {models_txt_path}")
    print(f"  Number of samples per trajectory: {args.num_samples}")
    print(f"  Random seed: {args_new.seed if args_new.seed is not None else 'None (random)'}")
    if args.gpu_id is not None:
        print(f"  GPU ID: {args.gpu_id}/{args.num_gpus}")

    # Find all trajectories in the dataset (both train and val)
    annotation_files = []

    for split in ['train', 'val']:
        split_dir = os.path.join(args.dataset_dir, 'annotation', split)
        if os.path.exists(split_dir):
            split_files = glob.glob(os.path.join(split_dir, '*.json'))
            for f in split_files:
                traj_id = os.path.basename(f).replace('.json', '')
                annotation_files.append((split, traj_id))

    # Sort by trajectory ID (convert to int for proper sorting)
    annotation_files.sort(key=lambda x: int(x[1]))

    # Filter trajectories based on GPU ID
    if args.gpu_id is not None:
        total_trajectories = len(annotation_files)
        filtered_annotation_files = []
        for traj_idx, (split, traj_id) in enumerate(annotation_files):
            if traj_idx % args.num_gpus == args.gpu_id:
                filtered_annotation_files.append((split, traj_id))
        annotation_files = filtered_annotation_files
        print(f"\nGPU {args.gpu_id} processing {len(annotation_files)}/{total_trajectories} trajectories")
        print(f"Trajectory IDs: {[traj_id for _, traj_id in annotation_files]}")
    else:
        print(f"\nFound {len(annotation_files)} trajectories to process")
        print(f"Trajectory IDs: {[traj_id for _, traj_id in annotation_files]}")

    # Create rollout agent
    Agent = agent(args)
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'\nRollout with {args.task_type}')

    # Process each trajectory
    successful_count = 0
    failed_count = 0
    failed_trajectories = []

    for split, val_id_i in tqdm(annotation_files, desc=f"Processing trajectories (GPU {args.gpu_id if args.gpu_id is not None else 'N/A'})"):
        try:
            print(f"\n{'='*80}")
            print(f"Processing trajectory {val_id_i}")
            print(f"{'='*80}")

            # Get the total trajectory length
            annotation_path = f"{args.val_dataset_dir}/annotation/{split}/{val_id_i}.json"
            with open(annotation_path) as f:
                anno = json.load(f)
                # Handle new dataset format
                if 'observation.state.cartesian_position' in anno:
                    total_length = len(anno['observation.state.cartesian_position'])
                elif 'action' in anno:
                    total_length = len(anno['action'])
                else:
                    total_length = anno["video_length"]

            # Calculate the number of frames available after downsampling
            downsample_factor = 1 if args.downsampled else 3
            actual_skip = args.skip_step * downsample_factor
            available_frames = (total_length - args.start_idx) // actual_skip

            # Calculate all possible non-overlapping starting positions
            # Starting positions: 0, pred_step, 2*pred_step, ...
            max_start_pos = available_frames - pred_step
            if max_start_pos < 0:
                print(f"Trajectory {val_id_i} is too short (available_frames={available_frames}, pred_step={pred_step}). Skipping.")
                continue

            possible_start_positions = list(range(0, max_start_pos + 1, pred_step))
            num_possible_samples = len(possible_start_positions)

            # Sample x positions randomly (or all if fewer than x available)
            num_samples_to_take = min(args.num_samples, num_possible_samples)
            if num_samples_to_take < args.num_samples:
                print(f"Warning: Only {num_samples_to_take} non-overlapping samples available (requested {args.num_samples})")

            sampled_indices = np.random.choice(len(possible_start_positions), size=num_samples_to_take, replace=False)
            sampled_positions = [possible_start_positions[i] for i in sampled_indices]

            # Sort positions chronologically and assign sample_ids
            sampled_positions.sort()

            print(f"Trajectory {val_id_i} ({split}): total_length={total_length}, available_frames={available_frames}")
            print(f"Sampled {num_samples_to_take} positions: {sampled_positions}")

            # Create output directory for this trajectory
            traj_output_dir = os.path.join(video_dir, val_id_i)
            gt_dir = os.path.join(traj_output_dir, "groundtruth")
            pred_dir = os.path.join(traj_output_dir, "prediction")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            # Process each sampled position
            for sample_id, start_pos in enumerate(sampled_positions):
                true_video, pred_video, instruction = process_single_sample(
                    Agent, val_id_i, start_pos, sample_id, pred_step, args
                )

                # true_video and pred_video have shape (3, pred_step, H, W, 3)
                # where the first dimension is the view (0=left, 1=right, 2=wrist)

                # Save each view separately
                for view_id in range(3):
                    # Extract frames for this view
                    gt_frames = true_video[view_id]  # (pred_step, H, W, 3)
                    pred_frames = pred_video[view_id]  # (pred_step, H, W, 3)

                    # Save GT video
                    gt_filename = os.path.join(gt_dir, f"{sample_id}_view{view_id}.mp4")
                    mediapy.write_video(gt_filename, gt_frames, fps=5)

                    # Save prediction video
                    pred_filename = os.path.join(pred_dir, f"{sample_id}_view{view_id}.mp4")
                    mediapy.write_video(pred_filename, pred_frames, fps=5)

                print(f"Saved sample {sample_id} (start_pos={start_pos}) to {traj_output_dir}")

            successful_count += 1

        except Exception as e:
            print(f"ERROR processing trajectory {val_id_i}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            failed_trajectories.append(val_id_i)
            continue

    # Print summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE (GPU {args.gpu_id if args.gpu_id is not None else 'N/A'})")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful_count}/{len(annotation_files)}")
    print(f"Failed: {failed_count}/{len(annotation_files)}")
    if failed_trajectories:
        print(f"Failed trajectories: {failed_trajectories}")
    print(f"\nVideos saved to: {video_dir}")
    print(f"Models registry: {models_txt_path}")
