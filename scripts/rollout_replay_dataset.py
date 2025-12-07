#!/usr/bin/env python3
"""
Rollout script that processes an entire dataset for trajectory replay.
Similar to rollout_replay_traj.py but iterates through all trajectories in a dataset.
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

                # IMPORTANT: LoRA adapters are deltas on top of a base checkpoint!
                # We must load the base checkpoint first, then apply LoRA adapters

                # Step 1: Load base checkpoint if specified
                if hasattr(args, 'base_ckpt_path') and args.base_ckpt_path is not None:
                    print(f"Loading base checkpoint from {args.base_ckpt_path}")
                    base_state_dict = torch.load(args.base_ckpt_path, map_location='cpu')

                    # Remap keys for PEFT LoRA compatibility (same as training script)
                    print("Remapping UNet keys for PEFT LoRA compatibility...")
                    from peft import LoraConfig
                    lora_target_modules = args.lora_target_modules

                    remapped_state_dict = {}
                    for key, value in base_state_dict.items():
                        if key.startswith('unet.'):
                            # Remove 'unet.' prefix
                            new_key = key[5:]  # "unet.down_blocks.0..." -> "down_blocks.0..."

                            # Check if this is a LoRA-targeted module
                            is_lora_target = any(target in new_key for target in lora_target_modules)

                            if is_lora_target and (new_key.endswith('.weight') or new_key.endswith('.bias')):
                                # Add base_model.model prefix and .base_layer
                                parts = new_key.rsplit('.', 1)  # Split at last dot
                                new_key = f"base_model.model.{parts[0]}.base_layer.{parts[1]}"
                            else:
                                # Just add base_model.model prefix
                                new_key = f"base_model.model.{new_key}"

                            remapped_state_dict[f"unet.{new_key}"] = value
                        else:
                            # Keep other keys as-is (action_encoder, etc.)
                            remapped_state_dict[key] = value

                    # Load the remapped base checkpoint
                    missing, unexpected = self.model.load_state_dict(remapped_state_dict, strict=False)
                    print(f"Base checkpoint loaded: {len(missing)} missing keys (LoRA adapters, expected), {len(unexpected)} unexpected keys")
                else:
                    print("WARNING: No base_ckpt_path specified! LoRA adapters need a base checkpoint to work correctly.")
                    print("         The model will use fresh SVD initialization, which may produce incorrect results.")

                # Step 2: Load LoRA adapters on top of base checkpoint
                adapter_name = "loaded_adapter"
                self.model.unet.load_adapter(args.val_model_path, adapter_name=adapter_name)
                self.model.unet.set_adapter(adapter_name)
                print(f"LoRA adapters loaded and activated successfully: {adapter_name}")
                print(f"Active adapters: {self.model.unet.active_adapters}")

                # Step 3: Load action encoder weights
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

    def get_traj_info(self, 
                      id, 
                      start_idx=0, # starting frame index
                      num_frames=8 # number of generated frames (downsampled)
                      ):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        
        # parse annotation path
        annotation_path = f"{args.val_dataset_dir}/annotation/train/{id}.json"
        if not os.path.exists(annotation_path):
            annotation_path = f"{args.val_dataset_dir}/annotation/val/{id}.json"

        # calculate raw length
        with open(annotation_path) as f:
            anno = json.load(f)

        # calculate true skip factor
        downsample_factor = 1 if args.downsampled else 3
        actual_skip = skip * downsample_factor

        # frame ids to read from gt actions (raw)
        frames_ids = np.arange(start_idx, start_idx + (num_frames+1) * actual_skip, actual_skip)
        
        # frame ids to read from gt video (downsampled)
        downsampled_frame_ids = frames_ids // actual_skip
        
        instruction = ""

        # get actions/states
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

        # get videos
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
            # print("length of video:", len(vr))
            try:
                true_video = vr.get_batch(downsampled_frame_ids).asnumpy()
                # true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(downsampled_frame_ids).numpy()
                # true_video = vr.get_batch(range(length)).numpy()
                
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

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video, videos], axis=-3)  # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


def process_single_trajectory(Agent, 
                              val_id_i, 
                              start_idx_i, # starting frame index (raw)
                              interact_num, # number of prediction rounds
                              pred_step, # number of (downsampled) frames for each prediction window
                              args):
    """Process a single trajectory and return the rollout video."""
    num_history = args.num_history
    num_frames = args.num_frames

    print(f"\n{'='*80}")
    print(f"Processing trajectory {val_id_i}")
    print(f"{'='*80}")

    # read ground truth trajectory informations
    eef_gt, joint_pos_gt, _, video_latents, instruction = Agent.get_traj_info(
        val_id_i, 
        start_idx=start_idx_i, 
        num_frames=int(interact_num * (pred_step - 1)) # total number of generated frames
    )
    text_i = instruction
    print("Instruction:", instruction)
    print("EEF pose at t=0:", eef_gt[0])
    print("Joint at t=0:", joint_pos_gt[0])

    # create buffers and push first frames to history buffer
    video_to_save = []
    his_cond = []
    his_joint = []
    his_eef = []
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
    assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
    for i in range(Agent.args.num_history*4):
        his_cond.append(first_latent)  # (1, 4, 72, 40)
        his_joint.append(joint_pos_gt[0:1])  # (1, 7)
        his_eef.append(eef_gt[0:1])  # (1, 7)

    # interact loop
    for i in range(interact_num):
        # ground truth video
        start_id = int(i*(pred_step-1))
        end_id = start_id + pred_step
        video_latent_true = [v[start_id:end_id] for v in video_latents]
        print(f"start={start_id}, end={end_id}, video_latent_true[0] shape = {video_latent_true[0].shape}")

        # prepare input for policy
        joint_first = his_joint[-1][0]
        state_first = his_eef[-1][0]
        assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
        assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"

        # forward policy
        print(f"\n--- Interaction step: {i+1}/{interact_num} ---")
        # in the trajectory replay model, we use action recorded in trajectory
        cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
        # print("Cartesian action (first):", cartesian_pose[0])
        # print("Cartesian action (last):", cartesian_pose[-1])
        print("cartesian action shape:", cartesian_pose.shape)

        # retrieve history cond and action cond
        history_idx = [0, 0, -8, -6, -4, -2]
        his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
        print("his_pose shape:", his_pose.shape)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        print("action_cond shape:", action_cond.shape)
        his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]  # (1, 4, 72, 40)
        try: 
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), 7), f"Expected action_cond shape ({int(num_history+num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
        except AssertionError as e:
            print("AssertionError:", e)
            print("current_latent:", current_latent.shape)
            print("action_cond:", action_cond.shape)
            print("his_cond_input:", his_cond_input.shape)
            raise e
        # forward world model
        videos_cat, _, _, predicted_latents = Agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_cond_input,
            text=text_i if Agent.args.text_cond else None
        )

        # push current step to history buffer
        his_eef.append(cartesian_pose[pred_step-1:pred_step])  #(1,7)
        his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
        if i == interact_num - 1:
            video_to_save.append(videos_cat)  # save all frames for the last interaction step
        else:
            video_to_save.append(videos_cat[:pred_step-1])  # last frame is the first frame of next step, so we remove it here

    # concatenate all video segments
    video = np.concatenate(video_to_save, axis=0)

    return video, instruction


if __name__ == "__main__":
    # from droid_inference_config import wm_args
    from droid_inference_config_lora import wm_args
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
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    args.__dict__['gripper_annotation'] = args_new.gripper_annotation

    # Set val_dataset_dir to the specified dataset directory
    args.val_dataset_dir = args.dataset_dir

    # Extract dataset name from dataset_dir path
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    # Setup evaluation directory structure
    eval_base_dir = "/n/fs/tom-project/video_models/Ctrl-World/dataset_eval"
    dataset_eval_dir = os.path.join(eval_base_dir, dataset_name)
    os.makedirs(dataset_eval_dir, exist_ok=True)

    models_txt_path = os.path.join(dataset_eval_dir, "models.txt")

    # Determine model index and update models.txt
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

    # Update models.txt with this checkpoint
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

    # Limit trajectories if max_trajectories is set
    # if args.max_trajectories is not None:
    #     annotation_files = annotation_files[:args.max_trajectories]

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

    for split, val_id_i in tqdm(annotation_files, desc="Processing trajectories"):
        try:
            # First, get the total trajectory length to determine interact_num dynamically
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

            # Calculate the maximum number of frames available after start_idx and downsampling
            downsample_factor = 1 if args.downsampled else 3
            actual_skip = args.skip_step * downsample_factor
            available_frames = (total_length - args.start_idx) // actual_skip

            # Calculate interact_num dynamically: need history frames (8) + frames for predictions
            # Each interaction advances by (pred_step - 1) frames
            history_frames = 8
            interact_num = max(1, (available_frames - history_frames - 1) // (pred_step - 1) - 1)

            print(f"\nTrajectory {val_id_i} ({split}): total_length={total_length}, start_idx={args.start_idx}, "
                  f"available_frames={available_frames}, calculated interact_num={interact_num}")

            # Process the trajectory
            video, instruction = process_single_trajectory(
                Agent, val_id_i, args.start_idx, interact_num, pred_step, args
            )

            # Save the video with simple naming: {traj_id}.mp4
            filename_video = os.path.join(video_dir, f"{val_id_i}.mp4")
            mediapy.write_video(filename_video, video, fps=4)
            print(f"Saved video to {filename_video}")

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
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful_count}/{len(annotation_files)}")
    print(f"Failed: {failed_count}/{len(annotation_files)}")
    if failed_trajectories:
        print(f"Failed trajectories: {failed_trajectories}")
    print(f"\nVideos saved to: {video_dir}")
    print(f"Models registry: {models_txt_path}")
