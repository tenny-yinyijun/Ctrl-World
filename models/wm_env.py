import torch
import numpy as np
import einops
import json

from configs.pi_inference_config import wm_args

import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from decord import VideoReader, cpu

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
from models.action_adapter.train2 import Dynamics


class WorldModelEnv(gym.Env):
    def __init__(self, wm_ckpt: str, control_mode="joint_velocity"):
        # Initialize the world model environment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.control_mode = control_mode

        self.args = wm_args()
        self.dtype = self.args.dtype
        self.pred_step = self.args.pred_step
        self.num_history = self.args.num_history
        self.num_frames = self.args.num_frames
        self.history_idx = self.args.history_idx

        # model
        self.model = CrtlWorld(self.args)
        self.model.load_state_dict(torch.load(wm_ckpt))
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        
        self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
        self.dynamics_model.load_state_dict(torch.load(self.args.action_adapter, map_location=self.device))        


        # load data statistics for normalization
        with open("/n/fs/iromdata/video_models/Ctrl-World/assets/stat.json", 'r') as f:
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
        """Normalize data to [-1, 1] range using min-max normalization."""
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, idx, start_idx=0, steps=None, skip=1):
        """
        Load trajectory information from the droid_ctrl_world dataset.

        Args:
            idx: trajectory ID (string or int)
            start_idx: starting frame index
            steps: number of frames to load (if None, load all available frames)
            skip: frame skip interval

        Returns:
            eef_gt: end-effector ground truth poses (cartesian states)
            joint_pos_gt: joint positions ground truth
            video_dict: list of video arrays for each camera view
            video_latents: list of encoded video latents for each camera view
            instruction: text instruction for the task
        """
        val_dataset_dir = "/n/fs/iromdata/droid_ctrl_world"
        annotation_path = f"{val_dataset_dir}/annotation/train/{idx}.json"

        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]

        # Determine frames to load
        if steps is None:
            # Load all frames from start_idx to end
            steps = (length - start_idx) // skip

        frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # Get action and joint pos
        instruction = anno['texts'][0]
        eef_gt = np.array(anno['states'])
        eef_gt = eef_gt[frames_ids]
        joint_pos_gt = np.array(anno['observation.state.joint_position'])
        joint_pos_gt = joint_pos_gt[frames_ids]
        gripper_pos_gt = np.array(anno['observation.state.gripper_position'])
        gripper_pos_gt = gripper_pos_gt[frames_ids]
        joint_pos_gt = np.concatenate([joint_pos_gt, gripper_pos_gt[:, None]], axis=-1)  # (num_frames, 8)

        # Get videos
        video_dict = []
        video_latents = []
        for view_id in range(len(anno['videos'])):
            video_path = anno['videos'][view_id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # Load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # Encode video
            true_video_tensor = torch.from_numpy(true_video).to(self.dtype).to(self.device)
            x = true_video_tensor.permute(0, 3, 1, 2).to(self.device) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)

            video_latents.append(x)

        return eef_gt, joint_pos_gt, video_dict, video_latents, instruction

    def reset(self, idx):
        # Reset the environment and return the initial observation and info
        self.eef_gt, self.joint_pos_gt, self.video_dict, self.video_latents, self.instruction = self.get_traj_info(idx)
        self.predicted_latents = None
        self.video_to_save = []
        self.info_to_save = []
        self.his_cond = []
        self.his_joint = []
        self.his_eef = []
        self.is_first_interact = True
        self.first_latent = torch.cat([v[0] for v in self.video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert self.first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {self.first_latent.shape}"
        for i in range(self.num_history*4):
            self.his_cond.append(self.first_latent)  # (1, 4, 72, 40)
            self.his_joint.append(self.joint_pos_gt[0:1])  # (1, 7)
            self.his_eef.append(self.eef_gt[0:1])  # (1, 7)
        self.video_dict_pred = [v[0:1] for v in self.video_dict]
        
        # TODO check
        current_obs = [v[-1] for v in self.video_dict_pred]
        return current_obs, self.his_joint[-1][0], self.instruction
        # return current_obs, self.his_eef[-1][0], self.instruction

    def step(self, action_chunk, is_last_interact=False):
        final_video = None
        ##### forward policy
        current_joint = self.his_joint[-1][0] # (1, 8)
        current_pose = self.his_eef[-1]  # (1, 7) eef pose
        if self.control_mode == "cartesian_pose":
            # action_chunk contains delta cartesian poses (15, 7) or (15, 8)
            # Format: (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_pos)

            # Extract current pose components
            current_pose_xyz_euler = current_pose[0]  # (7,) [x, y, z, roll, pitch, yaw, gripper]
            current_gripper = current_joint[7:8]  # (1,) get gripper from joint state

            # Separate delta poses and gripper positions from action_chunk
            if action_chunk.shape[1] == 8:
                delta_xyz_euler = action_chunk[:, :6]  # (15, 6) delta pose
                gripper_pos = action_chunk[:, 6:7]  # (15, 1) absolute gripper position
            else:
                delta_xyz_euler = action_chunk[:, :6]  # (15, 6) delta pose
                gripper_pos = action_chunk[:, 6:7]  # (15, 1) absolute gripper position

            # Clip gripper positions
            gripper_max = self.args.gripper_max
            gripper_pos = np.clip(gripper_pos, 0, gripper_max)

            # Integrate delta poses to get absolute cartesian poses
            cartesian_poses = []
            current_xyz_euler = current_pose_xyz_euler[:6].copy()  # (6,) [x, y, z, roll, pitch, yaw]

            # Add initial state (current gripper position)
            gripper_pos_full = np.concatenate([current_gripper[None, :], gripper_pos], axis=0)  # (16, 1)

            # First pose is current pose
            for i in range(len(delta_xyz_euler)):
                # Integrate delta to current pose
                current_xyz_euler = current_xyz_euler + delta_xyz_euler[i]
                # Combine xyz, euler, and gripper
                pose_with_gripper = np.concatenate([current_xyz_euler, gripper_pos_full[i+1]], axis=0)
                cartesian_poses.append(pose_with_gripper)

            cartesian_poses = np.array(cartesian_poses)  # (15, 7)

            # Downsample to match pred_step (same as joint_velocity mode)
            skip = self.args.policy_skip_step
            cartesian_pose = cartesian_poses[::skip][:self.pred_step]  # (5, 7)

        else:
            joint_pos, cartesian_pose = self.get_action_cond(current_joint, current_pose, action_chunk)
            self.his_joint.append(joint_pos[self.pred_step-1][None,:])  # (1, 8)

        ##### forward world model
        
        # history_idx = [0,0,-8,-6,-4,-2]
        his_pose = np.concatenate([self.his_eef[idx] for idx in self.history_idx], axis=0)  # (4, 7)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        his_cond_input = torch.cat([self.his_cond[idx] for idx in self.history_idx], dim=0).unsqueeze(0)
        current_latent = self.his_cond[-1]  # (1, 4, 72, 40)
        assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
        assert action_cond.shape == (int(self.num_history+self.num_frames), 7), f"Expected action_cond shape ({int(self.num_history+self.num_frames)}, 7), got {action_cond.shape}"
        assert his_cond_input.shape == (1, int(self.num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(self.num_history)}, 72, 40), got {his_cond_input.shape}"
        # forward world model
        video_dict_pred, predicted_latents = self.forward_wm(action_cond, current_latent, his_cond=his_cond_input)

        # predict_latents: [3, 5, 4, 24, 40]
        # video_dict_pred: [3, 5, 192, 320, 3]
        
        # push current step to history buffer
        self.his_eef.append(cartesian_pose[self.pred_step-1:self.pred_step]) #(1,7)
        self.his_cond.append(torch.cat([v[self.pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
        
        # if i == interact_num - 1:
        #     video_to_save.append(videos_cat)  # save all frames for the last interaction step
        # else:
        #     video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here
        current_obs = [v[-1] for v in video_dict_pred] 
        # current_pose = self.his_eef[-1][0]
        current_joint = self.his_joint[-1][0]
        
        if is_last_interact:
            self.video_to_save.append(video_dict_pred)
            # concatenate all frames
            final_video = self.video_to_save
        else:
            # remove the first frame to avoid duplication
            self.video_to_save.append([v[:self.pred_step-1] for v in video_dict_pred])

        return current_obs, current_joint, final_video
        
    
    def get_action_cond(self, joints, state, action_chunk):
        idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        current_joint = joints[None,:][:,:7] #(1,7)
        current_gripper = joints[None,:][:,7:] #(1,1)
        if self.control_mode == "cartesian_pose":
            raise NotImplementedError("Cartesian pose control mode not implemented yet.")
        elif self.control_mode == "joint_velocity":
            # policy output joint velocity and gripper position
            joint_vel = action_chunk[:,:7] # (15, 7)
            gripper_pos = action_chunk[:,7:] # (15, 1)
            joint_vel = joint_vel[idx]  # (15, 7)
            gripper_pos = gripper_pos[idx]  # (15, 1)
            gripper_max = self.args.gripper_max
            gripper_pos = np.clip(gripper_pos, 0, gripper_max)
            # calculate future joint positions
            joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)
            # fk
            state_fk = []
            joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]  # (15, 7)
            gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)
            joint_vel = joint_vel  # (15, 7)
            for i in range(joint_pos.shape[0]):
                current_state_fk = get_fk_solution(joint_pos[i,:7])
                xyz = current_state_fk[:3, 3]
                rotation_matrix = current_state_fk[:3, :3]
                r = R.from_matrix(rotation_matrix)
                euler = r.as_euler('xyz') 
                state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
            state_fk = np.array(state_fk) # (15,7)

            # prepare output
            skip = self.args.policy_skip_step
            valid_num = int(skip*(self.args.pred_step-1))
            state_fk_skip = state_fk[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = joint_pos[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:,-1:]], axis=-1) # (5, 8) add gripper pos

            return joint_pos_skip, state_fk_skip

            
    
    def forward_wm(self, action_cond, video_latent_cond, his_cond=None):
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)

        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
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
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        # videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        # videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8) 

        return videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)
