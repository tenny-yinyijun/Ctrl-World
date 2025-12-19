import torch
import numpy as np
import einops

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
    

def get_traj_info(idx):
    # TODO given the index, go to /n/fs/iromdata/droid_ctrl_world to find the corresponding video and annotation, and then parse the information similar to get_traj_info in @rollout_interact_pi.py
    pass
    # return eef_gt, joint_pos_gt, video_dict, video_latents, instruction

class WorldModelEnv(gym.Env):
    def __init__(self, args, wm_ckpt: str, control_mode="joint_velocity"):
        # Initialize the world model environment
        self.num_history = 5
        self.pred_step = 5
        self.policy_skip_step = 2 
        self.control_mode = control_mode
        
        # model
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(wm_ckpt))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()

    def reset_eval_env(self, idx):
        # Reset the environment and return the initial observation and info
        self.eef_gt, self.joint_pos_gt, self.video_dict, self.video_latents, self.instruction = get_traj_info(idx)
        self.predicted_latents = None
        self.video_to_save = []
        self.info_to_save = []
        self.his_cond = []
        self.his_joint = []
        self.his_eef = []
        self.first_latent = torch.cat([v[0] for v in self.video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert self.first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {self.first_latent.shape}"
        for i in range(self.num_history*4):
            self.his_cond.append(self.first_latent)  # (1, 4, 72, 40)
            self.his_joint.append(self.joint_pos_gt[0:1])  # (1, 7)
            self.his_eef.append(self.eef_gt[0:1])  # (1, 7)
        self.video_dict_pred = [v[0:1] for v in self.video_dict]
        
        # TODO check
        current_obs = [v[-1] for v in self.video_dict_pred]
        return current_obs, self.his_eef[-1][0]

    def step(self, action_chunk):
        ##### forward policy
        
        current_joint = self.his_joint[-1][0] # (1, 8)
        current_pose = self.his_eef[-1]  # (1, 7) eef pose
        if self.control_mode == "cartesian_pose":
            # downsample and compute delta TODO
            cartesian_pose = np.zeros((self.pred_step,7))
        else:
            cartesian_pose = self.get_cartesian_pose(current_joint, current_pose, action_chunk)

        ##### forward world model
        
        history_idx = [0,0,-8,-6,-4,-2]
        his_pose = np.concatenate([self.his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        his_cond_input = torch.cat([self.his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = self.his_cond[-1]  # (1, 4, 72, 40)
        assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
        assert action_cond.shape == (int(self.num_history+self.num_frames), 7), f"Expected action_cond shape ({int(self.num_history+self.num_frames)}, 7), got {action_cond.shape}"
        assert his_cond_input.shape == (1, int(self.num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(self.num_history)}, 72, 40), got {his_cond_input.shape}"
        # forward world model
        videos_cat, true_videos, video_dict_pred, predicted_latents = self.forward_wm(action_cond, current_latent, his_cond=his_cond_input)

        print("################ record information ################")
        # push current step to history buffer
        self.his_eef.append(cartesian_pose[self.pred_step-1:self.pred_step]) #(1,7)
        self.his_cond.append(torch.cat([v[self.pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
        
        # if i == interact_num - 1:
        #     video_to_save.append(videos_cat)  # save all frames for the last interaction step
        # else:
        #     video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here
        
        
        # return observation
        
    
    def get_action_cond(self, joints, state, action_chunk):
        idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        current_joint = joints[None,:][:,:7]
        current_gripper = joints[None,:][:,7:]
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
            policy_in_out = {
                'joint_pos': joint_pos[:valid_num],  # (12, 7)
                'joint_vel': joint_vel[:valid_num],  # (12, 7)
                'state_fk': state_fk[:valid_num],  # (12, 7)
            }
            state_fk_skip = state_fk[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = joint_pos[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:,-1:]], axis=-1) # (5, 8) add gripper pos

            return policy_in_out, joint_pos_skip, state_fk_skip

            
    
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
