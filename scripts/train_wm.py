# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import mediapy
from models.ctrl_world import CrtlWorld
# from droid_irom_finetune_lora import wm_args as wm_args_lora
from droid_irom_finetune import wm_args as wm_args_full
from droid_irom_finetune_curriculum import wm_args as wm_args_curriculum
# from droid_irom_finetune_continue import wm_args as wm_args_cont
# from droid_irom_finetune_small import wm_args as wm_args_small
import math
from dataset.curriculum_scheduler import CurriculumScheduler
from dataset.curriculum_sampler import DynamicCurriculumSampler


def main(args):
    logger = get_logger(__name__, log_level="INFO")

    # Use dispatch_batches to properly handle custom samplers (like DynamicCurriculumSampler)
    # in distributed training without replacing them with SequentialSampler
    from accelerate import DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(dispatch_batches=True) if getattr(args, 'use_curriculum', False) else None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir,
        dataloader_config=dataloader_config
    )

    # model and optimizer
    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')

        # Handle LoRA checkpoint loading
        if hasattr(args, 'use_lora') and args.use_lora:
            # Check if this is a LoRA checkpoint or full checkpoint
            if any('lora' in k for k in state_dict.keys()):
                print("Loading LoRA checkpoint...")
                model.load_state_dict(state_dict, strict=True)
            else:
                print("Loading full model checkpoint into LoRA model...")
                print("Remapping UNet keys for PEFT LoRA compatibility...")

                # Remap UNet keys: "unet.xxx" -> "unet.base_model.model.xxx"
                # For LoRA-targeted modules, also add ".base_layer" before weight/bias
                # This is needed because PEFT wraps the UNet and changes key names
                remapped_state_dict = {}
                unet_keys_remapped = 0
                unet_lora_keys_remapped = 0
                action_encoder_keys_loaded = 0
                other_keys_loaded = 0

                # LoRA target modules from config
                lora_targets = args.lora_target_modules if hasattr(args, 'lora_target_modules') else ["to_q", "to_k", "to_v", "to_out.0"]

                for key, value in state_dict.items():
                    if key.startswith('unet.'):
                        # First, apply base PEFT wrapping
                        new_key = key.replace('unet.', 'unet.base_model.model.', 1)

                        # Check if this key is for a LoRA-targeted module
                        # LoRA-targeted modules have their weights under .base_layer
                        is_lora_target = False
                        for target in lora_targets:
                            # Check if this key contains the target module
                            if f'.{target}.weight' in new_key or f'.{target}.bias' in new_key:
                                # Insert .base_layer before .weight or .bias
                                new_key = new_key.replace(f'.{target}.weight', f'.{target}.base_layer.weight')
                                new_key = new_key.replace(f'.{target}.bias', f'.{target}.base_layer.bias')
                                is_lora_target = True
                                unet_lora_keys_remapped += 1
                                break

                        remapped_state_dict[new_key] = value
                        if not is_lora_target:
                            unet_keys_remapped += 1
                    elif key.startswith('action_encoder.'):
                        # Action encoder keys don't need remapping
                        remapped_state_dict[key] = value
                        action_encoder_keys_loaded += 1
                    else:
                        # Other keys (vae, image_encoder, etc.) don't need remapping
                        remapped_state_dict[key] = value
                        other_keys_loaded += 1

                print(f"  UNet keys remapped (non-LoRA): {unet_keys_remapped}")
                print(f"  UNet keys remapped (LoRA-targeted): {unet_lora_keys_remapped}")
                print(f"  Action encoder keys: {action_encoder_keys_loaded}")
                print(f"  Other keys: {other_keys_loaded}")

                # Load the remapped state dict
                missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
                print(f"  Missing keys (LoRA adapters, expected): {len(missing)}")
                print(f"  Unexpected keys (should be 0): {len(unexpected)}")

                if unexpected:
                    print(f"  WARNING: Found {len(unexpected)} unexpected keys!")
                    print(f"  Sample unexpected keys: {list(unexpected)[:5]}")

                # Verify action_encoder was loaded
                action_missing = [k for k in missing if 'action_encoder' in k]
                if action_missing:
                    print(f"  ⚠️  WARNING: {len(action_missing)} action_encoder keys NOT loaded!")
                    for k in action_missing[:5]:
                        print(f"    - {k}")
                else:
                    print(f"  ✓ Action encoder weights successfully loaded from checkpoint")

                # Verify UNet base weights were loaded (exclude LoRA adapter weights)
                unet_missing = [k for k in missing if k.startswith('unet.base_model.model.') and not any(lora_key in k for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'])]
                if unet_missing:
                    print(f"  ⚠️  WARNING: {len(unet_missing)} UNet base weights NOT loaded!")
                    for k in unet_missing[:5]:
                        print(f"    - {k}")
                else:
                    print(f"  ✓ UNet base weights successfully loaded from checkpoint")
        else:
            model.load_state_dict(state_dict, strict=True)

    model.to(accelerator.device)
    model.train()

    # Setup optimizer - only optimize trainable parameters
    trainable_params = []
    if hasattr(args, 'use_lora') and args.use_lora:
        # Add LoRA parameters from UNet
        trainable_params.extend(filter(lambda p: p.requires_grad, model.unet.parameters()))
        # Add action_encoder parameters if training it
        if hasattr(args, 'train_action_encoder') and args.train_action_encoder:
            trainable_params.extend(model.action_encoder.parameters())
    else:
        # Full fine-tuning: train all model parameters that require grad
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Learning rate scheduler (if enabled)
    lr_scheduler = None
    if hasattr(args, 'use_lr_scheduler') and args.use_lr_scheduler:
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type if hasattr(args, 'lr_scheduler_type') else "cosine",
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps if hasattr(args, 'lr_warmup_steps') else 500,
            num_training_steps=args.max_train_steps,
            scheduler_specific_kwargs={
                "num_cycles": args.lr_num_cycles if hasattr(args, 'lr_num_cycles') else 0.5
            } if hasattr(args, 'lr_num_cycles') else {}
        )
        logger.info(f"Using LR scheduler: {args.lr_scheduler_type} with {args.lr_warmup_steps} warmup steps")

    # logs
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tag = args.tag
    run_name = f"train_{now}_{tag}"
    config = {
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "max_train_steps": args.max_train_steps,
        "max_grad_norm": args.max_grad_norm,
        "checkpointing_steps": args.checkpointing_steps,
        "validation_steps": args.validation_steps,
        "num_frames": args.num_frames,
        "num_history": args.num_history,
        "width": args.width,
        "height": args.height,
        "tag": args.tag,
    }
    accelerator.init_trackers(args.wandb_project_name, config=config, init_kwargs={"wandb":{"name":run_name}})

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n" + "="*60)
        print("MODEL PARAMETER COUNT")
        print("="*60)
        # count parameters num in each part
        num_params_total = sum(p.numel() for p in model.unet.parameters())
        num_params_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        print(f"UNet total parameters: {num_params_total/1e6:.2f}M")
        print(f"UNet trainable parameters: {num_params_trainable/1e6:.2f}M ({100*num_params_trainable/num_params_total:.2f}%)")

        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"VAE parameters: {num_params/1e6:.2f}M (frozen)")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Image encoder parameters: {num_params/1e6:.2f}M (frozen)")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Text encoder parameters: {num_params/1e6:.2f}M (frozen)")

        num_params_total_ae = sum(p.numel() for p in model.action_encoder.parameters())
        num_params_trainable_ae = sum(p.numel() for p in model.action_encoder.parameters() if p.requires_grad)
        status = "trainable" if num_params_trainable_ae > 0 else "frozen"
        print(f"Action encoder parameters: {num_params_total_ae/1e6:.2f}M ({status})")

        # Total trainable
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())
        print("-"*60)
        print(f"TOTAL trainable parameters: {total_trainable/1e6:.2f}M / {total_all/1e6:.2f}M ({100*total_trainable/total_all:.2f}%)")
        print("="*60 + "\n")

    # train and val datasets
    from dataset.dataset_droid_exp33 import Dataset_mix
    import copy
    train_dataset = Dataset_mix(args,mode='train')

    # Define multiple validation datasets with their metric prefixes
    # Format: (dataset_name, dataset_cfg, metric_prefix)
    validation_configs = [
        (args.dataset_names, "val"),  # Regular val split
        # ("droid_validation", "droid_validation", "droid_val"),  # Check for forgetting
        # ("irom_test", "irom_test", "irom_val"),  # Check for overfitting
        # Add more validation datasets here as needed, e.g.:
        # ("new_dataset_name", "new_dataset_cfg", "new_val"),
    ]

    # Create validation datasets
    val_datasets = []
    for dataset_name, prefix in validation_configs:
        val_args = copy.deepcopy(args)
        val_args.dataset_names = dataset_name
        # val_args.dataset_cfgs = dataset_cfg
        val_args.prob = [1.0]
        val_dataset = Dataset_mix(val_args, mode='val')
        val_datasets.append((val_dataset, prefix))

    # Initialize curriculum learning if enabled
    curriculum_sampler = None
    if getattr(args, 'use_curriculum', False):
        print(f"Initializing curriculum learning with {args.curriculum_schedule_type} schedule")

        # Create curriculum scheduler
        curriculum_total_steps = getattr(args, 'curriculum_total_steps', None) or args.max_train_steps
        scheduler = CurriculumScheduler(
            schedule_type=args.curriculum_schedule_type,
            total_steps=curriculum_total_steps,
            num_levels=5,
            warmup_steps=getattr(args, 'curriculum_warmup_steps', 1000),
            stabilization_steps=getattr(args, 'curriculum_stabilization_steps', 0),
            initial_dist=getattr(args, 'curriculum_initial_dist', None),
            final_dist=getattr(args, 'curriculum_final_dist', None),
            schedule_params=getattr(args, 'curriculum_schedule_params', None),
        )
        train_dataset.curriculum_scheduler = scheduler

        # Create initial sample weights
        initial_weights = train_dataset.get_sample_weights(current_step=0)
        if initial_weights is not None:
            curriculum_sampler = DynamicCurriculumSampler(
                train_dataset,
                initial_weights,
                num_samples=len(train_dataset)
            )
            print(f"Curriculum sampler created with {len(train_dataset)} samples")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=curriculum_sampler if curriculum_sampler else None,
        shuffle=args.shuffle if curriculum_sampler is None else False,  # Don't shuffle if using sampler
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Prepare everything with our accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # With dispatch_batches=True, the DynamicCurriculumSampler is preserved at
    # train_dataloader.batch_sampler.sampler (not train_dataloader.sampler)
    if curriculum_sampler is not None:
        if hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'sampler'):
            curriculum_sampler = train_dataloader.batch_sampler.sampler
            print(f"Found DynamicCurriculumSampler at batch_sampler.sampler (type: {type(curriculum_sampler).__name__})")
        else:
            print(f"Warning: Could not find DynamicCurriculumSampler in prepared dataloader!")
            print(f"Dataloader type: {type(train_dataloader)}, sampler type: {type(train_dataloader.sampler)}")
            curriculum_sampler = None
   
    ############################ training ##############################
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * args.gradient_accumulation_steps*total_batch_size / len(train_dataloader))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    global_step = 0
    forward_step=0
    train_loss = 0.0
    grad_norm = 0.0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()
                accelerator.backward(loss_gen)
                params_to_clip = model.parameters()
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    grad_norm += total_norm.item()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                forward_step += 1

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Update curriculum weights periodically
                curriculum_update_interval = getattr(args, 'curriculum_update_interval', 100)
                if curriculum_sampler and global_step % curriculum_update_interval == 0:
                    new_weights = train_dataset.get_sample_weights(global_step)
                    curriculum_sampler.update_weights(new_weights)
                    if accelerator.is_main_process and global_step % (curriculum_update_interval * 10) == 0:
                        logger.info(f"[Curriculum] Updated weights at step {global_step}, weights sum: {new_weights.sum().item():.2f}")

                # log loss and lr every N steps (configurable)
                logging_steps = args.logging_steps if hasattr(args, 'logging_steps') else 100
                if global_step % logging_steps == 0:
                    avg_train_loss = train_loss / logging_steps
                    avg_grad_norm = grad_norm / logging_steps
                    progress_bar.set_postfix({"loss": avg_train_loss})

                    # Prepare logging dictionary
                    logs = {
                        "train_loss": avg_train_loss,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "grad_norm": avg_grad_norm
                    }

                    # Add curriculum metrics if enabled
                    if getattr(args, 'use_curriculum', False) and train_dataset.curriculum_scheduler:
                        level_probs = train_dataset.curriculum_scheduler.get_level_probabilities(global_step)
                        curriculum_logs = {
                            f"curriculum/level_{i}_prob": prob
                            for i, prob in enumerate(level_probs)
                        }
                        # Add overall curriculum progress
                        curriculum_logs['curriculum/progress'] = min(1.0, global_step / args.max_train_steps)
                        logs.update(curriculum_logs)

                    accelerator.log(logs, step=global_step)
                    train_loss = 0.0
                    grad_norm = 0.0
                # save ckpt every checkpointing_steps
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)

                    if hasattr(args, 'use_lora') and args.use_lora:
                        # Optionally save full model state (includes LoRA weights + everything)
                        if args.save_full_checkpoint:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                            torch.save(unwrapped_model.state_dict(), save_path)
                            logger.info(f"Saved full checkpoint to {save_path}")

                        # Always save LoRA weights separately (lightweight)
                        lora_save_dir = os.path.join(args.output_dir, f"lora_only-{global_step}")
                        os.makedirs(lora_save_dir, exist_ok=True)
                        unwrapped_model.unet.save_pretrained(lora_save_dir)
                        logger.info(f"Saved LoRA adapters to {lora_save_dir}")

                        # Save action encoder if it's being trained
                        if hasattr(args, 'train_action_encoder') and args.train_action_encoder:
                            action_encoder_path = os.path.join(lora_save_dir, "action_encoder.pt")
                            torch.save(unwrapped_model.action_encoder.state_dict(), action_encoder_path)
                            logger.info(f"Saved action encoder to {action_encoder_path}")
                    else:
                        # Save full model for non-LoRA training
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        torch.save(unwrapped_model.state_dict(), save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                # generate video every validation_steps (including step 0)
                if global_step % args.validation_steps == 1: # or global_step == 1:
                    model.eval()  # All ranks must call this to stay synchronized

                    # Ensure all ranks wait before validation starts
                    accelerator.wait_for_everyone()

                    # Only rank 0 does validation, other ranks just wait
                    if accelerator.is_main_process:
                        print(f"[VALIDATION][Rank 0] Starting validation on {len(val_datasets)} datasets", flush=True)

                        with accelerator.autocast():
                            for dataset_idx, (val_dataset, metric_prefix) in enumerate(val_datasets):
                                print(f"[VALIDATION][Rank 0] Processing dataset {dataset_idx+1}/{len(val_datasets)}: {metric_prefix}", flush=True)

                                # Process all validation samples on rank 0
                                all_metrics = []
                                for sample_id in range(args.video_num):
                                    print(f"[VALIDATION][Rank 0] Processing sample {sample_id+1}/{args.video_num} for {metric_prefix}", flush=True)
                                    metrics = validate_video_generation(
                                        model, val_dataset, args, global_step, args.output_dir,
                                        sample_id, accelerator, metric_prefix=metric_prefix,
                                        save_videos=True
                                    )
                                    all_metrics.append(metrics)
                                    print(f"[VALIDATION][Rank 0] Completed sample {sample_id+1}/{args.video_num} for {metric_prefix}", flush=True)

                                # Compute average metrics
                                if all_metrics:
                                    avg_metrics = {
                                        f"{metric_prefix}/mse_overall": sum(m['overall_mse'] for m in all_metrics) / len(all_metrics),
                                        f"{metric_prefix}/mse_left": sum(m['left_mse'] for m in all_metrics) / len(all_metrics),
                                        f"{metric_prefix}/mse_right": sum(m['right_mse'] for m in all_metrics) / len(all_metrics),
                                        f"{metric_prefix}/mse_wrist": sum(m['wrist_mse'] for m in all_metrics) / len(all_metrics),
                                    }
                                    accelerator.log(avg_metrics, step=global_step)
                                    print(f"[VALIDATION][Rank 0] Logged metrics for {metric_prefix}: {avg_metrics}", flush=True)
                                else:
                                    print(f"[VALIDATION][Rank 0] WARNING: No validation metrics collected for {metric_prefix}", flush=True)

                        print(f"[VALIDATION][Rank 0] Completed all validation datasets", flush=True)
                    else:
                        print(f"[VALIDATION][Rank {accelerator.process_index}] Waiting for rank 0 to complete validation...", flush=True)

                    # Ensure all ranks wait for validation to complete before continuing training
                    accelerator.wait_for_everyone()
                    print(f"[VALIDATION][Rank {accelerator.process_index}] Validation complete, resuming training", flush=True)
                    model.train()  # All ranks must call this to stay synchronized



def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)
    
            

def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True, metric_prefix="val", save_videos=True):
    print(f"[VIDGEN][Rank {accelerator.process_index}] START for sample {id}, {metric_prefix}, save_videos={save_videos}", flush=True)

    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    print(f"[VIDGEN][Rank {accelerator.process_index}] Loading data from dataset for sample {id}", flush=True)
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    # text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print(f"[VIDGEN][Rank {accelerator.process_index}] Data loaded for sample {id}: image {current_latent.shape}, action {actions.shape}", flush=True)
    assert current_latent.shape[1:] == (4, 72, 40)
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    print(f"[VIDGEN][Rank {accelerator.process_index}] Starting inference for sample {id}", flush=True)
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = model.module.action_encoder(actions, frame_level_cond=args.frame_level_cond) if accelerator.num_processes > 1 else model.action_encoder(actions, frame_level_cond=args.frame_level_cond) # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
        print(f"[VIDGEN][Rank {accelerator.process_index}] Inference completed for sample {id}", flush=True)

    # Compute MSE metrics before rearranging
    print(f"[VIDGEN][Rank {accelerator.process_index}] Computing metrics for sample {id}", flush=True)
    overall_mse = torch.nn.functional.mse_loss(pred_latents, future_latent_ft).item()

    # Rearrange to split views for per-view metrics
    pred_latents_views = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
    future_latent_views = einops.rearrange(future_latent_ft, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

    # Compute per-view MSE (left, right, wrist)
    bsz_total = pred_latents_views.shape[0]
    left_mse = torch.nn.functional.mse_loss(pred_latents_views[0::3], future_latent_views[0::3]).item()
    right_mse = torch.nn.functional.mse_loss(pred_latents_views[1::3], future_latent_views[1::3]).item()
    wrist_mse = torch.nn.functional.mse_loss(pred_latents_views[2::3], future_latent_views[2::3]).item()

    # Store metrics to return for averaging
    metrics = {
        'overall_mse': overall_mse,
        'left_mse': left_mse,
        'right_mse': right_mse,
        'wrist_mse': wrist_mse,
    }

    # Only decode and save videos if requested (to save time on non-main ranks)
    if save_videos:
        print(f"[VIDGEN][Rank {accelerator.process_index}] Starting video decoding for sample {id}", flush=True)
        pred_latents = pred_latents_views
        video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
        video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)

        # decode latent
        if video_gt.shape[2] != 3:
            decoded_video = []
            bsz,frame_num = video_gt.shape[:2]
            video_gt = video_gt.flatten(0,1)
            decode_kwargs = {}
            for i in range(0,video_gt.shape[0],args.decode_chunk_size):
                chunk = video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
                decode_kwargs["num_frames"] = chunk.shape[0]
                decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
            video_gt = torch.cat(decoded_video,dim=0)
            video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])

            decoded_video = []
            bsz,frame_num = pred_latents.shape[:2]
            pred_latents = pred_latents.flatten(0,1)
            decode_kwargs = {}
            for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
                chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
                decode_kwargs["num_frames"] = chunk.shape[0]
                decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
            videos = torch.cat(decoded_video,dim=0)
            videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

        video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1)*255)
        video_gt = video_gt.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)
        videos = np.concatenate([video_gt[:, :args.num_history],videos],axis=1) #(2,16,512,256,3)
        videos = np.concatenate([video_gt,videos],axis=-3) #(2,16,512,256,3)
        videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)

        os.makedirs(f"{videos_dir}/samples", exist_ok=True)
        filename = f"{videos_dir}/samples/{metric_prefix}_train_steps_{train_steps}_{id}.mp4"
        print(f"[VIDGEN][Rank {accelerator.process_index}] Saving video to {filename}", flush=True)
        mediapy.write_video(filename, videos, fps=2)
        print(f"[VIDGEN][Rank {accelerator.process_index}] Video saved for sample {id}", flush=True)

    print(f"[VIDGEN][Rank {accelerator.process_index}] COMPLETE for sample {id}", flush=True)
    return metrics 



if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    # dataset_names
    parser.add_argument('--dataset_names', type=str, default=None)
    # tag
    parser.add_argument('--tag', type=str, default=None)
    args_new = parser.parse_args()
    
    if args_new.config == "droid_irom_finetune":
        wm_args = wm_args_full
    elif args_new.config == "droid_irom_finetune_curriculum":
        wm_args = wm_args_curriculum
    # elif args_new.config == "droid_irom_finetune_lora":
    #     wm_args = wm_args_lora
    # elif args_new.config == "droid_irom_finetune_small":
    #     wm_args = wm_args_small
    # elif args_new.config == "droid_irom_finetune_continue":
    #     wm_args = wm_args_cont
    else:
        raise NotImplementedError(f"Unknown config: {args_new.config}")
    args = wm_args()

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args
    
    args = merge_args(args, args_new)
    
    args.tag = f'{args.tag}-{args.dataset_names}'
    args.output_dir = f"model_ckpt/{args.tag}"
    args.wandb_run_name = args.tag
    
    # args.dataset_cfgs = args.dataset_names  # for simplicity, use the same names for cfgs
    
    # change args.prob if there are multiple datasets (connected by "+", e.g. "droid+irom")
    if '+' in args.dataset_names:
        dataset_list = args.dataset_names.split('+')
        prob = [1.0/len(dataset_list) for _ in dataset_list]
        args.prob = prob
    
    main(args)

    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info
    # CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 unit_test2.py

    # args = Args()
    # from video_dataset.dataset_droid_exp33 import Dataset_mix
    # dataset = Dataset_mix(args,mode='val')
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    # model = CrtlWorld(args).to('cuda')
    # # print model parameter num
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    # total_elements = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # print(f"Total number of learnable parameters: {total_elements}")
    # model.train()
    

    # for batch in dataloader:
    #     print(batch['latent'].shape)
    #     print(batch['text'])
    #     print(batch['action'].shape)

    #     loss,_ = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())





    # device = 'cuda'
    # video_encoder = VideoEncoder(hidden_size=1024).to(device)
    # # count the parameters of the model
    # num_params = sum(p.numel() for p in video_encoder.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # vae_latent = torch.randn(8, 1, 4, 32, 32).to(device)
    # clip_latent = torch.randn(8, 20, 512).to(device)
    # current_img = video_encoder(vae_latent, clip_latent)
    # print(current_img.shape)  # (8, 1, 4, 32, 32)


    # pos_emb = get_2d_sincos_pos_embed(1024, 16)
    # print(pos_emb.shape)  # (256, 1024)
    # clip_emb = get_1d_sincos_pos_embed_from_grid(1024, np.arange(20))
    # print(clip_emb.shape)  # (20, 512)
