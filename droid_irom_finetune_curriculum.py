"""
Curriculum Learning Configuration for Ctrl-World Training

This config extends droid_irom_finetune.py with curriculum learning enabled.
Use this for training with progressive difficulty sampling.
"""

from droid_irom_finetune import wm_args as wm_args_base
from dataclasses import dataclass


@dataclass
class wm_args(wm_args_base):
    """
    Curriculum learning configuration - inherits all settings from base config
    and overrides curriculum-specific parameters.
    """

    ########################### Curriculum Learning (ENABLED) ###########################
    # Enable curriculum learning
    use_curriculum = True

    # Curriculum schedule type: 'linear', 'exponential', 'step', 'polynomial'
    curriculum_schedule_type = 'linear'

    # Warmup period before curriculum starts (steps)
    curriculum_warmup_steps = 1000

    # Stabilization period - how many steps at the end to maintain final distribution
    # This allows training to stabilize with uniform (or final) sampling at the end
    # Recommended: 20-30% of total training steps for best results
    # Example: 20000 means last 20k steps will use uniform sampling (20% of 100k)
    curriculum_stabilization_steps = 20000

    # How often to update curriculum sample weights (steps)
    curriculum_update_interval = 2000

    # Total steps for curriculum progression (default: same as max_train_steps)
    curriculum_total_steps = None  # Will use max_train_steps (100k)

    # Initial difficulty distribution (bias toward easy samples)
    # [0.6, 0.3, 0.1, 0.0, 0.0] = 60% easiest, 30% easy, 10% medium
    curriculum_initial_dist = [0.6, 0.3, 0.1, 0.0, 0.0]

    # Final difficulty distribution (uniform across all levels)
    # [0.2, 0.2, 0.2, 0.2, 0.2] = 20% each level
    curriculum_final_dist = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Schedule-specific parameters (optional)
    curriculum_schedule_params = None

    ########################### Example Configurations ###########################
    # Uncomment one of the following to try different curriculum strategies:

    # # Strategy 1: Exponential schedule (rapid shift after mastering easy samples)
    # curriculum_schedule_type = 'exponential'
    # curriculum_schedule_params = {'decay_rate': 2.0}

    # # Strategy 2: Step-based schedule (discrete difficulty jumps)
    # curriculum_schedule_type = 'step'
    # curriculum_schedule_params = {
    #     'milestones': [10000, 30000, 60000],
    #     'distributions': [
    #         [0.7, 0.2, 0.1, 0.0, 0.0],  # 0-10k: mostly easy
    #         [0.4, 0.3, 0.2, 0.1, 0.0],  # 10k-30k: mixed
    #         [0.2, 0.2, 0.3, 0.2, 0.1],  # 30k-60k: more hard
    #         [0.2, 0.2, 0.2, 0.2, 0.2],  # 60k+: uniform
    #     ]
    # }

    # # Strategy 3: Polynomial schedule (slower start)
    # curriculum_schedule_type = 'polynomial'
    # curriculum_schedule_params = {'power': 2.0}  # quadratic progression

    # # Strategy 4: More aggressive easy bias initially
    # curriculum_initial_dist = [0.8, 0.15, 0.05, 0.0, 0.0]

    # # Strategy 5: End with hard bias (not uniform)
    # curriculum_final_dist = [0.0, 0.1, 0.2, 0.3, 0.4]

    # # Strategy 6: Shorter curriculum (first 50k steps only)
    # curriculum_total_steps = 50000
