"""
Curriculum Learning Scheduler

Maps training progress (steps) to difficulty level sampling probabilities.
Supports multiple schedule types: linear, exponential, step-based, and polynomial.
"""

import numpy as np
from typing import List, Dict, Optional


class CurriculumScheduler:
    """
    Curriculum scheduler that maps training steps to difficulty level probabilities.

    Supports multiple schedule types for flexible curriculum design:
    - Linear: Smooth interpolation from initial to final distribution
    - Exponential: Slow start, rapid shift to harder samples
    - Step: Discrete difficulty jumps at milestones
    - Polynomial: Configurable pacing via power parameter
    """

    def __init__(
        self,
        schedule_type: str = 'linear',
        total_steps: int = 500000,
        num_levels: int = 5,
        warmup_steps: int = 1000,
        stabilization_steps: int = 0,
        initial_dist: Optional[List[float]] = None,
        final_dist: Optional[List[float]] = None,
        schedule_params: Optional[Dict] = None
    ):
        """
        Initialize curriculum scheduler.

        Args:
            schedule_type: Type of schedule ('linear', 'exponential', 'step', 'polynomial')
            total_steps: Total training steps for curriculum progression
            num_levels: Number of difficulty levels (default: 5)
            warmup_steps: Number of warmup steps with uniform sampling
            stabilization_steps: Number of steps at the end to maintain final distribution (for stabilization)
            initial_dist: Initial difficulty distribution (bias toward easy)
            final_dist: Final difficulty distribution
            schedule_params: Additional parameters for specific schedule types
        """
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.num_levels = num_levels
        self.warmup_steps = warmup_steps
        self.stabilization_steps = stabilization_steps
        self.schedule_params = schedule_params or {}

        # Effective curriculum duration (excluding warmup and stabilization)
        self.curriculum_end_step = total_steps - stabilization_steps

        # Set default distributions if not provided
        if initial_dist is None:
            # Strong bias toward easy samples (level 0)
            self.initial_dist = [0.6, 0.3, 0.1, 0.0, 0.0]
        else:
            self.initial_dist = initial_dist

        if final_dist is None:
            # Uniform distribution across all levels
            self.final_dist = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            self.final_dist = final_dist

        # Validate distributions
        self._validate_distributions()

        # For step-based schedule, extract milestones and distributions
        if self.schedule_type == 'step':
            self.milestones = self.schedule_params.get('milestones', [])
            self.step_distributions = self.schedule_params.get('distributions', [])

            # Validate step-based schedule
            if not self.milestones or not self.step_distributions:
                raise ValueError("Step-based schedule requires 'milestones' and 'distributions' in schedule_params")
            if len(self.step_distributions) != len(self.milestones) + 1:
                raise ValueError(f"Number of distributions ({len(self.step_distributions)}) must be len(milestones) + 1 ({len(self.milestones) + 1})")

    def _validate_distributions(self):
        """Validate that distributions are valid probabilities"""
        for dist, name in [(self.initial_dist, 'initial'), (self.final_dist, 'final')]:
            if len(dist) != self.num_levels:
                raise ValueError(f"{name} distribution has {len(dist)} levels, expected {self.num_levels}")
            if not np.isclose(sum(dist), 1.0):
                raise ValueError(f"{name} distribution does not sum to 1.0 (sum={sum(dist)})")
            if any(p < 0 for p in dist):
                raise ValueError(f"{name} distribution has negative probabilities")

    def get_level_probabilities(self, current_step: int) -> List[float]:
        """
        Get difficulty level sampling probabilities for the current training step.

        Args:
            current_step: Current training step

        Returns:
            List of probabilities for each difficulty level (length = num_levels)
        """
        # Warmup period: uniform sampling
        if current_step < self.warmup_steps:
            return [1.0 / self.num_levels] * self.num_levels

        # Stabilization period: maintain final distribution
        if current_step >= self.curriculum_end_step:
            # Normalize final distribution (in case user provided non-normalized)
            total = sum(self.final_dist)
            if total > 0:
                return [p / total for p in self.final_dist]
            else:
                return [1.0 / self.num_levels] * self.num_levels

        # Compute curriculum progress (0.0 to 1.0)
        if self.schedule_type == 'linear':
            progress = self._linear_progress(current_step)
        elif self.schedule_type == 'exponential':
            progress = self._exponential_progress(current_step)
        elif self.schedule_type == 'step':
            return self._step_progress(current_step)
        elif self.schedule_type == 'polynomial':
            progress = self._polynomial_progress(current_step)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Interpolate between initial and final distributions
        level_probs = []
        for i in range(self.num_levels):
            prob = (1 - progress) * self.initial_dist[i] + progress * self.final_dist[i]
            level_probs.append(prob)

        # Normalize to ensure sum = 1.0 (account for floating point errors)
        total = sum(level_probs)
        if total > 0:
            level_probs = [p / total for p in level_probs]
        else:
            # Fallback to uniform if all zero
            level_probs = [1.0 / self.num_levels] * self.num_levels

        return level_probs

    def _linear_progress(self, current_step: int) -> float:
        """Linear progression from 0.0 to 1.0"""
        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.curriculum_end_step - self.warmup_steps
        progress = min(1.0, max(0.0, adjusted_step / adjusted_total))
        return progress

    def _exponential_progress(self, current_step: int) -> float:
        """
        Exponential progression: slow start, rapid shift to harder samples.

        Uses: progress = 1 - exp(-decay_rate * normalized_step)
        """
        decay_rate = self.schedule_params.get('decay_rate', 2.0)

        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.curriculum_end_step - self.warmup_steps
        normalized_step = min(1.0, max(0.0, adjusted_step / adjusted_total))

        # Exponential curve: 1 - exp(-decay_rate * x)
        progress = 1.0 - np.exp(-decay_rate * normalized_step)
        return progress

    def _polynomial_progress(self, current_step: int) -> float:
        """
        Polynomial progression with configurable convexity.

        Uses: progress = (normalized_step) ^ power
        power > 1: slower start, faster end
        power < 1: faster start, slower end
        power = 1: linear (same as linear schedule)
        """
        power = self.schedule_params.get('power', 2.0)

        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.curriculum_end_step - self.warmup_steps
        normalized_step = min(1.0, max(0.0, adjusted_step / adjusted_total))

        progress = normalized_step ** power
        return progress

    def _step_progress(self, current_step: int) -> List[float]:
        """
        Step-based progression: discrete difficulty jumps at milestones.

        Returns distribution directly (no interpolation).
        """
        adjusted_step = current_step - self.warmup_steps

        # Find which milestone phase we're in
        phase_idx = 0
        for i, milestone in enumerate(self.milestones):
            if adjusted_step >= milestone:
                phase_idx = i + 1
            else:
                break

        # Return the distribution for this phase
        distribution = self.step_distributions[phase_idx]

        # Normalize
        total = sum(distribution)
        if total > 0:
            distribution = [p / total for p in distribution]
        else:
            distribution = [1.0 / self.num_levels] * self.num_levels

        return distribution

    def get_progress(self, current_step: int) -> float:
        """
        Get overall curriculum progress (0.0 to 1.0).

        Args:
            current_step: Current training step

        Returns:
            Progress value between 0.0 (start) and 1.0 (complete)
        """
        if current_step < self.warmup_steps:
            return 0.0

        if current_step >= self.curriculum_end_step:
            return 1.0

        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.curriculum_end_step - self.warmup_steps
        progress = min(1.0, max(0.0, adjusted_step / adjusted_total))
        return progress

    def __repr__(self):
        return (f"CurriculumScheduler(schedule_type='{self.schedule_type}', "
                f"total_steps={self.total_steps}, num_levels={self.num_levels}, "
                f"warmup_steps={self.warmup_steps}, stabilization_steps={self.stabilization_steps})")
