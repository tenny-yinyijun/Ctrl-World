"""
Dynamic Curriculum Sampler

PyTorch sampler that supports runtime weight updates for curriculum learning.
"""

import torch
from torch.utils.data import Sampler


class DynamicCurriculumSampler(Sampler):
    """
    Weighted sampler with dynamic weight updates for curriculum learning.

    This sampler allows updating sample weights during training to implement
    curriculum learning schedules. Weights are updated periodically from the
    training loop to bias sampling toward different difficulty levels.
    """

    def __init__(self, dataset, initial_weights, num_samples):
        """
        Initialize the dynamic curriculum sampler.

        Args:
            dataset: The dataset to sample from
            initial_weights: Initial sample weights (torch.Tensor of shape [num_samples])
            num_samples: Number of samples per epoch
        """
        self.dataset = dataset
        self.weights = initial_weights
        self.num_samples = num_samples

        # Validate inputs
        if self.weights is not None:
            if not isinstance(self.weights, torch.Tensor):
                raise TypeError(f"initial_weights must be torch.Tensor, got {type(self.weights)}")
            if len(self.weights) != num_samples:
                raise ValueError(f"initial_weights length ({len(self.weights)}) != num_samples ({num_samples})")

    def update_weights(self, new_weights):
        """
        Update curriculum weights (called from training loop).

        Args:
            new_weights: New sample weights (torch.Tensor or None)
        """
        if new_weights is not None:
            if not isinstance(new_weights, torch.Tensor):
                raise TypeError(f"new_weights must be torch.Tensor, got {type(new_weights)}")
            if len(new_weights) != self.num_samples:
                raise ValueError(f"new_weights length ({len(new_weights)}) != num_samples ({self.num_samples})")
            self.weights = new_weights

    def __iter__(self):
        """
        Generate sample indices for one epoch.

        Returns:
            Iterator over sample indices
        """
        # Weighted random sampling with replacement
        if self.weights is not None and self.weights.sum() > 0:
            # Use torch.multinomial for weighted sampling
            # This samples indices according to the probability distribution defined by weights
            return iter(torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True
            ).tolist())
        else:
            # Fallback to uniform random sampling if weights are invalid
            return iter(torch.randperm(self.num_samples).tolist())

    def __len__(self):
        """Return the number of samples per epoch"""
        return self.num_samples

    def __repr__(self):
        return (f"DynamicCurriculumSampler(num_samples={self.num_samples}, "
                f"weights_sum={self.weights.sum().item() if self.weights is not None else 0})")
