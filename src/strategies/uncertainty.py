"""Uncertainty-based acquisition strategies."""
import torch
from .base import AcquisitionStrategy


class RandomSampling(AcquisitionStrategy):
    """Random sampling baseline."""
    
    def __init__(self):
        super().__init__(name="random")
    
    def select(self, probs, budget, **kwargs):
        """Select random samples.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            
        Returns:
            Tensor of selected indices
        """
        return torch.topk(torch.rand(len(probs)), budget)[1]


class EntropySampling(AcquisitionStrategy):
    """Entropy-based sampling - select most uncertain samples."""
    
    def __init__(self):
        super().__init__(name="entropy")
    
    def select(self, probs, budget, **kwargs):
        """Select samples with highest entropy.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            
        Returns:
            Tensor of selected indices
        """
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        return torch.topk(entropy, budget)[1]


class LeastConfidenceSampling(AcquisitionStrategy):
    """Least confidence sampling - select samples with lowest max probability."""
    
    def __init__(self):
        super().__init__(name="least_confidence")
    
    def select(self, probs, budget, **kwargs):
        """Select samples with lowest confidence.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            
        Returns:
            Tensor of selected indices
        """
        confidence = probs.max(dim=1)[0]
        return torch.topk(1 - confidence, budget)[1]


class MarginSampling(AcquisitionStrategy):
    """Margin sampling - select samples with smallest margin between top-2 predictions."""
    
    def __init__(self):
        super().__init__(name="margin")
    
    def select(self, probs, budget, **kwargs):
        """Select samples with smallest margin.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            
        Returns:
            Tensor of selected indices
        """
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return torch.topk(-margin, budget)[1]
