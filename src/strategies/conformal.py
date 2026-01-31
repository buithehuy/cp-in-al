"""Conformal Prediction based acquisition strategies."""
import torch
import numpy as np
from .base import AcquisitionStrategy


class CPSizeSampling(AcquisitionStrategy):
    """CP Size sampling - select samples with largest prediction sets."""
    
    def __init__(self):
        super().__init__(name="cp_size")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with largest prediction set sizes.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: Conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        set_sizes = (probs >= (1 - qhat)).sum(dim=1).float()
        return torch.topk(set_sizes, budget)[1]


class CPVShapedSampling(AcquisitionStrategy):
    """CP V-shaped sampling - prioritize both set_size=0 (overconfident) and large sets."""
    
    def __init__(self):
        super().__init__(name="cp_v_shaped")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with V-shaped scoring.
        
        Prioritizes:
        - set_size=0: Overconfident errors (highest priority)
        - set_size=1: Confident correct predictions (lowest priority)  
        - set_size>1: Uncertain predictions (increasing priority)
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: Conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        num_classes = probs.shape[1]
        set_sizes = (probs >= (1 - qhat)).sum(dim=1).float()
        
        # V-shaped scoring
        score = torch.where(
            set_sizes == 0,
            torch.tensor(num_classes + 1.0),  # Highest score for empty sets
            torch.where(
                set_sizes == 1,
                torch.tensor(0.0),  # Lowest score for singleton sets
                set_sizes  # Increasing score for larger sets
            )
        )
        return torch.topk(score, budget)[1]


class CombinedSampling(AcquisitionStrategy):
    """Combined sampling - Entropy + CP Size (equal weighting)."""
    
    def __init__(self):
        super().__init__(name="combined")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples using combined entropy and CP size score.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: Conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        num_classes = probs.shape[1]
        
        # Normalized entropy
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        entropy_norm = entropy / np.log(num_classes)
        
        # Normalized set sizes
        set_sizes = (probs >= (1 - qhat)).sum(dim=1).float()
        set_sizes_norm = set_sizes / num_classes
        
        # Combined score (equal weighting)
        combined_score = 0.5 * entropy_norm + 0.5 * set_sizes_norm
        return torch.topk(combined_score, budget)[1]


class CombinedVShapedSampling(AcquisitionStrategy):
    """Combined V-shaped sampling - Entropy + V-shaped CP score."""
    
    def __init__(self):
        super().__init__(name="combined_v_shaped")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples using combined entropy and V-shaped CP score.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: Conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        num_classes = probs.shape[1]
        
        # Normalized entropy
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        entropy_norm = entropy / np.log(num_classes)
        
        # V-shaped CP score
        set_sizes = (probs >= (1 - qhat)).sum(dim=1).float()
        cp_score = torch.where(
            set_sizes == 0,
            torch.tensor(1.0),  # Max score for empty sets
            torch.where(
                set_sizes == 1,
                torch.tensor(0.0),  # Min score for singleton sets
                set_sizes / num_classes  # Normalized for larger sets
            )
        )
        
        # Combined score (equal weighting)
        combined_score = 0.5 * entropy_norm + 0.5 * cp_score
        return torch.topk(combined_score, budget)[1]
