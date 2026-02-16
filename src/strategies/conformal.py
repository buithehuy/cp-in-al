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


class CPVShapedEntropySampling(AcquisitionStrategy):
    """CP V-shaped sampling with entropy-based prioritization for zero-setsize samples.
    
    Improves upon cp_v_shaped by using entropy to rank samples with set_size=0:
    - set_size=0: Ranked by entropy (higher entropy = higher priority)
    - set_size=1: Confident correct predictions (lowest priority)
    - set_size>1: Uncertain predictions (increasing priority)
    """
    
    def __init__(self):
        super().__init__(name="cp_v_shaped_entropy")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with V-shaped scoring and entropy-based zero-set ranking.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: Conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        num_classes = probs.shape[1]
        set_sizes = (probs >= (1 - qhat)).sum(dim=1).float()
        
        # Calculate entropy for all samples
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        entropy_norm = entropy / np.log(num_classes)
        
        # V-shaped scoring with entropy for zero-setsize samples
        score = torch.where(
            set_sizes == 0,
            # For empty sets: base score + entropy bonus
            torch.tensor(num_classes + 1.0) + entropy_norm,
            torch.where(
                set_sizes == 1,
                torch.tensor(0.0),  # Lowest score for singleton sets
                set_sizes  # Increasing score for larger sets
            )
        )
        return torch.topk(score, budget)[1]


# class CPAPSSampling(AcquisitionStrategy):
#     """Adaptive Prediction Sets (APS) - conformal prediction with cumulative probability.
    
#     APS differs from standard conformal prediction by using an adaptive threshold:
#     - Sort class probabilities in descending order
#     - Include classes cumulatively until sum exceeds 1 - qhat
#     - Produces smaller, more focused prediction sets
#     - Selection prioritizes samples with larger APS sets (more uncertain)
#     """
    
#     def __init__(self):
#         super().__init__(name="cp_aps")
    
#     def select(self, probs, budget, qhat, **kwargs):
#         """Select samples with largest APS prediction set sizes.
        
#         Args:
#             probs: Probability tensor of shape (n_samples, n_classes)
#             budget: Number of samples to select
#             qhat: APS conformity score threshold (cumulative probability)
            
#         Returns:
#             Tensor of selected indices
#         """
#         # Sort probabilities in descending order for each sample
#         sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        
#         # Compute cumulative sum of sorted probabilities
#         cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
#         # For APS, qhat is already a cumulative probability threshold
#         # Find where cumulative sum first exceeds qhat
#         threshold = qhat
        
#         # For each sample, find the index where cumsum first exceeds threshold
#         # Add 1 because we need to include that class
#         set_sizes = torch.zeros(probs.shape[0])
#         for i in range(probs.shape[0]):
#             # Find first index where cumsum exceeds threshold
#             exceeds = (cumsum_probs[i] >= threshold).nonzero(as_tuple=True)[0]
#             if len(exceeds) > 0:
#                 set_sizes[i] = exceeds[0].item() + 1  # +1 to include that class
#             else:
#                 # If never exceeds threshold, include all classes
#                 set_sizes[i] = probs.shape[1]
        
#         # Select samples with largest set sizes (most uncertain)
#         return torch.topk(set_sizes, budget)[1]

import torch

class CPAPSSampling(AcquisitionStrategy):
    def __init__(self):
        super().__init__(name="cp_aps")
    
    def select(self, probs, budget, qhat, **kwargs):
        # 1. Sort xác suất
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
        # 2. Tính set_sizes (như cũ)
        is_in_set = cumsum_probs < qhat
        set_sizes = is_in_set.sum(dim=1).float() + 1
        
        # 3. CẢI TIẾN: Tạo scoring liên tục thay vì jitter
        # Lấy tổng tích lũy ngay trước phần tử cuối cùng lọt vào set
        # shifted_cumsum giúp lấy giá trị tại (index - 1)
        shifted_cumsum = torch.cat([torch.zeros(probs.shape[0], 1).to(probs.device), cumsum_probs[:, :-1]], dim=1)
        prev_cumsum = torch.gather(shifted_cumsum, 1, (set_sizes.long() - 1).unsqueeze(1)).squeeze()
        
        # Lấy xác suất của chính phần tử khiến set size nhảy bậc
        current_prob = torch.gather(sorted_probs, 1, (set_sizes.long() - 1).unsqueeze(1)).squeeze()
        
        # Soft Score: Phần dư tỉ lệ thuận với độ mập mờ tại ngưỡng qhat
        # Càng gần qhat, score càng cao
        soft_score = (qhat - prev_cumsum) / (current_prob + 1e-9)
        
        # Kết hợp: Set size là ưu tiên 1, soft_score là ưu tiên 2 (liên tục)
        uncertainty_score = set_sizes + soft_score
        
        _, indices = torch.topk(uncertainty_score, budget)
        return indices


class RMCPSampling(AcquisitionStrategy):
    """RMCP (Relative Margin Conformal Prediction) sampling strategy.
    
    Uses relative margin scores: score_k = p_k - mean(p_j for j != k)
    Selects samples with largest prediction sets based on RMCP.
    
    This strategy combines margin-based uncertainty with conformal prediction,
    measuring how much each class exceeds the average of other classes.
    """
    
    def __init__(self):
        super().__init__(name="cp_rmcp")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with largest RMCP prediction set sizes.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: RMCP conformity score threshold
            
        Returns:
            Tensor of selected indices
        """
        n_samples, n_classes = probs.shape
        set_sizes = torch.zeros(n_samples)
        
        for i in range(n_samples):
            # Compute relative margin scores for all classes
            scores = torch.zeros(n_classes)
            
            for k in range(n_classes):
                # score_k = p_k - mean(p_j for j != k)
                other_probs = torch.cat([probs[i, :k], probs[i, k+1:]])
                scores[k] = probs[i, k] - other_probs.mean()
            
            # Count classes with score >= qhat
            set_sizes[i] = (scores >= qhat).sum().float()
        
        # Select samples with largest set sizes (most uncertain)
        return torch.topk(set_sizes, budget)[1]


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
