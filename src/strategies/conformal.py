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


class CPAPSSampling(AcquisitionStrategy):
    """Adaptive Prediction Sets (APS) - conformal prediction with cumulative probability.
    
    APS differs from standard conformal prediction by using an adaptive threshold:
    - Sort class probabilities in descending order
    - Include classes cumulatively until sum exceeds 1 - qhat
    - Produces smaller, more focused prediction sets
    - Selection prioritizes samples with larger APS sets (more uncertain)
    """
    
    def __init__(self):
        super().__init__(name="cp_aps")
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with largest APS prediction set sizes.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            qhat: APS conformity score threshold (cumulative probability)
            
        Returns:
            Tensor of selected indices
        """
        # Sort probabilities in descending order for each sample
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        
        # Compute cumulative sum of sorted probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
        # For APS, qhat is already a cumulative probability threshold
        # Find where cumulative sum first exceeds qhat
        threshold = qhat
        
        # For each sample, find the index where cumsum first exceeds threshold
        # Add 1 because we need to include that class
        set_sizes = torch.zeros(probs.shape[0])
        for i in range(probs.shape[0]):
            # Find first index where cumsum exceeds threshold
            exceeds = (cumsum_probs[i] >= threshold).nonzero(as_tuple=True)[0]
            if len(exceeds) > 0:
                set_sizes[i] = exceeds[0].item() + 1  # +1 to include that class
            else:
                # If never exceeds threshold, include all classes
                set_sizes[i] = probs.shape[1]
        
        # Select samples with largest set sizes (most uncertain)
        return torch.topk(set_sizes, budget)[1]

import torch
import torch.nn.functional as F

class ConformalBoundaryUncertaintySampling(AcquisitionStrategy):
    """Conformal Boundary Uncertainty (CBU) - APS Version.
    
    This strategy measures how many classes in each sample are 'uncertain' 
    relative to the APS threshold (qhat). It uses the variance of the 
    soft-inclusion Bernoulli distribution:
    
        U(x) = sum_{y} sigma(z) * (1 - sigma(z))
        where z = (qhat - cumsum_probs) / T
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__(name="cp_boundary")
        self.temperature = temperature
    
    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with highest uncertainty at the APS boundary.
        
        Args:
            probs: Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat: Marginal APS threshold from calibration
        """
        # 1. Tính APS Score (Cumulative Sum) - Vectorized
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
        # 2. Tính khoảng cách mềm đến biên qhat
        # z > 0: lớp nằm trong tập dự đoán
        # z < 0: lớp nằm ngoài tập dự đoán
        z = (qhat - cumsum_probs) / self.temperature
        
        # 3. Tính Conformal Boundary Uncertainty (CBU)
        # sigma * (1 - sigma) đạt cực đại tại z = 0 (tức là cumsum_probs = qhat)
        sig = torch.sigmoid(z)
        u_per_class = sig * (1.0 - sig)
        
        # 4. Aggregate: Mẫu nào có nhiều lớp 'mấp mé' biên APS nhất sẽ có score cao nhất
        uncertainty = u_per_class.sum(dim=1)
        
        # Trả về top K mẫu có Uncertainty cao nhất
        return torch.topk(uncertainty, budget)[1]

class CPSetPartitionMISampling(AcquisitionStrategy):
    """Conformal Set-Partition Mutual Information (CSPMI) — novel strategy.

    The conformal prediction set C(x) = {y : 1 - p_y <= qhat} partitions the
    label space into two groups: "plausible" classes (inside the set) and
    "rejected" classes (outside). This partition carries *information* about
    the true label y under the model's current belief p(y|x).

    CSPMI measures exactly that information gain — the mutual information
    between the label y and the binary conformal-membership indicator I[y∈C(x)]:

        CSPMI(x) = I(y ; I[y ∈ C(x)])
                 = H(p) - [P_in × H(p_in) + P_out × H(p_out)]

    where:
        P_in   = Σ_{y∈C} p_y          (total prob mass inside the set)
        p_in   = renormalized p over classes inside  the set
        p_out  = renormalized p over classes outside the set

    --- Why this beats entropy ---

    | Scenario                              | H(p) | CSPMI |
    |---------------------------------------|------|-------|
    | True decision boundary (2 classes)    | High | High  |  ← want this
    | Noisy/OOD  (set = all C classes)      | Max  | ~0    |  ← entropy fails
    | Overconfident-wrong (set is empty)    | Low  | High  |  ← entropy fails

    Key property: when C(x) = all classes (pure noise), P_in → 1,
    H(p_in) → H(p), so CSPMI → 0. Entropy would give H_max here.

    Fully conformal: qhat computed via marginal compute_qhat on calib set.
    """

    def __init__(self):
        super().__init__(name="cp_spm_info")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with highest conformal set-partition mutual information.

        Args:
            probs:  Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   Marginal conformal threshold from calibration

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        eps = 1e-9
        n, C = probs.shape

        # ── Conformal set membership: class y is "in" iff p_y >= 1 - qhat ──
        in_set  = (probs >= (1.0 - qhat)).float()   # (n, C)  {0, 1}
        out_set = 1.0 - in_set                       # (n, C)  {0, 1}

        # ── Total entropy H(p) ───────────────────────────────────────────────
        H_total = -(probs * torch.log(probs + eps)).sum(dim=1)   # (n,)

        # ── Within-set distribution and entropy ──────────────────────────────
        # P_in: total probability mass inside the conformal set
        P_in = (probs * in_set).sum(dim=1).clamp(min=eps)         # (n,)

        # p_in: renormalized distribution inside the set
        p_in  = probs * in_set / P_in.unsqueeze(1)                # (n, C)
        H_in  = -(p_in  * torch.log(p_in  + eps) * in_set ).sum(dim=1)  # (n,)

        # ── Out-of-set distribution and entropy ──────────────────────────────
        P_out = (probs * out_set).sum(dim=1).clamp(min=eps)       # (n,)

        p_out = probs * out_set / P_out.unsqueeze(1)              # (n, C)
        H_out = -(p_out * torch.log(p_out + eps) * out_set).sum(dim=1)  # (n,)

        # ── Conditional entropy H(y | set membership) ────────────────────────
        # P_in_raw and P_out_raw WITHOUT clamping, for correct weighting
        P_in_raw  = (probs * in_set ).sum(dim=1)   # (n,)  may be 0
        P_out_raw = (probs * out_set).sum(dim=1)   # (n,)  may be 0

        H_conditional = P_in_raw * H_in + P_out_raw * H_out      # (n,)

        # ── CSPMI = total entropy - conditional entropy ───────────────────────
        mutual_info = (H_total - H_conditional).clamp(min=0.0)    # (n,)

        return torch.topk(mutual_info, budget)[1]


# import torch

# class CPAPSSampling(AcquisitionStrategy):
#     def __init__(self):
#         super().__init__(name="cp_aps")
    
#     def select(self, probs, budget, qhat, **kwargs):
#         # 1. Sort xác suất
#         sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
#         cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
#         # 2. Tính set_sizes (như cũ)
#         is_in_set = cumsum_probs < qhat
#         set_sizes = is_in_set.sum(dim=1).float() + 1
        
#         # 3. CẢI TIẾN: Tạo scoring liên tục thay vì jitter
#         # Lấy tổng tích lũy ngay trước phần tử cuối cùng lọt vào set
#         # shifted_cumsum giúp lấy giá trị tại (index - 1)
#         shifted_cumsum = torch.cat([torch.zeros(probs.shape[0], 1).to(probs.device), cumsum_probs[:, :-1]], dim=1)
#         prev_cumsum = torch.gather(shifted_cumsum, 1, (set_sizes.long() - 1).unsqueeze(1)).squeeze()
        
#         # Lấy xác suất của chính phần tử khiến set size nhảy bậc
#         current_prob = torch.gather(sorted_probs, 1, (set_sizes.long() - 1).unsqueeze(1)).squeeze()
        
#         # Soft Score: Phần dư tỉ lệ thuận với độ mập mờ tại ngưỡng qhat
#         # Càng gần qhat, score càng cao
#         soft_score = (qhat - prev_cumsum) / (current_prob + 1e-9)
        
#         # Kết hợp: Set size là ưu tiên 1, soft_score là ưu tiên 2 (liên tục)
#         uncertainty_score = set_sizes + soft_score
        
#         _, indices = torch.topk(uncertainty_score, budget)
#         return indices


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


# class ConformalBoundaryUncertaintySampling(AcquisitionStrategy):
#     """Conformal Boundary Uncertainty (CBU) sampling strategy.

#     Measures how close each sample sits to the conformal decision boundary
#     using a soft sigmoid-based uncertainty score:

#         U(x) = sum_{y=1}^{C} sigma((q_hat - s(x,y)) / T_eff)
#                              * (1 - sigma((q_hat - s(x,y)) / T_eff))

#     where:
#         - s(x, y) = 1 - p_y  (marginal conformal non-conformity score)
#         - q_hat is the marginal conformal threshold (same as cp_size)
#         - sigma is the sigmoid function
#         - T_eff = std(q_hat - s) * T_scale  (adaptive temperature)

#     T_eff được tính **adaptive** từ std của toàn bộ tập unlabeled để tránh
#     sigmoid bão hòa (T_scale mặc định = 1.0, tăng → selection mượt hơn,
#     giảm → chỉ chọn sample cực kỳ sát biên).

#     The product sigma(...) * (1 - sigma(...)) peaks at 0.25 when the argument
#     is 0, i.e. exactly at the conformal boundary q_hat = s(x, y).
#     Samples with high U(x) have many classes hovering near the boundary,
#     indicating high structural uncertainty from a conformal perspective.
#     """

#     def __init__(self, T_scale: float = 1.0):
#         super().__init__(name="cp_boundary_uncertainty")
#         self.T_scale = T_scale

#     def select(self, probs, budget, qhat, **kwargs):
#         """Select K samples with the highest conformal boundary uncertainty.

#         Args:
#             probs:   Probability tensor of shape (n_samples, n_classes)
#             budget:  Number of samples to select (K)
#             qhat:    Marginal conformal threshold (scalar or 0-dim tensor)

#         Returns:
#             Tensor of selected indices (shape: [budget])
#         """
#         # Non-conformity scores: s(x, y) = 1 - p_y  →  shape (n, C)
#         scores = 1.0 - probs  # higher score ↔ model less confident about y

#         # Distance của mỗi (sample, class) tới biên conformal
#         dist = qhat - scores   # shape (n, C)
#         # dist > 0 → class này nằm trong prediction set
#         # dist < 0 → class này nằm ngoài prediction set
#         # dist = 0 → đúng trên biên → đóng góp cao nhất vào U(x)

#         # Adaptive temperature: scale theo std toàn bộ distances
#         # Tránh sigmoid bão hòa khi dist có dải rộng hơn T tĩnh nhiều lần
#         T_eff = dist.std().clamp(min=1e-6) * self.T_scale
#         z = dist / T_eff       # shape (n, C), chuẩn hóa về dải hợp lý

#         # Sigmoid uncertainty: đạt max 0.25 khi z=0 (đúng trên biên)
#         sig = torch.sigmoid(z)            # shape (n, C)
#         u_per_class = sig * (1.0 - sig)  # shape (n, C)

#         # Aggregate over classes → tổng uncertainty của sample
#         uncertainty = u_per_class.sum(dim=1)   # shape (n,)

#         return torch.topk(uncertainty, budget)[1]
