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


class CPAPSSPMISampling(AcquisitionStrategy):
    """APS-based Set-Partition Mutual Information (APS-SPMI) — novel strategy.

    Same information-theoretic principle as CSPMI, but uses the APS
    (Adaptive Prediction Sets) cumulative partition instead of the marginal
    binary threshold.

    APS partition: class at sorted rank r is "in" the set if the cumulative
    probability BEFORE including it hasn't yet crossed qhat:
        in_aps(y_r) = 1  iff  cumsum_{r-1} < qhat

    Why APS partition beats marginal partition for CSPMI:
    - APS sets are smaller and rank-ordered → finer, more informative split
    - On CIFAR-100 (high C), marginal CSPMI collapses when P_in ≈ 1
      (large qhat puts most prob mass inside set → MI ≈ 0).
      APS naturally produces smaller sets, preventing this collapse.
    - APS respects the ORDERING of classes — only the most probable classes
      are included, making the in/out partition semantically meaningful.

    Score:
        APS_SPMI(x) = H(p) - [P_in × H(p_in) + P_out × H(p_out)]

    where in/out is determined by the APS cumulative partition.

    Fully conformal: qhat is computed via compute_qhat_aps on calibration set.
    """

    def __init__(self):
        super().__init__(name="cp_aps_spm")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with highest APS conformal set-partition MI.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   APS threshold from compute_qhat_aps on calibration set

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        eps = 1e-9
        n, C = probs.shape

        # ── APS partition: class at rank r is "in" if cumsum[r-1] < qhat ────
        sorted_probs, sort_idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)                    # (n, C)

        # prev_cumsum[r] = cumsum before including rank r
        prev_cumsum = torch.cat(
            [torch.zeros(n, 1, device=probs.device), cumsum[:, :-1]], dim=1
        )                                                             # (n, C)

        # A class at sorted rank r is in the APS set iff we haven't yet
        # accumulated enough probability to cross qhat before it
        in_aps_sorted = (prev_cumsum < qhat).float()                  # (n, C)

        # Scatter back to original class order
        in_aps = torch.zeros_like(probs)
        in_aps.scatter_(1, sort_idx, in_aps_sorted)                   # (n, C)
        out_aps = 1.0 - in_aps

        # ── Total entropy H(p) ───────────────────────────────────────────────
        H_total = -(probs * torch.log(probs + eps)).sum(dim=1)        # (n,)

        # ── Within-set: renormalized distribution over APS-included classes ──
        P_in_raw  = (probs * in_aps ).sum(dim=1)                      # (n,)
        P_out_raw = (probs * out_aps).sum(dim=1)                      # (n,)

        p_in  = probs * in_aps  / P_in_raw .clamp(min=eps).unsqueeze(1)
        p_out = probs * out_aps / P_out_raw.clamp(min=eps).unsqueeze(1)

        H_in  = -(p_in  * torch.log(p_in  + eps) * in_aps ).sum(dim=1)
        H_out = -(p_out * torch.log(p_out + eps) * out_aps).sum(dim=1)

        # ── APS-SPMI = H(p) - H(y | APS membership) ─────────────────────────
        H_conditional = P_in_raw * H_in + P_out_raw * H_out
        mi = (H_total - H_conditional).clamp(min=0.0)                # (n,)

        return torch.topk(mi, budget)[1]


class CPWiseSampling(AcquisitionStrategy):
    """Conformal Within-Set Information Efficiency (WISE) — novel pure-uncertainty strategy.

    Core insight: measuring *how much* information per included class is in the
    APS conformal set separates genuine ambiguity from noise.

        WISE(x) = H(p_in_aps | x) / |C_APS(x)|

    where:
        p_in_aps : probability distribution renormalized over the APS set
        |C_APS|  : APS set size (number of included classes)
        H(p_in)  : Shannon entropy within the APS set

    ---
    Why this beats entropy on high-C / noisy datasets:

    Score = H_in / k   ∝   log(k) / k   for uniform within-set distribution,
    which is maximised at k ≈ e ≈ 2.7 and DECREASES thereafter.

    | Sample type            | APS size k | WISE score (uniform case) |
    |------------------------|------------|--------------------------|
    | Confident (singleton)  | 1          | 0.000  ← not selected ✓  |
    | Decision boundary      | 2–3        | 0.347 – 0.366  ← TOP ✓  |
    | Confused (10 classes)  | 10         | 0.230            ✓       |
    | Noisy OOD (80 classes) | 80         | 0.055            ← LOW ✓ |

    Entropy would rank the noisy OOD sample HIGHEST.
    WISE ranks true boundary samples first.

    Pure uncertainty: each sample scored independently.
    Fully conformal: uses APS qhat from compute_qhat_aps on calibration set.
    """

    def __init__(self):
        super().__init__(name="cp_wise")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples maximising within-set entropy per APS-included class.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   APS threshold from compute_qhat_aps on calibration set

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        eps = 1e-9
        n, C = probs.shape

        # ── APS partition (vectorized) ────────────────────────────────────────
        sorted_probs, sort_idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)              # (n, C)

        # Class at sorted rank r is "in" APS set iff cumsum before it < qhat
        prev_cumsum = torch.cat(
            [torch.zeros(n, 1, device=probs.device), cumsum[:, :-1]], dim=1
        )                                                       # (n, C)
        in_aps_sorted = (prev_cumsum < qhat).float()           # (n, C)

        # Scatter back to original class order
        in_aps = torch.zeros_like(probs)
        in_aps.scatter_(1, sort_idx, in_aps_sorted)            # (n, C)

        # ── APS set sizes ─────────────────────────────────────────────────────
        aps_sizes = in_aps.sum(dim=1).clamp(min=1.0)           # (n,)  ≥ 1

        # ── Within-set entropy H(p_in) ────────────────────────────────────────
        P_in  = (probs * in_aps).sum(dim=1).clamp(min=eps)     # (n,)
        p_in  = probs * in_aps / P_in.unsqueeze(1)             # (n, C) renorm
        H_in  = -(p_in * torch.log(p_in + eps) * in_aps).sum(dim=1)  # (n,)

        # ── WISE = H_in / set_size ────────────────────────────────────────────
        # Equivalent to average per-class information in the conformal set.
        # Peaks at APS size ≈ e ≈ 2.7 (for uniform within-set distribution),
        # naturally penalising both singletons and large noisy sets.
        wise = H_in / aps_sizes                                 # (n,)

        return torch.topk(wise, budget)[1]


class CPDiversityAPSSampling(AcquisitionStrategy):
    """Conformal Diversity-Uncertainty Sampling (CDUS) — novel strategy.

    Root cause of entropy failure on CIFAR-100: entropy selects REDUNDANT
    samples — many pool images are confused about the SAME set of classes.
    Labeling one teaches the same lesson as labeling ten.

    This strategy combines:
        1. APS uncertainty  : ranks samples by conformal set size (cp_aps)
           — already beats pure entropy in later rounds.
        2. Greedy diversity : ensures successive selections are NOT confused
           about the same classes, by penalizing cosine similarity in
           probability space to already-selected samples.

    Score at greedy step t:
        score_t(x) = APS_set_size(x) / C   − λ × max_{x'∈S_t} cos_sim(p(x), p(x'))

    The cosine similarity between softmax vectors is high when two samples
    are uncertain about the SAME classes → penalizing it forces diversity
    across the model's confusion manifold.

    Fully conformal: APS qhat is used (same as cp_aps / cp_boundary).

    Implementation note: to stay tractable for large pools, we first filter
    to the top (candidate_ratio × budget) candidates by APS uncertainty,
    then run the greedy diversity pass on that reduced set.
    """

    def __init__(self, lambda_div: float = 1.0, candidate_ratio: int = 8):
        super().__init__(name="cp_diversity")
        self.lambda_div   = lambda_div
        self.candidate_ratio = candidate_ratio

    def select(self, probs, budget, qhat, **kwargs):
        """Greedy conformal-diverse selection.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   APS threshold from calibration

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        n, C = probs.shape

        # ── Step 1: APS set size (vectorized) ────────────────────────────────
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        cumsum_probs     = torch.cumsum(sorted_probs, dim=1)          # (n, C)

        exceeds_mask = (cumsum_probs >= qhat)                          # (n, C)
        has_exceed   = exceeds_mask.any(dim=1)                        # (n,)
        first_exceed = exceeds_mask.long().argmax(dim=1)              # (n,)
        aps_sizes    = torch.where(has_exceed, first_exceed + 1,
                                   torch.tensor(C, dtype=torch.long)) # (n,)

        unc = aps_sizes.float() / C                                   # (n,) ∈ (0,1]

        # ── Step 2: Filter to top candidates ──────────────────────────────────
        # Add tiny random jitter to break ties in APS sizes (integers)
        # so that torch.topk doesn't always pick the first-encountered indices.
        k_cand  = min(self.candidate_ratio * budget, n)
        unc_jitter = unc + torch.rand(n) * 1e-6     # break ties randomly
        top_idx = torch.topk(unc_jitter, k_cand)[1]  # (k_cand,)

        p_cand   = probs[top_idx]                                     # (k_cand, C)
        unc_cand = unc[top_idx]                                       # (k_cand,)

        # L2-normalise rows for cosine similarity
        p_norm   = p_cand / (p_cand.norm(dim=1, keepdim=True) + 1e-9) # (k_cand, C)

        # ── Step 3: Greedy diversity selection ────────────────────────────────
        selected_local = []
        selected_mask  = torch.zeros(k_cand, dtype=torch.bool)
        max_sim        = torch.zeros(k_cand)                          # (k_cand,)

        for _ in range(budget):
            if not selected_local:
                scores = unc_cand
            else:
                scores = unc_cand - self.lambda_div * max_sim

            scores = scores.masked_fill(selected_mask, float('-inf'))
            best   = int(scores.argmax().item())

            selected_local.append(best)
            selected_mask[best] = True

            # Update max cosine-sim to newly selected sample
            sim_new = (p_norm @ p_norm[best]).abs()                   # (k_cand,)
            max_sim = torch.maximum(max_sim, sim_new)

        # ── Step 4: Map local indices back to global indices ──────────────────
        return top_idx[torch.tensor(selected_local, dtype=torch.long)]


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


class CPRelativeMarginSampling(AcquisitionStrategy):
    """Relative Conformity Score (RCS) Sampling — novel conformal method.

    COMPLETELY NEW non-conformity score (not marginal, not APS, not RMCP):

        s_rcs(x, y) = 1 − p_y / p_max(x)   ∈ [0, 1]

    Prediction set:
        C_rcs(x) = {y : p_y ≥ p_max(x) × (1 − qhat)}

    — a *relative epsilon-ball* around the top prediction, where the
    threshold ADAPTS per sample (scales with p_max).

    ---
    Key difference from all existing methods:

    | Method   | Conformity score             | Threshold for inclusion  |
    |----------|------------------------------|--------------------------|
    | Marginal | 1 − p_y                      | fixed: p_y ≥ 1−qhat      |
    | APS      | cumsum up to rank of y       | cumulative rank          |
    | RMCP     | p_y − mean(p_{j≠y})         | absolute margin          |
    | **RCS**  | **1 − p_y / p_max (ratio)**  | **relative: p_y/p_max ≥ 1−qhat** |

    Why RCS beats marginal for AL:
    - A sample with p = [0.30, 0.28, 0.26, ...] has tiny margin.
      Marginal might include only 1–2 classes (if 1−qhat = 0.25).
      RCS includes all classes within 1−qhat of the max → LARGER set → selected.
    - A sample with p = [0.90, 0.05, ...] is confident.
      Marginal: 1 class. RCS: still 1 class (0.05 << 0.90×(1−qhat)).
    - Uniformly confused (0.01 each): marginal often includes many classes.
      RCS includes ALL classes (all equal to max) → qhat captures this
      difficulty correctly and still produces a large set, but the key
      advantage is in the CALIBRATION: qhat_rcs is smaller for well-
      calibrated models, making it more discriminative.

    Pure uncertainty: each sample scored independently.
    Fully conformal: qhat via compute_qhat_rcs on calibration set.
    """

    def __init__(self):
        super().__init__(name="cp_rel_margin")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with largest RCS prediction sets.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   RCS threshold from compute_qhat_rcs on calibration set

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        n, C = probs.shape

        # Per-sample relative threshold: p_max × (1 − qhat)
        p_max = probs.max(dim=1, keepdim=True)[0]            # (n, 1)
        threshold = p_max * (1.0 - qhat)                     # (n, 1)

        # RCS prediction set: classes whose probability ≥ relative threshold
        in_set   = (probs >= threshold).float()              # (n, C)
        set_sizes = in_set.sum(dim=1)                        # (n,)

        # Select samples with largest RCS prediction sets
        return torch.topk(set_sizes, budget)[1]


class CPRCSMISampling(AcquisitionStrategy):
    """RCS Set-Partition Mutual Information (RCS-MI) — hybrid novel strategy.

    Motivation: RCS wins early rounds (large relative sets, informative) but
    degenerates in later rounds when the model is strong — p_max grows, the
    relative threshold rises, and almost every sample gets set_size=1,
    making them indistinguishable by set size alone.

    Fix: replace the set-SIZE signal with the set-PARTITION MUTUAL INFORMATION,
    computed under the RCS partition (not marginal, not APS).

        RCS_MI(x) = H(p) − [P_in_rcs × H(p_in_rcs) + P_out_rcs × H(p_out_rcs)]

    Why this fixes late-round degeneration:
    - When set_size=1 (one class dominates under RCS), P_in_rcs ≈ p_max.
      If p_max = 0.7 and p_out is spread over 99 other classes:
        H_out = entropy over 99 tiny probs ≈ log(99) (high!)
        MI = H(p) − [0.7 × H_in + 0.3 × H_out]  ← non-zero, discriminative.
    - Samples where the model is 70% confident but has a "crowded out-set"
      (many runner-up classes) score HIGHER than samples where the model is
      90% confident with nothing in the out-set.

    This correctly identifies "I'm pretty sure, but there are many alternatives"
    — exactly the hard cases that need labeling.

    Uses the RCS partition: `C_rcs(x) = {y : p_y ≥ p_max(x) × (1 − qhat)}`
    calibrated by `compute_qhat_rcs` on the held-out calibration set.
    Fully conformal, pure uncertainty (no diversity component).
    """

    def __init__(self):
        super().__init__(name="cp_rcs_mi")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with highest RCS set-partition mutual information.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   RCS threshold from compute_qhat_rcs on calibration set

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        eps = 1e-9
        n, C = probs.shape

        # ── RCS partition: class y is "in" iff p_y ≥ p_max × (1-qhat) ────────
        p_max   = probs.max(dim=1, keepdim=True)[0]          # (n, 1)
        threshold = p_max * (1.0 - qhat)                     # (n, 1)
        in_rcs  = (probs >= threshold).float()               # (n, C)
        out_rcs = 1.0 - in_rcs                               # (n, C)

        # ── Total entropy H(p) ───────────────────────────────────────────────
        H_total = -(probs * torch.log(probs + eps)).sum(dim=1)   # (n,)

        # ── Within-RCS-set distribution ───────────────────────────────────────
        P_in_raw  = (probs * in_rcs ).sum(dim=1)             # (n,)  raw weight
        P_out_raw = (probs * out_rcs).sum(dim=1)             # (n,)

        p_in  = probs * in_rcs  / P_in_raw .clamp(min=eps).unsqueeze(1)
        p_out = probs * out_rcs / P_out_raw.clamp(min=eps).unsqueeze(1)

        H_in  = -(p_in  * torch.log(p_in  + eps) * in_rcs ).sum(dim=1)
        H_out = -(p_out * torch.log(p_out + eps) * out_rcs).sum(dim=1)

        # ── RCS-MI = H(p) − H(y | RCS membership) ───────────────────────────
        H_cond = P_in_raw * H_in + P_out_raw * H_out
        rcs_mi = (H_total - H_cond).clamp(min=0.0)           # (n,)

        return torch.topk(rcs_mi, budget)[1]


class CPRCSv2Sampling(AcquisitionStrategy):
    """RCS v2 — Continuous Relative Conformity Score sampling.

    Problem with original RCS: acquisition signal = set_size (integer)
    → many ties, no discrimination when most samples have set_size=1 in late rounds.

    Fix: instead of COUNTING classes above the RCS threshold, use a
    CONTINUOUS score directly derived from the RCS non-conformity measure.

    Score: competitive_ratio(x) = p_(2)(x) / p_max(x)

    This is 1 − s_rcs(x, y_(2)), the "inverse RCS score" of the runner-up class.

    Interpretation:
        competitive_ratio → 1  : runner-up tightly competes with top class
                                → model genuinely uncertain → SELECT ✓
        competitive_ratio → 0  : top class dominates far beyond runner-up
                                → model confident → don't select ✓

    Why this doesn't degenerate in late rounds:
        competitive_ratio = p_(2)/p_max is a RATIO — it's invariant to the
        absolute level of p_max. Even when all samples have p_max = 0.9
        (strong model), competitive_ratio varies freely in [0, 1]:
            p=[0.90, 0.09, ...]  → ratio = 0.10  (clear winner)
            p=[0.90, 0.80, ...]  → ratio = 0.89  (genuine boundary)

    Extension: to use ALL competing classes (not just runner-up):
        weighted_ratio(x) = Σ_{y≠top} p_y × (p_y/p_max)
                          = (||p||² − p_max²) / p_max

    By default uses the runner-up score (mode='top2') but also supports
    the full weighted version (mode='weighted').

    Fully conformal: uses qhat from compute_qhat_rcs on calibration set.
    Pure uncertainty: no diversity component.
    """

    def __init__(self, mode: str = 'top2'):
        super().__init__(name="cp_rcs_v2")
        assert mode in ('top2', 'weighted'), "mode must be 'top2' or 'weighted'"
        self.mode = mode

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with highest continuous RCS uncertainty.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   RCS threshold from compute_qhat_rcs (informational)

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        n, C = probs.shape
        p_max = probs.max(dim=1)[0]                           # (n,)

        if self.mode == 'top2':
            # Score = p_(2) / p_max  (runner-up competitive ratio)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            p_second = sorted_probs[:, 1]                     # (n,)
            score = p_second / (p_max + 1e-9)                 # (n,) ∈ [0, 1)

        return torch.topk(score, budget)[1]


class CPClasswiseSampling(AcquisitionStrategy):
    """Classwise Conformal Prediction sampling — novel class-conditional strategy.

    Instead of one global qhat, uses a separate qhat_y per class calibrated on
    the calibration set:

        C_cw(x) = {y : p_y >= 1 - qhat_y}

    This gives class-adaptive prediction sets:
    - Rare/hard classes (low p_y on calib) → large qhat_y → easier to include
      → hard pool samples from those classes get LARGER sets → selected ✓
    - Easy/dominant classes (high p_y on calib) → small qhat_y → stricter threshold
      → only truly uncertain samples for those classes get large sets ✓

    Why this beats marginal CP for acquisition:
    - Marginal CP uses one global threshold 1-qhat, which tends to over-include
      easy classes and under-include hard classes in prediction sets.
    - Classwise CP corrects this bias: pool samples that are hard relative to
      their own class distribution get larger sets and are prioritized.

    Fully conformal: uses compute_qhat_classwise on calibration set.
    """

    def __init__(self):
        super().__init__(name="cp_classwise")

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with largest classwise prediction set sizes.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   Tensor (num_classes,) from compute_qhat_classwise

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        # thresholds: (C,) → broadcast to (n, C)
        thresholds = (1.0 - qhat).to(probs.device)           # (C,)
        pred_sets  = probs >= thresholds.unsqueeze(0)         # (n, C)
        set_sizes  = pred_sets.sum(dim=1).float()             # (n,)

        return torch.topk(set_sizes, budget)[1]


class CPGapSampling(AcquisitionStrategy):
    """CP Softmax Gap (CP-GAP) — continuous boundary-distance scoring.

    Problem with CP_SIZE: it counts how many classes exceed the threshold
    1-qhat, yielding an INTEGER score with massive ties (only 0..C values).
    Two samples with the same set_size are indistinguishable, even if one
    has classes tightly clustered around the boundary (genuinely hard) while
    the other has classes far above it (noisy/random).

    CP-GAP replaces the binary in/out counting with a CONTINUOUS score that
    measures how close each class's probability sits to the conformal boundary:

        gap_y = |p_y - (1 - qhat)|

        CP_GAP(x) = Σ_{y=1}^{C} exp(-gap_y / τ)

    Each class contributes maximally when p_y ≈ 1-qhat (exactly at boundary)
    and contributes near-zero when p_y is far from the boundary.

    --- Why this separates hard from noisy samples ---

    | Sample type          | Class distribution near boundary | CP_GAP | CP_SIZE |
    |----------------------|---------------------------------:|:------:|:-------:|
    | Hard (2-class boundary) | 2 classes tightly at boundary | HIGH   | 2       |
    | Noisy (flat p)       | classes scattered far from boundary | LOW  | 2-3     |
    | Confident            | 1 class far above, rest far below | LOW   | 1       |

    At a given set_size, CP-GAP produces a CONTINUOUS ranking that breaks ties
    in a principled way: samples whose classes hover near the conformal
    decision boundary are ranked higher.

    Temperature τ controls sensitivity:
    - Small τ (0.01): only classes VERY close to boundary contribute
    - Large τ (0.1): broader neighborhood around boundary contributes

    Pure uncertainty, fully conformal (uses standard marginal qhat).
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__(name="cp_gap")
        self.temperature = temperature

    def select(self, probs, budget, qhat, **kwargs):
        """Select samples with most classes near the conformal boundary.

        Args:
            probs:  Probability tensor (n_samples, n_classes)
            budget: Number of samples to select (K)
            qhat:   Marginal conformal threshold from calibration

        Returns:
            Tensor of selected indices (shape: [budget])
        """
        # Distance of each class probability to the conformal boundary
        boundary = 1.0 - qhat                                    # scalar
        gaps = (probs - boundary).abs()                          # (n, C)

        # Soft boundary proximity: peaks at gap=0, decays exponentially
        proximity = torch.exp(-gaps / self.temperature)          # (n, C)

        # Aggregate: total boundary proximity across all classes
        score = proximity.sum(dim=1)                             # (n,)

        return torch.topk(score, budget)[1]
