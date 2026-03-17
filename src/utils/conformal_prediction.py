"""Conformal Prediction utilities."""
import torch
import torch.nn.functional as F
import numpy as np


def get_probs(model, loader, device='cuda'):
    """Extract softmax probabilities from model.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device to use
        
    Returns:
        Tuple of (probs, labels) as tensors
    """
    model.eval()
    probs_list, labels_list = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu()
            probs_list.append(probs)
            labels_list.append(y)
    
    return torch.cat(probs_list), torch.cat(labels_list)


def get_features_and_probs(model, loader, device='cuda'):
    """Extract softmax probabilities and final features from model.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device to use
        
    Returns:
        Tuple of (features, probs, labels) as tensors
    """
    model.eval()
    features_list, probs_list, labels_list = [], [], []
    
    captured_features = {}
    def hook(module, input, output):
        captured_features['features'] = input[0].detach()
        
    if hasattr(model, 'model') and hasattr(model.model, 'fc'):
        handle = model.model.fc.register_forward_hook(hook)
    else:
        raise ValueError("Could not find final fully connected layer (model.model.fc) to attach hook.")
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu()
            
            # The hook will have populated captured_features['features']
            # Reshape it to (batch_size, feature_dim) in case it's (batch_size, feature_dim, 1, 1)
            feat = captured_features['features'].cpu()
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            features_list.append(feat)
            probs_list.append(probs)
            labels_list.append(y)
            
    handle.remove()
    return torch.cat(features_list), torch.cat(probs_list), torch.cat(labels_list)


def compute_qhat(model, loader, alpha=0.1, device='cuda'):
    """Compute conformity score threshold qhat.
    
    Args:
        model: Trained model
        loader: Calibration DataLoader
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        device: Device to use
        
    Returns:
        qhat: Conformity score threshold
    """
    probs, labels = get_probs(model, loader, device)
    
    # Conformity scores: 1 - p(true_class)
    scores = 1 - probs[torch.arange(len(labels)), labels]
    
    # Compute quantile
    n = len(labels)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k - 1, n - 1)  # Index for k-th smallest
    
    qhat = torch.sort(scores)[0][k].item()
    return qhat


def evaluate_conformal_prediction(model, loader, qhat, device='cuda'):
    """Evaluate conformal prediction metrics.
    
    Args:
        model: Trained model
        loader: Test DataLoader  
        qhat: Conformity score threshold
        device: Device to use
        
    Returns:
        Dictionary with CP metrics (coverage, avg_set_size, zero_sets)
    """
    probs, labels = get_probs(model, loader, device)
    
    # Generate prediction sets
    pred_sets = probs >= (1 - qhat)
    
    # Coverage: fraction of test samples where true label is in prediction set
    coverage = pred_sets[torch.arange(len(labels)), labels].float().mean().item()
    
    # Average set size
    avg_set_size = pred_sets.sum(dim=1).float().mean().item()
    
    # Number of empty sets (overconfident errors)
    zero_sets = (pred_sets.sum(dim=1) == 0).sum().item()
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'zero_sets': zero_sets
    }


def compute_qhat_aps(model, loader, alpha=0.1, device='cuda'):
    """Compute APS conformity score threshold.
    
    For APS (Adaptive Prediction Sets), the conformity score is based on
    the cumulative probability up to and including the true class.
    
    Args:
        model: Trained model
        loader: Calibration DataLoader
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        device: Device to use
        
    Returns:
        qhat: APS conformity score threshold
    """
    probs, labels = get_probs(model, loader, device)
    
    # For each sample, sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
    
    # Find the rank of the true label in the sorted order
    # and compute cumulative sum up to that rank
    n_samples = len(labels)
    scores = torch.zeros(n_samples)
    
    for i in range(n_samples):
        # Find where true label appears in sorted indices
        true_label = labels[i]
        rank = (sorted_indices[i] == true_label).nonzero(as_tuple=True)[0].item()
        
        # Conformity score is cumulative prob up to and including true class
        scores[i] = sorted_probs[i, :rank+1].sum().item()
    
    # CRITICAL: For APS, higher score = easier to cover (true class appears early)
    # So we need the ALPHA quantile (not 1-alpha) to get the threshold that
    # ensures (1-alpha) coverage
    n = len(labels)
    k = int(np.ceil((n + 1) * alpha))  # Changed from (1-alpha) to alpha
    k = min(max(k - 1, 0), n - 1)  # Ensure valid index
    
    qhat = torch.sort(scores)[0][k].item()
    return qhat


def evaluate_aps(model, loader, qhat, device='cuda'):
    """Evaluate APS (Adaptive Prediction Sets) metrics.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        qhat: APS conformity score threshold
        device: Device to use
        
    Returns:
        Dictionary with APS metrics (coverage, avg_set_size, zero_sets)
    """
    probs, labels = get_probs(model, loader, device)
    
    # For each sample, generate APS prediction set
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=1)
    
    # Prediction set: smallest set where cumulative prob >= qhat
    n_samples, n_classes = probs.shape
    pred_set_sizes = torch.zeros(n_samples)
    
    for i in range(n_samples):
        # Find first index where cumsum >= qhat
        exceeds = (cumsum_probs[i] >= qhat).nonzero(as_tuple=True)[0]
        if len(exceeds) > 0:
            pred_set_sizes[i] = exceeds[0].item() + 1
        else:
            pred_set_sizes[i] = n_classes
    
    # For coverage, check if true label is in the prediction set
    # A label is in the APS set if it's among the top-k classes where k is the set size
    coverage_count = 0
    for i in range(n_samples):
        sorted_probs_i, sorted_indices = torch.sort(probs[i], descending=True)
        set_size = int(pred_set_sizes[i].item())
        pred_set = sorted_indices[:set_size]
        
        if labels[i] in pred_set:
            coverage_count += 1
    
    coverage = coverage_count / n_samples
    avg_set_size = pred_set_sizes.mean().item()
    zero_sets = (pred_set_sizes == 0).sum().item()
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'zero_sets': zero_sets
    }


def compute_qhat_rmcp(model, loader, alpha=0.1, device='cuda'):
    """Compute RMCP (Relative Margin Conformal Prediction) conformity score threshold.
    
    RMCP uses relative margin scores: score_k = p_k - mean(p_j for j != k)
    This measures how much a class probability exceeds the average of others.
    
    Args:
        model: Trained model
        loader: Calibration DataLoader
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        device: Device to use
        
    Returns:
        qhat: RMCP conformity score threshold
    """
    probs, labels = get_probs(model, loader, device)
    n_samples, n_classes = probs.shape
    
    # Compute conformity scores for true class on calibration set
    scores = torch.zeros(n_samples)
    
    for i in range(n_samples):
        true_class = labels[i].item()
        
        # Compute relative margin score for true class
        # score = p_true - mean(p_j for j != true)
        other_probs = torch.cat([probs[i, :true_class], probs[i, true_class+1:]])
        scores[i] = probs[i, true_class] - other_probs.mean()
    
    # CRITICAL: RMCP score interpretation
    # High score (e.g., +0.7) = p_true >> mean(others) = CONFIDENT = easy to cover
    # Low score (e.g., -0.2) = p_true << mean(others) = UNCERTAIN = hard to cover
    # This is similar to APS (reversed from standard CP)!
    # So we need ALPHA quantile (not 1-alpha) to get proper coverage
    n = len(scores)
    k = int(np.ceil((n + 1) * alpha))  # Changed from (1-alpha) to alpha
    k = min(max(k - 1, 0), n - 1)
    
    qhat = torch.sort(scores)[0][k].item()
    return qhat


def evaluate_rmcp(model, loader, qhat, device='cuda'):
    """Evaluate RMCP (Relative Margin Conformal Prediction) metrics.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        qhat: RMCP conformity score threshold
        device: Device to use
        
    Returns:
        Dictionary with RMCP metrics (coverage, avg_set_size, zero_sets)
    """
    probs, labels = get_probs(model, loader, device)
    n_samples, n_classes = probs.shape
    
    coverage_count = 0
    set_sizes = []
    zero_count = 0
    
    for i in range(n_samples):
        # Compute relative margin scores for all classes
        scores = torch.zeros(n_classes)
        
        for k in range(n_classes):
            # score_k = p_k - mean(p_j for j != k)
            other_probs = torch.cat([probs[i, :k], probs[i, k+1:]])
            scores[k] = probs[i, k] - other_probs.mean()
        
        # Generate prediction set: classes with score >= qhat
        pred_set = (scores >= qhat).nonzero(as_tuple=True)[0]
        set_size = len(pred_set)
        set_sizes.append(set_size)
        
        if set_size == 0:
            zero_count += 1
        
        # Check if true label is in prediction set
        if labels[i] in pred_set:
            coverage_count += 1
    
    coverage = coverage_count / n_samples
    avg_set_size = np.mean(set_sizes)
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'zero_sets': zero_count
    }


def compute_qhat_rcs(model, loader, alpha=0.1, device='cuda'):
    """Compute Relative Conformity Score (RCS) threshold qhat.

    NOVEL Non-Conformity Score:
        s_rcs(x, y) = 1 - p_y / p_max(x)   in [0, 1]

    Unlike marginal (absolute) or APS (cumulative rank), RCS is a *relative*
    score measuring how far each class sits from the top prediction, as a
    fraction of the top prediction itself.

    Prediction set:
        C_rcs(x) = {y : p_y >= p_max(x) * (1 - qhat)}

    = all classes whose probability is at least (1-qhat) of the maximum.
    The threshold adapts PER SAMPLE: confident samples get a tight relative
    neighbourhood; uncertain samples automatically include more competitors.

    Args:
        model:  Trained model
        loader: Calibration DataLoader
        alpha:  Miscoverage level (e.g. 0.1 for 90% coverage)
        device: Device

    Returns:
        qhat: RCS threshold in (0, 1)
    """
    probs, labels = get_probs(model, loader, device)
    p_max  = probs.max(dim=1)[0]                              # (n,)
    p_true = probs[torch.arange(len(labels)), labels]         # (n,)

    # s_rcs for the true label: 0 when model tops y*, ~1 when model is wrong
    scores = 1.0 - p_true / (p_max + 1e-9)                   # (n,) in [0,1)

    n = len(labels)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k - 1, n - 1)

    qhat = torch.sort(scores)[0][k].item()
    return qhat


def evaluate_rcs(model, loader, qhat, device='cuda'):
    """Evaluate Relative Conformity Score (RCS) prediction sets.

    C_rcs(x) = {y : p_y >= p_max(x) * (1 - qhat)}

    Args:
        model:  Trained model
        loader: Test DataLoader
        qhat:   RCS threshold from compute_qhat_rcs
        device: Device

    Returns:
        dict with coverage, avg_set_size, zero_sets
    """
    probs, labels = get_probs(model, loader, device)
    p_max = probs.max(dim=1, keepdim=True)[0]                 # (n, 1)

    threshold = p_max * (1.0 - qhat)                          # (n, 1) per-sample
    pred_sets = (probs >= threshold)                           # (n, C)

    coverage     = pred_sets[torch.arange(len(labels)), labels].float().mean().item()
    avg_set_size = pred_sets.sum(dim=1).float().mean().item()
    zero_sets    = (pred_sets.sum(dim=1) == 0).sum().item()

    return {'coverage': coverage, 'avg_set_size': avg_set_size, 'zero_sets': zero_sets}


def compute_qhat_classwise(model, loader, alpha=0.1, device='cuda'):
    """Compute class-conditional (classwise) conformal thresholds.

    Instead of one global qhat, computes a separate qhat_y for each class y
    on the calibration set, using only samples whose true label is y.

    This gives tighter, class-adaptive prediction sets and guarantees
    class-conditional coverage: P(y ∈ C(x) | Y=y) >= 1-alpha for each y.

    Args:
        model:  Trained model
        loader: Calibration DataLoader
        alpha:  Miscoverage level (e.g. 0.1 for 90% coverage)
        device: Device

    Returns:
        qhat_per_class: Tensor of shape (num_classes,) with per-class thresholds
    """
    probs, labels = get_probs(model, loader, device)
    num_classes = probs.shape[1]

    # Non-conformity score: 1 - p(true class)
    scores = 1.0 - probs[torch.arange(len(labels)), labels]  # (n,)

    qhat_per_class = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = (labels == c)
        n_c = mask.sum().item()

        if n_c == 0:
            # No calibration examples for class c → use global fallback
            qhat_per_class[c] = 1.0
            continue

        scores_c = scores[mask]
        k = int(np.ceil((n_c + 1) * (1 - alpha)))
        k = min(k - 1, n_c - 1)
        qhat_per_class[c] = torch.sort(scores_c)[0][k].item()

    return qhat_per_class  # shape: (num_classes,)


def evaluate_classwise(model, loader, qhat_per_class, device='cuda'):
    """Evaluate classwise conformal prediction metrics.

    Prediction set: C(x) = {y : 1 - p_y <= qhat_y}
                         = {y : p_y >= 1 - qhat_y}

    Args:
        model:           Trained model
        loader:          Test DataLoader
        qhat_per_class:  Tensor (num_classes,) from compute_qhat_classwise
        device:          Device

    Returns:
        dict with coverage, avg_set_size, zero_sets
    """
    probs, labels = get_probs(model, loader, device)

    # threshold_y = 1 - qhat_y for each class
    thresholds = (1.0 - qhat_per_class).to(probs.device)  # (C,)

    # class y is in prediction set iff p_y >= threshold_y
    pred_sets = probs >= thresholds.unsqueeze(0)           # (n, C)

    coverage     = pred_sets[torch.arange(len(labels)), labels].float().mean().item()
    avg_set_size = pred_sets.sum(dim=1).float().mean().item()
    zero_sets    = (pred_sets.sum(dim=1) == 0).sum().item()

    return {'coverage': coverage, 'avg_set_size': avg_set_size, 'zero_sets': zero_sets}


def compute_qhat_feature_cp(model, loader, alpha=0.1, device='cuda'):
    """Compute conformity score threshold qhat based on Feature CP distance.
    
    Args:
        model: Trained model
        loader: Calibration DataLoader
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        device: Device to use
        
    Returns:
        qhat: Conformity score threshold
    """
    features, probs, labels = get_features_and_probs(model, loader, device)
    
    if hasattr(model, 'model') and hasattr(model.model, 'fc'):
        weights = model.model.fc.weight.data.cpu()
    else:
        raise ValueError("Could not find weights of final fully connected layer.")
        
    n_samples = len(labels)
    scores = torch.zeros(n_samples)
    
    for i in range(n_samples):
        y = labels[i]
        scores[i] = torch.norm(features[i] - weights[y], p=2).item()
        
    n = len(labels)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k - 1, n - 1)
    
    qhat = torch.sort(scores)[0][k].item()
    return qhat


def evaluate_feature_cp(model, loader, qhat, device='cuda'):
    """Evaluate Feature CP metrics.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        qhat: Feature CP conformity score threshold
        device: Device to use
        
    Returns:
        Dictionary with CP metrics (coverage, avg_set_size, zero_sets)
    """
    features, probs, labels = get_features_and_probs(model, loader, device)
    if hasattr(model, 'model') and hasattr(model.model, 'fc'):
        weights = model.model.fc.weight.data.cpu()
    else:
        raise ValueError("Could not find weights of final fully connected layer.")
        
    n_samples, n_classes = probs.shape
    features_expanded = features.unsqueeze(1)
    weights_expanded = weights.unsqueeze(0)
    
    # Distance from each feature to each class weight
    distances = torch.norm(features_expanded - weights_expanded, p=2, dim=2)
    pred_sets = distances <= qhat
    
    coverage = pred_sets[torch.arange(n_samples), labels].float().mean().item()
    avg_set_size = pred_sets.sum(dim=1).float().mean().item()
    zero_sets = (pred_sets.sum(dim=1) == 0).sum().item()
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'zero_sets': zero_sets
    }
