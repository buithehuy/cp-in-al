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

