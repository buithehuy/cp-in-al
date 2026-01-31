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
