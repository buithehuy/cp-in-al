"""Training utilities."""
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_round(model, loader, epochs, lr=0.01, momentum=0.9, weight_decay=5e-4, device='cuda', use_amp=True):
    """Train model for one active learning round.
    
    Args:
        model: Model to train
        loader: Training DataLoader
        epochs: Number of epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        device: Device to use
        use_amp: Whether to use Automatic Mixed Precision (default: True for 2-3x speedup)
    """
    model.train()
    model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize AMP GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()



def eval_acc(model, loader, device='cuda'):
    """Evaluate model accuracy.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        device: Device to use
        
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    model.to(device)
    
    correct, total = 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    
    return 100.0 * correct / total
