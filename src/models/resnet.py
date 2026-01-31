"""ResNet model adapted for CIFAR-10."""
import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes=10, pretrained=True):
    """Get ResNet18 model adapted for CIFAR-10.
    
    CIFAR-10 images are 32x32, so we modify the first conv layer and remove maxpool
    to preserve spatial dimensions better.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Modified ResNet18 model
    """
    # Load pretrained ResNet18
    if pretrained:
        model = models.resnet18(weights='IMAGENET1K_V1')
    else:
        model = models.resnet18(weights=None)
    
    # Modify first conv layer for 32x32 images
    # Original: kernel_size=7, stride=2, padding=3
    # Modified: kernel_size=3, stride=1, padding=1 (preserves spatial dim)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool to preserve spatial dimensions
    model.maxpool = nn.Identity()
    
    # Modify fc layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


class ResNet18(nn.Module):
    """ResNet18 wrapper for CIFAR-10."""
    
    def __init__(self, num_classes=10, pretrained=True):
        """Initialize ResNet18 model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        self.model = get_resnet18(num_classes=num_classes, pretrained=pretrained)
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.model(x)
