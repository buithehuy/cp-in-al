"""Base class for acquisition strategies."""
from abc import ABC, abstractmethod


class AcquisitionStrategy(ABC):
    """Abstract base class for acquisition strategies."""
    
    def __init__(self, name):
        """Initialize acquisition strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    @abstractmethod
    def select(self, probs, budget, **kwargs):
        """Select samples based on acquisition function.
        
        Args:
            probs: Probability tensor of shape (n_samples, n_classes)
            budget: Number of samples to select
            **kwargs: Additional arguments (e.g., qhat for CP-based methods)
            
        Returns:
            Tensor of selected indices
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
