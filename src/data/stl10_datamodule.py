from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

class STL10DataModule:
    """Data module for STL-10 with Active Learning setup."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        
        # STL-10 specific normalization
        self.mean = cfg.get('mean', [0.4467, 0.4398, 0.4066])
        self.std = cfg.get('std', [0.2603, 0.2566, 0.2713])
        
        # Transforms - STL-10 images are 96x96
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Load STL-10 (using train split for active learning)
        self.train_set = datasets.STL10(
            root=self.root, 
            split='train',
            download=True, 
            transform=self.transform_train
        )
        self.test_set = datasets.STL10(
            root=self.root, 
            split='test',
            download=True, 
            transform=self.transform_test
        )
        
    def setup_active_learning(self, initial_labeled, calibration_size, seed=42):
        """Set up active learning data splits.
        
        Args:
            initial_labeled: Number of initial labeled samples
            calibration_size: Number of calibration samples
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (labeled_idx, calibration_idx, pool_idx)
        """
        np.random.seed(seed)
        idx = np.random.permutation(len(self.train_set))
        
        labeled_idx = idx[:initial_labeled].tolist()
        calib_idx = idx[initial_labeled:initial_labeled + calibration_size].tolist()
        pool_idx = idx[initial_labeled + calibration_size:].tolist()
        
        return labeled_idx, calib_idx, pool_idx
    
    def get_loader(self, indices, shuffle=False, transform=None):
        """Get data loader for specified indices.
        
        Args:
            indices: List of sample indices
            shuffle: Whether to shuffle data
            transform: Optional transform override
            
        Returns:
            DataLoader object
        """
        if transform is not None:
            # Create dataset copy with new transform
            dataset = datasets.STL10(
                root=self.root, 
                split='train',
                download=False, 
                transform=transform
            )
        else:
            dataset = self.train_set
            
        subset = Subset(dataset, indices)
        return DataLoader(
            subset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def get_test_loader(self):
        """Get test data loader.
        
        Returns:
            DataLoader for test set
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
