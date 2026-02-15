"""CIFAR-100-C DataModule with dynamic corruption generation."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import torch
from imagecorruptions import corrupt
from PIL import Image

class CIFAR100CDataModule:
    """Data module for CIFAR-100-C with Active Learning setup.
    
    Dynamically generates corruption on a subset of the pool data.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)
        
        # Corruption settings
        self.corruption_name = cfg.get('corruption', 'gaussian_noise')
        self.severity = cfg.get('severity', 1)
        self.corruption_ratio = cfg.get('corruption_ratio', 0.0) # Ratio of pool to corrupt
        
        # CIFAR-100 specific normalization
        self.mean = cfg.get('mean', [0.5071, 0.4867, 0.4408])
        self.std = cfg.get('std', [0.2675, 0.2565, 0.2761])
        
        # Transforms (Standard CIFAR-100 transforms)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Load standard CIFAR-100
        self.train_set = datasets.CIFAR100(
            root=self.root, 
            train=True, 
            download=True, 
            transform=self.transform_train
        )
        self.test_set = datasets.CIFAR100(
            root=self.root, 
            train=False, 
            download=True, 
            transform=self.transform_test
        )
        
    
    def setup_active_learning(self, initial_labeled, calibration_size, seed=42):
        """Set up active learning data splits and inject corruption."""
        if hasattr(self, '_is_corrupted') and self._is_corrupted:
            print("Warning: Corruption already applied. Skipping re-injection.")
            # We need to reconstruct indices since we don't return them before
            # But normally setup_active_learning is called ONCE. 
            # If called again, it Re-shuffles. If we re-shuffle, we might mix corrupted and clean data!
            # So, if already corrupted, we should probably reload the dataset OR throw error.
            # Ideally, reload dataset to be safe.
            print("Reloading clean dataset to ensure correct split...")
            self.train_set = datasets.CIFAR100(
                root=self.root, 
                train=True, 
                download=True, 
                transform=self.transform_train
            )
            self._is_corrupted = False
            
        np.random.seed(seed)
        idx = np.random.permutation(len(self.train_set))
        
        labeled_idx = idx[:initial_labeled].tolist()
        calib_idx = idx[initial_labeled:initial_labeled + calibration_size].tolist()
        pool_idx = idx[initial_labeled + calibration_size:].tolist()
        
        print(f"Initial labeled set: {len(labeled_idx)} clean images (not corrupted)")
        print(f"Calibration set: {len(calib_idx)} clean images (not corrupted)")
        
        # Inject corruption into the Pool
        if self.corruption_ratio > 0:
            print(f"Pool size: {len(pool_idx)}")
            self._inject_corruption(pool_idx, seed)
            self._is_corrupted = True
            
        return labeled_idx, calib_idx, pool_idx
    
    def _inject_corruption(self, pool_idx, seed):
        """Apply corruption to a subset of pool images in self.train_set."""
        np.random.seed(seed)
        
        # Calculate number of images to corrupt based on pool size
        num_corrupt = int(len(pool_idx) * self.corruption_ratio)
        
        if num_corrupt == 0:
            print("Corruption ratio too small, no images corrupted.")
            return
            
        # Select random indices from the pool to corrupt
        corrupt_indices = np.random.choice(pool_idx, num_corrupt, replace=False)
        
        print(f"Injecting {self.corruption_name} (severity={self.severity}) into {num_corrupt} pool images ({self.corruption_ratio:.1%})...")
        
        # Apply corruption
        # self.train_set.data is a numpy array [N, 32, 32, 3] of uint8
        
        # We process in batches to show progress if needed, but for 5-10k images it's fast enough
        count = 0
        for idx in corrupt_indices:
            img = self.train_set.data[idx] # uint8 [32, 32, 3]
            
            # Apply corruption using imagecorruptions library
            # It expects numpy array and returns numpy array
            corrupted_img = corrupt(img, severity=self.severity, corruption_name=self.corruption_name)
            
            # Update the dataset in-place
            self.train_set.data[idx] = corrupted_img
            count += 1
            
        print(f"✓ Successfully corrupted {count} images in the training set.")
        
    def get_loader(self, indices, shuffle=False, transform=None):
        """Get data loader for specified indices, preserving corrupted data."""
        if transform is not None:
            # We must use a shallow copy to preserve the modified .data attribute
            # If we create a new dataset from scratch, we lose the corruptions!
            import copy
            dataset = copy.copy(self.train_set)
            dataset.transform = transform
        else:
            dataset = self.train_set
            
        subset = Subset(dataset, indices)
        return DataLoader(
            subset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
