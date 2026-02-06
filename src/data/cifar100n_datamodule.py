"""CIFAR-100N DataModule with configurable noise levels."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import torch
import os
import urllib.request


class CIFAR100NDataModule:
    """Data module for CIFAR-100N with noisy labels and Active Learning setup.
    
    Supports three noise types:
    - clean: Original CIFAR-100 labels (no noise)
    - noisy: Human-annotated noisy labels from CIFAR-100N (~40% noise)
    - worst: Worst-case aggregated labels (higher noise)
    """
    
    # Multiple mirror URLs for CIFAR-100N noisy labels
    NOISY_LABELS_URLS = [
        "http://www.yliuu.com/web-cifarN/files/CIFAR-100_human.pt",
        "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt",
        "https://huggingface.co/datasets/nateraw/cifar-100n/resolve/main/CIFAR-100_human.pt",
    ]
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)
        self.noise_type = cfg.get('noise_type', 'clean')  # clean, noisy, worst
        self.noisy_labels_path = cfg.get('noisy_labels_path', 'data/CIFAR-100_human.pt')
        
        # Validate noise type
        if self.noise_type not in ['clean', 'noisy', 'worst']:
            raise ValueError(f"Invalid noise_type: {self.noise_type}. Must be 'clean', 'noisy', or 'worst'")
        
        # CIFAR-100 specific normalization
        self.mean = cfg.get('mean', [0.5071, 0.4867, 0.4408])
        self.std = cfg.get('std', [0.2675, 0.2565, 0.2761])
        
        # Transforms
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
        
        # Load CIFAR-100
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
        
        # Apply noisy labels if needed
        if self.noise_type != 'clean':
            self._load_and_apply_noisy_labels()
    
    def _download_noisy_labels(self):
        """Download CIFAR-100N noisy labels if not present."""
        os.makedirs(os.path.dirname(self.noisy_labels_path), exist_ok=True)
        
        if not os.path.exists(self.noisy_labels_path):
            print(f"Downloading CIFAR-100N noisy labels to {self.noisy_labels_path}...")
            
            # Try multiple mirror URLs
            success = False
            last_error = None
            
            for i, url in enumerate(self.NOISY_LABELS_URLS):
                try:
                    print(f"  Attempting mirror {i+1}/{len(self.NOISY_LABELS_URLS)}: {url}")
                    urllib.request.urlretrieve(url, self.noisy_labels_path)
                    print("  ✓ Download complete!")
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    print(f"  ✗ Failed: {e}")
                    continue
            
            if not success:
                # Provide detailed manual download instructions
                raise RuntimeError(
                    f"\n{'='*70}\n"
                    f"CIFAR-100N noisy labels download failed from all mirrors.\n"
                    f"{'='*70}\n\n"
                    f"Please download manually using ONE of these methods:\n\n"
                    f"METHOD 1 - Direct download:\n"
                    f"  wget https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt \\\n"
                    f"       -O {self.noisy_labels_path}\n\n"
                    f"METHOD 2 - Using curl:\n"
                    f"  curl -L https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt \\\n"
                    f"       -o {self.noisy_labels_path}\n\n"
                    f"METHOD 3 - Clone repository:\n"
                    f"  git clone https://github.com/UCSC-REAL/cifar-10-100n.git\n"
                    f"  cp cifar-10-100n/data/CIFAR-100_human.pt {self.noisy_labels_path}\n\n"
                    f"Then re-run your command.\n"
                    f"{'='*70}\n"
                    f"Last error: {last_error}\n"
                    f"{'='*70}"
                )

    
    def _load_and_apply_noisy_labels(self):
        """Load and apply noisy labels based on noise_type."""
        # Download if needed
        self._download_noisy_labels()
        
        # Load noisy labels file
        print(f"Loading CIFAR-100N noisy labels (type: {self.noise_type})...")
        noisy_data = torch.load(self.noisy_labels_path)
        
        # Select appropriate labels
        if self.noise_type == 'noisy':
            # Use human-annotated noisy labels
            labels = noisy_data['noisy_label']
        elif self.noise_type == 'worst':
            # Use worst-case labels (try different keys)
            if 'worse_label' in noisy_data:
                labels = noisy_data['worse_label']
            elif 'aggre_label' in noisy_data:
                labels = noisy_data['aggre_label']
            elif 'worst_label' in noisy_data:
                labels = noisy_data['worst_label']
            else:
                # Fallback to random_label1 if worst not available
                print("Warning: 'worst' label not found, using 'random_label1'")
                labels = noisy_data.get('random_label1', noisy_data['noisy_label'])
        
        # Convert to list and apply
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        
        # Replace labels in training set
        self.train_set.targets = labels
        
        # Compute noise statistics
        clean_labels = noisy_data['clean_label']
        if isinstance(clean_labels, torch.Tensor):
            clean_labels = clean_labels.numpy()
        noise_rate = (np.array(labels) != clean_labels).mean()
        
        print(f"Noisy labels loaded successfully!")
        print(f"  Noise type: {self.noise_type}")
        print(f"  Noise rate: {noise_rate:.1%}")
        print(f"  Training samples: {len(self.train_set)}")
    
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
            dataset = datasets.CIFAR100(
                root=self.root, 
                train=True, 
                download=False, 
                transform=transform
            )
            # Apply same noisy labels if using noise
            if self.noise_type != 'clean':
                dataset.targets = self.train_set.targets
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
        """Get test data loader.
        
        Returns:
            DataLoader for test set (always uses clean labels)
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
