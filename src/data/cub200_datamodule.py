from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import requests
import tarfile
from pathlib import Path

class CUB200Dataset(Dataset):
    """CUB-200-2011 dataset."""
    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        
        # Dataset paths
        self.dataset_path = self.root / 'CUB_200_2011'
        self.images_path = self.dataset_path / 'images'
        
        if download:
            self._download()
        
        if not self.dataset_path.exists():
            raise RuntimeError('Dataset not found. Use download=True to download it.')
        
        # Load metadata
        self._load_metadata()
        
    def _download(self):
        """Download CUB-200-2011 dataset."""
        if self.dataset_path.exists():
            return
        
        url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        self.root.mkdir(parents=True, exist_ok=True)
        
        print(f'Downloading CUB-200-2011 dataset from {url}...')
        tgz_path = self.root / 'CUB_200_2011.tgz'
        
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(tgz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print('Extracting...')
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(self.root)
        
        # Clean up
        tgz_path.unlink()
        print('Download complete!')
    
    def _load_metadata(self):
        """Load image paths and labels."""
        # Read images file
        images_file = self.dataset_path / 'images.txt'
        with open(images_file, 'r') as f:
            images = [line.strip().split() for line in f]
        
        # Read labels file
        labels_file = self.dataset_path / 'image_class_labels.txt'
        with open(labels_file, 'r') as f:
            labels = [line.strip().split() for line in f]
        
        # Read train/test split
        split_file = self.dataset_path / 'train_test_split.txt'
        with open(split_file, 'r') as f:
            splits = [line.strip().split() for line in f]
        
        # Filter by train/test
        self.data = []
        self.targets = []
        
        for (img_id, img_path), (_, label), (_, is_train) in zip(images, labels, splits):
            is_train = int(is_train)
            if (self.train and is_train == 1) or (not self.train and is_train == 0):
                self.data.append(self.images_path / img_path)
                # Convert to 0-indexed
                self.targets.append(int(label) - 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class CUB200DataModule:
    """Data module for CUB-200-2011 with Active Learning setup."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 64)
        self.num_workers = cfg.get('num_workers', 2)
        
        # CUB-200 specific normalization (calculated from dataset)
        self.mean = cfg.get('mean', [0.4859, 0.4996, 0.4318])
        self.std = cfg.get('std', [0.2321, 0.2277, 0.2665])
        
        # Image size
        self.image_size = cfg.get('image_size', 224)
        
        # Transforms
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Load CUB-200-2011
        self.train_set = CUB200Dataset(
            root=self.root, 
            train=True, 
            download=True, 
            transform=self.transform_train
        )
        self.test_set = CUB200Dataset(
            root=self.root, 
            train=False, 
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
            dataset = CUB200Dataset(
                root=self.root, 
                train=True, 
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
            num_workers=self.num_workers,
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
            num_workers=self.num_workers,
            pin_memory=True
        )
