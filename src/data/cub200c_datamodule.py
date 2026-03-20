"""CUB200-C DataModule with dynamic corruption generation."""
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import numpy as np
import copy
import os
from PIL import Image
from pathlib import Path
from imagecorruptions import corrupt

import requests
import tarfile


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

        self._load_metadata()

        # Cache for corrupted images: {idx: PIL.Image or np.ndarray}
        self._corrupted_cache = {}

    def _download(self):
        """Download CUB-200-2011 dataset."""
        if self.dataset_path.exists():
            return

        url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        self.root.mkdir(parents=True, exist_ok=True)

        print(f'Downloading CUB-200-2011 dataset from {url}...')
        tgz_path = self.root / 'CUB_200_2011.tgz'

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(tgz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print('Extracting...')
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(self.root)

        tgz_path.unlink()
        print('Download complete!')

    def _load_metadata(self):
        """Load image paths and labels."""
        images_file = self.dataset_path / 'images.txt'
        with open(images_file, 'r') as f:
            images = [line.strip().split() for line in f]

        labels_file = self.dataset_path / 'image_class_labels.txt'
        with open(labels_file, 'r') as f:
            labels = [line.strip().split() for line in f]

        split_file = self.dataset_path / 'train_test_split.txt'
        with open(split_file, 'r') as f:
            splits = [line.strip().split() for line in f]

        self.data = []
        self.targets = []

        for (img_id, img_path), (_, label), (_, is_train) in zip(images, labels, splits):
            is_train = int(is_train)
            if (self.train and is_train == 1) or (not self.train and is_train == 0):
                self.data.append(self.images_path / img_path)
                self.targets.append(int(label) - 1)  # 0-indexed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]

        # Use cached corrupted image if available, else load from file
        if idx in self._corrupted_cache:
            img = self._corrupted_cache[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        else:
            img_path = self.data[idx]
            img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CUB200CDataModule:
    """Data module for CUB-200-C with Active Learning setup.

    Dynamically generates corruption on a subset of the pool data.
    CUB-200 loads images from files, so corruption is applied and cached
    in-memory as numpy arrays.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 64)
        self.num_workers = cfg.get('num_workers', 2)

        # Corruption settings
        self.corruption_name = cfg.get('corruption', 'gaussian_noise')
        self.severity = cfg.get('severity', 1)
        self.corruption_ratio = cfg.get('corruption_ratio', 0.0)

        # CUB-200 specific normalization
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
        """Set up active learning data splits and inject corruption."""
        if hasattr(self, '_is_corrupted') and self._is_corrupted:
            print("Warning: Corruption already applied. Reloading clean dataset...")
            self.train_set = CUB200Dataset(
                root=self.root,
                train=True,
                download=False,
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
        else:
            self.corrupted_indices = set()  # No corruption

        return labeled_idx, calib_idx, pool_idx

    def _inject_corruption(self, pool_idx, seed):
        """Apply corruption to a subset of pool images.

        CUB-200 images are loaded from files, so we load, corrupt, and cache
        them as numpy arrays in self.train_set._corrupted_cache.
        imagecorruptions expects [H, W, 3] uint8 numpy arrays.
        """
        np.random.seed(seed)

        num_corrupt = int(len(pool_idx) * self.corruption_ratio)

        if num_corrupt == 0:
            print("Corruption ratio too small, no images corrupted.")
            self.corrupted_indices = set()
            return

        corrupt_indices = np.random.choice(pool_idx, num_corrupt, replace=False)
        self.corrupted_indices = set(corrupt_indices.tolist())

        print(f"Injecting {self.corruption_name} (severity={self.severity}) into "
              f"{num_corrupt} pool images ({self.corruption_ratio:.1%})...")

        count = 0
        for idx in corrupt_indices:
            img_path = self.train_set.data[idx]
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)  # [H, W, 3] uint8
            corrupted_np = corrupt(img_np, severity=self.severity,
                                   corruption_name=self.corruption_name)
            # Cache as numpy array; __getitem__ will convert back to PIL before transform
            self.train_set._corrupted_cache[idx] = corrupted_np
            count += 1

        print(f"✓ Successfully corrupted {count} images in the training set.")

    def get_loader(self, indices, shuffle=False, transform=None):
        """Get data loader for specified indices, preserving corrupted data."""
        if transform is not None:
            dataset = copy.copy(self.train_set)
            dataset.transform = transform
            # Keep the same corrupted cache reference (shallow copy is fine)
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
