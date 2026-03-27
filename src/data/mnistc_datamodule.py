"""MNIST-C DataModule with dynamic corruption generation."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
import torch
from imagecorruptions import corrupt


class MNISTCDataModule:
    """Data module for MNIST-C with Active Learning setup.

    Dynamically generates corruption on a subset of the pool data.
    MNIST images are grayscale (H, W) – they are converted to uint8 RGB
    for compatibility with the imagecorruptions library, then converted back.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)

        # Corruption settings
        self.corruption_name = cfg.get('corruption', 'gaussian_noise')
        self.severity = cfg.get('severity', 1)
        self.corruption_ratio = cfg.get('corruption_ratio', 0.0)

        # MNIST specific normalization
        self.mean = cfg.get('mean', [0.1307])
        self.std = cfg.get('std', [0.3081])

        # Transforms
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # Load standard MNIST
        self.train_set = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform_train
        )
        self.test_set = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform_test
        )

        self._is_corrupted = False
        self.corrupted_indices = set()

    def setup_active_learning(self, initial_labeled, calibration_size, seed=42):
        """Set up active learning data splits and inject corruption."""
        if self._is_corrupted:
            print("Warning: Corruption already applied. Reloading clean dataset...")
            self.train_set = datasets.MNIST(
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
        else:
            self.corrupted_indices = set()

        return labeled_idx, calib_idx, pool_idx

    def _inject_corruption(self, pool_idx, seed):
        """Apply corruption to a subset of pool images in self.train_set.

        MNIST .data is a torch.Tensor of shape [N, 28, 28] with values in [0, 255].
        The imagecorruptions library expects uint8 RGB images, so we temporarily
        convert each grayscale image to a (28, 28, 3) uint8 numpy array, apply
        corruption, and store the average channel back to grayscale.
        """
        np.random.seed(seed)

        num_corrupt = int(len(pool_idx) * self.corruption_ratio)
        if num_corrupt == 0:
            print("Corruption ratio too small, no images corrupted.")
            self.corrupted_indices = set()
            return

        corrupt_indices = np.random.choice(pool_idx, num_corrupt, replace=False)
        self.corrupted_indices = set(corrupt_indices.tolist())

        print(
            f"Injecting {self.corruption_name} (severity={self.severity}) "
            f"into {num_corrupt} pool images ({self.corruption_ratio:.1%})..."
        )

        count = 0
        for idx in corrupt_indices:
            # self.train_set.data: torch.Tensor [N, 28, 28], uint8
            img_gray = self.train_set.data[idx].numpy()  # (28, 28) uint8

            # Convert grayscale to RGB for imagecorruptions
            img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)  # (28, 28, 3)

            corrupted_rgb = corrupt(
                img_rgb,
                severity=self.severity,
                corruption_name=self.corruption_name
            )  # (28, 28, 3) uint8

            # Convert back to grayscale by averaging channels
            corrupted_gray = corrupted_rgb.mean(axis=-1).astype(np.uint8)  # (28, 28)

            self.train_set.data[idx] = torch.from_numpy(corrupted_gray)
            count += 1

        print(f"\u2713 Successfully corrupted {count} images in the training set.")

    def get_loader(self, indices, shuffle=False, transform=None):
        """Get data loader for specified indices, preserving corrupted data."""
        if transform is not None:
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
