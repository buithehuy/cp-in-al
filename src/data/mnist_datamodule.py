from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


class MNISTDataModule:
    """Data module for MNIST with Active Learning setup."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)

        # MNIST specific normalization (grayscale, single channel)
        self.mean = cfg.get('mean', [0.1307])
        self.std = cfg.get('std', [0.3081])

        # Transforms – expand grayscale to 3 channels so ResNet18 works out of the box
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(self.mean * 3, self.std * 3)
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(self.mean * 3, self.std * 3)
        ])

        # Load MNIST
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
            dataset = datasets.MNIST(
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
