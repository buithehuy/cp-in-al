"""STL10-C DataModule with dynamic corruption generation."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
from imagecorruptions import corrupt


class STL10CDataModule:
    """Data module for STL10-C with Active Learning setup.

    Dynamically generates corruption on a subset of the pool data.

    Note: STL10's .data attribute is [N, 3, H, W] (channels-first uint8).
    We transpose to [N, H, W, 3] before applying imagecorruptions, then
    transpose back.
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

        # Load STL-10
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
        """Set up active learning data splits and inject corruption."""
        if hasattr(self, '_is_corrupted') and self._is_corrupted:
            print("Warning: Corruption already applied. Reloading clean dataset...")
            self.train_set = datasets.STL10(
                root=self.root,
                split='train',
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
            self.corrupted_indices = set()  # No corruption

        return labeled_idx, calib_idx, pool_idx

    def _inject_corruption(self, pool_idx, seed):
        """Apply corruption to a subset of pool images in self.train_set.

        STL10 .data is [N, 3, H, W] uint8. imagecorruptions expects [H, W, 3],
        so we transpose in/out.
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

        # self.train_set.data is [N, 3, H, W] uint8
        count = 0
        for idx in corrupt_indices:
            img_chw = self.train_set.data[idx]          # [3, H, W] uint8
            img_hwc = img_chw.transpose(1, 2, 0)        # [H, W, 3] uint8
            corrupted_hwc = corrupt(img_hwc, severity=self.severity,
                                    corruption_name=self.corruption_name)
            self.train_set.data[idx] = corrupted_hwc.transpose(2, 0, 1)  # back to [3, H, W]
            count += 1

        print(f"✓ Successfully corrupted {count} images in the training set.")

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
