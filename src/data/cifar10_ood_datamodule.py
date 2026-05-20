"""CIFAR-10 OOD DataModule — injects Out-of-Distribution images into the pool.

Supported OOD sources: MNIST, FashionMNIST, SVHN.
OOD images are resized to 32×32 and converted to 3-channel RGB so they
share the same tensor shape as CIFAR-10.  Each OOD image receives a
**random** CIFAR-10 class label (simulating a human annotator who is
forced to pick a class even though the image doesn't belong to any).
"""
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


class CIFAR10OODDataModule:
    """CIFAR-10 + OOD contamination in the unlabeled pool."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)

        # OOD settings
        self.ood_dataset_name = cfg.get('ood_dataset', 'MNIST')
        self.ood_ratio = cfg.get('ood_ratio', 0.5)

        # CIFAR-10 normalization
        self.mean = cfg.get('mean', [0.4914, 0.4822, 0.4465])
        self.std = cfg.get('std', [0.2023, 0.1994, 0.2010])

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

        # Load standard CIFAR-10
        self.train_set = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform_train
        )
        self.test_set = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform_test
        )

        # Will be populated during setup_active_learning
        self.ood_indices = set()

    # ------------------------------------------------------------------
    # Active-learning split + OOD injection
    # ------------------------------------------------------------------
    def setup_active_learning(self, initial_labeled, calibration_size, seed=42):
        """Split data and inject OOD images into the pool."""
        np.random.seed(seed)
        n_total = len(self.train_set)
        idx = np.random.permutation(n_total)

        labeled_idx = idx[:initial_labeled].tolist()
        calib_idx = idx[initial_labeled:initial_labeled + calibration_size].tolist()
        pool_idx = idx[initial_labeled + calibration_size:].tolist()

        print(f"Initial labeled set: {len(labeled_idx)} clean images")
        print(f"Calibration set: {len(calib_idx)} clean images")

        if self.ood_ratio > 0:
            pool_idx = self._inject_ood(pool_idx, seed)
        else:
            self.ood_indices = set()
            print("OOD ratio = 0 → no OOD injection.")

        return labeled_idx, calib_idx, pool_idx

    # ------------------------------------------------------------------
    # OOD injection
    # ------------------------------------------------------------------
    def _inject_ood(self, pool_idx, seed):
        """Replace part of the pool with OOD images.

        With ood_ratio = 0.5 and original pool of 40 000:
            → keep 20 000 clean CIFAR-10 images
            → add  20 000 OOD images
            → total pool = 40 000
        """
        np.random.seed(seed)
        random.seed(seed)

        original_pool_size = len(pool_idx)
        n_clean = int(original_pool_size * (1 - self.ood_ratio))
        n_ood = original_pool_size - n_clean

        if n_ood == 0:
            self.ood_indices = set()
            return pool_idx

        # Trim pool to keep only the clean portion
        pool_idx = pool_idx[:n_clean]

        # Load OOD images
        ood_images = self._load_ood_images(n_ood, seed)

        # Current dataset size (= first available index for appended data)
        base_idx = len(self.train_set)
        num_classes = self.cfg.get('num_classes', 10)

        # Append OOD images to self.train_set.data / .targets
        # CIFAR-10: self.data is np.ndarray (N, 32, 32, 3) uint8
        #           self.targets is list of int
        self.train_set.data = np.concatenate(
            [self.train_set.data, ood_images], axis=0
        )

        ood_labels = [random.randint(0, num_classes - 1) for _ in range(n_ood)]
        self.train_set.targets.extend(ood_labels)

        # Record OOD indices & add them to pool
        new_indices = list(range(base_idx, base_idx + n_ood))
        self.ood_indices = set(new_indices)
        pool_idx.extend(new_indices)

        # Shuffle pool so OOD images are mixed in
        np.random.shuffle(pool_idx)

        print(f"Pool: {len(pool_idx)} total  "
              f"({n_clean} clean + {n_ood} OOD [{self.ood_dataset_name}])  "
              f"ood_ratio={self.ood_ratio:.0%}")

        return pool_idx

    # ------------------------------------------------------------------
    def _load_ood_images(self, n_ood, seed):
        """Load & preprocess OOD images → np.ndarray (N, 32, 32, 3) uint8."""
        name = self.ood_dataset_name.upper()
        print(f"Loading {n_ood} OOD images from {name}...")

        if name == 'MNIST':
            ds = datasets.MNIST(root=self.root, train=True, download=True)
            raw = ds.data.numpy()  # (60000, 28, 28) uint8
            images = self._grayscale28_to_rgb32(raw, n_ood, seed)

        elif name == 'FASHIONMNIST':
            ds = datasets.FashionMNIST(root=self.root, train=True, download=True)
            raw = ds.data.numpy()  # (60000, 28, 28) uint8
            images = self._grayscale28_to_rgb32(raw, n_ood, seed)

        elif name == 'SVHN':
            ds = datasets.SVHN(root=self.root, split='train', download=True)
            raw = ds.data  # (73257, 3, 32, 32) uint8
            raw = raw.transpose(0, 2, 3, 1)  # → (N, 32, 32, 3)
            np.random.seed(seed)
            chosen = np.random.choice(len(raw), size=min(n_ood, len(raw)), replace=n_ood > len(raw))
            images = raw[chosen][:n_ood]

        else:
            raise ValueError(f"Unknown OOD dataset: {self.ood_dataset_name}. "
                             f"Supported: MNIST, FashionMNIST, SVHN")

        print(f"✓ Loaded {len(images)} OOD images, shape per image: {images.shape[1:]}")
        return images

    # ------------------------------------------------------------------
    @staticmethod
    def _grayscale28_to_rgb32(raw, n_ood, seed):
        """Convert (N, 28, 28) grayscale uint8 → (n_ood, 32, 32, 3) RGB uint8."""
        np.random.seed(seed)
        chosen_idx = np.random.choice(len(raw), size=min(n_ood, len(raw)), replace=n_ood > len(raw))
        selected = raw[chosen_idx][:n_ood]

        result = np.empty((len(selected), 32, 32, 3), dtype=np.uint8)
        for i, img in enumerate(selected):
            pil = Image.fromarray(img, mode='L')
            pil = pil.resize((32, 32), Image.BILINEAR)
            pil = pil.convert('RGB')
            result[i] = np.array(pil)
        return result

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------
    def get_loader(self, indices, shuffle=False, transform=None):
        """Get data loader for specified indices."""
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
        """Get test data loader (clean CIFAR-10 test set only)."""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
