"""Data modules package."""
from .datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule
from .cifar100n_datamodule import CIFAR100NDataModule
from .stl10_datamodule import STL10DataModule
from .svhn_datamodule import SVHNDataModule

__all__ = [
    'CIFAR10DataModule',
    'CIFAR100DataModule',
    'CIFAR100NDataModule',
    'STL10DataModule',
    'SVHNDataModule',
]
