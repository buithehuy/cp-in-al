"""Data module package."""
from .datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule
from .stl10_datamodule import STL10DataModule
from .svhn_datamodule import SVHNDataModule

__all__ = ['CIFAR10DataModule', 'CIFAR100DataModule', 'STL10DataModule', 'SVHNDataModule']
