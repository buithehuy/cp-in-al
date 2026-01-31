"""Data module package."""
from .datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule

__all__ = ['CIFAR10DataModule', 'CIFAR100DataModule']
