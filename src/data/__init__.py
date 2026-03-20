"""Data modules package."""
from .datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule
from .cifar100n_datamodule import CIFAR100NDataModule
from .cifar100c_datamodule import CIFAR100CDataModule
from .cifar10c_datamodule import CIFAR10CDataModule
from .stl10_datamodule import STL10DataModule
from .svhn_datamodule import SVHNDataModule
from .cub200_datamodule import CUB200DataModule
from .svhnc_datamodule import SVHNCDataModule
from .stl10c_datamodule import STL10CDataModule
from .cub200c_datamodule import CUB200CDataModule

__all__ = [
    'CIFAR10DataModule',
    'CIFAR100DataModule',
    'CIFAR100NDataModule',
    'CIFAR100CDataModule',
    'CIFAR10CDataModule',
    'STL10DataModule',
    'SVHNDataModule',
    'CUB200DataModule',
    'SVHNCDataModule',
    'STL10CDataModule',
    'CUB200CDataModule',
]
