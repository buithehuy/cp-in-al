"""Data modules package."""
from .datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule
from .cifar100n_datamodule import CIFAR100NDataModule
from .cifar100c_datamodule import CIFAR100CDataModule
from .cifar10c_datamodule import CIFAR10CDataModule
from .cifar10_ood_datamodule import CIFAR10OODDataModule
from .stl10_datamodule import STL10DataModule
from .svhn_datamodule import SVHNDataModule
from .cub200_datamodule import CUB200DataModule
from .svhnc_datamodule import SVHNCDataModule
from .stl10c_datamodule import STL10CDataModule
from .cub200c_datamodule import CUB200CDataModule
from .mnist_datamodule import MNISTDataModule
from .mnistc_datamodule import MNISTCDataModule

__all__ = [
    'CIFAR10DataModule',
    'CIFAR100DataModule',
    'CIFAR100NDataModule',
    'CIFAR100CDataModule',
    'CIFAR10CDataModule',
    'CIFAR10OODDataModule',
    'STL10DataModule',
    'SVHNDataModule',
    'CUB200DataModule',
    'SVHNCDataModule',
    'STL10CDataModule',
    'CUB200CDataModule',
    'MNISTDataModule',
    'MNISTCDataModule',
]
