"""Datasets module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from torchvision import datasets
from ..benchmark_gradient_augmentation.constants import FINE_IDX_TO_COARSE


def _limit_dataset(ds, n: int | None, seed: int = 0) -> torch.utils.data.Dataset:
    """Return a random Subset of ds with n samples (or ds itself if n is None/invalid)."""
    if n is None or n <= 0 or n >= len(ds):
        return ds

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]

    return Subset(ds, idx.tolist())


class CIFAR100To10(Dataset):
    """Wrapper to the adapted CIFAR 100 dataset."""
    def __init__(self, base_dataset, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []

        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            if label in FINE_IDX_TO_COARSE:
                self.data.append(img)
                self.targets.append(FINE_IDX_TO_COARSE[label])

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def _load_cifar100_to_10(root, train=True, download=True, transform=None):
    """Helper to load the adapted CIFAR 100 dataset."""
    base = datasets.CIFAR100(root=root, train=train, download=download)
    return CIFAR100To10(base_dataset=base, transform=transform)


def _to_tensor(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    task: str = "regression",
    *,
    x_dtype: torch.dtype = torch.float32,
    y_dtype: torch.dtype | None = None,
    regression_as_2d: bool = True,
):
    """
    Convert numpy arrays to torch tensors with safe dtypes and shapes.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input features. Cast to `x_dtype` (default: torch.float32).
    y : np.ndarray or torch.Tensor
        Targets. Casting/shape handling depends on `task`, unless `y_dtype` is provided.
    task : {"classification", "regression"}, optional
        - "classification": y -> torch.int64, ensure 1D (N,)
        - "regression"   : y -> torch.float32, shape (N,1) if `regression_as_2d=True`
    x_dtype : torch.dtype, optional
        Target dtype for `x`. Default is torch.float32.
    y_dtype : torch.dtype or None, optional
        If provided, forces `y` to this dtype regardless of `task`.
    regression_as_2d : bool, optional
        If True (default), reshape regression targets to (N,1).
        If False, keep as 1D (N,).

    Returns
    -------
    x : torch.Tensor
    y : torch.Tensor
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    x = x.to(x_dtype)

    if y_dtype is not None:
        y = y.to(y_dtype)
        return x, y

    if task == "classification":
        if y.ndim == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        if y.dtype != torch.int64:
            y = y.to(torch.int64)
        return x, y

    if task == "regression":
        y = y.to(torch.float32)
        if regression_as_2d:
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            elif y.ndim == 2 and y.size(-1) != 1:
                raise ValueError(f"Expected regression y with last dim 1, got shape {tuple(y.shape)}.")
        else:
            if y.ndim == 2 and y.size(-1) == 1:
                y = y.squeeze(-1)
        return x, y

    raise ValueError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")
