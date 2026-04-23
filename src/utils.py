"""Utility helpers for reproducible clustering experiments."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
try:
    import torchvision as _torchvision
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ArrayPair = Tuple[np.ndarray, np.ndarray]
TrainTestSplit = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def set_random_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (if available) for reproducible experiments.

    Args:
        seed: Integer seed value. Defaults to 42.

    Returns:
        None. Seeds are set as a side effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)


def generate_synthetic_dataset(
    kind: str,
    n_samples: int = 2000,
    data_noise: float = 0.03,
    random_state: int = 42,
    test_size: float = 0.3,
) -> TrainTestSplit:
    """Generate the moons or circles benchmark used in the paper's synthetic study.

    Args:
        kind: Either "moons" or "circles".
        n_samples: Number of total samples to generate.
        data_noise: Standard deviation of the label-generating noise applied by
            sklearn when placing data points (i.e. spread of each cluster).
            Note: this is distinct from the augmentation noise used inside
            GEDI training in ``train_gedi()`` (Gaussian noise with std=0.03).
        random_state: Seed used for data generation and splitting.
        test_size: Fraction of examples reserved for testing.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
            - X_train: Standardized training features, shape (n_train, 2).
            - X_test:  Standardized test features,     shape (n_test,  2).
            - y_train: Integer class labels for train, shape (n_train,).
            - y_test:  Integer class labels for test,  shape (n_test,).
    """
    if kind == "moons":
        X, y = make_moons(n_samples=n_samples, noise=data_noise, random_state=random_state)
    elif kind == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=data_noise,
            factor=0.5,
            random_state=random_state,
        )
    else:
        raise ValueError("kind must be either 'moons' or 'circles'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_additional_dataset(name: str = "digits") -> ArrayPair:
    """Load a labeled dataset not used in the paper for external validation.

    Unlike ``generate_synthetic_dataset``, this function returns the **full**
    dataset without a train/test split. Clustering is an unsupervised task so
    all samples are evaluated together; callers that need a held-out split can
    apply ``sklearn.model_selection.train_test_split`` on the returned arrays.

    Args:
        name: One of "digits", "wine", or "iris".

    Returns:
        A tuple (X, y) containing standardized features and labels.
    """
    name = name.lower()
    if name == "digits":
        dataset = load_digits()
    elif name == "wine":
        dataset = load_wine()
    elif name == "iris":
        dataset = load_iris()
    elif name in ("fashion_mnist", "fashionmnist"):
        return load_fashion_mnist()
    elif name == "svhn":
        return load_svhn()
    else:
        raise ValueError("Supported datasets are: digits, wine, iris, fashion_mnist, svhn.")

    scaler = StandardScaler()
    X = scaler.fit_transform(dataset.data)
    y = dataset.target.astype(int)
    return X, y


def load_svhn(
    data_dir: str = "data",
    max_samples: int = 5000,
    pca_components: int = 50,
    random_state: int = 42,
    raw: bool = False,
    split: str = "test",
) -> ArrayPair:
    """Load a subset of SVHN via torchvision for clustering evaluation.

    Images are flattened from 32x32x3 to 3072 dimensions and normalised to [-1, 1].
    By default, dimensionality is further reduced via PCA for MLP-based models.
    Set ``raw=True`` to skip StandardScaler and PCA and return the flat normalised
    pixels directly — required when using the ResNet-8 encoder (Appendix M, Table 8).

    Args:
        data_dir:       Directory where torchvision will cache the raw files.
        max_samples:    Maximum number of samples to retain (random subsample).
            Pass ``None`` to use the full split.
        pca_components: Number of PCA components used to reduce dimensionality.
            Ignored when ``raw=True``.
        random_state:   Seed for subsampling and PCA.
        raw:            If True, skip StandardScaler and PCA and return flat
            normalised pixels, shape (n, 3072).  Required for ResNet-8 encoder.
        split:          SVHN split to load: ``'train'`` (73 257 samples) or
            ``'test'`` (26 032 samples).  The paper trains on ``'train'`` and
            evaluates on ``'test'``.

    Returns:
        A tuple (X, y) where X has shape (n, pca_components) for PCA mode or
        (n, 3072) for raw mode, and y has shape (n,).
    """
    if not _TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for load_svhn. "
            "Install with: pip install torchvision"
        )
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_tv = torchvision.datasets.SVHN(
        root=data_dir, split=split, download=True, transform=transform,
    )
    loader = DataLoader(dataset_tv, batch_size=4096, shuffle=False, num_workers=0)
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb.numpy().reshape(xb.shape[0], -1))
        y_list.append(yb.numpy())
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    rng = np.random.RandomState(random_state)
    if max_samples is not None:
        idx = rng.choice(len(X_all), size=min(max_samples, len(X_all)), replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    if raw:
        # Return flat [-1, 1] pixels as-is — for ResNet-8 encoder
        return X_all.astype(np.float32), y_all.astype(int)

    X_all = StandardScaler().fit_transform(X_all)
    if pca_components and pca_components < X_all.shape[1]:
        X_all = PCA(n_components=pca_components, random_state=random_state).fit_transform(X_all)
    return X_all, y_all.astype(int)


def load_fashion_mnist(
    data_dir: str = "data",
    split: str = "test",
    max_samples: int | None = None,
    pca_components: int = 50,
    random_state: int = 42,
) -> ArrayPair:
    """Load Fashion-MNIST via torchvision for clustering evaluation.

    Images are flattened from 28x28 to 784 dimensions, normalised to [-1, 1],
    then reduced via PCA.  The test split (10 000 samples) is used by default.

    Args:
        data_dir:       Directory where torchvision will cache the raw files.
        split:          ``"train"`` (60 000) or ``"test"`` (10 000).
        max_samples:    Cap on number of samples; ``None`` keeps all.
        pca_components: Number of PCA components.
        random_state:   Seed for subsampling and PCA.

    Returns:
        A tuple (X, y) where X has shape (n, pca_components) and y has shape (n,).
    """
    if not _TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for load_fashion_mnist. "
            "Install with: pip install torchvision"
        )
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset_tv = torchvision.datasets.FashionMNIST(
        root=data_dir, train=(split == "train"), download=True, transform=transform,
    )
    loader = DataLoader(dataset_tv, batch_size=4096, shuffle=False, num_workers=0)
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb.numpy().reshape(xb.shape[0], -1))
        y_list.append(yb.numpy())
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if max_samples is not None and max_samples < len(X_all):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_all), size=max_samples, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    X_all = StandardScaler().fit_transform(X_all)
    if pca_components and pca_components < X_all.shape[1]:
        X_all = PCA(n_components=pca_components, random_state=random_state).fit_transform(X_all)
    return X_all, y_all.astype(int)


def get_paper_reference_scores() -> Dict[str, Dict[str, float]]:
    """Return NMI scores for all baselines reported in Table 3 of the paper.

    Source: Table 3, Section 6.2 of Sansone & Manhaeve (TMLR 2025).
    Mean values only; standard deviations are available in the paper.

    Returns:
        A nested dictionary keyed by dataset name, then by method name, with
        the NMI value reported in the paper's synthetic comparison table.
        Methods: JEM, Barlow (Barlow Twins), SwAV, GEDI_no_gen, GEDI.
        Example: result['moons']['GEDI'] == 0.94
    """
    return {
        "moons": {
            "JEM": 0.00,
            "Barlow": 0.22,
            "SwAV": 0.76,
            "GEDI_no_gen": 0.98,
            "GEDI": 0.94,
        },
        "circles": {
            "JEM": 0.00,
            "Barlow": 0.13,
            "SwAV": 0.00,
            "GEDI_no_gen": 0.83,
            "GEDI": 1.00,
        },
        # Source: Table 4, Sansone & Manhaeve (TMLR 2025) — NMI, SVHN test split
        "svhn": {
            "JEM": 0.00,
            "Barlow": 0.20,
            "SwAV": 0.21,
            "GEDI_no_gen": 0.27,
            "GEDI": 0.25,
        },
    }
