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
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
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
    else:
        raise ValueError("Supported datasets are: digits, wine, iris.")

    scaler = StandardScaler()
    X = scaler.fit_transform(dataset.data)
    y = dataset.target.astype(int)
    return X, y


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
    }
