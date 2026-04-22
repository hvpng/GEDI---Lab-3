"""Evaluation metrics for clustering experiments.

This module provides the label-aware and label-free metrics required by the
lab instructions. All important functions include docstrings and are designed
to be deterministic when combined with a fixed random seed.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering accuracy with Hungarian label matching.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted cluster assignments.

    Returns:
        Accuracy after the best one-to-one label permutation is applied.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    n_labels = max(y_true.max(), y_pred.max()) + 1
    contingency = np.zeros((n_labels, n_labels), dtype=np.int64)
    for pred_label, true_label in zip(y_pred, y_true):
        contingency[pred_label, true_label] += 1

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / y_true.size)


def evaluate_clustering(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a clustering solution with six common metrics.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        y_true: Ground-truth labels used for supervised metrics.
        y_pred: Predicted cluster assignments.

    Returns:
        A dictionary containing ACC, NMI, ARI, silhouette, DBI and CHI.
        Metrics that cannot be computed safely are returned as NaN.
    """
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
    }

    unique_clusters = np.unique(y_pred)
    if X.ndim == 2 and unique_clusters.size > 1 and unique_clusters.size < len(X):
        results["Silhouette"] = float(silhouette_score(X, y_pred))
        results["DBI"] = float(davies_bouldin_score(X, y_pred))
        results["CHI"] = float(calinski_harabasz_score(X, y_pred))
    else:
        results["Silhouette"] = np.nan
        results["DBI"] = np.nan
        results["CHI"] = np.nan

    return results


def relative_deviation(reproduced: float, reported: float) -> float:
    """Compute the relative deviation between reproduced and reported scores.

    Args:
        reproduced: Score obtained by the current implementation.
        reported: Score reported in the paper.

    Returns:
        The relative deviation in percent. If the reference value is zero,
        NaN is returned to avoid division-by-zero artifacts.
    """
    if reported == 0:
        return float("nan")
    return float(((reproduced - reported) / reported) * 100.0)


def build_paper_comparison_table(
    reported_scores: Mapping[str, float],
    reproduced_scores: Mapping[str, float],
    metric_name: str = "NMI",
) -> pd.DataFrame:
    """Create a comparison table between paper scores and reproduced scores.

    Args:
        reported_scores: Mapping from dataset name to paper-reported metric.
        reproduced_scores: Mapping from dataset name to current metric.
        metric_name: Name of the metric to display in the output table.

    Returns:
        A pandas DataFrame with paper score, reproduced score and relative
        deviation for every shared dataset key.
    """
    shared_datasets = sorted(set(reported_scores) & set(reproduced_scores))
    rows = []
    for dataset in shared_datasets:
        reported = float(reported_scores[dataset])
        reproduced = float(reproduced_scores[dataset])
        rows.append(
            {
                "Dataset": dataset,
                f"Paper {metric_name}": reported,
                f"Our {metric_name}": reproduced,
                "Relative deviation (%)": relative_deviation(reproduced, reported),
            }
        )

    return pd.DataFrame(rows)
