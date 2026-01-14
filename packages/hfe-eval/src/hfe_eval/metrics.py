"""Metrics for hallucination detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


@dataclass
class EvaluationMetrics:
    auroc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    exact_match: float | None = None


def compute_metrics(
    labels: list[int],
    predictions: list[int],
    scores: list[float] | None = None,
) -> EvaluationMetrics:
    """
    Compute evaluation metrics for hallucination detection.

    Parameters
    ----------
    labels : list[int]
        Ground truth labels (0=valid, 1=hallucination)
    predictions : list[int]
        Binary predictions
    scores : list[float], optional
        Continuous scores for AUROC calculation

    Returns
    -------
    EvaluationMetrics
        Computed metrics
    """
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)

    auroc = 0.0
    if scores is not None:
        scores_arr = np.array(scores)
        if len(np.unique(labels_arr)) > 1:
            auroc = roc_auc_score(labels_arr, scores_arr)

    return EvaluationMetrics(
        auroc=auroc,
        f1=f1_score(labels_arr, preds_arr, zero_division=0),
        precision=precision_score(labels_arr, preds_arr, zero_division=0),
        recall=recall_score(labels_arr, preds_arr, zero_division=0),
        accuracy=accuracy_score(labels_arr, preds_arr),
    )


def compute_exact_match(predictions: list[str], ground_truths: list[str]) -> float:
    """Compute exact match score for QA tasks."""
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.lower().strip()
        gt_clean = gt.lower().strip()

        if "so the answer is" in pred_clean:
            pred_clean = pred_clean.split("so the answer is")[-1].strip()
            pred_clean = pred_clean.rstrip(".")

        if gt_clean in pred_clean or pred_clean == gt_clean:
            correct += 1

    return correct / len(predictions)


def compute_f1_token(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)
