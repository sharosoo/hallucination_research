"""
Metrics Module for Hallucination Detection Evaluation

중앙 집중식 메트릭 계산 모듈.
모든 실험에서 일관된 방식으로 AUROC, AUPRC 등을 계산합니다.

핵심 정책:
- Semantic Energy: energy = -mean(logits), 높은 값(덜 음수) = uncertain = hallucination
- Semantic Entropy: 높은 값 = uncertain = hallucination
- 모든 메트릭: 높은 score = hallucination으로 통일 (flip 불필요)

Usage:
    from hfe_core.metrics import compute_auroc, compute_all_metrics

    # 개별 AUROC 계산
    se_auroc = compute_auroc(labels, se_scores, score_type="semantic_entropy")
    energy_auroc = compute_auroc(labels, energy_scores, score_type="semantic_energy")

    # 전체 메트릭 계산
    metrics = compute_all_metrics(labels, se_scores, energy_scores)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np


@dataclass
class MetricsResult:
    """메트릭 계산 결과"""

    auroc: float | None
    auprc: float | None
    n_samples: int
    n_positive: int  # hallucination count
    n_negative: int  # normal count
    positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllMetricsResult:
    """SE + Energy 통합 메트릭 결과"""

    semantic_entropy: MetricsResult
    semantic_energy: MetricsResult
    winner: str  # "SE", "Energy", "Tie", "N/A"
    auroc_diff: float | None  # SE - Energy

    def to_dict(self) -> dict:
        return {
            "semantic_entropy": self.semantic_entropy.to_dict(),
            "semantic_energy": self.semantic_energy.to_dict(),
            "winner": self.winner,
            "auroc_diff": self.auroc_diff,
        }


ScoreType = Literal["semantic_entropy", "semantic_energy", "raw"]


def _validate_inputs(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """입력 검증 및 통계 계산"""
    labels = np.array(labels)
    scores = np.array(scores)

    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: labels={len(labels)}, scores={len(scores)}")

    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))

    return labels, scores, n_positive, n_negative


def _transform_scores(
    scores: np.ndarray,
    score_type: ScoreType,
) -> np.ndarray:
    """
    Energy 부호 정책 (중요!):

    energy = -mean(logits) 에서:
    - 높은 logit (confident) → 매우 음수 energy (예: -45)
    - 낮은 logit (uncertain) → 덜 음수 energy (예: -20)

    따라서:
    - 높은 energy (덜 음수) = uncertain = hallucination
    - 낮은 energy (더 음수) = confident = normal

    AUROC: 높은 score = hallucination 예측
    → Energy는 flip 없이 그대로 사용!
    """
    return scores


def compute_auroc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    score_type: ScoreType = "raw",
) -> float | None:
    """
    AUROC 계산

    Args:
        labels: 정답 레이블 (1=hallucination, 0=normal)
        scores: 예측 점수
        score_type: 점수 유형 ("semantic_entropy", "semantic_energy", "raw")
            - semantic_energy: 그대로 사용 (높은 값 = uncertain = hallucination)
            - semantic_entropy, raw: 그대로 사용

    Returns:
        AUROC 값 또는 None (계산 불가 시)
    """
    from sklearn.metrics import roc_auc_score

    labels, scores, n_pos, n_neg = _validate_inputs(labels, scores)

    # 두 클래스 모두 필요
    if n_pos == 0 or n_neg == 0:
        return None

    # 최소 샘플 수
    if len(labels) < 5:
        return None

    # Score 변환
    transformed = _transform_scores(scores, score_type)

    try:
        return float(roc_auc_score(labels, transformed))
    except Exception:
        return None


def compute_auprc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    score_type: ScoreType = "raw",
) -> float | None:
    """
    AUPRC (Average Precision) 계산

    Args:
        labels: 정답 레이블 (1=hallucination, 0=normal)
        scores: 예측 점수
        score_type: 점수 유형

    Returns:
        AUPRC 값 또는 None (계산 불가 시)
    """
    from sklearn.metrics import average_precision_score

    labels, scores, n_pos, n_neg = _validate_inputs(labels, scores)

    if n_pos == 0 or n_neg == 0:
        return None

    if len(labels) < 5:
        return None

    transformed = _transform_scores(scores, score_type)

    try:
        return float(average_precision_score(labels, transformed))
    except Exception:
        return None


def compute_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    score_type: ScoreType = "raw",
) -> MetricsResult:
    """
    단일 score 유형에 대한 메트릭 계산

    Args:
        labels: 정답 레이블
        scores: 예측 점수
        score_type: 점수 유형

    Returns:
        MetricsResult
    """
    labels_arr, scores_arr, n_pos, n_neg = _validate_inputs(labels, scores)

    auroc = compute_auroc(labels, scores, score_type)
    auprc = compute_auprc(labels, scores, score_type)

    n_total = len(labels_arr)
    positive_rate = n_pos / n_total if n_total > 0 else 0.0

    return MetricsResult(
        auroc=auroc,
        auprc=auprc,
        n_samples=n_total,
        n_positive=n_pos,
        n_negative=n_neg,
        positive_rate=positive_rate,
    )


def compute_all_metrics(
    labels: list[int] | np.ndarray,
    se_scores: list[float] | np.ndarray,
    energy_scores: list[float] | np.ndarray,
) -> AllMetricsResult:
    """SE + Energy AUROC 계산. 높은 score = hallucination."""
    se_result = compute_metrics(labels, se_scores, score_type="semantic_entropy")
    energy_result = compute_metrics(labels, energy_scores, score_type="semantic_energy")

    # Winner 결정
    if se_result.auroc is not None and energy_result.auroc is not None:
        auroc_diff = se_result.auroc - energy_result.auroc
        if abs(auroc_diff) < 0.001:
            winner = "Tie"
        elif auroc_diff > 0:
            winner = "SE"
        else:
            winner = "Energy"
    else:
        auroc_diff = None
        if se_result.auroc is not None:
            winner = "SE (Energy N/A)"
        elif energy_result.auroc is not None:
            winner = "Energy (SE N/A)"
        else:
            winner = "N/A"

    return AllMetricsResult(
        semantic_entropy=se_result,
        semantic_energy=energy_result,
        winner=winner,
        auroc_diff=auroc_diff,
    )


def compute_combined_score(
    se_score: float,
    energy_score: float,
    weight_energy: float,
    normalize: bool = True,
    se_min: float | None = None,
    se_max: float | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> float:
    """Score = w * Energy_norm + (1-w) * SE_norm. 높은 값 = hallucination."""
    if normalize:
        if se_min is not None and se_max is not None and se_max > se_min:
            se_norm = (se_score - se_min) / (se_max - se_min)
        else:
            se_norm = se_score

        if (
            energy_min is not None
            and energy_max is not None
            and energy_max > energy_min
        ):
            energy_norm = (energy_score - energy_min) / (energy_max - energy_min)
        else:
            energy_norm = energy_score
    else:
        se_norm = se_score
        energy_norm = energy_score

    return weight_energy * energy_norm + (1 - weight_energy) * se_norm


def compute_combined_auroc(
    labels: list[int] | np.ndarray,
    se_scores: list[float] | np.ndarray,
    energy_scores: list[float] | np.ndarray,
    weight_energy: float,
    normalize: bool = True,
) -> float | None:
    """
    결합된 점수의 AUROC 계산

    Args:
        labels: 정답 레이블
        se_scores: SE 점수 리스트
        energy_scores: Energy 점수 리스트 (raw)
        weight_energy: Energy 가중치
        normalize: 정규화 여부

    Returns:
        AUROC 또는 None
    """
    labels_arr = np.array(labels)
    se_arr = np.array(se_scores)
    energy_arr = np.array(energy_scores)

    if normalize:
        se_min, se_max = float(np.min(se_arr)), float(np.max(se_arr))
        energy_min, energy_max = float(np.min(energy_arr)), float(np.max(energy_arr))
    else:
        se_min = se_max = energy_min = energy_max = None

    combined = [
        compute_combined_score(
            se_arr[i],
            energy_arr[i],
            weight_energy,
            normalize,
            se_min,
            se_max,
            energy_min,
            energy_max,
        )
        for i in range(len(labels_arr))
    ]

    return compute_auroc(labels, combined, score_type="raw")


# Statistics helpers
def get_score_statistics(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> dict:
    """점수 통계 계산"""
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    hall_scores = scores_arr[labels_arr == 1]
    norm_scores = scores_arr[labels_arr == 0]

    result = {
        "overall": {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
        }
    }

    if len(hall_scores) > 0:
        result["hallucination"] = {
            "mean": float(np.mean(hall_scores)),
            "std": float(np.std(hall_scores)),
            "n": len(hall_scores),
        }

    if len(norm_scores) > 0:
        result["normal"] = {
            "mean": float(np.mean(norm_scores)),
            "std": float(np.std(norm_scores)),
            "n": len(norm_scores),
        }

    return result
