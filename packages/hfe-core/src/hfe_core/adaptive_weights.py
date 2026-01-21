"""Adaptive weight functions for SE/Energy combination based on corpus coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class WeightFunction(Protocol):
    """Protocol for weight functions."""

    def __call__(self, coverage: float) -> float:
        """Compute energy weight from corpus coverage (0~1)."""
        ...


class WeightType(Enum):
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    THRESHOLD = "threshold"
    FIXED = "fixed"
    SE_FALLBACK = "se_fallback"


@dataclass
class AdaptiveWeightResult:
    """Result of adaptive weight computation."""

    coverage: float
    w_energy: float
    w_se: float
    weight_type: WeightType

    @property
    def combined_score(self) -> float | None:
        return None


class SigmoidWeight:
    """
    Sigmoid weight function: w(C) = 1 / (1 + exp(k * (C - μ)))

    - High coverage → low energy weight (use SE)
    - Low coverage → high energy weight (use Energy)

    Parameters:
        k: Steepness of sigmoid (default: 10)
        mu: Center point (default: 0.5)
    """

    def __init__(self, k: float = 10.0, mu: float = 0.5):
        self.k = k
        self.mu = mu

    def __call__(self, coverage: float) -> float:
        return 1.0 / (1.0 + math.exp(self.k * (coverage - self.mu)))


class LinearWeight:
    """
    Linear weight function: w(C) = w_max - (w_max - w_min) * C

    Parameters:
        w_min: Minimum energy weight at coverage=1 (default: 0.1)
        w_max: Maximum energy weight at coverage=0 (default: 0.9)
    """

    def __init__(self, w_min: float = 0.1, w_max: float = 0.9):
        self.w_min = w_min
        self.w_max = w_max

    def __call__(self, coverage: float) -> float:
        coverage = max(0.0, min(1.0, coverage))
        return self.w_max - (self.w_max - self.w_min) * coverage


class ThresholdWeight:
    """
    Threshold (step) weight function.

    Parameters:
        tau1: First threshold (default: 0.3)
        tau2: Second threshold (default: 0.7)
        w_low: Weight when coverage < tau1 (default: 0.9)
        w_mid: Weight when tau1 <= coverage < tau2 (default: 0.5)
        w_high: Weight when coverage >= tau2 (default: 0.1)
    """

    def __init__(
        self,
        tau1: float = 0.3,
        tau2: float = 0.7,
        w_low: float = 0.9,
        w_mid: float = 0.5,
        w_high: float = 0.1,
    ):
        self.tau1 = tau1
        self.tau2 = tau2
        self.w_low = w_low
        self.w_mid = w_mid
        self.w_high = w_high

    def __call__(self, coverage: float) -> float:
        if coverage < self.tau1:
            return self.w_low
        elif coverage < self.tau2:
            return self.w_mid
        else:
            return self.w_high


class FixedWeight:
    """Fixed weight (baseline)."""

    def __init__(self, w_energy: float = 0.5):
        self._w_energy = w_energy

    def __call__(self, coverage: float) -> float:
        return self._w_energy


class SEFallbackWeight:
    """
    SE-based fallback: Use SE when confident (high SE), fallback to Energy when SE is low.

    Rationale: Zero-SE analysis shows Energy AUROC=0.768 for samples where SE<0.1.
    When SE is near zero, the model is "confidently wrong" - Energy detects this better.

    Parameters:
        se_threshold: SE value below which we fallback to Energy (default: 0.1)
        w_energy_low_se: Energy weight when SE < threshold (default: 0.9)
        w_energy_high_se: Energy weight when SE >= threshold (default: 0.1)
    """

    def __init__(
        self,
        se_threshold: float = 0.1,
        w_energy_low_se: float = 0.9,
        w_energy_high_se: float = 0.1,
    ):
        self.se_threshold = se_threshold
        self.w_energy_low_se = w_energy_low_se
        self.w_energy_high_se = w_energy_high_se
        self._current_se: float | None = None

    def set_se(self, se_value: float) -> None:
        self._current_se = se_value

    def __call__(self, coverage: float) -> float:
        if self._current_se is None:
            return 0.5
        if self._current_se < self.se_threshold:
            return self.w_energy_low_se
        return self.w_energy_high_se


class AdaptiveWeightCalculator:
    """
    Calculator for adaptive SE/Energy weights based on corpus coverage.

    Usage:
        calc = AdaptiveWeightCalculator(weight_type=WeightType.SIGMOID)
        result = calc.compute(coverage=0.3)
        # result.w_energy ≈ 0.88, result.w_se ≈ 0.12
    """

    WEIGHT_FUNCTIONS: dict[WeightType, type] = {
        WeightType.SIGMOID: SigmoidWeight,
        WeightType.LINEAR: LinearWeight,
        WeightType.THRESHOLD: ThresholdWeight,
        WeightType.FIXED: FixedWeight,
        WeightType.SE_FALLBACK: SEFallbackWeight,
    }

    def __init__(
        self,
        weight_type: WeightType = WeightType.SIGMOID,
        **kwargs,
    ):
        self.weight_type = weight_type
        self.weight_fn = self.WEIGHT_FUNCTIONS[weight_type](**kwargs)

    def compute(self, coverage: float) -> AdaptiveWeightResult:
        w_energy = self.weight_fn(coverage)
        w_se = 1.0 - w_energy

        return AdaptiveWeightResult(
            coverage=coverage,
            w_energy=w_energy,
            w_se=w_se,
            weight_type=self.weight_type,
        )

    def compute_score(
        self,
        coverage: float,
        semantic_entropy: float,
        semantic_energy: float,
        normalize: bool = True,
    ) -> float:
        """
        Compute combined score using adaptive weights.

        Args:
            coverage: Corpus coverage (0~1)
            semantic_entropy: SE value (higher = more uncertain)
            semantic_energy: Energy value (more negative = more confident)
            normalize: Whether to normalize scores before combining

        Returns:
            Combined score (higher = more likely hallucination)
        """
        if isinstance(self.weight_fn, SEFallbackWeight):
            self.weight_fn.set_se(semantic_entropy)

        result = self.compute(coverage)

        if normalize:
            se_norm = semantic_entropy
            energy_norm = semantic_energy / 100.0
        else:
            se_norm = semantic_entropy
            energy_norm = semantic_energy

        return result.w_energy * energy_norm + result.w_se * se_norm


def create_weight_calculator(
    weight_type: str | WeightType,
    **kwargs,
) -> AdaptiveWeightCalculator:
    """Factory function for creating weight calculators."""
    if isinstance(weight_type, str):
        weight_type = WeightType(weight_type)
    return AdaptiveWeightCalculator(weight_type=weight_type, **kwargs)


BASELINE_CONFIGS = {
    "se_only": {"weight_type": WeightType.FIXED, "w_energy": 0.0},
    "energy_only": {"weight_type": WeightType.FIXED, "w_energy": 1.0},
    "fixed_0.1": {"weight_type": WeightType.FIXED, "w_energy": 0.1},
    "fixed_0.3": {"weight_type": WeightType.FIXED, "w_energy": 0.3},
    "fixed_0.5": {"weight_type": WeightType.FIXED, "w_energy": 0.5},
    "fixed_0.7": {"weight_type": WeightType.FIXED, "w_energy": 0.7},
    "fixed_0.9": {"weight_type": WeightType.FIXED, "w_energy": 0.9},
    "adaptive_sigmoid": {"weight_type": WeightType.SIGMOID},
    "adaptive_linear": {"weight_type": WeightType.LINEAR},
    "adaptive_threshold": {"weight_type": WeightType.THRESHOLD},
    "se_fallback": {"weight_type": WeightType.SE_FALLBACK},
    "se_fallback_strict": {"weight_type": WeightType.SE_FALLBACK, "se_threshold": 0.05},
    "se_fallback_relaxed": {"weight_type": WeightType.SE_FALLBACK, "se_threshold": 0.2},
}


def create_baseline_calculator(name: str) -> AdaptiveWeightCalculator:
    """Create a calculator from predefined baseline configs."""
    if name not in BASELINE_CONFIGS:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {list(BASELINE_CONFIGS.keys())}"
        )
    return AdaptiveWeightCalculator(**BASELINE_CONFIGS[name])
