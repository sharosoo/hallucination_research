"""
Adaptive SE-Energy Cascade

조교님 제안: input data를 보고 adaptive하게, input에 대한 function으로 τ 결정

핵심 아이디어:
- 고정 τ의 한계: cross-dataset에서 일반화 실패
- 해결: 각 샘플의 특성(SE 값, 클러스터 수 등)을 보고 가중치 결정

방법:
1. Rule-based adaptive: SE 값과 클러스터 수 기반 규칙
2. Sigmoid adaptive: SE를 sigmoid에 통과시켜 부드럽게 전환
3. Cluster-aware: 클러스터가 1개면 Energy, 아니면 SE

장점:
- 학습 불필요 (cross-dataset 일반화)
- 해석 가능한 규칙
- 각 샘플에 맞는 최적 메트릭 선택
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable


class AdaptiveMethod(Enum):
    """Adaptive 방법 종류"""
    RULE_BASED = "rule_based"       # 규칙 기반
    SIGMOID = "sigmoid"              # Sigmoid 함수
    CLUSTER_AWARE = "cluster_aware"  # 클러스터 수 기반
    HYBRID = "hybrid"                # 복합


@dataclass
class AdaptiveCascadeResult:
    """Adaptive cascade 결과"""
    
    # 최종 점수
    score: float
    
    # 개별 지표
    semantic_entropy: float
    semantic_energy: float
    
    # 적응형 가중치
    w_energy: float
    w_se: float
    
    # 메타 정보
    num_clusters: int
    method: AdaptiveMethod
    reason: str  # 왜 이 가중치를 선택했는지


class AdaptiveCascade:
    """
    Adaptive SE-Energy Cascade
    
    입력 특성을 보고 SE와 Energy의 가중치를 동적으로 결정
    
    Usage:
        cascade = AdaptiveCascade(method=AdaptiveMethod.HYBRID)
        
        result = cascade.compute(
            semantic_entropy=0.05,
            semantic_energy=2.3,
            num_clusters=1,
        )
        
        print(f"Score: {result.score}")
        print(f"Weights: SE={result.w_se:.2f}, Energy={result.w_energy:.2f}")
        print(f"Reason: {result.reason}")
    """
    
    def __init__(
        self,
        method: AdaptiveMethod = AdaptiveMethod.HYBRID,
        # Rule-based parameters
        se_threshold_low: float = 0.1,   # SE가 이 값 미만이면 Energy 선호
        se_threshold_high: float = 0.5,  # SE가 이 값 이상이면 SE 선호
        # Sigmoid parameters
        sigmoid_center: float = 0.3,     # Sigmoid 중심점
        sigmoid_steepness: float = 10.0,  # Sigmoid 기울기
        # Weight bounds
        w_energy_max: float = 0.9,
        w_energy_min: float = 0.1,
    ):
        self.method = method
        self.se_threshold_low = se_threshold_low
        self.se_threshold_high = se_threshold_high
        self.sigmoid_center = sigmoid_center
        self.sigmoid_steepness = sigmoid_steepness
        self.w_energy_max = w_energy_max
        self.w_energy_min = w_energy_min
    
    def compute(
        self,
        semantic_entropy: float,
        semantic_energy: float,
        num_clusters: int = 1,
        normalize: bool = True,
    ) -> AdaptiveCascadeResult:
        """
        Adaptive cascade 계산
        
        Args:
            semantic_entropy: SE 값 (높을수록 다양한 응답)
            semantic_energy: Energy 값 (높을수록 낮은 확신)
            num_clusters: NLI 클러스터 수
            normalize: 점수 정규화 여부
        
        Returns:
            AdaptiveCascadeResult
        """
        
        # 가중치 계산
        if self.method == AdaptiveMethod.RULE_BASED:
            w_energy, reason = self._rule_based(semantic_entropy, num_clusters)
        elif self.method == AdaptiveMethod.SIGMOID:
            w_energy, reason = self._sigmoid(semantic_entropy)
        elif self.method == AdaptiveMethod.CLUSTER_AWARE:
            w_energy, reason = self._cluster_aware(num_clusters)
        elif self.method == AdaptiveMethod.HYBRID:
            w_energy, reason = self._hybrid(semantic_entropy, num_clusters)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        w_se = 1.0 - w_energy
        
        # 정규화 (옵션)
        if normalize:
            # SE와 Energy를 0-1 범위로 정규화
            se_norm = min(semantic_entropy / 2.0, 1.0)  # SE는 보통 0-2 범위
            energy_norm = min(max(semantic_energy, 0) / 5.0, 1.0)  # Energy는 보통 0-5 범위
        else:
            se_norm = semantic_entropy
            energy_norm = semantic_energy
        
        # 최종 점수 계산
        score = w_energy * energy_norm + w_se * se_norm
        
        return AdaptiveCascadeResult(
            score=score,
            semantic_entropy=semantic_entropy,
            semantic_energy=semantic_energy,
            w_energy=w_energy,
            w_se=w_se,
            num_clusters=num_clusters,
            method=self.method,
            reason=reason,
        )
    
    def _rule_based(self, se: float, num_clusters: int) -> tuple[float, str]:
        """
        규칙 기반 가중치 결정
        
        Rules:
        1. SE < 0.1 (Zero-SE) → Energy 선호 (w_energy = 0.9)
        2. SE > 0.5 (High-SE) → SE 선호 (w_energy = 0.1)
        3. 중간 → 선형 보간
        """
        if se < self.se_threshold_low:
            return self.w_energy_max, f"Zero-SE (SE={se:.3f}<{self.se_threshold_low})"
        elif se > self.se_threshold_high:
            return self.w_energy_min, f"High-SE (SE={se:.3f}>{self.se_threshold_high})"
        else:
            # 선형 보간
            t = (se - self.se_threshold_low) / (self.se_threshold_high - self.se_threshold_low)
            w = self.w_energy_max - t * (self.w_energy_max - self.w_energy_min)
            return w, f"Medium-SE (SE={se:.3f}, interpolated)"
    
    def _sigmoid(self, se: float) -> tuple[float, str]:
        """
        Sigmoid 기반 부드러운 전환
        
        w_energy = 1 / (1 + exp(k * (SE - μ)))
        
        - SE가 낮으면 → w_energy ≈ 1 (Energy 선호)
        - SE가 높으면 → w_energy ≈ 0 (SE 선호)
        """
        # Sigmoid
        w = 1.0 / (1.0 + math.exp(self.sigmoid_steepness * (se - self.sigmoid_center)))
        
        # Bound to [w_min, w_max]
        w = self.w_energy_min + w * (self.w_energy_max - self.w_energy_min)
        
        return w, f"Sigmoid (SE={se:.3f}, center={self.sigmoid_center})"
    
    def _cluster_aware(self, num_clusters: int) -> tuple[float, str]:
        """
        클러스터 수 기반 결정
        
        - 클러스터 1개 (Zero-SE 정의) → Energy 사용
        - 클러스터 2개 이상 → SE 사용
        """
        if num_clusters == 1:
            return self.w_energy_max, f"Single cluster (n={num_clusters})"
        elif num_clusters == 2:
            return 0.5, f"Two clusters (n={num_clusters})"
        else:
            return self.w_energy_min, f"Multiple clusters (n={num_clusters})"
    
    def _hybrid(self, se: float, num_clusters: int) -> tuple[float, str]:
        """
        복합 방법: 클러스터 수와 SE 값 모두 고려
        
        1. 클러스터 1개 → Energy 강하게 선호
        2. 클러스터 2개 이상이지만 SE 낮음 → Energy 약하게 선호
        3. 클러스터 2개 이상이고 SE 높음 → SE 선호
        """
        # 클러스터가 1개면 무조건 Energy
        if num_clusters == 1:
            return self.w_energy_max, f"Single cluster (forced Energy)"
        
        # 클러스터 2개 이상이면 SE 값으로 판단
        if se < self.se_threshold_low:
            return 0.7, f"Low SE despite multiple clusters (SE={se:.3f})"
        elif se > self.se_threshold_high:
            return self.w_energy_min, f"High SE with multiple clusters (SE={se:.3f})"
        else:
            # 선형 보간 (클러스터가 있으므로 Energy 가중치 낮춤)
            t = (se - self.se_threshold_low) / (self.se_threshold_high - self.se_threshold_low)
            w = 0.7 - t * (0.7 - self.w_energy_min)
            return w, f"Medium SE with clusters (SE={se:.3f}, n={num_clusters})"


def compute_adaptive_score(
    semantic_entropy: float,
    semantic_energy: float,
    num_clusters: int = 1,
    method: str = "hybrid",
) -> float:
    """
    편의 함수: Adaptive cascade 점수 계산
    
    Args:
        semantic_entropy: SE 값
        semantic_energy: Energy 값
        num_clusters: 클러스터 수
        method: "rule_based", "sigmoid", "cluster_aware", "hybrid"
    
    Returns:
        Adaptive cascade 점수 (높을수록 환각 가능성)
    """
    method_enum = AdaptiveMethod(method)
    cascade = AdaptiveCascade(method=method_enum)
    result = cascade.compute(semantic_entropy, semantic_energy, num_clusters)
    return result.score
