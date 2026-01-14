"""
AHSFE: Adaptive Hybrid Semantic Free Energy

기존 고정 가중치 HSFE의 한계를 극복하는 적응형 환각 탐지 프레임워크

핵심 아이디어:
- HSFE = α × Energy + β × Entropy (α, β 고정)
- AHSFE = w₁(x) × Energy + w₂(x) × Entropy (w₁, w₂ 학습)

w₁(x), w₂(x)는 입력 특성에 따라 Weight Predictor가 예측:
- 다양한 응답 → w_entropy ↑
- 일관된 응답 (제로-엔트로피) → w_energy ↑
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nli_clusterer import NLIClusterer, Response, Cluster
from .semantic_entropy import SemanticEntropyCalculator
from .semantic_energy import SemanticEnergyCalculator
from .feature_extractor import FeatureExtractor, Features
from .weight_predictor import WeightPredictor, WeightPrediction


@dataclass
class AHSFEResult:
    """AHSFE 계산 결과"""

    # 최종 점수
    ahsfe: float  # 높을수록 환각 가능성 높음

    # 개별 지표
    semantic_entropy: float
    semantic_energy: float

    # 적응형 가중치
    w_energy: float
    w_entropy: float

    # 부가 정보
    num_clusters: int
    clusters: list[Cluster]
    features: Features


class AHSFE:
    """
    Adaptive Hybrid Semantic Free Energy

    입력별로 Energy와 Entropy의 가중치를 동적으로 조정하는 환각 탐지기

    Usage:
        # 기본 사용 (학습 전, 고정 가중치)
        ahsfe = AHSFE()

        result = ahsfe.compute(
            question="What is the capital of France?",
            responses=responses,  # Response 리스트 (logits 포함)
        )

        print(f"AHSFE: {result.ahsfe}")
        print(f"w_energy: {result.w_energy}, w_entropy: {result.w_entropy}")

        # 학습된 Weight Predictor 사용
        ahsfe = AHSFE(weight_predictor_path="weights.pt")
    """

    def __init__(
        self,
        nli_model: str = "microsoft/deberta-large-mnli",
        weight_predictor_path: Optional[str] = None,
        default_weights: tuple[float, float] = (0.5, 0.5),
        device: Optional[str] = None,
        normalize_scores: bool = True,
    ):
        """
        Args:
            nli_model: NLI 클러스터링용 모델
            weight_predictor_path: 학습된 Weight Predictor 경로 (None이면 기본 가중치)
            default_weights: (w_energy, w_entropy) 기본 가중치
            device: 'cuda' or 'cpu'
            normalize_scores: SE와 Energy를 정규화하여 결합할지 여부
        """
        self.device = device
        self.normalize_scores = normalize_scores

        # NLI 클러스터러 (SE와 Energy 공유)
        self.clusterer = NLIClusterer(model_name=nli_model, device=device)

        # 특성 추출기
        self.feature_extractor = FeatureExtractor()

        # Weight Predictor
        self.weight_predictor = WeightPredictor(
            model_path=weight_predictor_path,
            default_weights=default_weights,
            device=device or "cpu",
        )

    def compute(
        self,
        question: str,
        responses: list[Response],
        knowledge: Optional[str] = None,
    ) -> AHSFEResult:
        """
        AHSFE 계산

        Args:
            question: 질문 텍스트
            responses: Response 객체 리스트 (logits 필수)
            knowledge: knowledge 컨텍스트 (있으면)

        Returns:
            AHSFEResult
        """
        if not responses:
            return AHSFEResult(
                ahsfe=0.0,
                semantic_entropy=0.0,
                semantic_energy=0.0,
                w_energy=0.5,
                w_entropy=0.5,
                num_clusters=0,
                clusters=[],
                features=Features(),
            )

        # 1. NLI 클러스터링 (한 번만)
        clusters = self.clusterer.cluster(responses)

        # 2. Semantic Entropy 계산
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))

        # 3. Semantic Energy 계산
        energy = SemanticEnergyCalculator.compute_energy_only(responses)

        # 4. 특성 추출
        features = self.feature_extractor.extract(
            question=question,
            responses=responses,
            clusters=clusters,
            semantic_entropy=se,
            semantic_energy=energy,
            knowledge=knowledge,
        )

        # 5. 적응형 가중치 예측
        weight_pred = self.weight_predictor.predict(features.to_vector())
        w_energy = weight_pred.w_energy
        w_entropy = weight_pred.w_entropy

        # 6. AHSFE 계산
        if self.normalize_scores:
            # Energy와 SE를 [0, 1] 범위로 정규화
            # Energy: 일반적으로 -60 ~ -30 범위 → 0 ~ 1
            # SE: 일반적으로 0 ~ 2 범위 → 0 ~ 1
            norm_energy = (energy + 60) / 30  # [-60, -30] → [0, 1]
            norm_energy = max(0, min(1, norm_energy))

            norm_se = se / 2  # [0, 2] → [0, 1]
            norm_se = max(0, min(1, norm_se))

            ahsfe = w_energy * norm_energy + w_entropy * norm_se
        else:
            ahsfe = w_energy * energy + w_entropy * se

        return AHSFEResult(
            ahsfe=float(ahsfe),
            semantic_entropy=se,
            semantic_energy=energy,
            w_energy=w_energy,
            w_entropy=w_entropy,
            num_clusters=len(clusters),
            clusters=clusters,
            features=features,
        )

    def compute_with_clusters(
        self,
        question: str,
        responses: list[Response],
        clusters: list[Cluster],
        knowledge: Optional[str] = None,
    ) -> AHSFEResult:
        """
        이미 클러스터링된 결과로 AHSFE 계산 (클러스터링 중복 방지)

        외부에서 클러스터링을 수행한 경우 사용
        """
        if not responses:
            return AHSFEResult(
                ahsfe=0.0,
                semantic_entropy=0.0,
                semantic_energy=0.0,
                w_energy=0.5,
                w_entropy=0.5,
                num_clusters=0,
                clusters=[],
                features=Features(),
            )

        # SE 계산
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))

        # Energy 계산
        energy = SemanticEnergyCalculator.compute_energy_only(responses)

        # 특성 추출
        features = self.feature_extractor.extract(
            question=question,
            responses=responses,
            clusters=clusters,
            semantic_entropy=se,
            semantic_energy=energy,
            knowledge=knowledge,
        )

        # 가중치 예측
        weight_pred = self.weight_predictor.predict(features.to_vector())
        w_energy = weight_pred.w_energy
        w_entropy = weight_pred.w_entropy

        # AHSFE 계산
        if self.normalize_scores:
            norm_energy = (energy + 60) / 30
            norm_energy = max(0, min(1, norm_energy))
            norm_se = se / 2
            norm_se = max(0, min(1, norm_se))
            ahsfe = w_energy * norm_energy + w_entropy * norm_se
        else:
            ahsfe = w_energy * energy + w_entropy * se

        return AHSFEResult(
            ahsfe=float(ahsfe),
            semantic_entropy=se,
            semantic_energy=energy,
            w_energy=w_energy,
            w_entropy=w_entropy,
            num_clusters=len(clusters),
            clusters=clusters,
            features=features,
        )

    def is_hallucination(
        self,
        question: str,
        responses: list[Response],
        threshold: float = 0.5,
        knowledge: Optional[str] = None,
    ) -> bool:
        """
        환각 여부 판단

        Args:
            question: 질문
            responses: 응답 리스트
            threshold: AHSFE 임계값 (이상이면 환각)
            knowledge: knowledge 컨텍스트

        Returns:
            True if hallucination, False otherwise
        """
        result = self.compute(question, responses, knowledge)
        return result.ahsfe >= threshold
