"""
Feature Extractor for AHSFE Weight Predictor

입력 특성 추출기: 질문과 응답들로부터 Weight Predictor의 입력 특성을 추출

특성 목록:
1. 응답 특성
   - num_responses: 응답 수
   - avg_response_length: 평균 응답 길이 (토큰 수)
   - response_length_std: 응답 길이 표준편차

2. 클러스터 특성
   - num_clusters: 클러스터 수
   - largest_cluster_ratio: 최대 클러스터 비율
   - cluster_entropy: 클러스터 분포 엔트로피

3. Semantic 지표 (raw)
   - raw_entropy: Semantic Entropy 값
   - raw_energy: Semantic Energy 값
   - entropy_energy_ratio: SE / |Energy|

4. 질문 특성 (선택적)
   - question_length: 질문 길이
   - has_knowledge: knowledge 컨텍스트 유무
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .nli_clusterer import Response, Cluster


@dataclass
class Features:
    """추출된 특성"""

    # 응답 특성
    num_responses: int = 0
    avg_response_length: float = 0.0
    response_length_std: float = 0.0

    # 클러스터 특성
    num_clusters: int = 0
    largest_cluster_ratio: float = 0.0
    cluster_entropy: float = 0.0

    # Semantic 지표
    raw_entropy: float = 0.0
    raw_energy: float = 0.0
    entropy_energy_ratio: float = 0.0

    # 질문 특성
    question_length: int = 0
    has_knowledge: bool = False

    def to_vector(self) -> np.ndarray:
        """신경망 입력용 벡터로 변환"""
        return np.array(
            [
                self.num_responses,
                self.avg_response_length,
                self.response_length_std,
                self.num_clusters,
                self.largest_cluster_ratio,
                self.cluster_entropy,
                self.raw_entropy,
                self.raw_energy,
                self.entropy_energy_ratio,
                self.question_length,
                float(self.has_knowledge),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> list[str]:
        """특성 이름 목록"""
        return [
            "num_responses",
            "avg_response_length",
            "response_length_std",
            "num_clusters",
            "largest_cluster_ratio",
            "cluster_entropy",
            "raw_entropy",
            "raw_energy",
            "entropy_energy_ratio",
            "question_length",
            "has_knowledge",
        ]

    @staticmethod
    def num_features() -> int:
        """특성 수"""
        return 11


class FeatureExtractor:
    """
    AHSFE Weight Predictor를 위한 특성 추출기

    Usage:
        extractor = FeatureExtractor()

        features = extractor.extract(
            question="What is the capital of France?",
            responses=responses,
            clusters=clusters,
            semantic_entropy=0.5,
            semantic_energy=-45.0,
        )

        # 신경망 입력용
        feature_vector = features.to_vector()  # shape: (11,)
    """

    def extract(
        self,
        question: str,
        responses: list[Response],
        clusters: list[Cluster],
        semantic_entropy: float,
        semantic_energy: float,
        knowledge: Optional[str] = None,
    ) -> Features:
        """
        특성 추출

        Args:
            question: 질문 텍스트
            responses: Response 객체 리스트
            clusters: Cluster 객체 리스트 (NLI 클러스터링 결과)
            semantic_entropy: 계산된 SE 값
            semantic_energy: 계산된 Energy 값
            knowledge: knowledge 컨텍스트 (있으면)

        Returns:
            Features 객체
        """
        features = Features()

        # 1. 응답 특성
        features.num_responses = len(responses)

        if responses:
            lengths = [len(r.text.split()) for r in responses]
            features.avg_response_length = float(np.mean(lengths))
            features.response_length_std = float(np.std(lengths))

        # 2. 클러스터 특성
        features.num_clusters = len(clusters)

        if clusters and responses:
            cluster_sizes = [len(c.members) for c in clusters]
            features.largest_cluster_ratio = max(cluster_sizes) / len(responses)

            # 클러스터 분포 엔트로피
            probs = [s / len(responses) for s in cluster_sizes]
            features.cluster_entropy = -sum(p * np.log(p) for p in probs if p > 0)

        # 3. Semantic 지표
        features.raw_entropy = semantic_entropy
        features.raw_energy = semantic_energy

        # entropy / |energy| ratio (energy가 음수이므로 절대값)
        if abs(semantic_energy) > 1e-8:
            features.entropy_energy_ratio = semantic_entropy / abs(semantic_energy)

        # 4. 질문 특성
        features.question_length = len(question.split())
        features.has_knowledge = knowledge is not None and len(knowledge) > 0

        return features

    def extract_batch(
        self,
        questions: list[str],
        responses_list: list[list[Response]],
        clusters_list: list[list[Cluster]],
        entropies: list[float],
        energies: list[float],
        knowledges: Optional[list[Optional[str]]] = None,
    ) -> np.ndarray:
        """
        배치 특성 추출

        Returns:
            shape: (batch_size, num_features)
        """
        if knowledges is None:
            knowledges = [None] * len(questions)

        features_list = []
        for q, r, c, se, e, k in zip(
            questions, responses_list, clusters_list, entropies, energies, knowledges
        ):
            f = self.extract(q, r, c, se, e, k)
            features_list.append(f.to_vector())

        return np.stack(features_list, axis=0)
