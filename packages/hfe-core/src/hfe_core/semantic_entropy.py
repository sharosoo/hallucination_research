"""
Semantic Entropy Calculator for Hallucination Detection

Based on: Detecting hallucinations in large language models using
semantic entropy (Farquhar et al., 2024, Nature)

논문 공식:
1. 응답 확률: p(x^(i)|q) = Π p(x_t | x_<t, q)
2. 정규화: p̄(x^(i)) = p(x^(i)|q) / Σ p(x^(j)|q)
3. 클러스터 확률: p(C_k) = Σ_{x^(i) ∈ C_k} p̄(x^(i))
4. Semantic Entropy: H_SE = -Σ p(C_k) log p(C_k)

Discrete 버전 (black-box 호환):
- p(C_k) = |C_k| / N  (클러스터 내 샘플 수 / 전체 샘플 수)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nli_clusterer import NLIClusterer, Response, Cluster


@dataclass
class SemanticEntropyResult:
    """Semantic Entropy 계산 결과"""

    entropy: float  # H_SE
    num_clusters: int  # K
    clusters: list[Cluster]  # 클러스터 리스트
    cluster_probabilities: list[float]  # p(C_k) 리스트


class SemanticEntropyCalculator:
    """
    Semantic Entropy 계산기 (Farquhar et al., 2024 Nature)

    Usage:
        calculator = SemanticEntropyCalculator()

        # 응답 리스트 준비
        responses = [
            Response(text="Paris", log_probability=-1.2),
            Response(text="Paris is the capital", log_probability=-2.5),
            Response(text="Berlin", log_probability=-3.0),
            ...
        ]

        result = calculator.compute(responses)
        print(f"Semantic Entropy: {result.entropy}")
    """

    def __init__(
        self,
        nli_model: str = "microsoft/deberta-large-mnli",
        device: Optional[str] = None,
    ):
        """
        Args:
            nli_model: NLI 모델 (논문: DeBERTa-Large)
            device: 'cuda' or 'cpu'
        """
        self.clusterer = NLIClusterer(
            model_name=nli_model,
            device=device,
        )

    def compute(
        self,
        responses: list[Response],
        use_discrete: bool = True,
    ) -> SemanticEntropyResult:
        """
        Semantic Entropy 계산

        Args:
            responses: Response 객체 리스트
                - text: 응답 텍스트 (필수)
                - log_probability: log p(x|q) (continuous 버전에 필요)
            use_discrete: True면 discrete 버전 (클러스터 내 샘플 비율)
                         False면 continuous 버전 (확률 기반)

        Returns:
            SemanticEntropyResult
        """
        if not responses:
            return SemanticEntropyResult(
                entropy=0.0,
                num_clusters=0,
                clusters=[],
                cluster_probabilities=[],
            )

        # 1. 클러스터링 (bidirectional entailment 기반)
        clusters = self.clusterer.cluster(responses)
        n_total = len(responses)

        # 2. 클러스터 확률 계산
        cluster_probs = []

        if use_discrete:
            # Discrete SE: p(C_k) = |C_k| / N
            for cluster in clusters:
                prob = len(cluster.members) / n_total
                cluster.probability = prob
                cluster_probs.append(prob)
        else:
            # Continuous SE: p(C_k) = Σ p̄(x^(i))
            # 먼저 정규화된 확률 계산
            log_probs = []
            for r in responses:
                # log_probability 속성이 있으면 사용, 없으면 probability에서 계산
                if hasattr(r, "log_probability") and r.log_probability is not None:
                    log_probs.append(r.log_probability)
                elif r.probability > 0:
                    log_probs.append(np.log(r.probability))
                else:
                    log_probs.append(-100.0)  # 매우 낮은 확률

            # Softmax로 정규화: p̄(x^(i)) = exp(log_p) / Σ exp(log_p)
            log_probs = np.array(log_probs)
            max_log_prob = np.max(log_probs)
            normalized_probs = np.exp(log_probs - max_log_prob)
            normalized_probs = normalized_probs / np.sum(normalized_probs)

            # 클러스터 확률 = 멤버 확률의 합
            for cluster in clusters:
                prob = 0.0
                for member in cluster.members:
                    # 해당 멤버의 인덱스 찾기
                    for idx, r in enumerate(responses):
                        if r is member:
                            prob += normalized_probs[idx]
                            break
                cluster.probability = prob
                cluster_probs.append(prob)

        # 3. Semantic Entropy 계산: H_SE = -Σ p(C_k) log p(C_k)
        entropy = 0.0
        for prob in cluster_probs:
            if prob > 0:
                entropy -= prob * np.log(prob)

        return SemanticEntropyResult(
            entropy=float(entropy),
            num_clusters=len(clusters),
            clusters=clusters,
            cluster_probabilities=cluster_probs,
        )

    def compute_from_texts(
        self,
        texts: list[str],
        log_probabilities: Optional[list[float]] = None,
        use_discrete: bool = True,
    ) -> SemanticEntropyResult:
        """
        텍스트 리스트로부터 Semantic Entropy 계산

        Args:
            texts: 응답 텍스트 리스트
            log_probabilities: 각 응답의 log probability (continuous 버전용)
            use_discrete: discrete 버전 사용 여부

        Returns:
            SemanticEntropyResult
        """
        responses = []
        for i, text in enumerate(texts):
            r = Response(text=text)
            if log_probabilities is not None:
                r.log_probability = log_probabilities[i]
            responses.append(r)

        return self.compute(responses, use_discrete=use_discrete)

    @staticmethod
    def compute_from_clusters(
        clusters: list[Cluster],
        n_total: int,
    ) -> float:
        """
        이미 클러스터링된 결과로부터 Semantic Entropy 계산 (discrete)

        클러스터링을 별도로 수행한 경우 사용.

        Args:
            clusters: Cluster 리스트
            n_total: 전체 응답 수

        Returns:
            entropy 값
        """
        if not clusters or n_total == 0:
            return 0.0

        entropy = 0.0
        for cluster in clusters:
            p = len(cluster.members) / n_total
            if p > 0:
                entropy -= p * np.log(p)

        return float(entropy)
