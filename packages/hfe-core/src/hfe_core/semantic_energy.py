"""
Semantic Energy Calculator for Hallucination Detection

Based on: Semantic Energy: Detecting LLM Hallucination Beyond Entropy
(Ma et al., 2025, arXiv:2508.14496)

논문 공식:
1. 토큰 에너지: Ẽ(x_t) = -z_θ(x_t)  (negative logit)
2. 응답 에너지: E(x^(i)) = (1/T_i) Σ E_t
3. 클러스터 에너지 (Boltzmann): E_Bolt(C) = Σ E(x^(i))
4. Semantic Energy (최종):
   U(x) = (1/nT_i) Σ_{x^(i) ∈ C_k} Σ_{t=1}^{T_i} -z_θ(x_t)

핵심 차이점 (vs Semantic Entropy):
- Semantic Entropy: softmax 후 확률 사용 → 정규화로 logit 크기 정보 손실
- Semantic Energy: raw logit 사용 → 모델의 내재적 확신도 보존

장점:
- 제로-엔트로피 문제 해결 (모델이 일관되게 틀려도 구분 가능)
- 낮은 에너지 = 높은 확신 = 더 신뢰할 수 있는 응답
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nli_clusterer import NLIClusterer, Response, Cluster


@dataclass
class SemanticEnergyResult:
    """Semantic Energy 계산 결과"""

    energy: float  # U(x) - 낮을수록 confident
    num_clusters: int  # K
    clusters: list[Cluster]  # 클러스터 리스트
    cluster_energies: list[float]  # E_Bolt(C_k) 리스트
    response_energies: list[float]  # E(x^(i)) 리스트


class SemanticEnergyCalculator:
    """
    Semantic Energy 계산기 (Ma et al., 2025)

    논문 Equation 14:
    U(x) = (1/nT_i) Σ Σ -z_θ(x_t)

    Usage:
        calculator = SemanticEnergyCalculator()

        # 응답 리스트 준비 (logits 필수!)
        responses = [
            Response(
                text="Paris",
                logits=[8.2, 7.5, 9.1]  # 각 토큰의 raw logit (softmax 전!)
            ),
            ...
        ]

        result = calculator.compute(responses)
        print(f"Semantic Energy: {result.energy}")  # 낮을수록 confident
    """

    def __init__(
        self,
        nli_model: str = "microsoft/deberta-large-mnli",
        device: Optional[str] = None,
    ):
        """
        Args:
            nli_model: NLI 모델 (논문: TIGER-Lab/general-verifier)
            device: 'cuda' or 'cpu'
        """
        self.clusterer = NLIClusterer(
            model_name=nli_model,
            device=device,
        )

    def compute(self, responses: list[Response]) -> SemanticEnergyResult:
        """
        Semantic Energy 계산 (논문 Eq. 14)

        U(x) = (1/nT_i) Σ Σ -z_θ(x_t)

        Args:
            responses: Response 객체 리스트
                - text: 응답 텍스트 (필수)
                - logits: 각 토큰의 raw logit (필수!) - softmax 전 값

        Returns:
            SemanticEnergyResult
        """
        if not responses:
            return SemanticEnergyResult(
                energy=0.0,
                num_clusters=0,
                clusters=[],
                cluster_energies=[],
                response_energies=[],
            )

        # logits 확인
        has_logits = any(r.logits for r in responses)
        if not has_logits:
            raise ValueError(
                "Semantic Energy requires raw logits for each response. "
                "Set Response.logits = [token_logit_1, token_logit_2, ...] "
                "where each logit is the raw value BEFORE softmax."
            )

        # 1. 응답별 에너지 계산: E(x^(i)) = (1/T_i) Σ_{t=1}^{T_i} -z_θ(x_t)
        response_energies = []
        for r in responses:
            if r.logits:
                # E(x^(i)) = mean(-z_θ(x_t)) = -mean(z_θ(x_t))
                energy = -np.mean(r.logits)
            else:
                energy = 0.0
            response_energies.append(energy)

        # 2. 클러스터링 (Semantic Entropy와 동일한 NLI 기반)
        clusters = self.clusterer.cluster(responses)

        # 3. 클러스터별 에너지 (Boltzmann): E_Bolt(C) = Σ E(x^(i))
        cluster_energies = []
        for cluster in clusters:
            # 클러스터 내 모든 응답의 에너지 합
            cluster_energy = 0.0
            for member in cluster.members:
                # 해당 멤버의 인덱스 찾기
                for idx, r in enumerate(responses):
                    if r is member:
                        cluster_energy += response_energies[idx]
                        break
            cluster.energy = cluster_energy
            cluster_energies.append(cluster_energy)

        # 4. 최종 Semantic Energy (논문 Eq. 14)
        # U = (1/nT_i) Σ Σ -z_θ(x_t)
        # = 모든 응답의 모든 토큰에 대한 평균 negative logit
        total_tokens = 0
        total_neg_logit_sum = 0.0

        for r in responses:
            if r.logits:
                total_tokens += len(r.logits)
                total_neg_logit_sum += sum(-z for z in r.logits)

        if total_tokens > 0:
            final_energy = total_neg_logit_sum / total_tokens
        else:
            final_energy = 0.0

        return SemanticEnergyResult(
            energy=float(final_energy),
            num_clusters=len(clusters),
            clusters=clusters,
            cluster_energies=cluster_energies,
            response_energies=response_energies,
        )

    @staticmethod
    def compute_energy_only(responses: list[Response]) -> float:
        """
        클러스터링 없이 Energy만 계산 (논문 Eq. 14)

        U = (1/nT) Σ Σ -z_θ(x_t)

        Args:
            responses: Response 객체 리스트 (logits 필수)

        Returns:
            energy 값 (낮을수록 confident)
        """
        if not responses:
            return 0.0

        total_neg_logit = 0.0
        total_tokens = 0

        for r in responses:
            if r.logits:
                total_neg_logit += sum(-z for z in r.logits)
                total_tokens += len(r.logits)

        if total_tokens == 0:
            return 0.0

        return float(total_neg_logit / total_tokens)

    def compute_per_cluster(self, responses: list[Response]) -> SemanticEnergyResult:
        """
        클러스터별 평균 에너지 계산 (논문의 변형)

        각 클러스터의 평균 에너지를 계산하고,
        가장 낮은 에너지(가장 confident)를 가진 클러스터 기준으로 판단.

        Args:
            responses: Response 객체 리스트

        Returns:
            SemanticEnergyResult (energy = 최소 클러스터 에너지)
        """
        if not responses:
            return SemanticEnergyResult(
                energy=0.0,
                num_clusters=0,
                clusters=[],
                cluster_energies=[],
                response_energies=[],
            )

        # 응답별 에너지
        response_energies = []
        for r in responses:
            if r.logits:
                energy = -np.mean(r.logits)
            else:
                energy = 0.0
            response_energies.append(energy)

        # 클러스터링
        clusters = self.clusterer.cluster(responses)

        # 클러스터별 평균 에너지
        cluster_energies = []
        for cluster in clusters:
            member_energies = []
            for member in cluster.members:
                for idx, r in enumerate(responses):
                    if r is member:
                        member_energies.append(response_energies[idx])
                        break

            if member_energies:
                avg_energy = np.mean(member_energies)
            else:
                avg_energy = 0.0

            cluster.energy = avg_energy
            cluster_energies.append(avg_energy)

        # 최소 에너지 (가장 confident한 클러스터)
        min_energy = min(cluster_energies) if cluster_energies else 0.0

        return SemanticEnergyResult(
            energy=float(min_energy),
            num_clusters=len(clusters),
            clusters=clusters,
            cluster_energies=cluster_energies,
            response_energies=response_energies,
        )
