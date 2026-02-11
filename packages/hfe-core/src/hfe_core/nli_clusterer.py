"""
NLI 기반 의미적 클러스터링 (Semantic Clustering via NLI)

Farquhar et al. (2024) Nature 논문의 핵심 구성요소.
양방향 entailment를 사용하여 의미적으로 동일한 응답을 클러스터링.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Response:
    """샘플링된 응답"""

    text: str
    probability: float = 1.0
    log_probability: Optional[float] = None  # log p(x|q) - continuous SE용
    logits: list[float] = field(
        default_factory=list
    )  # 생성된 각 토큰의 raw logit (softmax 전)


@dataclass
class Cluster:
    """의미적 클러스터"""

    representative: str  # 대표 텍스트
    members: list[Response] = field(default_factory=list)
    probability: float = 0.0  # 클러스터 확률 (멤버 확률 합)
    energy: float = 0.0  # 평균 에너지 (Semantic Energy용)

    def __post_init__(self):
        if not self.members:
            self.members = []

    def add(self, response: Response):
        self.members.append(response)

    @classmethod
    def from_response(cls, response: Response) -> "Cluster":
        """응답으로부터 새 클러스터 생성"""
        cluster = cls(representative=response.text)
        cluster.add(response)
        return cluster


class NLIClusterer:
    """
    NLI 기반 의미적 클러스터링

    양방향 entailment를 사용하여 의미적으로 동일한 응답을 그룹화.
    DeBERTa-v3-large-mnli 모델 사용 권장.
    """

    # MNLI 라벨: 0=contradiction, 1=neutral, 2=entailment
    ENTAILMENT_IDX = 2

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",  # gated 아닌 모델
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            model_name: NLI 모델 이름
            device: 'cuda' or 'cpu'
            threshold: entailment 판정 임계값
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        premise → hypothesis entailment 점수 계산

        Returns:
            entailment 확률 (0~1)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        return probs[0, self.ENTAILMENT_IDX].item()

    def get_nli_prediction(self, premise: str, hypothesis: str) -> str:
        """
        NLI 예측 (argmax)

        Returns:
            'contradiction', 'neutral', or 'entailment'
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)
        pred_idx = outputs.logits.argmax(dim=-1).item()

        # MNLI 라벨: 0=contradiction, 1=neutral, 2=entailment
        labels = ["contradiction", "neutral", "entailment"]
        return labels[pred_idx]

    def bidirectional_entailment(self, text1: str, text2: str) -> bool:
        """
        양방향 entailment 확인 (Farquhar et al. 2024 방식)

        논문에서는 확률 threshold가 아닌 argmax 예측 사용!
        두 텍스트가 서로 entail하면 의미적으로 동일하다고 판단.

        Returns:
            True if text1 ↔ text2 (bidirectional entailment)
        """
        # text1 → text2: argmax가 entailment인지
        pred1 = self.get_nli_prediction(text1, text2)
        if pred1 != "entailment":
            return False

        # text2 → text1: argmax가 entailment인지
        pred2 = self.get_nli_prediction(text2, text1)
        return pred2 == "entailment"

    def cluster(self, responses: list[Response]) -> list[Cluster]:
        """
        응답들을 의미적으로 클러스터링 (Farquhar et al. 2024 방식)

        Greedy 알고리즘:
        - 각 응답에 대해, 기존 클러스터의 **어떤 멤버**와든
          bidirectional entailment가 성립하면 해당 클러스터에 추가
        - 아니면 새 클러스터 생성

        Args:
            responses: Response 객체 리스트

        Returns:
            Cluster 리스트
        """
        if not responses:
            return []

        clusters: list[Cluster] = []

        for response in responses:
            found_cluster = False

            # 기존 클러스터들을 순회
            for cluster in clusters:
                # 클러스터 내 ANY member와 bidirectional entailment 확인
                for member in cluster.members:
                    if self.bidirectional_entailment(response.text, member.text):
                        cluster.add(response)
                        found_cluster = True
                        break
                if found_cluster:
                    break

            if not found_cluster:
                clusters.append(Cluster.from_response(response))

        return clusters

    def cluster_with_stats(self, responses: list[Response]) -> list[Cluster]:
        """
        클러스터링 + 확률/에너지 통계 계산

        Returns:
            통계가 계산된 Cluster 리스트
        """
        clusters = self.cluster(responses)

        for cluster in clusters:
            # 클러스터 확률 = 멤버 확률 합
            cluster.probability = sum(r.probability for r in cluster.members)

            # 클러스터 에너지 = 멤버 logit 평균의 평균
            if any(r.logits for r in cluster.members):
                energies = []
                for r in cluster.members:
                    if r.logits:
                        energies.append(sum(r.logits) / len(r.logits))
                if energies:
                    cluster.energy = sum(energies) / len(energies)

        return clusters
