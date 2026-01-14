"""
Weight Predictor for AHSFE

입력 특성으로부터 Semantic Entropy와 Semantic Energy의 가중치를 예측하는 신경망

핵심 아이디어:
- 기존 HSFE: HSFE = α × Energy + β × Entropy (α, β 고정)
- AHSFE: AHSFE = w₁(x) × Energy + w₂(x) × Entropy (w₁, w₂ 학습)

학습 방법:
- Margin Ranking Loss: 환각 샘플의 AHSFE > 정상 샘플의 AHSFE
- 또는 Binary Cross Entropy: AHSFE를 환각 확률로 해석
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class WeightPrediction:
    """가중치 예측 결과"""

    w_energy: float  # Energy 가중치
    w_entropy: float  # Entropy 가중치

    def __post_init__(self):
        # 정규화 확인
        total = self.w_energy + self.w_entropy
        if abs(total - 1.0) > 1e-6:
            self.w_energy = self.w_energy / total
            self.w_entropy = self.w_entropy / total


if TORCH_AVAILABLE:

    class WeightPredictorNet(nn.Module):
        """
        Weight Predictor 신경망

        입력: 특성 벡터 (11차원)
        출력: (w_energy, w_entropy) - softmax로 합이 1

        Architecture:
            Input(11) → Linear(32) → ReLU → Dropout →
            Linear(32) → ReLU → Linear(2) → Softmax
        """

        def __init__(
            self,
            input_dim: int = 11,
            hidden_dim: int = 32,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (batch_size, input_dim) 특성 벡터

            Returns:
                w_energy: (batch_size,) Energy 가중치
                w_entropy: (batch_size,) Entropy 가중치
            """
            logits = self.network(x)  # (batch_size, 2)
            weights = F.softmax(logits, dim=-1)  # 합이 1

            return weights[:, 0], weights[:, 1]  # w_energy, w_entropy

        def predict_single(self, features: np.ndarray) -> WeightPrediction:
            """단일 샘플 예측"""
            self.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                w_e, w_s = self.forward(x)
                return WeightPrediction(
                    w_energy=w_e.item(),
                    w_entropy=w_s.item(),
                )


class WeightPredictor:
    """
    Weight Predictor 래퍼 클래스

    학습된 모델이 없으면 기본 가중치 사용 (0.5, 0.5)

    Usage:
        # 학습 전 (기본 가중치)
        predictor = WeightPredictor()
        w = predictor.predict(features)  # w_energy=0.5, w_entropy=0.5

        # 학습 후
        predictor = WeightPredictor(model_path="weights.pt")
        w = predictor.predict(features)  # 학습된 가중치
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        default_weights: Tuple[float, float] = (0.5, 0.5),
        device: str = "cpu",
    ):
        """
        Args:
            model_path: 학습된 모델 경로 (None이면 기본 가중치 사용)
            default_weights: (w_energy, w_entropy) 기본 가중치
            device: 'cuda' or 'cpu'
        """
        self.default_weights = default_weights
        self.device = device
        self.model: Optional["WeightPredictorNet"] = None

        if model_path is not None and TORCH_AVAILABLE:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """학습된 모델 로드"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for loading model")

        self.model = WeightPredictorNet()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> WeightPrediction:
        """
        가중치 예측

        Args:
            features: (num_features,) 또는 (1, num_features) 특성 벡터

        Returns:
            WeightPrediction
        """
        if self.model is None:
            # 기본 가중치 반환
            return WeightPrediction(
                w_energy=self.default_weights[0],
                w_entropy=self.default_weights[1],
            )

        return self.model.predict_single(features)

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 가중치 예측

        Args:
            features: (batch_size, num_features) 특성 벡터

        Returns:
            w_energy: (batch_size,)
            w_entropy: (batch_size,)
        """
        if self.model is None:
            batch_size = features.shape[0]
            return (
                np.full(batch_size, self.default_weights[0]),
                np.full(batch_size, self.default_weights[1]),
            )

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            w_e, w_s = self.model(x)
            return w_e.cpu().numpy(), w_s.cpu().numpy()


if TORCH_AVAILABLE:

    class AHSFELoss(nn.Module):
        """
        AHSFE 학습용 손실 함수

        Margin Ranking Loss:
        - 환각 샘플의 AHSFE > 정상 샘플의 AHSFE + margin

        또는 Binary Cross Entropy:
        - AHSFE를 sigmoid로 환각 확률로 변환 후 BCE
        """

        def __init__(
            self,
            loss_type: str = "margin",  # "margin" or "bce"
            margin: float = 0.5,
        ):
            super().__init__()
            self.loss_type = loss_type
            self.margin = margin

        def forward(
            self,
            ahsfe_scores: torch.Tensor,  # (batch_size,)
            labels: torch.Tensor,  # (batch_size,) 0=normal, 1=hallucination
        ) -> torch.Tensor:
            """
            Args:
                ahsfe_scores: AHSFE 점수 (높을수록 환각)
                labels: 환각 레이블 (0 또는 1)

            Returns:
                loss 스칼라
            """
            if self.loss_type == "bce":
                # AHSFE를 확률로 변환
                probs = torch.sigmoid(ahsfe_scores)
                return F.binary_cross_entropy(probs, labels.float())

            elif self.loss_type == "margin":
                # Margin Ranking Loss
                hall_mask = labels == 1
                norm_mask = labels == 0

                if not hall_mask.any() or not norm_mask.any():
                    return torch.tensor(0.0, device=ahsfe_scores.device)

                hall_scores = ahsfe_scores[hall_mask]
                norm_scores = ahsfe_scores[norm_mask]

                # 모든 (환각, 정상) 쌍에 대해 margin loss
                # hall_score > norm_score + margin 이어야 함
                loss = 0.0
                count = 0

                for h in hall_scores:
                    for n in norm_scores:
                        # max(0, margin - (h - n))
                        loss += F.relu(self.margin - (h - n))
                        count += 1

                if count > 0:
                    loss = loss / count

                return loss

            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
