"""
HFE-Core: 핵심 환각 탐지 알고리즘

주요 구성요소:
- NLIClusterer: NLI 기반 의미적 클러스터링
- SemanticEntropyCalculator: Semantic Entropy 계산
- SemanticEnergyCalculator: Semantic Energy 계산
- AHSFE: Adaptive Hybrid Semantic Free Energy
- FeatureExtractor: 입력 특성 추출기
- WeightPredictor: 적응형 가중치 예측기
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "NLIClusterer":
        from hfe_core.nli_clusterer import NLIClusterer

        return NLIClusterer
    elif name == "Response":
        from hfe_core.nli_clusterer import Response

        return Response
    elif name == "Cluster":
        from hfe_core.nli_clusterer import Cluster

        return Cluster
    elif name == "SemanticEntropyCalculator":
        from hfe_core.semantic_entropy import SemanticEntropyCalculator

        return SemanticEntropyCalculator
    elif name == "SemanticEntropyResult":
        from hfe_core.semantic_entropy import SemanticEntropyResult

        return SemanticEntropyResult
    elif name == "SemanticEnergyCalculator":
        from hfe_core.semantic_energy import SemanticEnergyCalculator

        return SemanticEnergyCalculator
    elif name == "SemanticEnergyResult":
        from hfe_core.semantic_energy import SemanticEnergyResult

        return SemanticEnergyResult
    elif name == "AHSFE":
        from hfe_core.ahsfe import AHSFE

        return AHSFE
    elif name == "AHSFEResult":
        from hfe_core.ahsfe import AHSFEResult

        return AHSFEResult
    elif name == "FeatureExtractor":
        from hfe_core.feature_extractor import FeatureExtractor

        return FeatureExtractor
    elif name == "Features":
        from hfe_core.feature_extractor import Features

        return Features
    elif name == "WeightPredictor":
        from hfe_core.weight_predictor import WeightPredictor

        return WeightPredictor
    elif name == "WeightPrediction":
        from hfe_core.weight_predictor import WeightPrediction

        return WeightPrediction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # NLI Clusterer
    "NLIClusterer",
    "Response",
    "Cluster",
    # Semantic Entropy
    "SemanticEntropyCalculator",
    "SemanticEntropyResult",
    # Semantic Energy
    "SemanticEnergyCalculator",
    "SemanticEnergyResult",
    # AHSFE
    "AHSFE",
    "AHSFEResult",
    # Feature Extractor
    "FeatureExtractor",
    "Features",
    # Weight Predictor
    "WeightPredictor",
    "WeightPrediction",
]
