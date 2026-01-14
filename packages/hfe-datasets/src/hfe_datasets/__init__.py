"""HFE-Datasets: 환각 탐지 벤치마크용 데이터셋 로더"""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TruthfulQALoader":
        from hfe_datasets.truthfulqa import TruthfulQALoader

        return TruthfulQALoader
    elif name == "HaluEvalLoader":
        from hfe_datasets.halueval import HaluEvalLoader

        return HaluEvalLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TruthfulQALoader",
    "HaluEvalLoader",
]
