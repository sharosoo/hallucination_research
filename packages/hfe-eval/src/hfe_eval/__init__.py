"""HFE-Eval: 환각 탐지 평가 도구"""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "compute_metrics":
        from hfe_eval.metrics import compute_metrics

        return compute_metrics
    elif name == "EvaluationMetrics":
        from hfe_eval.metrics import EvaluationMetrics

        return EvaluationMetrics
    elif name == "compute_exact_match":
        from hfe_eval.metrics import compute_exact_match

        return compute_exact_match
    elif name == "compute_f1_token":
        from hfe_eval.metrics import compute_f1_token

        return compute_f1_token
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_metrics",
    "EvaluationMetrics",
    "compute_exact_match",
    "compute_f1_token",
]
