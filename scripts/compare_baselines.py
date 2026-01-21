#!/usr/bin/env python3
"""
Baseline 비교 실험 스크립트.

비교 대상:
- SE only (w_energy=0.0)
- Energy only (w_energy=1.0)
- Fixed-0.1, Fixed-0.5, Fixed-0.9
- Adaptive (Sigmoid, Linear, Threshold)

Usage:
    python scripts/compare_baselines.py
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "packages/hfe-core/src"))

from hfe_core.adaptive_weights import (
    AdaptiveWeightCalculator,
    WeightType,
    BASELINE_CONFIGS,
    create_baseline_calculator,
)


@dataclass
class BaselineResult:
    name: str
    auroc: float | None
    auprc: float | None


def compute_combined_scores(
    samples: list[dict],
    calculator: AdaptiveWeightCalculator,
) -> list[float]:
    scores = []
    for s in samples:
        coverage = s.get("corpus_stats", {}).get("coverage", 0.5)
        se = s["semantic_entropy"]
        energy = s["semantic_energy"]

        score = calculator.compute_score(coverage, se, energy, normalize=True)
        scores.append(score)

    return scores


def evaluate_baseline(
    name: str,
    samples: list[dict],
    calculator: AdaptiveWeightCalculator,
) -> BaselineResult:
    labels = [s["is_hallucination"] for s in samples]
    scores = compute_combined_scores(samples, calculator)

    if len(set(labels)) < 2:
        return BaselineResult(name=name, auroc=None, auprc=None)

    try:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        return BaselineResult(name=name, auroc=auroc, auprc=auprc)
    except Exception:
        return BaselineResult(name=name, auroc=None, auprc=None)


def run_comparison(dataset_name: str, samples: list[dict]) -> list[BaselineResult]:
    results = []

    baselines = [
        "se_only",
        "energy_only",
        "fixed_0.1",
        "fixed_0.5",
        "fixed_0.9",
        "adaptive_sigmoid",
        "adaptive_linear",
        "adaptive_threshold",
    ]

    for name in baselines:
        calc = create_baseline_calculator(name)
        result = evaluate_baseline(name, samples, calc)
        results.append(result)

    return results


def print_comparison(dataset_name: str, results: list[BaselineResult]):
    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 50}")
    print(f"{'Method':<22} {'AUROC':>10} {'AUPRC':>10}")
    print("-" * 50)

    sorted_results = sorted(results, key=lambda r: r.auroc or 0, reverse=True)

    for r in sorted_results:
        auroc_str = f"{r.auroc:.4f}" if r.auroc else "N/A"
        auprc_str = f"{r.auprc:.4f}" if r.auprc else "N/A"
        print(f"{r.name:<22} {auroc_str:>10} {auprc_str:>10}")

    best = sorted_results[0] if sorted_results else None
    if best and best.auroc:
        print(f"\nBest: {best.name} (AUROC: {best.auroc:.4f})")


def main():
    base = Path(__file__).parent.parent / "experiment_notes"

    experiments = [
        ("TruthfulQA", base / "exp01_truthfulqa/results_with_corpus.json"),
        ("HaluEval", base / "exp02_halueval/results_with_corpus.json"),
    ]

    all_results = {}

    for name, path in experiments:
        if not path.exists():
            print(f"\nSkipping {name}: {path} not found")
            print(f"  Run 'python scripts/add_corpus_stats.py' first")
            continue

        with open(path) as f:
            data = json.load(f)

        results = run_comparison(name, data["samples"])
        all_results[name] = results
        print_comparison(name, results)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("\nExpected result if hypothesis holds:")
    print("  - Adaptive methods should outperform fixed baselines")
    print("  - Adaptive should adapt: SE-like on TruthfulQA, Energy-like on HaluEval")


if __name__ == "__main__":
    main()
