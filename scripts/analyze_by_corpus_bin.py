#!/usr/bin/env python3
"""
Corpus coverage bin별 AUROC 분석 스크립트.

Usage:
    python scripts/analyze_by_corpus_bin.py

Requirements:
    먼저 add_corpus_stats.py를 실행하여 corpus stats가 추가된 데이터 필요
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class BinAnalysis:
    bin_name: str
    coverage_range: tuple[float, float]
    n_samples: int
    n_hallucination: int
    se_auroc: float | None
    energy_auroc: float | None
    se_auprc: float | None
    energy_auprc: float | None


def compute_metrics(
    labels: list[int], scores: list[float]
) -> tuple[float | None, float | None]:
    if len(set(labels)) < 2:
        return None, None
    try:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        return auroc, auprc
    except Exception:
        return None, None


def analyze_by_bins(
    samples: list[dict], bins: list[tuple[str, float, float]]
) -> list[BinAnalysis]:
    results = []

    for bin_name, low, high in bins:
        bin_samples = [
            s
            for s in samples
            if "corpus_stats" in s and low <= s["corpus_stats"]["coverage"] < high
        ]

        if not bin_samples:
            results.append(
                BinAnalysis(
                    bin_name=bin_name,
                    coverage_range=(low, high),
                    n_samples=0,
                    n_hallucination=0,
                    se_auroc=None,
                    energy_auroc=None,
                    se_auprc=None,
                    energy_auprc=None,
                )
            )
            continue

        labels = [s["is_hallucination"] for s in bin_samples]
        se_scores = [s["semantic_entropy"] for s in bin_samples]
        # Energy: 높은 값 = uncertain = hallucination (flip 불필요)
        energy_scores = [s["semantic_energy"] for s in bin_samples]

        se_auroc, se_auprc = compute_metrics(labels, se_scores)
        energy_auroc, energy_auprc = compute_metrics(labels, energy_scores)

        results.append(
            BinAnalysis(
                bin_name=bin_name,
                coverage_range=(low, high),
                n_samples=len(bin_samples),
                n_hallucination=sum(labels),
                se_auroc=se_auroc,
                energy_auroc=energy_auroc,
                se_auprc=se_auprc,
                energy_auprc=energy_auprc,
            )
        )

    return results


def print_results(dataset_name: str, results: list[BinAnalysis]):
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")
    print(
        f"{'Bin':<12} {'Range':<12} {'N':>6} {'Hall':>6} {'SE AUROC':>10} {'E AUROC':>10} {'Winner':<8}"
    )
    print("-" * 70)

    for r in results:
        range_str = f"[{r.coverage_range[0]:.1f}, {r.coverage_range[1]:.1f})"
        se_str = f"{r.se_auroc:.3f}" if r.se_auroc else "N/A"
        e_str = f"{r.energy_auroc:.3f}" if r.energy_auroc else "N/A"

        winner = ""
        if r.se_auroc and r.energy_auroc:
            winner = "SE" if r.se_auroc > r.energy_auroc else "Energy"

        print(
            f"{r.bin_name:<12} {range_str:<12} {r.n_samples:>6} {r.n_hallucination:>6} {se_str:>10} {e_str:>10} {winner:<8}"
        )


def main():
    base = Path(__file__).parent.parent / "experiment_notes"

    bins = [
        ("Very Low", 0.0, 0.2),
        ("Low", 0.2, 0.4),
        ("Medium", 0.4, 0.6),
        ("High", 0.6, 0.8),
        ("Very High", 0.8, 1.01),
    ]

    experiments = [
        ("TruthfulQA", base / "exp01_truthfulqa/results_with_corpus.json"),
        ("HaluEval", base / "exp02_halueval/results_with_corpus.json"),
    ]

    for name, path in experiments:
        if not path.exists():
            print(f"\nSkipping {name}: {path} not found")
            print(f"  Run 'python scripts/add_corpus_stats.py' first")
            continue

        with open(path) as f:
            data = json.load(f)

        results = analyze_by_bins(data["samples"], bins)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("\nExpected pattern (if hypothesis holds):")
    print("  - Low coverage bins: Energy AUROC > SE AUROC")
    print("  - High coverage bins: SE AUROC > Energy AUROC")


if __name__ == "__main__":
    main()
