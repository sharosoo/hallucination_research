#!/usr/bin/env python3
"""
Exp04: Corpus-Aware Adaptive Hallucination Detection - Full Analysis

실험 목적:
    Corpus coverage에 따라 SE와 Energy의 효과성이 어떻게 변하는지 분석하고,
    적응형 가중치 조합의 효과를 검증한다.

핵심 가설:
    - Low coverage (모델이 모르는 지식) → Energy가 효과적 (confabulation 탐지)
    - High coverage (모델이 아는 지식) → SE가 효과적 (confusion 탐지)

데이터:
    - truthfulqa_with_corpus.json: TruthfulQA 200 샘플 + corpus stats
    - halueval_with_corpus.json: HaluEval QA 200 샘플 + corpus stats

Energy 부호 정책:
    - energy = -mean(logits)
    - 높은 energy (덜 음수) = uncertain = hallucination
    - 따라서 AUROC 계산 시 flip 필요 없음!
    - 이전 step2_bin_analysis.py의 -energy flip은 잘못됨

측정 지표:
    - AUROC: Area Under ROC Curve (높은 score = hallucination 예측)
    - AUPRC: Area Under Precision-Recall Curve

출력:
    - analysis_results_YYYYMMDD_HHMMSS.json: 전체 분석 결과
    - coverage_vs_auroc.png: Coverage bin별 AUROC 비교 그래프
    - adaptive_comparison.png: Baseline vs Adaptive 비교

실행:
    python run_full_analysis.py [--dataset truthfulqa|halueval|both]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "hfe-core" / "src"))

from hfe_core.metrics import (
    compute_auroc,
    compute_all_metrics,
    compute_combined_auroc,
    get_score_statistics,
)


EXPERIMENT_DIR = Path(__file__).parent


@dataclass
class BinAnalysisResult:
    bin_name: str
    coverage_range: tuple[float, float]
    n_samples: int
    n_hallucination: int
    hallucination_rate: float
    se_auroc: float | None
    energy_auroc: float | None
    winner: str
    auroc_diff: float | None


@dataclass
class BaselineResult:
    method: str
    auroc: float | None
    auprc: float | None
    description: str


@dataclass
class DatasetAnalysis:
    dataset_name: str
    n_samples: int
    n_hallucination: int
    hallucination_rate: float
    coverage_stats: dict
    overall_metrics: dict
    bin_analysis: list[dict]
    baseline_comparison: list[dict]
    zero_se_analysis: dict | None


@dataclass
class FullAnalysisResult:
    timestamp: str
    config: dict
    datasets: dict[str, dict]
    summary: dict


def load_data(dataset_name: str) -> dict | None:
    if dataset_name == "truthfulqa":
        path = EXPERIMENT_DIR / "truthfulqa_with_corpus.json"
    elif dataset_name == "halueval":
        path = EXPERIMENT_DIR / "halueval_with_corpus.json"
    else:
        return None

    if not path.exists():
        print(f"  [WARN] {path} not found")
        return None

    with open(path) as f:
        return json.load(f)


def get_coverage_stats(samples: list[dict]) -> dict:
    coverages = [s["corpus_stats"]["coverage"] for s in samples if "corpus_stats" in s]
    if not coverages:
        return {}

    coverages = np.array(coverages)
    return {
        "min": float(np.min(coverages)),
        "max": float(np.max(coverages)),
        "mean": float(np.mean(coverages)),
        "median": float(np.median(coverages)),
        "std": float(np.std(coverages)),
        "percentiles": {
            "p10": float(np.percentile(coverages, 10)),
            "p25": float(np.percentile(coverages, 25)),
            "p50": float(np.percentile(coverages, 50)),
            "p75": float(np.percentile(coverages, 75)),
            "p90": float(np.percentile(coverages, 90)),
        },
    }


def analyze_by_bins(
    samples: list[dict],
    bin_type: str = "equal_width",
    n_bins: int = 5,
) -> list[BinAnalysisResult]:
    valid_samples = [s for s in samples if "corpus_stats" in s]
    if not valid_samples:
        return []

    coverages = np.array([s["corpus_stats"]["coverage"] for s in valid_samples])

    if bin_type == "equal_width":
        bin_edges = np.linspace(0, 1.01, n_bins + 1)
        bin_names = [
            f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(n_bins)
        ]
    elif bin_type == "percentile":
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(coverages, percentiles)
        bin_edges[0] = 0
        bin_edges[-1] = 1.01
        bin_names = [f"Q{i + 1}" for i in range(n_bins)]
    else:
        raise ValueError(f"Unknown bin_type: {bin_type}")

    results = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        bin_samples = [
            s for s in valid_samples if low <= s["corpus_stats"]["coverage"] < high
        ]

        if len(bin_samples) < 5:
            results.append(
                BinAnalysisResult(
                    bin_name=bin_names[i],
                    coverage_range=(float(low), float(high)),
                    n_samples=len(bin_samples),
                    n_hallucination=sum(s["is_hallucination"] for s in bin_samples)
                    if bin_samples
                    else 0,
                    hallucination_rate=0.0,
                    se_auroc=None,
                    energy_auroc=None,
                    winner="N/A (insufficient)",
                    auroc_diff=None,
                )
            )
            continue

        labels = [s["is_hallucination"] for s in bin_samples]
        se_scores = [s["semantic_entropy"] for s in bin_samples]
        energy_scores = [s["semantic_energy"] for s in bin_samples]

        se_auroc = compute_auroc(labels, se_scores, score_type="semantic_entropy")
        energy_auroc = compute_auroc(
            labels, energy_scores, score_type="semantic_energy"
        )

        if se_auroc is not None and energy_auroc is not None:
            auroc_diff = se_auroc - energy_auroc
            if abs(auroc_diff) < 0.01:
                winner = "Tie"
            elif auroc_diff > 0:
                winner = "SE"
            else:
                winner = "Energy"
        else:
            auroc_diff = None
            winner = "N/A"

        n_hall = sum(labels)
        results.append(
            BinAnalysisResult(
                bin_name=bin_names[i],
                coverage_range=(float(low), float(high)),
                n_samples=len(bin_samples),
                n_hallucination=n_hall,
                hallucination_rate=n_hall / len(bin_samples),
                se_auroc=se_auroc,
                energy_auroc=energy_auroc,
                winner=winner,
                auroc_diff=auroc_diff,
            )
        )

    return results


def compare_baselines(samples: list[dict]) -> list[BaselineResult]:
    valid_samples = [s for s in samples if "corpus_stats" in s]
    if not valid_samples:
        return []

    labels = [s["is_hallucination"] for s in valid_samples]
    se_scores = [s["semantic_entropy"] for s in valid_samples]
    energy_scores = [s["semantic_energy"] for s in valid_samples]
    coverages = [s["corpus_stats"]["coverage"] for s in valid_samples]

    results = []

    se_auroc = compute_auroc(labels, se_scores, score_type="semantic_entropy")
    from hfe_core.metrics import compute_auprc

    se_auprc = compute_auprc(labels, se_scores, score_type="semantic_entropy")
    results.append(
        BaselineResult("se_only", se_auroc, se_auprc, "Semantic Entropy only")
    )

    energy_auroc = compute_auroc(labels, energy_scores, score_type="semantic_energy")
    energy_auprc = compute_auprc(labels, energy_scores, score_type="semantic_energy")
    results.append(
        BaselineResult(
            "energy_only", energy_auroc, energy_auprc, "Semantic Energy only"
        )
    )

    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        auroc = compute_combined_auroc(
            labels, se_scores, energy_scores, w, normalize=True
        )
        results.append(
            BaselineResult(
                f"fixed_{w:.1f}",
                auroc,
                None,
                f"Fixed weight: w_energy={w:.1f}",
            )
        )

    def adaptive_sigmoid(coverage, k=10.0, midpoint=0.3):
        import math

        return 1 / (1 + math.exp(k * (coverage - midpoint)))

    def adaptive_linear(coverage, w_max=0.8, w_min=0.2):
        return w_max - (w_max - w_min) * coverage

    def adaptive_threshold(coverage):
        if coverage < 0.2:
            return 0.8
        elif coverage < 0.5:
            return 0.5
        else:
            return 0.2

    for name, weight_fn in [
        ("adaptive_sigmoid", adaptive_sigmoid),
        ("adaptive_linear", adaptive_linear),
        ("adaptive_threshold", adaptive_threshold),
    ]:
        se_arr = np.array(se_scores)
        energy_arr = np.array(energy_scores)

        se_min, se_max = float(np.min(se_arr)), float(np.max(se_arr))
        energy_min, energy_max = float(np.min(energy_arr)), float(np.max(energy_arr))

        combined_scores = []
        for i in range(len(labels)):
            w = weight_fn(coverages[i])
            se_norm = (se_arr[i] - se_min) / (se_max - se_min) if se_max > se_min else 0
            energy_norm = (
                (energy_arr[i] - energy_min) / (energy_max - energy_min)
                if energy_max > energy_min
                else 0
            )
            score = w * energy_norm + (1 - w) * se_norm
            combined_scores.append(score)

        auroc = compute_auroc(labels, combined_scores, score_type="raw")
        results.append(BaselineResult(name, auroc, None, f"Adaptive: {name}"))

    def se_fallback(se_val, threshold=0.1, w_low=0.9, w_high=0.1):
        return w_low if se_val < threshold else w_high

    for name, threshold in [
        ("se_fallback_0.05", 0.05),
        ("se_fallback_0.1", 0.1),
        ("se_fallback_0.2", 0.2),
        ("se_fallback_0.3", 0.3),
    ]:
        se_arr = np.array(se_scores)
        energy_arr = np.array(energy_scores)

        se_min, se_max = float(np.min(se_arr)), float(np.max(se_arr))
        energy_min, energy_max = float(np.min(energy_arr)), float(np.max(energy_arr))

        combined_scores = []
        for i in range(len(labels)):
            w = se_fallback(se_arr[i], threshold=threshold)
            se_norm = (se_arr[i] - se_min) / (se_max - se_min) if se_max > se_min else 0
            energy_norm = (
                (energy_arr[i] - energy_min) / (energy_max - energy_min)
                if energy_max > energy_min
                else 0
            )
            score = w * energy_norm + (1 - w) * se_norm
            combined_scores.append(score)

        auroc = compute_auroc(labels, combined_scores, score_type="raw")
        results.append(
            BaselineResult(name, auroc, None, f"SE-Fallback: threshold={threshold}")
        )

    return results


def analyze_zero_se_cases(samples: list[dict], threshold: float = 0.1) -> dict | None:
    zero_se_samples = [s for s in samples if s["semantic_entropy"] < threshold]

    if len(zero_se_samples) < 10:
        return None

    labels = [s["is_hallucination"] for s in zero_se_samples]
    energy_scores = [s["semantic_energy"] for s in zero_se_samples]

    n_hall = sum(labels)
    n_norm = len(labels) - n_hall

    if n_hall == 0 or n_norm == 0:
        return {
            "threshold": threshold,
            "n_samples": len(zero_se_samples),
            "n_hallucination": n_hall,
            "n_normal": n_norm,
            "energy_auroc": None,
            "note": "Cannot compute AUROC (single class)",
        }

    energy_auroc = compute_auroc(labels, energy_scores, score_type="semantic_energy")

    return {
        "threshold": threshold,
        "n_samples": len(zero_se_samples),
        "n_hallucination": n_hall,
        "n_normal": n_norm,
        "hallucination_rate": n_hall / len(zero_se_samples),
        "energy_auroc": energy_auroc,
    }


def analyze_dataset(data: dict) -> DatasetAnalysis:
    samples = data["samples"]
    dataset_name = data["config"]["dataset"]

    valid_samples = [s for s in samples if "corpus_stats" in s]

    labels = [s["is_hallucination"] for s in valid_samples]
    se_scores = [s["semantic_entropy"] for s in valid_samples]
    energy_scores = [s["semantic_energy"] for s in valid_samples]

    overall = compute_all_metrics(labels, se_scores, energy_scores)

    se_stats = get_score_statistics(se_scores, labels)
    energy_stats = get_score_statistics(energy_scores, labels)

    n_hall = sum(labels)

    bin_results_equal = analyze_by_bins(valid_samples, bin_type="equal_width", n_bins=5)
    bin_results_percentile = analyze_by_bins(
        valid_samples, bin_type="percentile", n_bins=5
    )
    bin_results_fine = analyze_by_bins(valid_samples, bin_type="percentile", n_bins=10)

    baseline_results = compare_baselines(valid_samples)

    zero_se = analyze_zero_se_cases(valid_samples, threshold=0.1)

    return DatasetAnalysis(
        dataset_name=dataset_name,
        n_samples=len(valid_samples),
        n_hallucination=n_hall,
        hallucination_rate=n_hall / len(valid_samples) if valid_samples else 0,
        coverage_stats=get_coverage_stats(valid_samples),
        overall_metrics={
            "semantic_entropy": overall.semantic_entropy.to_dict(),
            "semantic_energy": overall.semantic_energy.to_dict(),
            "winner": overall.winner,
            "auroc_diff": overall.auroc_diff,
            "se_statistics": se_stats,
            "energy_statistics": energy_stats,
        },
        bin_analysis={
            "equal_width_5bins": [asdict(r) for r in bin_results_equal],
            "percentile_5bins": [asdict(r) for r in bin_results_percentile],
            "percentile_10bins": [asdict(r) for r in bin_results_fine],
        },
        baseline_comparison=[asdict(r) for r in baseline_results],
        zero_se_analysis=zero_se,
    )


def create_plots(results: dict, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    for dataset_name, analysis in results["datasets"].items():
        bins_5 = analysis["bin_analysis"]["percentile_5bins"]

        bin_names = [b["bin_name"] for b in bins_5]
        se_aurocs = [b["se_auroc"] if b["se_auroc"] else 0 for b in bins_5]
        energy_aurocs = [b["energy_auroc"] if b["energy_auroc"] else 0 for b in bins_5]

        x = np.arange(len(bin_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(
            x - width / 2, se_aurocs, width, label="SE AUROC", color="steelblue"
        )
        bars2 = ax.bar(
            x + width / 2, energy_aurocs, width, label="Energy AUROC", color="coral"
        )

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
        ax.set_xlabel("Coverage Bin (Percentile)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{dataset_name}: SE vs Energy AUROC by Coverage Bin")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names)
        ax.legend()
        ax.set_ylim(0, 1)

        for bar, val in zip(bars1, se_aurocs):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for bar, val in zip(bars2, energy_aurocs):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{dataset_name.lower()}_coverage_vs_auroc.png", dpi=150
        )
        plt.close()

        baselines = analysis["baseline_comparison"]
        methods = [b["method"] for b in baselines if b["auroc"] is not None]
        aurocs = [b["auroc"] for b in baselines if b["auroc"] is not None]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = [
            "steelblue"
            if "se" in m
            else "coral"
            if "energy" in m
            else "green"
            if "adaptive" in m
            else "gray"
            for m in methods
        ]
        bars = ax.bar(methods, aurocs, color=colors)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Method")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{dataset_name}: Baseline Comparison")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")

        for bar, val in zip(bars, aurocs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{dataset_name.lower()}_baseline_comparison.png", dpi=150
        )
        plt.close()

    print(f"  Plots saved to {output_dir}")


def print_summary(results: dict):
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    for dataset_name, analysis in results["datasets"].items():
        print(f"\n{'─' * 40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─' * 40}")

        print(
            f"  Samples: {analysis['n_samples']} (Hallucination: {analysis['n_hallucination']}, Rate: {analysis['hallucination_rate']:.1%})"
        )

        cov = analysis["coverage_stats"]
        print(
            f"  Coverage: min={cov['min']:.3f}, max={cov['max']:.3f}, mean={cov['mean']:.3f}, median={cov['median']:.3f}"
        )

        overall = analysis["overall_metrics"]
        se_auroc = overall["semantic_entropy"]["auroc"]
        energy_auroc = overall["semantic_energy"]["auroc"]
        print(f"\n  Overall AUROC:")
        print(f"    SE:     {se_auroc:.4f}" if se_auroc else "    SE:     N/A")
        print(f"    Energy: {energy_auroc:.4f}" if energy_auroc else "    Energy: N/A")
        print(f"    Winner: {overall['winner']}")

        print(f"\n  Bin Analysis (Percentile 5-bins):")
        print(
            f"    {'Bin':<8} {'N':>5} {'Hall%':>7} {'SE':>8} {'Energy':>8} {'Winner':<8}"
        )
        print(f"    {'-' * 50}")
        for b in analysis["bin_analysis"]["percentile_5bins"]:
            se = f"{b['se_auroc']:.3f}" if b["se_auroc"] else "N/A"
            en = f"{b['energy_auroc']:.3f}" if b["energy_auroc"] else "N/A"
            hr = (
                f"{b['hallucination_rate'] * 100:.1f}%" if b["n_samples"] > 0 else "N/A"
            )
            print(
                f"    {b['bin_name']:<8} {b['n_samples']:>5} {hr:>7} {se:>8} {en:>8} {b['winner']:<8}"
            )

        print(f"\n  Baseline Comparison (Top 5):")
        baselines = sorted(
            [b for b in analysis["baseline_comparison"] if b["auroc"] is not None],
            key=lambda x: x["auroc"],
            reverse=True,
        )[:5]
        for i, b in enumerate(baselines, 1):
            print(f"    {i}. {b['method']}: {b['auroc']:.4f}")

        if analysis["zero_se_analysis"]:
            zse = analysis["zero_se_analysis"]
            print(f"\n  Zero-SE Analysis (SE < {zse['threshold']}):")
            print(
                f"    Samples: {zse['n_samples']} (Hall: {zse['n_hallucination']}, Norm: {zse['n_normal']})"
            )
            if zse.get("energy_auroc"):
                print(f"    Energy AUROC: {zse['energy_auroc']:.4f}")


def main(datasets: list[str] = None):
    if datasets is None:
        datasets = ["truthfulqa", "halueval"]

    print("=" * 80)
    print("Exp04: Corpus-Aware Adaptive Hallucination Detection - Full Analysis")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Datasets: {datasets}")

    print("\n[1/4] Loading data...")
    all_data = {}
    for ds in datasets:
        data = load_data(ds)
        if data:
            all_data[ds] = data
            print(f"  {ds}: {len(data['samples'])} samples loaded")

    if not all_data:
        print("No data found. Run step1_add_corpus_stats.py first.")
        return

    print("\n[2/4] Analyzing datasets...")
    dataset_analyses = {}
    for ds_name, data in all_data.items():
        print(f"  Analyzing {ds_name}...")
        analysis = analyze_dataset(data)
        dataset_analyses[ds_name] = asdict(analysis)

    best_methods = {}
    for ds_name, analysis in dataset_analyses.items():
        baselines = analysis["baseline_comparison"]
        best = max([b for b in baselines if b["auroc"]], key=lambda x: x["auroc"])
        best_methods[ds_name] = {"method": best["method"], "auroc": best["auroc"]}

    results = FullAnalysisResult(
        timestamp=datetime.now().isoformat(),
        config={
            "datasets": datasets,
            "energy_sign_policy": "NO FLIP - higher energy = uncertain = hallucination",
            "bin_types": ["equal_width_5bins", "percentile_5bins", "percentile_10bins"],
        },
        datasets=dataset_analyses,
        summary={
            "best_methods": best_methods,
            "hypothesis_check": "See bin analysis for coverage-based patterns",
        },
    )

    print("\n[3/4] Saving results...")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EXPERIMENT_DIR / f"analysis_results_{timestamp_str}.json"

    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"  Results saved to: {output_path}")

    latest_path = EXPERIMENT_DIR / "analysis_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"  Latest link: {latest_path}")

    print("\n[4/4] Creating plots...")
    create_plots(asdict(results), EXPERIMENT_DIR)

    print_summary(asdict(results))

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp04: Full Corpus-Adaptive Analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["truthfulqa", "halueval", "both"],
        default="both",
        help="Dataset to analyze",
    )
    args = parser.parse_args()

    if args.dataset == "both":
        datasets = ["truthfulqa", "halueval"]
    else:
        datasets = [args.dataset]

    main(datasets)
