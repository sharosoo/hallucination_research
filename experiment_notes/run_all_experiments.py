#!/usr/bin/env python3
"""
통합 실험 스크립트 - Corpus 기반 적응형 환각 탐지 연구

실행 순서:
1. Corpus stats 추가 (Infini-gram API)
2. Bin별 AUROC 분석 (가설 검증)
3. Baseline 비교 실험 (적응형 vs 고정 가중치)
4. 결과 저장 및 리포트 생성
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "packages/hfe-core/src"))


EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "exp04_corpus_adaptive"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ExperimentConfig:
    timestamp: str
    datasets: list[str]
    corpus_index: str
    weight_functions: list[str]
    bins: list[tuple[str, float, float]]


@dataclass
class BinResult:
    bin_name: str
    coverage_range: tuple[float, float]
    n_samples: int
    n_hallucination: int
    hallucination_rate: float
    se_auroc: float | None
    energy_auroc: float | None
    winner: str


@dataclass
class BaselineResult:
    method: str
    auroc: float | None
    auprc: float | None


@dataclass
class DatasetResult:
    dataset: str
    total_samples: int
    bin_analysis: list[BinResult]
    baseline_comparison: list[BaselineResult]
    best_method: str
    best_auroc: float


def compute_auroc_auprc(
    labels: list[int], scores: list[float]
) -> tuple[float | None, float | None]:
    if len(set(labels)) < 2 or len(labels) < 5:
        return None, None
    try:
        return roc_auc_score(labels, scores), average_precision_score(labels, scores)
    except:
        return None, None


def add_corpus_stats_to_samples(
    samples: list[dict], max_samples: int | None = None
) -> list[dict]:
    from hfe_core.corpus_stats import InfiniGramClient, CorpusCoverageCalculator
    from hfe_core.triplet_extractor import TripletExtractor

    client = InfiniGramClient()
    calculator = CorpusCoverageCalculator(client)
    extractor = TripletExtractor()

    processed = []
    total = min(len(samples), max_samples) if max_samples else len(samples)

    print(f"  Adding corpus stats to {total} samples...")

    for i, sample in enumerate(samples[:total]):
        if i % 20 == 0:
            print(f"    Progress: {i}/{total}")

        question = sample.get("question", "")
        responses = sample.get("responses", [])
        answer = responses[0] if responses else ""

        try:
            q_result = extractor.extract(question, is_question=True)
            a_result = extractor.extract(answer, is_question=False)
            entities_q = q_result.entities
            entities_a = a_result.entities

            if entities_q or entities_a:
                coverage = calculator.compute(entities_q, entities_a)
                sample["corpus_stats"] = {
                    "entities_q": entities_q,
                    "entities_a": entities_a,
                    "freq_score": coverage.freq_score,
                    "cooc_score": coverage.cooc_score,
                    "coverage": coverage.coverage,
                    "entity_frequencies": coverage.entity_frequencies,
                }
            else:
                sample["corpus_stats"] = {
                    "entities_q": [],
                    "entities_a": [],
                    "freq_score": 0.0,
                    "cooc_score": 1.0,
                    "coverage": 0.5,
                    "entity_frequencies": {},
                }
        except Exception as e:
            print(f"    Error at sample {i}: {e}")
            sample["corpus_stats"] = {
                "entities_q": [],
                "entities_a": [],
                "freq_score": 0.0,
                "cooc_score": 1.0,
                "coverage": 0.5,
                "entity_frequencies": {},
            }

        processed.append(sample)
        time.sleep(0.1)

    return processed


def analyze_by_bins(
    samples: list[dict], bins: list[tuple[str, float, float]]
) -> list[BinResult]:
    """Coverage bin별 AUROC 분석"""
    results = []

    for bin_name, low, high in bins:
        bin_samples = [
            s
            for s in samples
            if "corpus_stats" in s and low <= s["corpus_stats"]["coverage"] < high
        ]

        if len(bin_samples) < 3:
            results.append(
                BinResult(
                    bin_name=bin_name,
                    coverage_range=(low, high),
                    n_samples=len(bin_samples),
                    n_hallucination=sum(s["is_hallucination"] for s in bin_samples)
                    if bin_samples
                    else 0,
                    hallucination_rate=0.0,
                    se_auroc=None,
                    energy_auroc=None,
                    winner="N/A",
                )
            )
            continue

        labels = [s["is_hallucination"] for s in bin_samples]
        se_scores = [s["semantic_entropy"] for s in bin_samples]
        # Energy: 높은 값 = uncertain = hallucination (flip 불필요)
        energy_scores = [s["semantic_energy"] for s in bin_samples]

        se_auroc, _ = compute_auroc_auprc(labels, se_scores)
        energy_auroc, _ = compute_auroc_auprc(labels, energy_scores)

        winner = "N/A"
        if se_auroc and energy_auroc:
            winner = "SE" if se_auroc > energy_auroc else "Energy"
        elif se_auroc:
            winner = "SE"
        elif energy_auroc:
            winner = "Energy"

        results.append(
            BinResult(
                bin_name=bin_name,
                coverage_range=(low, high),
                n_samples=len(bin_samples),
                n_hallucination=sum(labels),
                hallucination_rate=sum(labels) / len(labels),
                se_auroc=se_auroc,
                energy_auroc=energy_auroc,
                winner=winner,
            )
        )

    return results


def compare_baselines(samples: list[dict]) -> list[BaselineResult]:
    """Baseline 방법들 비교"""
    from hfe_core.adaptive_weights import create_baseline_calculator

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

    results = []
    labels = [s["is_hallucination"] for s in samples]

    for name in baselines:
        calc = create_baseline_calculator(name)

        scores = []
        for s in samples:
            coverage = s.get("corpus_stats", {}).get("coverage", 0.5)
            se = s["semantic_entropy"]
            energy = s["semantic_energy"]
            score = calc.compute_score(coverage, se, energy, normalize=True)
            scores.append(score)

        auroc, auprc = compute_auroc_auprc(labels, scores)
        results.append(BaselineResult(method=name, auroc=auroc, auprc=auprc))

    return results


def run_experiment(
    dataset_name: str, data_path: Path, bins: list, max_samples: int | None = None
) -> DatasetResult:
    """단일 데이터셋 실험 실행"""
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {dataset_name}")
    print(f"{'=' * 60}")

    with open(data_path) as f:
        data = json.load(f)

    samples = data["samples"]
    if max_samples:
        samples = samples[:max_samples]

    print(f"\n[Step 1] Adding corpus statistics...")
    samples_with_stats = add_corpus_stats_to_samples(samples, max_samples)

    print(f"\n[Step 2] Analyzing by coverage bins...")
    bin_results = analyze_by_bins(samples_with_stats, bins)

    print(f"\n[Step 3] Comparing baselines...")
    baseline_results = compare_baselines(samples_with_stats)

    best = max(
        [r for r in baseline_results if r.auroc], key=lambda x: x.auroc, default=None
    )

    return DatasetResult(
        dataset=dataset_name,
        total_samples=len(samples_with_stats),
        bin_analysis=[asdict(r) for r in bin_results],
        baseline_comparison=[asdict(r) for r in baseline_results],
        best_method=best.method if best else "N/A",
        best_auroc=best.auroc if best else 0.0,
    )


def print_report(results: list[DatasetResult]):
    """결과 리포트 출력"""
    print("\n" + "=" * 70)
    print("EXPERIMENT REPORT: Corpus-Aware Adaptive Hallucination Detection")
    print("=" * 70)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"Dataset: {r.dataset} (N={r.total_samples})")
        print(f"{'─' * 70}")

        print("\n[Bin Analysis] Coverage → AUROC")
        print(
            f"{'Bin':<12} {'N':>5} {'Hall%':>7} {'SE':>8} {'Energy':>8} {'Winner':<8}"
        )
        print("-" * 55)

        for b in r.bin_analysis:
            se = f"{b['se_auroc']:.3f}" if b["se_auroc"] else "N/A"
            en = f"{b['energy_auroc']:.3f}" if b["energy_auroc"] else "N/A"
            hr = f"{b['hallucination_rate'] * 100:.1f}%"
            print(
                f"{b['bin_name']:<12} {b['n_samples']:>5} {hr:>7} {se:>8} {en:>8} {b['winner']:<8}"
            )

        print("\n[Baseline Comparison]")
        print(f"{'Method':<22} {'AUROC':>10} {'AUPRC':>10}")
        print("-" * 45)

        sorted_baselines = sorted(
            r.baseline_comparison, key=lambda x: x["auroc"] or 0, reverse=True
        )
        for b in sorted_baselines:
            auroc = f"{b['auroc']:.4f}" if b["auroc"] else "N/A"
            auprc = f"{b['auprc']:.4f}" if b["auprc"] else "N/A"
            marker = " *" if b["method"] == r.best_method else ""
            print(f"{b['method']:<22} {auroc:>10} {auprc:>10}{marker}")

        print(f"\nBest: {r.best_method} (AUROC: {r.best_auroc:.4f})")


def save_results(results: list[DatasetResult], config: ExperimentConfig):
    """결과 저장"""
    output = {
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }

    output_path = OUTPUT_DIR / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    config = ExperimentConfig(
        timestamp=datetime.now().isoformat(),
        datasets=["TruthfulQA", "HaluEval"],
        corpus_index="v4_dolma-v1_7_llama",
        weight_functions=["sigmoid", "linear", "threshold", "fixed"],
        bins=[
            ("Very Low", 0.0, 0.2),
            ("Low", 0.2, 0.4),
            ("Medium", 0.4, 0.6),
            ("High", 0.6, 0.8),
            ("Very High", 0.8, 1.01),
        ],
    )

    experiments = [
        ("TruthfulQA", EXPERIMENT_DIR / "exp01_truthfulqa/results.json"),
        ("HaluEval", EXPERIMENT_DIR / "exp02_halueval/results.json"),
    ]

    results = []

    for name, path in experiments:
        if path.exists():
            result = run_experiment(name, path, config.bins, max_samples=200)
            results.append(result)
        else:
            print(f"Skipping {name}: {path} not found")

    print_report(results)
    save_results(results, config)

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
