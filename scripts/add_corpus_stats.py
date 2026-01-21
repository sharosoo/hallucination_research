#!/usr/bin/env python3
"""
기존 실험 데이터에 corpus statistics 추가하는 스크립트.

Usage:
    python scripts/add_corpus_stats.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages/hfe-core/src"))

from hfe_core.corpus_stats import InfiniGramClient, CorpusCoverageCalculator
from hfe_core.triplet_extractor import TripletExtractor


def process_experiment(
    input_path: Path,
    output_path: Path,
    extractor: TripletExtractor,
    calculator: CorpusCoverageCalculator,
):
    print(f"\nProcessing: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    total = len(samples)

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        responses = sample.get("responses", [])
        answer = responses[0] if responses else ""

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
                "num_cooc_pairs": coverage.num_cooc_pairs,
            }
        else:
            sample["corpus_stats"] = {
                "entities_q": [],
                "entities_a": [],
                "freq_score": 0.0,
                "cooc_score": 1.0,
                "coverage": 0.5,
                "entity_frequencies": {},
                "num_cooc_pairs": 0,
            }

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total}")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")


def main():
    print("Initializing TripletExtractor (local model)...")
    extractor = TripletExtractor()

    print("Initializing Infini-gram Client...")
    client = InfiniGramClient()
    calculator = CorpusCoverageCalculator(client)

    base = Path(__file__).parent.parent / "experiment_notes"

    experiments = [
        (
            base / "exp01_truthfulqa/results.json",
            base / "exp01_truthfulqa/results_with_corpus.json",
        ),
        (
            base / "exp02_halueval/results.json",
            base / "exp02_halueval/results_with_corpus.json",
        ),
    ]

    for input_path, output_path in experiments:
        if input_path.exists():
            process_experiment(input_path, output_path, extractor, calculator)
        else:
            print(f"Skipping (not found): {input_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
