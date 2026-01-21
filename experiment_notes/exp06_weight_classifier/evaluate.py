"""
exp06: 분류기 평가 및 Cross-dataset 일반화 테스트
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from dataclasses import dataclass
from train_classifier import (
    TrainedModel,
    SEThresholdBaseline,
    EnergyThresholdBaseline,
    FixedWeightCombiner,
)


OUTPUT_DIR = Path(__file__).parent


@dataclass
class EvalResult:
    model_name: str
    train_dataset: str
    test_dataset: str
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_models(dataset: str) -> List:
    with open(OUTPUT_DIR / f"{dataset}_models.pkl", "rb") as f:
        return pickle.load(f)


def load_test_data(dataset: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"{dataset}_test.csv")


def load_full_data(dataset: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"{dataset}_features.csv")


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)

    try:
        auroc = roc_auc_score(y, proba)
    except ValueError:
        auroc = 0.5

    accuracy = accuracy_score(y, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prepare_features(
    df: pd.DataFrame, feature_cols: List[str], scaler=None
) -> np.ndarray:
    X = df[feature_cols].values
    if scaler is not None:
        X = scaler.transform(X)
    return X


def evaluate_on_test(
    trained_models: List, test_df: pd.DataFrame, train_name: str, test_name: str
) -> List[EvalResult]:
    results = []
    y_test = test_df["is_hallucination"].values

    for tm in trained_models:
        X_test = prepare_features(test_df, tm.feature_cols, tm.scaler)
        metrics = evaluate_model(tm.model, X_test, y_test)

        results.append(
            EvalResult(
                model_name=tm.name,
                train_dataset=train_name,
                test_dataset=test_name,
                **metrics,
            )
        )

    return results


def print_results(results: List[EvalResult], title: str):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<25} {'Train→Test':<20} {'AUROC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    )
    print("-" * 85)

    for r in sorted(results, key=lambda x: x.auroc, reverse=True):
        transfer = f"{r.train_dataset}→{r.test_dataset}"
        print(
            f"{r.model_name:<25} {transfer:<20} {r.auroc:>8.4f} {r.accuracy:>8.4f} {r.precision:>8.4f} {r.recall:>8.4f} {r.f1:>8.4f}"
        )


def main():
    print("=" * 80)
    print("exp06: 분류기 평가")
    print("=" * 80)

    all_results = []

    print("\n[1/3] 각 데이터셋 Test 평가...")
    for dataset in ["truthfulqa", "halueval", "combined"]:
        models = load_models(dataset)
        test_df = load_test_data(dataset)
        results = evaluate_on_test(models, test_df, dataset, dataset)
        all_results.extend(results)
        print_results(results, f"{dataset.upper()} Test Set")

    print("\n[2/3] Cross-dataset 일반화 테스트...")

    tqa_models = load_models("truthfulqa")
    halu_full = load_full_data("halueval")
    results_tqa_to_halu = evaluate_on_test(
        tqa_models, halu_full, "truthfulqa", "halueval_full"
    )
    all_results.extend(results_tqa_to_halu)
    print_results(results_tqa_to_halu, "TruthfulQA → HaluEval (Full)")

    halu_models = load_models("halueval")
    tqa_full = load_full_data("truthfulqa")
    results_halu_to_tqa = evaluate_on_test(
        halu_models, tqa_full, "halueval", "truthfulqa_full"
    )
    all_results.extend(results_halu_to_tqa)
    print_results(results_halu_to_tqa, "HaluEval → TruthfulQA (Full)")

    combined_models = load_models("combined")
    results_combined_to_tqa = evaluate_on_test(
        combined_models, tqa_full, "combined", "truthfulqa_full"
    )
    results_combined_to_halu = evaluate_on_test(
        combined_models, halu_full, "combined", "halueval_full"
    )
    all_results.extend(results_combined_to_tqa)
    all_results.extend(results_combined_to_halu)
    print_results(results_combined_to_tqa, "Combined → TruthfulQA (Full)")
    print_results(results_combined_to_halu, "Combined → HaluEval (Full)")

    print("\n[3/3] 결과 요약...")

    results_df = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "train": r.train_dataset,
                "test": r.test_dataset,
                "auroc": r.auroc,
                "accuracy": r.accuracy,
                "f1": r.f1,
            }
            for r in all_results
        ]
    )
    results_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

    print("\n" + "=" * 80)
    print("핵심 비교: SE_only vs Energy_only vs Best Learned")
    print("=" * 80)

    key_models = [
        "SE_only",
        "Energy_only",
        "LogisticRegression",
        "LogisticRegression_ext",
    ]
    key_results = [r for r in all_results if r.model_name in key_models]

    summary = {}
    for r in key_results:
        key = (r.train_dataset, r.test_dataset)
        if key not in summary:
            summary[key] = {}
        summary[key][r.model_name] = r.auroc

    print(f"\n{'Train→Test':<30} {'SE':>10} {'Energy':>10} {'LR':>10} {'LR_ext':>10}")
    print("-" * 72)
    for (train, test), models in sorted(summary.items()):
        transfer = f"{train}→{test}"
        se = models.get("SE_only", 0)
        energy = models.get("Energy_only", 0)
        lr = models.get("LogisticRegression", 0)
        lr_ext = models.get("LogisticRegression_ext", 0)
        print(f"{transfer:<30} {se:>10.4f} {energy:>10.4f} {lr:>10.4f} {lr_ext:>10.4f}")

    print("\n" + "=" * 80)
    print("평가 완료!")
    print(f"결과 저장: {OUTPUT_DIR / 'evaluation_results.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
