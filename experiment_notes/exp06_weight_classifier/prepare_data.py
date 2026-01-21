"""
exp06: SE/Energy 가중치 분류기 학습 데이터 준비
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent.parent / "exp04_corpus_adaptive"
OUTPUT_DIR = Path(__file__).parent
RANDOM_STATE = 42
NUM_RESPONSES = 5


def load_experiment_data(filepath: Path) -> Tuple[Dict, List[Dict]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = {
        "experiment": data.get("experiment"),
        "config": data.get("config"),
        "metrics": data.get("metrics"),
        "statistics": data.get("statistics"),
    }
    return metadata, data.get("samples", [])


def extract_features(sample: Dict) -> Dict:
    se = sample.get("semantic_entropy", 0.0)
    energy_total = sample.get("semantic_energy", 0.0)
    energy_mean = (
        energy_total / NUM_RESPONSES if abs(energy_total) > 20 else energy_total
    )

    responses = sample.get("responses", [])
    lengths = [len(r) for r in responses] if responses else [0]

    return {
        "semantic_entropy": se,
        "energy_mean": energy_mean,
        "is_hallucination": sample.get("is_hallucination", 0),
        "response_length_mean": np.mean(lengths),
        "response_length_std": np.std(lengths),
        "num_clusters": sample.get("num_clusters", 1),
        "se_is_zero": 1 if se < 0.01 else 0,
        "question_length": len(sample.get("question", "")),
        "question": sample.get("question", ""),
    }


def samples_to_dataframe(samples: List[Dict], dataset_name: str) -> pd.DataFrame:
    features_list = [extract_features(sample) for sample in samples]
    for f in features_list:
        f["dataset"] = dataset_name
    return pd.DataFrame(features_list)


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["semantic_entropy", "energy_mean"]:
        mean, std = df[col].mean(), df[col].std()
        df[f"{col}_zscore"] = (df[col] - mean) / std if std > 0 else 0.0

        min_val, max_val = df[col].min(), df[col].max()
        df[f"{col}_minmax"] = (
            (df[col] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        )

    return df


def add_oracle_labels(df: pd.DataFrame) -> pd.DataFrame:
    """각 샘플에서 SE vs Energy 중 어느 것이 정답을 더 잘 예측하는지 라벨링"""
    df = df.copy()

    for dataset in df["dataset"].unique():
        mask = df["dataset"] == dataset
        subset = df[mask]

        se_median = subset["semantic_entropy"].median()
        energy_median = subset["energy_mean"].median()

        se_pred = (subset["semantic_entropy"] > se_median).astype(int)
        energy_pred = (subset["energy_mean"] > energy_median).astype(int)
        actual = subset["is_hallucination"]

        se_correct = (se_pred == actual).astype(int)
        energy_correct = (energy_pred == actual).astype(int)

        oracle = np.where(
            (energy_correct == 1) & (se_correct == 0),
            1,
            np.where((se_correct == 1) & (energy_correct == 0), 0, 0.5),
        )

        df.loc[mask, "se_correct"] = se_correct.values
        df.loc[mask, "energy_correct"] = energy_correct.values
        df.loc[mask, "oracle_prefer_energy"] = oracle

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_col: str = "is_hallucination",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE, stratify=df[stratify_col]
    )

    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=train_val[stratify_col],
    )

    return train, val, test


def print_dataset_stats(name: str, df: pd.DataFrame):
    print(f"\n  {name}:")
    print(f"    SE mean: {df['semantic_entropy'].mean():.3f}")
    print(f"    SE zero ratio: {df['se_is_zero'].mean() * 100:.1f}%")
    print(f"    Energy mean: {df['energy_mean'].mean():.3f}")
    print(f"    Hallucination rate: {df['is_hallucination'].mean() * 100:.1f}%")


def print_oracle_stats(name: str, df: pd.DataFrame):
    se_win = (df["oracle_prefer_energy"] == 0).sum()
    energy_win = (df["oracle_prefer_energy"] == 1).sum()
    tie = (df["oracle_prefer_energy"] == 0.5).sum()
    total = len(df)
    print(f"\n  {name} Oracle 분포:")
    print(f"    SE wins: {se_win} ({se_win / total * 100:.1f}%)")
    print(f"    Energy wins: {energy_win} ({energy_win / total * 100:.1f}%)")
    print(f"    Tie: {tie} ({tie / total * 100:.1f}%)")


def save_datasets(output_dir: Path, **datasets):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def main():
    print("=" * 60)
    print("exp06: 가중치 분류기 데이터 준비")
    print("=" * 60)

    print("\n[1/5] 데이터 로드...")
    _, truthfulqa_samples = load_experiment_data(
        DATA_DIR / "truthfulqa_with_corpus.json"
    )
    _, halueval_samples = load_experiment_data(DATA_DIR / "halueval_with_corpus.json")
    print(f"  TruthfulQA: {len(truthfulqa_samples)} samples")
    print(f"  HaluEval: {len(halueval_samples)} samples")

    print("\n[2/5] Feature 추출...")
    df_truthfulqa = samples_to_dataframe(truthfulqa_samples, "truthfulqa")
    df_halueval = samples_to_dataframe(halueval_samples, "halueval")

    print_dataset_stats("TruthfulQA", df_truthfulqa)
    print_dataset_stats("HaluEval", df_halueval)

    print("\n[3/5] 정규화 및 Oracle label 추가...")
    df_truthfulqa = add_oracle_labels(add_normalized_features(df_truthfulqa))
    df_halueval = add_oracle_labels(add_normalized_features(df_halueval))

    print_oracle_stats("TruthfulQA", df_truthfulqa)
    print_oracle_stats("HaluEval", df_halueval)

    print("\n[4/5] Train/Val/Test 분할...")
    train_tqa, val_tqa, test_tqa = split_data(df_truthfulqa)
    train_halu, val_halu, test_halu = split_data(df_halueval)

    df_combined = pd.concat([df_truthfulqa, df_halueval], ignore_index=True)
    train_combined, val_combined, test_combined = split_data(df_combined)

    print(
        f"  TruthfulQA: train={len(train_tqa)}, val={len(val_tqa)}, test={len(test_tqa)}"
    )
    print(
        f"  HaluEval: train={len(train_halu)}, val={len(val_halu)}, test={len(test_halu)}"
    )
    print(
        f"  Combined: train={len(train_combined)}, val={len(val_combined)}, test={len(test_combined)}"
    )

    print("\n[5/5] 저장...")
    save_datasets(
        OUTPUT_DIR,
        truthfulqa_features=df_truthfulqa,
        halueval_features=df_halueval,
        truthfulqa_train=train_tqa,
        truthfulqa_val=val_tqa,
        truthfulqa_test=test_tqa,
        halueval_train=train_halu,
        halueval_val=val_halu,
        halueval_test=test_halu,
        combined_features=df_combined,
        combined_train=train_combined,
        combined_val=val_combined,
        combined_test=test_combined,
    )
    print(f"  저장 완료: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("데이터 준비 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
