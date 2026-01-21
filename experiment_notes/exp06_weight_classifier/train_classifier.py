"""
exp06: SE/Energy 결합 분류기 학습
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


OUTPUT_DIR = Path(__file__).parent
RANDOM_STATE = 42

FEATURE_COLS_BASIC = ["semantic_entropy", "energy_mean"]
FEATURE_COLS_EXTENDED = [
    "semantic_entropy",
    "energy_mean",
    "response_length_mean",
    "response_length_std",
    "num_clusters",
    "se_is_zero",
    "question_length",
]


@dataclass
class TrainedModel:
    name: str
    model: object
    scaler: Optional[StandardScaler]
    feature_cols: List[str]
    train_auroc: float
    val_auroc: float


def load_splits(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(OUTPUT_DIR / f"{dataset}_train.csv")
    val = pd.read_csv(OUTPUT_DIR / f"{dataset}_val.csv")
    test = pd.read_csv(OUTPUT_DIR / f"{dataset}_test.csv")
    return train, val, test


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    X = df[feature_cols].values
    y = df["is_hallucination"].values

    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    return X, y, scaler


class SEThresholdBaseline:
    """SE > threshold → hallucination 예측"""

    def __init__(self):
        self.threshold = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        se_values = X[:, 0]
        best_threshold, best_auroc = 0.0, 0.0

        for threshold in np.percentile(se_values, range(10, 91, 10)):
            pred_proba = (se_values > threshold).astype(float)
            try:
                auroc = roc_auc_score(y, pred_proba)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_threshold = threshold
            except ValueError:
                continue

        self.threshold = best_threshold
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        se_values = X[:, 0]
        proba_1 = (se_values - se_values.min()) / (
            se_values.max() - se_values.min() + 1e-8
        )
        return np.column_stack([1 - proba_1, proba_1])


class EnergyThresholdBaseline:
    """Energy > threshold → hallucination 예측 (덜 음수 = 더 uncertain)"""

    def __init__(self):
        self.threshold = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        energy_values = X[:, 1]
        best_threshold, best_auroc = 0.0, 0.0

        for threshold in np.percentile(energy_values, range(10, 91, 10)):
            pred_proba = (energy_values > threshold).astype(float)
            try:
                auroc = roc_auc_score(y, pred_proba)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_threshold = threshold
            except ValueError:
                continue

        self.threshold = best_threshold
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        energy_values = X[:, 1]
        proba_1 = (energy_values - energy_values.min()) / (
            energy_values.max() - energy_values.min() + 1e-8
        )
        return np.column_stack([1 - proba_1, proba_1])


class FixedWeightCombiner:
    """Score = w × Energy_norm + (1-w) × SE_norm"""

    def __init__(self, weight: float = 0.5):
        self.weight = weight
        self.se_min, self.se_max = 0.0, 1.0
        self.energy_min, self.energy_max = 0.0, 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.se_min, self.se_max = X[:, 0].min(), X[:, 0].max()
        self.energy_min, self.energy_max = X[:, 1].min(), X[:, 1].max()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        se_norm = (X[:, 0] - self.se_min) / (self.se_max - self.se_min + 1e-8)
        energy_norm = (X[:, 1] - self.energy_min) / (
            self.energy_max - self.energy_min + 1e-8
        )

        combined = self.weight * energy_norm + (1 - self.weight) * se_norm
        return np.column_stack([1 - combined, combined])


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    proba = model.predict_proba(X)[:, 1]
    try:
        auroc = roc_auc_score(y, proba)
    except ValueError:
        auroc = 0.5

    pred = (proba > 0.5).astype(int)
    accuracy = accuracy_score(y, pred)
    return auroc, accuracy


def train_all_models(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> List[TrainedModel]:
    models = []

    X_train_basic, y_train, scaler_basic = prepare_features(
        train_df, FEATURE_COLS_BASIC, fit_scaler=True
    )
    X_val_basic, y_val, _ = prepare_features(
        val_df, FEATURE_COLS_BASIC, scaler=scaler_basic
    )

    X_train_ext, _, scaler_ext = prepare_features(
        train_df, FEATURE_COLS_EXTENDED, fit_scaler=True
    )
    X_val_ext, _, _ = prepare_features(val_df, FEATURE_COLS_EXTENDED, scaler=scaler_ext)

    X_train_raw, _, _ = prepare_features(train_df, FEATURE_COLS_BASIC)
    X_val_raw, _, _ = prepare_features(val_df, FEATURE_COLS_BASIC)

    se_baseline = SEThresholdBaseline().fit(X_train_raw, y_train)
    train_auroc, _ = evaluate_model(se_baseline, X_train_raw, y_train)
    val_auroc, _ = evaluate_model(se_baseline, X_val_raw, y_val)
    models.append(
        TrainedModel(
            "SE_only", se_baseline, None, FEATURE_COLS_BASIC, train_auroc, val_auroc
        )
    )

    energy_baseline = EnergyThresholdBaseline().fit(X_train_raw, y_train)
    train_auroc, _ = evaluate_model(energy_baseline, X_train_raw, y_train)
    val_auroc, _ = evaluate_model(energy_baseline, X_val_raw, y_val)
    models.append(
        TrainedModel(
            "Energy_only",
            energy_baseline,
            None,
            FEATURE_COLS_BASIC,
            train_auroc,
            val_auroc,
        )
    )

    for w in [0.3, 0.5, 0.7]:
        combiner = FixedWeightCombiner(weight=w).fit(X_train_raw, y_train)
        train_auroc, _ = evaluate_model(combiner, X_train_raw, y_train)
        val_auroc, _ = evaluate_model(combiner, X_val_raw, y_val)
        models.append(
            TrainedModel(
                f"Fixed_w{w}",
                combiner,
                None,
                FEATURE_COLS_BASIC,
                train_auroc,
                val_auroc,
            )
        )

    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train_basic, y_train)
    train_auroc, _ = evaluate_model(lr, X_train_basic, y_train)
    val_auroc, _ = evaluate_model(lr, X_val_basic, y_val)
    models.append(
        TrainedModel(
            "LogisticRegression",
            lr,
            scaler_basic,
            FEATURE_COLS_BASIC,
            train_auroc,
            val_auroc,
        )
    )

    lr_ext = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_ext.fit(X_train_ext, y_train)
    train_auroc, _ = evaluate_model(lr_ext, X_train_ext, y_train)
    val_auroc, _ = evaluate_model(lr_ext, X_val_ext, y_val)
    models.append(
        TrainedModel(
            "LogisticRegression_ext",
            lr_ext,
            scaler_ext,
            FEATURE_COLS_EXTENDED,
            train_auroc,
            val_auroc,
        )
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        random_state=RANDOM_STATE,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train_ext, y_train)
    train_auroc, _ = evaluate_model(mlp, X_train_ext, y_train)
    val_auroc, _ = evaluate_model(mlp, X_val_ext, y_val)
    models.append(
        TrainedModel(
            "MLP", mlp, scaler_ext, FEATURE_COLS_EXTENDED, train_auroc, val_auroc
        )
    )

    return models


def print_results(models: List[TrainedModel], dataset_name: str):
    print(f"\n{'=' * 60}")
    print(f"{dataset_name} 결과")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'Train AUROC':>12} {'Val AUROC':>12}")
    print("-" * 51)

    for m in sorted(models, key=lambda x: x.val_auroc, reverse=True):
        print(f"{m.name:<25} {m.train_auroc:>12.4f} {m.val_auroc:>12.4f}")


def save_models(models: List[TrainedModel], dataset_name: str):
    save_path = OUTPUT_DIR / f"{dataset_name}_models.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n모델 저장: {save_path}")


def main():
    print("=" * 60)
    print("exp06: SE/Energy 결합 분류기 학습")
    print("=" * 60)

    for dataset in ["truthfulqa", "halueval", "combined"]:
        print(f"\n[{dataset.upper()}] 학습 시작...")
        train_df, val_df, _ = load_splits(dataset)

        models = train_all_models(train_df, val_df)
        print_results(models, dataset)
        save_models(models, dataset)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
