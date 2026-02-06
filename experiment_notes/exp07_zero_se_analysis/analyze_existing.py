#!/usr/bin/env python
"""
E1: Zero-SE 현상 정량화 (기존 데이터 분석)

기존 exp01_truthfulqa/results.json, exp02_halueval/results.json에서
Zero-SE 현상을 다양한 ε 값으로 정밀하게 분석한다.

출력:
1. Zero-SE 정량화 테이블 (ε × dataset)
2. SE 구간별 성능 분석
3. Cascade threshold sweep
4. 상보성 분석 (Venn diagram 데이터)
5. Bootstrap confidence intervals

GPU 불필요 - 기존 결과 JSON만 사용.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent.parent
EXP_DIR = ROOT / "experiment_notes"


def load_existing_results():
    """기존 exp01/exp02 결과 로드"""
    datasets = {}

    # TruthfulQA
    tqa_path = EXP_DIR / "exp01_truthfulqa" / "results.json"
    if tqa_path.exists():
        with open(tqa_path) as f:
            tqa = json.load(f)
        datasets["TruthfulQA"] = tqa["samples"]
        print(f"  TruthfulQA: {len(tqa['samples'])} samples loaded")

    # HaluEval
    he_path = EXP_DIR / "exp02_halueval" / "results.json"
    if he_path.exists():
        with open(he_path) as f:
            he = json.load(f)
        datasets["HaluEval"] = he["samples"]
        print(f"  HaluEval: {len(he['samples'])} samples loaded")

    return datasets


def load_new_results():
    """exp07에서 새로 생성된 데이터셋 결과 로드.
    _llm_judge.json 파일이 존재하면 해당 데이터셋의 원본 대신 사용."""
    datasets = {}
    new_dir = EXP_DIR / "exp07_zero_se_analysis"

    all_files = sorted(new_dir.glob("results_*.json"))

    # Build set of base names that have _llm_judge versions
    llm_judge_bases = set()
    for f in all_files:
        if f.stem.endswith("_llm_judge"):
            base = f.stem.replace("_llm_judge", "")
            llm_judge_bases.add(base)

    for json_file in all_files:
        stem = json_file.stem
        # Skip old file if _llm_judge version exists
        if not stem.endswith("_llm_judge") and stem in llm_judge_bases:
            print(f"  (skipped {json_file.name} — _llm_judge version exists)")
            continue

        with open(json_file) as f:
            data = json.load(f)
        name = data.get(
            "dataset_name", stem.replace("results_", "").replace("_llm_judge", "")
        )
        datasets[name] = data["samples"]
        label_src = "LLM-judge" if stem.endswith("_llm_judge") else "original"
        print(f"  {name}: {len(data['samples'])} samples loaded ({label_src})")

    return datasets


def bootstrap_auroc(labels, scores, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for AUROC"""
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2 or len(labels) < 5:
        return None, None, None

    aurocs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            continue

    if len(aurocs) < 100:
        return None, None, None

    alpha = (1 - ci) / 2
    lower = np.percentile(aurocs, alpha * 100)
    upper = np.percentile(aurocs, (1 - alpha) * 100)
    mean = np.mean(aurocs)
    return mean, lower, upper


# ─────────────────────────────────────────────────────────────
# E1: Zero-SE 정량화
# ─────────────────────────────────────────────────────────────
def analyze_zero_se(samples, dataset_name, epsilons=(0.001, 0.01, 0.05, 0.1)):
    """Zero-SE 현상을 다양한 ε으로 정량화"""
    results = []

    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    total = len(samples)
    total_hall = labels.sum()
    total_hall_rate = total_hall / total

    for eps in epsilons:
        mask = entropies <= eps
        n_zero = mask.sum()

        if n_zero == 0:
            results.append(
                {
                    "dataset": dataset_name,
                    "epsilon": eps,
                    "n_total": total,
                    "n_zero_se": 0,
                    "zero_se_pct": 0.0,
                    "hall_rate_in_zero_se": None,
                    "energy_auroc_in_zero_se": None,
                    "se_auroc_in_zero_se": None,
                }
            )
            continue

        zero_labels = labels[mask]
        zero_energies = energies[mask]
        zero_entropies = entropies[mask]

        n_hall = zero_labels.sum()
        hall_rate = n_hall / n_zero

        # Energy AUROC in Zero-SE
        energy_auroc = None
        energy_auroc_ci = (None, None)
        if len(np.unique(zero_labels)) == 2:
            energy_auroc = roc_auc_score(zero_labels, zero_energies)
            _, lo, hi = bootstrap_auroc(zero_labels, zero_energies)
            energy_auroc_ci = (lo, hi)

        # SE AUROC in Zero-SE (expected to be ~0.5 / degenerate)
        se_auroc = None
        if len(np.unique(zero_labels)) == 2 and len(np.unique(zero_entropies)) > 1:
            try:
                se_auroc = roc_auc_score(zero_labels, zero_entropies)
            except ValueError:
                pass

        results.append(
            {
                "dataset": dataset_name,
                "epsilon": eps,
                "n_total": int(total),
                "n_zero_se": int(n_zero),
                "zero_se_pct": float(n_zero / total * 100),
                "n_hall_in_zero_se": int(n_hall),
                "n_normal_in_zero_se": int(n_zero - n_hall),
                "hall_rate_overall": float(total_hall_rate * 100),
                "hall_rate_in_zero_se": float(hall_rate * 100),
                "energy_auroc_in_zero_se": energy_auroc,
                "energy_auroc_ci": energy_auroc_ci,
                "se_auroc_in_zero_se": se_auroc,
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# E3: SE 구간별 분석
# ─────────────────────────────────────────────────────────────
def analyze_se_bins(samples, dataset_name, epsilon=0.05):
    """SE 구간별 성능 분석"""
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    # SE bins
    bins = [
        ("Zero-SE", 0, epsilon),
        ("Low", epsilon, 0.3),
        ("Medium", 0.3, 0.6),
        ("High", 0.6, 1.0),
        ("Very High", 1.0, float("inf")),
    ]

    results = []
    for bin_name, lo, hi in bins:
        if lo == 0:
            mask = (entropies >= lo) & (entropies <= hi)
        else:
            mask = (entropies > lo) & (entropies <= hi)

        n = mask.sum()
        if n == 0:
            results.append(
                {
                    "dataset": dataset_name,
                    "bin": bin_name,
                    "range": f"({lo}, {hi}]",
                    "n": 0,
                }
            )
            continue

        bin_labels = labels[mask]
        bin_energies = energies[mask]
        bin_entropies = entropies[mask]

        hall_rate = bin_labels.sum() / n

        se_auroc = None
        energy_auroc = None
        energy_ci = (None, None)

        if len(np.unique(bin_labels)) == 2:
            try:
                energy_auroc = roc_auc_score(bin_labels, bin_energies)
                _, lo_ci, hi_ci = bootstrap_auroc(bin_labels, bin_energies)
                energy_ci = (lo_ci, hi_ci)
            except ValueError:
                pass

            if len(np.unique(bin_entropies)) > 1:
                try:
                    se_auroc = roc_auc_score(bin_labels, bin_entropies)
                except ValueError:
                    pass

        results.append(
            {
                "dataset": dataset_name,
                "bin": bin_name,
                "range": f"({lo}, {hi}]" if lo > 0 else f"[{lo}, {hi}]",
                "n": int(n),
                "n_hall": int(bin_labels.sum()),
                "hall_rate": float(hall_rate * 100),
                "se_auroc": se_auroc,
                "energy_auroc": energy_auroc,
                "energy_auroc_ci": energy_ci,
                "winner": "Energy"
                if (energy_auroc or 0) > (se_auroc or 0)
                else "SE"
                if se_auroc
                else "N/A",
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# E4: Cascade Threshold Sweep
# ─────────────────────────────────────────────────────────────
def cascade_sweep(samples, dataset_name, n_thresholds=50):
    """SE-gated cascade: SE < τ → Energy, else → SE"""
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    if len(np.unique(labels)) < 2:
        return []

    # Normalize SE and Energy to [0, 1] range (higher = more likely hallucination)
    se_norm = (entropies - entropies.min()) / (
        entropies.max() - entropies.min() + 1e-10
    )
    # Energy: higher (less negative) = less confident = more likely hallucination
    energy_norm = (energies - energies.min()) / (
        energies.max() - energies.min() + 1e-10
    )

    # Baselines
    se_auroc = roc_auc_score(labels, se_norm)
    energy_auroc = roc_auc_score(labels, energy_norm)
    se_auprc = average_precision_score(labels, se_norm)
    energy_auprc = average_precision_score(labels, energy_norm)

    # Sweep τ on raw SE values
    tau_values = np.linspace(0, np.percentile(entropies, 95), n_thresholds)
    # Add key thresholds
    tau_values = np.unique(
        np.concatenate([tau_values, [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]])
    )
    tau_values.sort()

    results = []
    for tau in tau_values:
        # Cascade score
        cascade_score = np.where(entropies <= tau, energy_norm, se_norm)

        try:
            cascade_auroc = roc_auc_score(labels, cascade_score)
            cascade_auprc = average_precision_score(labels, cascade_score)
        except ValueError:
            continue

        n_energy_used = (entropies <= tau).sum()
        n_se_used = len(samples) - n_energy_used

        results.append(
            {
                "tau": float(tau),
                "cascade_auroc": float(cascade_auroc),
                "cascade_auprc": float(cascade_auprc),
                "se_auroc": float(se_auroc),
                "energy_auroc": float(energy_auroc),
                "se_auprc": float(se_auprc),
                "energy_auprc": float(energy_auprc),
                "delta_auroc_vs_se": float(cascade_auroc - se_auroc),
                "delta_auroc_vs_energy": float(cascade_auroc - energy_auroc),
                "n_energy_used": int(n_energy_used),
                "n_se_used": int(n_se_used),
                "pct_energy_used": float(n_energy_used / len(samples) * 100),
            }
        )

    return results


def find_best_tau(sweep_results):
    """Sweep 결과에서 최적 τ 찾기"""
    if not sweep_results:
        return None
    return max(sweep_results, key=lambda x: x["cascade_auroc"])


# ─────────────────────────────────────────────────────────────
# 상보성 분석
# ─────────────────────────────────────────────────────────────
def complementarity_analysis(
    samples, dataset_name, se_threshold_pct=80, energy_threshold_pct=80
):
    """SE와 Energy가 각각 잡는 것이 다른지 분석"""
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    if labels.sum() == 0 or labels.sum() == len(labels):
        return None

    # Threshold at given percentile of hallucination scores
    hall_se = entropies[labels == 1]
    hall_energy = energies[labels == 1]

    se_thresh = (
        np.percentile(hall_se, 100 - se_threshold_pct) if len(hall_se) > 0 else 0
    )
    energy_thresh = (
        np.percentile(hall_energy, 100 - energy_threshold_pct)
        if len(hall_energy) > 0
        else 0
    )

    # SE predicts hallucination if SE > threshold
    se_pred = entropies > se_thresh
    # Energy predicts hallucination if Energy > threshold (less negative = less confident)
    energy_pred = energies > energy_thresh

    hall_mask = labels == 1

    se_catches = se_pred & hall_mask
    energy_catches = energy_pred & hall_mask
    both_catch = se_catches & energy_catches
    neither_catch = ~se_catches & ~energy_catches & hall_mask

    # Oracle: either catches it
    oracle_catches = se_catches | energy_catches

    return {
        "dataset": dataset_name,
        "n_hallucinations": int(hall_mask.sum()),
        "se_catches": int(se_catches.sum()),
        "energy_catches": int(energy_catches.sum()),
        "both_catch": int(both_catch.sum()),
        "se_only": int((se_catches & ~energy_catches).sum()),
        "energy_only": int((energy_catches & ~se_catches).sum()),
        "neither_catch": int(neither_catch.sum()),
        "oracle_catches": int(oracle_catches.sum()),
        "oracle_catch_rate": float(oracle_catches.sum() / hall_mask.sum() * 100),
    }


# ─────────────────────────────────────────────────────────────
# Cross-dataset τ Transfer
# ─────────────────────────────────────────────────────────────
def cross_dataset_tau_transfer(all_datasets):
    """한 데이터셋에서 최적 τ를 찾아 다른 데이터셋에 적용"""
    results = []
    dataset_names = list(all_datasets.keys())

    for train_name in dataset_names:
        # Find best τ on train dataset
        sweep = cascade_sweep(all_datasets[train_name], train_name)
        best = find_best_tau(sweep)
        if best is None:
            continue

        best_tau = best["tau"]
        train_auroc = best["cascade_auroc"]

        for test_name in dataset_names:
            # Apply τ to test dataset
            test_samples = all_datasets[test_name]
            labels = np.array([s["is_hallucination"] for s in test_samples])
            entropies = np.array([s["semantic_entropy"] for s in test_samples])
            energies = np.array([s["semantic_energy"] for s in test_samples])

            if len(np.unique(labels)) < 2:
                continue

            se_norm = (entropies - entropies.min()) / (
                entropies.max() - entropies.min() + 1e-10
            )
            energy_norm = (energies - energies.min()) / (
                energies.max() - energies.min() + 1e-10
            )

            cascade_score = np.where(entropies <= best_tau, energy_norm, se_norm)
            cascade_auroc = roc_auc_score(labels, cascade_score)

            se_only_auroc = roc_auc_score(labels, se_norm)
            energy_only_auroc = roc_auc_score(labels, energy_norm)

            results.append(
                {
                    "train_dataset": train_name,
                    "test_dataset": test_name,
                    "tau": float(best_tau),
                    "cascade_auroc": float(cascade_auroc),
                    "se_only_auroc": float(se_only_auroc),
                    "energy_only_auroc": float(energy_only_auroc),
                    "delta_vs_se": float(cascade_auroc - se_only_auroc),
                    "delta_vs_energy": float(cascade_auroc - energy_only_auroc),
                    "is_cross": train_name != test_name,
                }
            )

    return results


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E1/E3/E4: Zero-SE 현상 정량화 + SE 구간별 분석 + Cascade Sweep")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] 기존 결과 로드...")
    all_datasets = load_existing_results()

    # 2. 새 데이터셋 결과 로드 (있으면)
    print("\n[2] 새 데이터셋 결과 로드...")
    new_datasets = load_new_results()
    all_datasets.update(new_datasets)

    if not all_datasets:
        print("ERROR: No datasets found!")
        return

    print(f"\n총 {len(all_datasets)}개 데이터셋: {list(all_datasets.keys())}")

    # === E1: Zero-SE 정량화 ===
    print("\n" + "=" * 70)
    print("E1: Zero-SE 정량화")
    print("=" * 70)

    all_zero_se_results = []
    for name, samples in all_datasets.items():
        results = analyze_zero_se(samples, name)
        all_zero_se_results.extend(results)

        print(f"\n--- {name} ---")
        print(
            f"{'ε':>8} {'Zero-SE%':>10} {'n':>5} {'Hall%':>8} {'Energy AUROC':>14} {'SE AUROC':>10}"
        )
        print("-" * 60)
        for r in results:
            energy_str = (
                f"{r['energy_auroc_in_zero_se']:.4f}"
                if r["energy_auroc_in_zero_se"]
                else "N/A"
            )
            se_str = (
                f"{r['se_auroc_in_zero_se']:.4f}" if r["se_auroc_in_zero_se"] else "N/A"
            )
            hall_str = (
                f"{r['hall_rate_in_zero_se']:.1f}%"
                if r["hall_rate_in_zero_se"] is not None
                else "N/A"
            )
            print(
                f"{r['epsilon']:>8.3f} {r['zero_se_pct']:>9.1f}% {r['n_zero_se']:>5} {hall_str:>8} {energy_str:>14} {se_str:>10}"
            )

    # === E3: SE 구간별 분석 ===
    print("\n" + "=" * 70)
    print("E3: SE 구간별 분석")
    print("=" * 70)

    all_bin_results = []
    for name, samples in all_datasets.items():
        results = analyze_se_bins(samples, name)
        all_bin_results.extend(results)

        print(f"\n--- {name} ---")
        print(
            f"{'Bin':>12} {'n':>5} {'Hall%':>8} {'SE AUROC':>10} {'Energy AUROC':>14} {'Winner':>8}"
        )
        print("-" * 65)
        for r in results:
            se_str = f"{r.get('se_auroc', 0):.4f}" if r.get("se_auroc") else "N/A"
            en_str = (
                f"{r.get('energy_auroc', 0):.4f}" if r.get("energy_auroc") else "N/A"
            )
            hall_str = (
                f"{r.get('hall_rate', 0):.1f}%"
                if r.get("hall_rate") is not None
                else "N/A"
            )
            print(
                f"{r['bin']:>12} {r['n']:>5} {hall_str:>8} {se_str:>10} {en_str:>14} {r.get('winner', 'N/A'):>8}"
            )

    # === E4: Cascade Sweep ===
    print("\n" + "=" * 70)
    print("E4: Cascade Threshold Sweep")
    print("=" * 70)

    all_sweep_results = {}
    for name, samples in all_datasets.items():
        sweep = cascade_sweep(samples, name)
        all_sweep_results[name] = sweep

        best = find_best_tau(sweep)
        if best:
            print(f"\n--- {name} ---")
            print(f"  Best τ = {best['tau']:.4f}")
            print(f"  Cascade AUROC: {best['cascade_auroc']:.4f}")
            print(f"  SE-only AUROC: {best['se_auroc']:.4f}")
            print(f"  Energy-only AUROC: {best['energy_auroc']:.4f}")
            print(f"  Δ vs SE: {best['delta_auroc_vs_se']:+.4f}")
            print(f"  Δ vs Energy: {best['delta_auroc_vs_energy']:+.4f}")
            print(f"  Energy used: {best['pct_energy_used']:.1f}%")

    # === Cross-dataset τ Transfer ===
    print("\n" + "=" * 70)
    print("Cross-dataset τ Transfer")
    print("=" * 70)

    transfer_results = cross_dataset_tau_transfer(all_datasets)
    print(
        f"\n{'Train→Test':>30} {'τ':>8} {'Cascade':>10} {'SE-only':>10} {'Energy-only':>12} {'ΔvsSE':>8}"
    )
    print("-" * 80)
    for r in transfer_results:
        arrow = f"{r['train_dataset']}→{r['test_dataset']}"
        cross = " *" if r["is_cross"] else ""
        print(
            f"{arrow:>30} {r['tau']:>8.4f} {r['cascade_auroc']:>10.4f} {r['se_only_auroc']:>10.4f} {r['energy_only_auroc']:>12.4f} {r['delta_vs_se']:>+8.4f}{cross}"
        )

    # === 상보성 분석 ===
    print("\n" + "=" * 70)
    print("상보성 분석")
    print("=" * 70)

    all_comp_results = []
    for name, samples in all_datasets.items():
        comp = complementarity_analysis(samples, name)
        if comp:
            all_comp_results.append(comp)
            print(f"\n--- {name} ---")
            print(f"  전체 환각: {comp['n_hallucinations']}")
            print(
                f"  SE만 탐지:     {comp['se_only']} ({comp['se_only'] / comp['n_hallucinations'] * 100:.1f}%)"
            )
            print(
                f"  Energy만 탐지: {comp['energy_only']} ({comp['energy_only'] / comp['n_hallucinations'] * 100:.1f}%)"
            )
            print(
                f"  둘 다 탐지:    {comp['both_catch']} ({comp['both_catch'] / comp['n_hallucinations'] * 100:.1f}%)"
            )
            print(
                f"  못 잡음:       {comp['neither_catch']} ({comp['neither_catch'] / comp['n_hallucinations'] * 100:.1f}%)"
            )
            print(f"  Oracle 탐지율: {comp['oracle_catch_rate']:.1f}%")

    # === 결과 저장 ===
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "datasets": list(all_datasets.keys()),
        "zero_se_analysis": all_zero_se_results,
        "se_bin_analysis": all_bin_results,
        "cascade_sweep": {k: v for k, v in all_sweep_results.items()},
        "cross_dataset_transfer": transfer_results,
        "complementarity": all_comp_results,
    }

    output_path = Path(__file__).parent / "analysis_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
