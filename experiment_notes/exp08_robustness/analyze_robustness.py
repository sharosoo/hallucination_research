#!/usr/bin/env python
"""
exp08: 실험 결과 보강 분석 (GPU 불필요)

기존 exp01/exp02/exp07 결과 JSON을 재분석하여:
1. 상보성 분석 threshold sensitivity (60/70/80/90th percentile)
2. Cascade AUROC delta에 대한 paired bootstrap 통계 검정
3. Zero-SE = single-cluster 확인 (num_clusters==1 vs SE≤ε 비교)
4. Cascade 정규화 대안 (rank-based normalization으로 leakage 제거)

출력: exp08_robustness/robustness_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
EXP_DIR = ROOT / "experiment_notes"


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
def load_all_datasets():
    """모든 실험 결과 로드"""
    datasets = {}

    # exp01 TruthfulQA
    p = EXP_DIR / "exp01_truthfulqa" / "results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        datasets["TruthfulQA"] = d["samples"]
        print(f"  TruthfulQA: {len(d['samples'])} samples")

    # exp02 HaluEval
    p = EXP_DIR / "exp02_halueval" / "results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        datasets["HaluEval"] = d["samples"]
        print(f"  HaluEval: {len(d['samples'])} samples")

    # exp07 new datasets — prefer _llm_judge.json when available
    exp07_dir = EXP_DIR / "exp07_zero_se_analysis"
    all_files = sorted(exp07_dir.glob("results_*.json"))

    llm_judge_bases = set()
    for f in all_files:
        if f.stem.endswith("_llm_judge"):
            llm_judge_bases.add(f.stem.replace("_llm_judge", ""))

    for json_file in all_files:
        stem = json_file.stem
        if not stem.endswith("_llm_judge") and stem in llm_judge_bases:
            print(f"  (skipped {json_file.name} — _llm_judge version exists)")
            continue

        with open(json_file) as f:
            d = json.load(f)
        name = d.get(
            "dataset_name", stem.replace("results_", "").replace("_llm_judge", "")
        )
        datasets[name] = d["samples"]
        label_src = "LLM-judge" if stem.endswith("_llm_judge") else "original"
        print(f"  {name}: {len(d['samples'])} samples ({label_src})")

    return datasets


# ─────────────────────────────────────────────────────────────
# 1. 상보성 분석 Threshold Sensitivity
# ─────────────────────────────────────────────────────────────
def complementarity_sensitivity(samples, dataset_name, percentiles=(60, 70, 80, 90)):
    """다양한 percentile threshold에서 상보성 분석 반복"""
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    if labels.sum() == 0 or labels.sum() == len(labels):
        return []

    results = []
    for pct in percentiles:
        hall_se = entropies[labels == 1]
        hall_energy = energies[labels == 1]

        se_thresh = np.percentile(hall_se, 100 - pct) if len(hall_se) > 0 else 0
        energy_thresh = (
            np.percentile(hall_energy, 100 - pct) if len(hall_energy) > 0 else 0
        )

        se_pred = entropies > se_thresh
        energy_pred = energies > energy_thresh
        hall_mask = labels == 1

        se_catches = se_pred & hall_mask
        energy_catches = energy_pred & hall_mask
        both_catch = se_catches & energy_catches
        neither_catch = ~se_catches & ~energy_catches & hall_mask
        se_only = se_catches & ~energy_catches
        energy_only = energy_catches & ~se_catches

        n_hall = hall_mask.sum()
        results.append(
            {
                "dataset": dataset_name,
                "percentile": pct,
                "n_hallucinations": int(n_hall),
                "se_only": int(se_only.sum()),
                "se_only_pct": float(se_only.sum() / n_hall * 100),
                "energy_only": int(energy_only.sum()),
                "energy_only_pct": float(energy_only.sum() / n_hall * 100),
                "both_catch": int(both_catch.sum()),
                "both_catch_pct": float(both_catch.sum() / n_hall * 100),
                "neither_catch": int(neither_catch.sum()),
                "neither_catch_pct": float(neither_catch.sum() / n_hall * 100),
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# 2. Paired Bootstrap 통계 검정 (Cascade vs SE-only AUROC)
# ─────────────────────────────────────────────────────────────
def paired_bootstrap_test(labels, se_scores, cascade_scores, n_bootstrap=5000, seed=42):
    """
    Paired bootstrap test: H0: AUROC(cascade) <= AUROC(se)
    Returns p-value and CI for delta.
    """
    labels = np.array(labels)
    se_scores = np.array(se_scores)
    cascade_scores = np.array(cascade_scores)

    if len(np.unique(labels)) < 2:
        return None

    base_se_auroc = roc_auc_score(labels, se_scores)
    base_cascade_auroc = roc_auc_score(labels, cascade_scores)
    observed_delta = base_cascade_auroc - base_se_auroc

    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(labels), len(labels))
        boot_labels = labels[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        try:
            se_auc = roc_auc_score(boot_labels, se_scores[idx])
            cas_auc = roc_auc_score(boot_labels, cascade_scores[idx])
            deltas.append(cas_auc - se_auc)
        except ValueError:
            continue

    if len(deltas) < 100:
        return None

    deltas = np.array(deltas)
    # Two-sided p-value: proportion of bootstrap deltas with opposite sign
    p_value = np.mean(deltas <= 0) if observed_delta > 0 else np.mean(deltas >= 0)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    mean_delta = np.mean(deltas)

    return {
        "observed_delta": float(observed_delta),
        "bootstrap_mean_delta": float(mean_delta),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_010": bool(p_value < 0.10),
        "n_bootstrap_valid": len(deltas),
    }


def cascade_with_bootstrap(samples, dataset_name, tau):
    """특정 τ에서 cascade vs SE-only AUROC delta의 bootstrap 검정"""
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    if len(np.unique(labels)) < 2:
        return None

    # Normalize
    se_norm = (entropies - entropies.min()) / (
        entropies.max() - entropies.min() + 1e-10
    )
    energy_norm = (energies - energies.min()) / (
        energies.max() - energies.min() + 1e-10
    )

    cascade_score = np.where(entropies <= tau, energy_norm, se_norm)

    result = paired_bootstrap_test(labels, se_norm, cascade_score)
    if result is None:
        return None

    result["dataset"] = dataset_name
    result["tau"] = float(tau)
    result["se_auroc"] = float(roc_auc_score(labels, se_norm))
    result["cascade_auroc"] = float(roc_auc_score(labels, cascade_score))
    return result


# ─────────────────────────────────────────────────────────────
# 3. Zero-SE = Single-Cluster 확인
# ─────────────────────────────────────────────────────────────
def verify_zero_se_is_single_cluster(samples, dataset_name, epsilon=0.001):
    """SE≤ε 인 샘플과 num_clusters==1 인 샘플이 동일한지 확인"""
    n_total = len(samples)
    se_zero_mask = [s["semantic_entropy"] <= epsilon for s in samples]
    single_cluster_mask = [s["num_clusters"] == 1 for s in samples]

    n_se_zero = sum(se_zero_mask)
    n_single_cluster = sum(single_cluster_mask)
    n_overlap = sum(a and b for a, b in zip(se_zero_mask, single_cluster_mask))

    # SE=0 but multi-cluster (should not exist mathematically)
    se_zero_multi = sum(a and not b for a, b in zip(se_zero_mask, single_cluster_mask))
    # Single cluster but SE>ε (should not exist either)
    single_but_nonzero = sum(
        not a and b for a, b in zip(se_zero_mask, single_cluster_mask)
    )

    # Also check: what cluster counts appear?
    cluster_counts = {}
    for s in samples:
        nc = s["num_clusters"]
        if nc not in cluster_counts:
            cluster_counts[nc] = {"total": 0, "hall": 0, "normal": 0}
        cluster_counts[nc]["total"] += 1
        if s["is_hallucination"]:
            cluster_counts[nc]["hall"] += 1
        else:
            cluster_counts[nc]["normal"] += 1

    return {
        "dataset": dataset_name,
        "n_total": n_total,
        "n_se_zero_eps001": n_se_zero,
        "n_single_cluster": n_single_cluster,
        "n_overlap": n_overlap,
        "exact_match": n_se_zero == n_single_cluster == n_overlap,
        "se_zero_but_multi_cluster": se_zero_multi,
        "single_cluster_but_se_nonzero": single_but_nonzero,
        "cluster_distribution": {str(k): v for k, v in sorted(cluster_counts.items())},
    }


# ─────────────────────────────────────────────────────────────
# 4. Rank-Based Normalization (Leakage-Free) Cascade
# ─────────────────────────────────────────────────────────────
def cascade_rank_normalized(samples, dataset_name, tau):
    """
    Rank-based normalization으로 min-max leakage 없는 cascade.
    각 score를 rank/N으로 변환 — 분포에 무관하게 [0,1] 범위.
    """
    labels = np.array([s["is_hallucination"] for s in samples])
    entropies = np.array([s["semantic_entropy"] for s in samples])
    energies = np.array([s["semantic_energy"] for s in samples])

    if len(np.unique(labels)) < 2:
        return None

    n = len(labels)
    # Rank normalization: rank(x) / N
    se_ranks = entropies.argsort().argsort() / n
    energy_ranks = energies.argsort().argsort() / n

    # Min-max (original)
    se_minmax = (entropies - entropies.min()) / (
        entropies.max() - entropies.min() + 1e-10
    )
    energy_minmax = (energies - energies.min()) / (
        energies.max() - energies.min() + 1e-10
    )

    # Cascade scores
    cascade_minmax = np.where(entropies <= tau, energy_minmax, se_minmax)
    cascade_rank = np.where(entropies <= tau, energy_ranks, se_ranks)

    se_auroc_minmax = roc_auc_score(labels, se_minmax)
    se_auroc_rank = roc_auc_score(labels, se_ranks)
    cascade_auroc_minmax = roc_auc_score(labels, cascade_minmax)
    cascade_auroc_rank = roc_auc_score(labels, cascade_rank)

    return {
        "dataset": dataset_name,
        "tau": float(tau),
        "se_auroc_minmax": float(se_auroc_minmax),
        "se_auroc_rank": float(se_auroc_rank),
        "cascade_auroc_minmax": float(cascade_auroc_minmax),
        "cascade_auroc_rank": float(cascade_auroc_rank),
        "delta_minmax": float(cascade_auroc_minmax - se_auroc_minmax),
        "delta_rank": float(cascade_auroc_rank - se_auroc_rank),
        "rank_vs_minmax_consistent": bool(
            (cascade_auroc_minmax - se_auroc_minmax)
            * (cascade_auroc_rank - se_auroc_rank)
            >= 0
        ),
    }


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("exp08: Robustness Analysis (GPU 불필요)")
    print("=" * 70)

    print("\n[1] 데이터 로드...")
    all_datasets = load_all_datasets()
    print(f"  총 {len(all_datasets)}개 데이터셋")

    # Cross-dataset τ from TruthfulQA
    TAU_TRANSFER = 0.526

    # ─── 1. 상보성 Threshold Sensitivity ───
    print("\n" + "=" * 70)
    print("1. Complementarity Threshold Sensitivity")
    print("=" * 70)

    all_comp_sensitivity = []
    for name, samples in all_datasets.items():
        results = complementarity_sensitivity(samples, name)
        all_comp_sensitivity.extend(results)

        print(f"\n--- {name} ---")
        print(
            f"  {'Pct':>5} {'SE-only':>10} {'Energy-only':>13} {'Both':>10} {'Neither':>10}"
        )
        print("  " + "-" * 50)
        for r in results:
            print(
                f"  {r['percentile']:>5} "
                f"{r['se_only_pct']:>9.1f}% "
                f"{r['energy_only_pct']:>12.1f}% "
                f"{r['both_catch_pct']:>9.1f}% "
                f"{r['neither_catch_pct']:>9.1f}%"
            )

    # ─── 2. Paired Bootstrap Test ───
    print("\n" + "=" * 70)
    print("2. Paired Bootstrap Test (Cascade vs SE-only)")
    print("=" * 70)

    # Test with: (a) best-τ per dataset, (b) cross-dataset τ=0.526
    # For best-τ, we load from exp07 analysis
    with open(EXP_DIR / "exp07_zero_se_analysis" / "analysis_results.json") as f:
        exp07_data = json.load(f)

    bootstrap_results = []
    for name, samples in all_datasets.items():
        # Best τ per dataset
        sweep = exp07_data["cascade_sweep"].get(name, [])
        if sweep:
            best = max(sweep, key=lambda s: s["cascade_auroc"])
            best_tau = best["tau"]
        else:
            best_tau = TAU_TRANSFER

        # (a) Best τ
        res_best = cascade_with_bootstrap(samples, name, best_tau)
        if res_best:
            res_best["tau_source"] = "best_per_dataset"
            bootstrap_results.append(res_best)
            print(f"\n--- {name} (best τ={best_tau:.3f}) ---")
            print(f"  Δ AUROC: {res_best['observed_delta']:+.4f}")
            print(
                f"  95% CI: [{res_best['ci_95_lower']:+.4f}, {res_best['ci_95_upper']:+.4f}]"
            )
            print(
                f"  p-value: {res_best['p_value']:.4f}"
                f"  {'✅ sig@0.05' if res_best['significant_at_005'] else '❌ not sig@0.05'}"
            )

        # (b) Cross-dataset τ=0.526
        res_cross = cascade_with_bootstrap(samples, name, TAU_TRANSFER)
        if res_cross:
            res_cross["tau_source"] = "cross_dataset_0.526"
            bootstrap_results.append(res_cross)
            print(f"\n--- {name} (cross τ=0.526) ---")
            print(f"  Δ AUROC: {res_cross['observed_delta']:+.4f}")
            print(
                f"  95% CI: [{res_cross['ci_95_lower']:+.4f}, {res_cross['ci_95_upper']:+.4f}]"
            )
            print(
                f"  p-value: {res_cross['p_value']:.4f}"
                f"  {'✅ sig@0.05' if res_cross['significant_at_005'] else '❌ not sig@0.05'}"
            )

    # ─── 3. Zero-SE = Single-Cluster 확인 ───
    print("\n" + "=" * 70)
    print("3. Zero-SE = Single-Cluster Verification")
    print("=" * 70)

    zero_se_verification = []
    for name, samples in all_datasets.items():
        result = verify_zero_se_is_single_cluster(samples, name)
        zero_se_verification.append(result)
        print(f"\n--- {name} ---")
        print(
            f"  SE≤0.001: {result['n_se_zero_eps001']}, "
            f"Single-cluster: {result['n_single_cluster']}, "
            f"Overlap: {result['n_overlap']}"
        )
        print(f"  Exact match: {'✅' if result['exact_match'] else '❌'}")
        print(f"  Cluster distribution:")
        for k, v in result["cluster_distribution"].items():
            hall_pct = v["hall"] / v["total"] * 100 if v["total"] > 0 else 0
            print(
                f"    K={k}: {v['total']} samples (hall={v['hall']}, {hall_pct:.1f}%)"
            )

    # ─── 4. Rank-Based Normalization ───
    print("\n" + "=" * 70)
    print("4. Rank-Based Normalization vs Min-Max")
    print("=" * 70)

    rank_results = []
    for name, samples in all_datasets.items():
        # Best τ
        sweep = exp07_data["cascade_sweep"].get(name, [])
        if sweep:
            best = max(sweep, key=lambda s: s["cascade_auroc"])
            best_tau = best["tau"]
        else:
            best_tau = TAU_TRANSFER

        res = cascade_rank_normalized(samples, name, best_tau)
        if res:
            rank_results.append(res)
            print(f"\n--- {name} (τ={best_tau:.3f}) ---")
            print(
                f"  Min-Max:  Cascade={res['cascade_auroc_minmax']:.4f}, "
                f"SE={res['se_auroc_minmax']:.4f}, Δ={res['delta_minmax']:+.4f}"
            )
            print(
                f"  Rank:     Cascade={res['cascade_auroc_rank']:.4f}, "
                f"SE={res['se_auroc_rank']:.4f}, Δ={res['delta_rank']:+.4f}"
            )
            print(
                f"  Consistent direction: "
                f"{'✅' if res['rank_vs_minmax_consistent'] else '❌'}"
            )

        # Also cross-dataset τ
        res_cross = cascade_rank_normalized(samples, name, TAU_TRANSFER)
        if res_cross:
            res_cross["tau_source"] = "cross_dataset_0.526"
            rank_results.append(res_cross)

    # ─── 결과 저장 ───
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "description": "exp08: Robustness analysis for exp07 claims",
        "complementarity_sensitivity": all_comp_sensitivity,
        "bootstrap_tests": bootstrap_results,
        "zero_se_single_cluster_verification": zero_se_verification,
        "rank_normalization_comparison": rank_results,
    }

    output_path = Path(__file__).parent / "robustness_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")

    # ─── 요약 ───
    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)

    # Complementarity: Energy-only range across thresholds
    print("\n[상보성] Energy-only 환각 비율 범위:")
    for name in all_datasets:
        ds_results = [r for r in all_comp_sensitivity if r["dataset"] == name]
        if ds_results:
            eonly_min = min(r["energy_only_pct"] for r in ds_results)
            eonly_max = max(r["energy_only_pct"] for r in ds_results)
            print(
                f"  {name}: {eonly_min:.1f}% ~ {eonly_max:.1f}% "
                f"(across {len(ds_results)} thresholds)"
            )

    # Bootstrap significance summary
    print("\n[통계 검정] Cascade vs SE-only:")
    for r in bootstrap_results:
        sig = "✅" if r["significant_at_005"] else "❌"
        print(
            f"  {r['dataset']} (τ={r['tau']:.3f}, {r.get('tau_source', '?')}): "
            f"Δ={r['observed_delta']:+.4f}, p={r['p_value']:.3f} {sig}"
        )

    # Zero-SE verification
    print("\n[Zero-SE 검증] SE=0 ↔ 단일 클러스터:")
    for r in zero_se_verification:
        print(
            f"  {r['dataset']}: {'✅ 완전 일치' if r['exact_match'] else '❌ 불일치'}"
        )

    # Rank normalization consistency
    print("\n[정규화] Rank vs Min-Max 방향 일관성:")
    for r in rank_results:
        if "tau_source" not in r:  # best-τ only
            consistent = "✅" if r["rank_vs_minmax_consistent"] else "❌"
            print(
                f"  {r['dataset']}: {consistent} "
                f"(MinMax Δ={r['delta_minmax']:+.4f}, "
                f"Rank Δ={r['delta_rank']:+.4f})"
            )


if __name__ == "__main__":
    main()
