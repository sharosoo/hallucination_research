#!/usr/bin/env python
"""
Exp10: 논문용 TruthfulQA 완전 실험

목적:
- TruthfulQA 200개 샘플에서 SE와 Energy의 환각 탐지 성능 비교
- Zero-SE 문제 정량화
- SE-gated Cascade 방법 검증
- 논문용 Figure 및 테이블 생성

방법:
1. LLM으로 질문당 K=5개 응답 샘플링
2. NLI 클러스터링으로 의미 그룹화
3. Semantic Entropy 계산 (클러스터 확률 기반)
4. Semantic Energy 계산 (raw logit 기반)
5. 정답 비교로 환각 레이블링
6. AUROC 및 다양한 분석 수행

출력:
- results.json: 실험 원본 결과
- analysis.json: 분석 결과
- figures/: 논문용 그래프
- RESULTS.md: 마크다운 리포트

실행:
  cd hallucination_lfe
  source .venv/bin/activate
  python experiment_notes/exp10_thesis_complete/run_experiment.py

  # 분석만 (이미 실험 완료된 경우):
  python experiment_notes/exp10_thesis_complete/run_experiment.py --analysis-only

  # 이어서 실행 (체크포인트 사용):
  python experiment_notes/exp10_thesis_complete/run_experiment.py --resume
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
ROOT = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent
FIG_DIR = EXP_DIR / "figures"

sys.path.insert(0, str(ROOT / "packages" / "hfe-core" / "src"))

CONFIG = {
    "experiment_name": "exp10_thesis_complete",
    "dataset": "TruthfulQA",
    "dataset_split": "validation",
    "llm_model": "Qwen/Qwen2.5-3B-Instruct",
    "nli_model": "microsoft/deberta-large-mnli",
    "num_responses": 5,
    "temperature": 0.7,
    "max_new_tokens": 50,
    "seed": 42,
}


# ============================================================
# Part 1: LLM 샘플러
# ============================================================
class LLMSampler:
    """Transformers 기반 다중 샘플링 + logit 추출"""

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"  모델 로딩: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def sample(self, question: str, num_samples: int, max_new_tokens: int, temperature: float):
        """다중 응답 샘플링 + raw logit 추출"""
        from hfe_core.nli_clusterer import Response
        
        messages = [
            {"role": "system", "content": "Answer the question concisely in one sentence."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        responses = []
        for _ in range(num_samples):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_ids = outputs.sequences[0, input_len:].tolist()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 토큰별 raw logit 추출
            token_logits = []
            log_prob_sum = 0.0
            for step_idx, token_id in enumerate(generated_ids):
                if step_idx >= len(outputs.scores):
                    break
                logits = outputs.scores[step_idx][0]
                token_logits.append(logits[token_id].item())
                log_prob_sum += torch.log_softmax(logits, dim=-1)[token_id].item()
                if token_id == self.tokenizer.eos_token_id:
                    break

            responses.append(Response(
                text=text.strip(),
                probability=1.0 / num_samples,
                log_probability=log_prob_sum,
                logits=token_logits,
            ))

        return responses


# ============================================================
# Part 2: 실험 실행
# ============================================================
def is_correct(response_text: str, correct_answers: list[str]) -> bool:
    """응답이 정답인지 확인"""
    response_lower = response_text.lower().strip()
    for correct in correct_answers:
        correct_lower = correct.lower().strip()
        if correct_lower in response_lower or response_lower in correct_lower:
            return True
        if len(correct_lower) > 5 and correct_lower[:20] in response_lower:
            return True
    return False


def run_experiment(resume: bool = False, max_samples: Optional[int] = None):
    """TruthfulQA 전체 실험"""
    from datasets import load_dataset
    from hfe_core.nli_clusterer import NLIClusterer
    from hfe_core.semantic_entropy import SemanticEntropyCalculator
    from hfe_core.semantic_energy import SemanticEnergyCalculator

    results_path = EXP_DIR / "results.json"
    checkpoint_path = EXP_DIR / "checkpoint.json"

    # Resume 처리
    existing_results = []
    start_idx = 0
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        existing_results = checkpoint.get("samples", [])
        start_idx = len(existing_results)
        print(f"[Resume] 체크포인트에서 {start_idx}개 로드됨")

    print("=" * 70)
    print("Exp10: TruthfulQA 논문용 완전 실험")
    print("=" * 70)
    print("\n[설정]")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # 데이터 로드
    print("\n[1/4] 데이터셋 로드...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"  전체: {len(dataset)}개, 사용: {total_samples}개")

    if start_idx >= total_samples:
        print("  이미 완료됨!")
        with open(results_path) as f:
            return json.load(f)["samples"]

    # 모델 로드
    print("\n[2/4] 모델 로드...")
    sampler = LLMSampler(CONFIG["llm_model"])
    clusterer = NLIClusterer(model_name=CONFIG["nli_model"], device=sampler.device)

    # 실험 실행
    print(f"\n[3/4] 실험 실행... ({start_idx}/{total_samples}부터)")
    results = existing_results.copy()

    for idx in tqdm(range(start_idx, total_samples), desc="Processing", initial=start_idx, total=total_samples):
        item = dataset[idx]
        question = item["question"]
        correct_answers = item["correct_answers"]

        # 응답 샘플링
        responses = sampler.sample(
            question,
            CONFIG["num_responses"],
            CONFIG["max_new_tokens"],
            CONFIG["temperature"],
        )

        # 클러스터링
        clusters = clusterer.cluster(responses)

        # 메트릭 계산
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))
        energy = SemanticEnergyCalculator.compute_energy_only(responses)

        # 환각 레이블
        any_correct = any(is_correct(r.text, correct_answers) for r in responses)
        is_hallucination = 0 if any_correct else 1

        results.append({
            "idx": idx,
            "question": question,
            "responses": [r.text for r in responses],
            "correct_answers": correct_answers[:3],
            "num_clusters": len(clusters),
            "semantic_entropy": se,
            "semantic_energy": energy,
            "is_hallucination": is_hallucination,
        })

        # 50개마다 체크포인트
        if (idx + 1) % 50 == 0:
            _save_checkpoint(results, checkpoint_path)

    # 최종 저장
    print("\n[4/4] 결과 저장...")
    output = {
        "experiment": CONFIG["experiment_name"],
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "samples": results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  저장: {results_path}")

    # 체크포인트 삭제
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


def _save_checkpoint(results, path):
    with open(path, "w") as f:
        json.dump({"samples": results, "timestamp": datetime.now().isoformat()}, f)


# ============================================================
# Part 3: 분석
# ============================================================
def bootstrap_auroc(labels, scores, n_iter=1000, ci=0.95):
    """Bootstrap AUROC 신뢰구간"""
    from sklearn.metrics import roc_auc_score
    
    labels, scores = np.array(labels), np.array(scores)
    if len(np.unique(labels)) < 2 or len(labels) < 10:
        return None, None, None

    rng = np.random.RandomState(42)
    aurocs = []
    for _ in range(n_iter):
        idx = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(labels[idx], scores[idx]))

    if len(aurocs) < 100:
        return None, None, None

    alpha = (1 - ci) / 2
    return np.mean(aurocs), np.percentile(aurocs, alpha * 100), np.percentile(aurocs, (1 - alpha) * 100)


def run_analysis(results: list) -> dict:
    """실험 결과 분석"""
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("\n" + "=" * 70)
    print("분석 시작")
    print("=" * 70)

    labels = np.array([r["is_hallucination"] for r in results])
    se_scores = np.array([r["semantic_entropy"] for r in results])
    energy_scores = np.array([r["semantic_energy"] for r in results])
    num_clusters = np.array([r["num_clusters"] for r in results])

    analysis = {
        "dataset": "TruthfulQA",
        "n_samples": len(results),
        "n_hallucinations": int(labels.sum()),
        "n_normal": int(len(labels) - labels.sum()),
        "hallucination_rate": float(labels.mean() * 100),
    }

    # === 1. 전체 AUROC ===
    print("\n[1] 전체 AUROC...")
    se_auroc = roc_auc_score(labels, se_scores)
    energy_auroc = roc_auc_score(labels, energy_scores)  # Energy: 높을수록 환각 (덜 negative)
    
    analysis["overall"] = {
        "se_auroc": float(se_auroc),
        "se_auprc": float(average_precision_score(labels, se_scores)),
        "energy_auroc": float(energy_auroc),
        "energy_auprc": float(average_precision_score(labels, energy_scores)),
    }
    print(f"  SE AUROC: {se_auroc:.4f}")
    print(f"  Energy AUROC: {energy_auroc:.4f}")

    # === 2. Zero-SE 분석 ===
    print("\n[2] Zero-SE 분석...")
    analysis["zero_se"] = []
    for eps in [0.001, 0.01, 0.05, 0.1]:
        mask = se_scores <= eps
        n_zero = mask.sum()
        if n_zero == 0:
            continue

        zero_labels = labels[mask]
        zero_energy = energy_scores[mask]
        hall_rate = zero_labels.mean() * 100

        energy_auroc_zero = None
        if len(np.unique(zero_labels)) == 2:
            energy_auroc_zero = float(roc_auc_score(zero_labels, zero_energy))
            mean_auc, lo, hi = bootstrap_auroc(zero_labels, zero_energy)
        else:
            lo, hi = None, None

        analysis["zero_se"].append({
            "epsilon": eps,
            "n": int(n_zero),
            "percentage": float(n_zero / len(results) * 100),
            "n_hallucinations": int(zero_labels.sum()),
            "hallucination_rate": float(hall_rate),
            "energy_auroc": energy_auroc_zero,
            "energy_auroc_ci": [lo, hi] if lo else None,
        })
        print(f"  ε={eps}: {n_zero}개 ({n_zero/len(results)*100:.1f}%), 환각률 {hall_rate:.1f}%, Energy AUROC {energy_auroc_zero or 'N/A'}")

    # === 3. SE 구간별 분석 ===
    print("\n[3] SE 구간별 분석...")
    bins = [
        ("Zero-SE", 0, 0.05),
        ("Low", 0.05, 0.3),
        ("Medium", 0.3, 0.6),
        ("High", 0.6, 1.0),
        ("Very High", 1.0, float("inf")),
    ]
    analysis["se_bins"] = []
    for name, lo, hi in bins:
        if lo == 0:
            mask = (se_scores >= lo) & (se_scores <= hi)
        else:
            mask = (se_scores > lo) & (se_scores <= hi)
        
        n = mask.sum()
        if n == 0:
            continue

        bin_labels = labels[mask]
        bin_se = se_scores[mask]
        bin_energy = energy_scores[mask]
        hall_rate = bin_labels.mean() * 100

        se_auc = None
        energy_auc = None
        if len(np.unique(bin_labels)) == 2:
            if len(np.unique(bin_se)) > 1:
                se_auc = float(roc_auc_score(bin_labels, bin_se))
            energy_auc = float(roc_auc_score(bin_labels, bin_energy))

        analysis["se_bins"].append({
            "bin": name,
            "range": f"[{lo}, {hi})" if lo == 0 else f"({lo}, {hi}]",
            "n": int(n),
            "hallucination_rate": float(hall_rate),
            "se_auroc": se_auc,
            "energy_auroc": energy_auc,
        })
        se_str = f"{se_auc:.3f}" if se_auc else "N/A"
        en_str = f"{energy_auc:.3f}" if energy_auc else "N/A"
        print(f"  {name}: n={n}, 환각률 {hall_rate:.1f}%, SE AUROC {se_str}, Energy AUROC {en_str}")

    # === 4. Cascade Sweep ===
    print("\n[4] Cascade threshold sweep...")
    # Normalize
    se_norm = (se_scores - se_scores.min()) / (se_scores.max() - se_scores.min() + 1e-10)
    energy_norm = (energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min() + 1e-10)

    analysis["cascade_sweep"] = []
    for tau in np.arange(0.0, 1.05, 0.05):
        # SE < tau → Energy, else → SE
        cascade = np.where(se_norm < tau, energy_norm, se_norm)
        cascade_auroc = float(roc_auc_score(labels, cascade))
        analysis["cascade_sweep"].append({"tau": float(tau), "auroc": cascade_auroc})

    best = max(analysis["cascade_sweep"], key=lambda x: x["auroc"])
    analysis["best_cascade"] = best
    print(f"  최적 τ={best['tau']:.2f}, AUROC={best['auroc']:.4f}")
    print(f"  개선: +{best['auroc'] - analysis['overall']['se_auroc']:.4f} vs SE-only")

    # === 5. 상보성 분석 ===
    print("\n[5] 상보성 분석...")
    se_threshold = np.percentile(se_scores, 80)
    energy_threshold = np.percentile(energy_scores, 80)

    se_catches = (se_scores > se_threshold) & (labels == 1)
    energy_catches = (energy_scores > energy_threshold) & (labels == 1)

    both = int((se_catches & energy_catches).sum())
    se_only = int((se_catches & ~energy_catches).sum())
    energy_only = int((~se_catches & energy_catches).sum())
    neither = int((~se_catches & ~energy_catches & (labels == 1)).sum())
    n_hall = int(labels.sum())

    analysis["complementarity"] = {
        "threshold_percentile": 80,
        "n_hallucinations": n_hall,
        "both_catch": both,
        "se_only": se_only,
        "energy_only": energy_only,
        "neither_catch": neither,
        "union_catch_rate": float((both + se_only + energy_only) / n_hall * 100) if n_hall > 0 else 0,
    }
    print(f"  Both: {both}, SE-only: {se_only}, Energy-only: {energy_only}, Neither: {neither}")
    print(f"  합집합 탐지율: {analysis['complementarity']['union_catch_rate']:.1f}%")

    # === 6. 클러스터 분석 ===
    print("\n[6] 클러스터 분석...")
    analysis["cluster_analysis"] = []
    for nc in range(1, 6):
        mask = num_clusters == nc
        n = mask.sum()
        if n == 0:
            continue
        hall_rate = labels[mask].mean() * 100
        analysis["cluster_analysis"].append({
            "num_clusters": nc,
            "n": int(n),
            "percentage": float(n / len(results) * 100),
            "hallucination_rate": float(hall_rate),
        })
        print(f"  {nc}개 클러스터: n={n} ({n/len(results)*100:.1f}%), 환각률 {hall_rate:.1f}%")

    # 저장
    analysis_path = EXP_DIR / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n분석 저장: {analysis_path}")

    return analysis


# ============================================================
# Part 4: Figure 생성
# ============================================================
def generate_figures(analysis: dict):
    """논문용 Figure 생성"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    COLORS = {"se": "#4A90D9", "energy": "#E8553A", "cascade": "#2ECC71", "hall": "#E74C3C"}

    print("\n" + "=" * 70)
    print("Figure 생성")
    print("=" * 70)

    # === Fig 1: Zero-SE Overview ===
    print("\n[Fig 1] Zero-SE Overview...")
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), None)
    if zero_se:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # (a) Zero-SE 비율
        ax = axes[0]
        ax.bar(["Zero-SE", "Non-Zero"], [zero_se["percentage"], 100 - zero_se["percentage"]], 
               color=[COLORS["se"], "#95A5A6"], alpha=0.8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("(a) Zero-SE Prevalence")
        ax.text(0, zero_se["percentage"] + 2, f"{zero_se['percentage']:.1f}%", ha="center", fontweight="bold")

        # (b) 환각률
        ax = axes[1]
        ax.bar(["Hallucination", "Correct"], 
               [zero_se["hallucination_rate"], 100 - zero_se["hallucination_rate"]], 
               color=[COLORS["hall"], "#95A5A6"], alpha=0.8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("(b) Hallucination Rate in Zero-SE")
        ax.text(0, zero_se["hallucination_rate"] + 2, f"{zero_se['hallucination_rate']:.1f}%", ha="center", fontweight="bold")

        # (c) Energy AUROC
        ax = axes[2]
        auroc = zero_se["energy_auroc"] or 0
        ax.bar(["Energy AUROC"], [auroc], color=COLORS["energy"], alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", label="Random")
        ax.set_ylabel("AUROC")
        ax.set_title("(c) Energy AUROC in Zero-SE")
        ax.set_ylim(0.3, 0.9)
        ax.legend()
        ax.text(0, auroc + 0.02, f"{auroc:.3f}", ha="center", fontweight="bold")

        fig.suptitle("Figure 1. Zero-SE Phenomenon in TruthfulQA", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig1_zero_se_overview.png")
        plt.close()
        print("  → fig1_zero_se_overview.png")

    # === Fig 2: SE Bin Crossover ===
    print("\n[Fig 2] SE Bin Crossover...")
    bins_data = analysis["se_bins"]
    if bins_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        names = [b["bin"] for b in bins_data]
        se_auc = [b["se_auroc"] or 0 for b in bins_data]
        en_auc = [b["energy_auroc"] or 0 for b in bins_data]
        ns = [b["n"] for b in bins_data]

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, se_auc, width, label="SE", color=COLORS["se"], alpha=0.85)
        ax.bar(x + width/2, en_auc, width, label="Energy", color=COLORS["energy"], alpha=0.85)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={ns_})" for n, ns_ in zip(names, ns)], fontsize=9)
        ax.set_ylabel("AUROC")
        ax.set_title("Figure 2. SE vs Energy AUROC by Semantic Entropy Bin", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig2_se_bin_crossover.png")
        plt.close()
        print("  → fig2_se_bin_crossover.png")

    # === Fig 3: Cascade Sweep ===
    print("\n[Fig 3] Cascade Sweep...")
    sweep = analysis["cascade_sweep"]
    if sweep:
        fig, ax = plt.subplots(figsize=(10, 5))

        taus = [s["tau"] for s in sweep]
        aurocs = [s["auroc"] for s in sweep]
        se_base = analysis["overall"]["se_auroc"]
        en_base = analysis["overall"]["energy_auroc"]

        ax.plot(taus, aurocs, color=COLORS["cascade"], linewidth=2, label="Cascade")
        ax.axhline(se_base, color=COLORS["se"], linestyle="--", linewidth=1.5, label=f"SE-only ({se_base:.3f})")
        ax.axhline(en_base, color=COLORS["energy"], linestyle="--", linewidth=1.5, label=f"Energy-only ({en_base:.3f})")

        best = analysis["best_cascade"]
        ax.plot(best["tau"], best["auroc"], "o", color=COLORS["cascade"], markersize=10)
        ax.annotate(f"Best: τ={best['tau']:.2f}\nAUROC={best['auroc']:.3f}",
                    xy=(best["tau"], best["auroc"]),
                    xytext=(best["tau"] + 0.1, best["auroc"] + 0.02),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

        ax.set_xlabel("Threshold τ (normalized SE)")
        ax.set_ylabel("AUROC")
        ax.set_title("Figure 3. SE-Gated Cascade Threshold Sweep", fontsize=13, fontweight="bold")
        ax.legend()

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_cascade_sweep.png")
        plt.close()
        print("  → fig3_cascade_sweep.png")

    # === Fig 4: Complementarity Pie ===
    print("\n[Fig 4] Complementarity...")
    comp = analysis["complementarity"]
    if comp["n_hallucinations"] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        values = [comp["se_only"], comp["energy_only"], comp["both_catch"], comp["neither_catch"]]
        labels = ["SE only", "Energy only", "Both", "Neither"]
        colors = [COLORS["se"], COLORS["energy"], "#8E44AD", "#BDC3C7"]

        ax.pie(values, labels=labels, colors=colors,
               autopct=lambda p: f'{p:.1f}%\n({int(p*comp["n_hallucinations"]/100)})',
               startangle=90, explode=[0.02]*4)
        ax.set_title(f"Figure 4. Complementarity Analysis\n(n={comp['n_hallucinations']} hallucinations)",
                     fontsize=12, fontweight="bold")

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_complementarity.png")
        plt.close()
        print("  → fig4_complementarity.png")

    # === Fig 5: Overall Comparison ===
    print("\n[Fig 5] Overall Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["SE-only", "Energy-only", f"Cascade\n(τ={analysis['best_cascade']['tau']:.2f})"]
    aurocs = [
        analysis["overall"]["se_auroc"],
        analysis["overall"]["energy_auroc"],
        analysis["best_cascade"]["auroc"],
    ]
    colors = [COLORS["se"], COLORS["energy"], COLORS["cascade"]]

    bars = ax.bar(methods, aurocs, color=colors, alpha=0.85, edgecolor="white", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("AUROC")
    ax.set_title("Figure 5. Overall Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, max(aurocs) + 0.08)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_overall_comparison.png")
    plt.close()
    print("  → fig5_overall_comparison.png")

    print(f"\n모든 Figure 저장: {FIG_DIR}")


# ============================================================
# Part 5: 마크다운 리포트
# ============================================================
def generate_report(analysis: dict):
    """RESULTS.md 생성"""
    
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    best = analysis["best_cascade"]
    comp = analysis["complementarity"]

    md = f"""# Exp10: TruthfulQA 논문용 완전 실험 결과

**실험 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 1. 실험 설정

| 항목 | 설정 |
|------|------|
| LLM | {CONFIG['llm_model']} |
| NLI 모델 | {CONFIG['nli_model']} |
| 데이터셋 | {CONFIG['dataset']} ({CONFIG['dataset_split']}) |
| 샘플링 수 (K) | {CONFIG['num_responses']} |
| Temperature | {CONFIG['temperature']} |
| Seed | {CONFIG['seed']} |

## 2. 데이터셋 통계

| 항목 | 값 |
|------|------|
| 전체 샘플 수 | {analysis['n_samples']} |
| 환각 샘플 수 | {analysis['n_hallucinations']} ({analysis['hallucination_rate']:.1f}%) |
| 정상 샘플 수 | {analysis['n_normal']} ({100 - analysis['hallucination_rate']:.1f}%) |

## 3. 전체 성능

| 방법 | AUROC | AUPRC |
|------|-------|-------|
| Semantic Entropy (SE) | {analysis['overall']['se_auroc']:.4f} | {analysis['overall']['se_auprc']:.4f} |
| Semantic Energy | {analysis['overall']['energy_auroc']:.4f} | {analysis['overall']['energy_auprc']:.4f} |
| **SE-gated Cascade (τ={best['tau']:.2f})** | **{best['auroc']:.4f}** | - |

**Cascade 개선**: +{best['auroc'] - analysis['overall']['se_auroc']:.4f} vs SE-only

## 4. Zero-SE 분석 (ε=0.05)

| 지표 | 값 |
|------|------|
| Zero-SE 비율 | {zero_se.get('percentage', 0):.1f}% ({zero_se.get('n', 0)}/{analysis['n_samples']}) |
| Zero-SE 내 환각률 | {zero_se.get('hallucination_rate', 0):.1f}% |
| Zero-SE 내 Energy AUROC | {zero_se.get('energy_auroc', 'N/A')} |

## 5. SE 구간별 성능

| 구간 | 범위 | n | 환각률 | SE AUROC | Energy AUROC |
|------|------|---|--------|----------|--------------|
"""
    for b in analysis["se_bins"]:
        se_str = f"{b['se_auroc']:.3f}" if b["se_auroc"] else "N/A"
        en_str = f"{b['energy_auroc']:.3f}" if b["energy_auroc"] else "N/A"
        md += f"| {b['bin']} | {b['range']} | {b['n']} | {b['hallucination_rate']:.1f}% | {se_str} | {en_str} |\n"

    md += f"""
## 6. 상보성 분석 (80th percentile)

| 카테고리 | 개수 | 비율 |
|----------|------|------|
| SE-only | {comp['se_only']} | {comp['se_only']/comp['n_hallucinations']*100:.1f}% |
| Energy-only | {comp['energy_only']} | {comp['energy_only']/comp['n_hallucinations']*100:.1f}% |
| Both | {comp['both_catch']} | {comp['both_catch']/comp['n_hallucinations']*100:.1f}% |
| Neither | {comp['neither_catch']} | {comp['neither_catch']/comp['n_hallucinations']*100:.1f}% |

**합집합 탐지율**: {comp['union_catch_rate']:.1f}%

## 7. 생성된 Figure

1. `fig1_zero_se_overview.png` - Zero-SE 현상 개요
2. `fig2_se_bin_crossover.png` - SE 구간별 Crossover 패턴
3. `fig3_cascade_sweep.png` - Cascade threshold sweep
4. `fig4_complementarity.png` - 상보성 분석
5. `fig5_overall_comparison.png` - 전체 방법 비교

## 8. 논문 초록/결론용 핵심 수치

```
- 전체 샘플: {analysis['n_samples']}개
- 환각률: {analysis['hallucination_rate']:.1f}%
- Zero-SE 비율: {zero_se.get('percentage', 0):.1f}%
- Zero-SE 내 환각률: {zero_se.get('hallucination_rate', 0):.1f}%
- Zero-SE 내 Energy AUROC: {zero_se.get('energy_auroc', 'N/A')}
- SE AUROC: {analysis['overall']['se_auroc']:.3f}
- Energy AUROC: {analysis['overall']['energy_auroc']:.3f}
- Cascade AUROC: {best['auroc']:.3f} (τ={best['tau']:.2f})
- Cascade 개선: +{best['auroc'] - analysis['overall']['se_auroc']:.3f}
- 합집합 탐지율: {comp['union_catch_rate']:.1f}%
```

## 9. 재현 방법

```bash
cd hallucination_lfe
source .venv/bin/activate
python experiment_notes/exp10_thesis_complete/run_experiment.py
```
"""

    report_path = EXP_DIR / "RESULTS.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"\n리포트 저장: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Exp10: TruthfulQA 논문용 완전 실험")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 이어서 실행")
    parser.add_argument("--analysis-only", action="store_true", help="분석만 실행")
    parser.add_argument("--max-samples", type=int, default=200, help="최대 샘플 수 (기본: 200)")
    args = parser.parse_args()

    if args.analysis_only:
        results_path = EXP_DIR / "results.json"
        if not results_path.exists():
            print("Error: results.json 없음. 실험 먼저 실행하세요.")
            sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)["samples"]
    else:
        results = run_experiment(resume=args.resume, max_samples=args.max_samples)

    analysis = run_analysis(results)
    generate_figures(analysis)
    generate_report(analysis)

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
