#!/usr/bin/env python3
"""
SE-gated Cascade 환각 탐지 실험 재현 스크립트

논문: "Semantic Entropy와 Semantic Energy의 상보성을 활용한 LLM 환각 탐지"

사용법:
    python run_experiment.py                    # 기본 200샘플 실험
    python run_experiment.py --samples 100      # 샘플 수 지정
    python run_experiment.py --seed 123         # 시드 변경

요구사항:
    - Python 3.10+
    - PyTorch 2.0+
    - transformers, datasets, scikit-learn
    - GPU 권장 (CPU도 가능하나 느림)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 경로 설정
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "packages" / "hfe-core" / "src"))

from hfe_core.nli_clusterer import NLIClusterer, Response


# ============================================================
# 설정
# ============================================================
CONFIG = {
    "llm_model": "Qwen/Qwen2.5-3B-Instruct",
    "nli_model": "microsoft/deberta-large-mnli",
    "num_samples_k": 5,          # 질문당 샘플링 수
    "temperature": 0.7,
    "max_new_tokens": 50,
    "default_n_samples": 200,    # 기본 실험 샘플 수
    "seed": 42,
}


# ============================================================
# 데이터 클래스
# ============================================================
@dataclass
class ExperimentResult:
    question: str
    responses: list[str]
    logits: list[list[float]]
    clusters: list[list[int]]
    num_clusters: int
    semantic_entropy: float
    semantic_energy: float
    is_hallucination: int
    cascade_score: float
    cascade_method: str


# ============================================================
# LLM 샘플러
# ============================================================
class LLMSampler:
    """LLM 다중 응답 샘플링 + logit 추출"""

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LLM] {model_name} 로딩 중... (device: {self.device})")

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
    def sample(self, question: str, k: int, temperature: float, max_tokens: int) -> list[Response]:
        """K개 응답 샘플링 + raw logit 추출"""
        messages = [
            {"role": "system", "content": "Answer the question concisely in one sentence."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        responses = []
        for _ in range(k):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_ids = outputs.sequences[0, input_len:].tolist()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Raw logit 추출 (각 토큰에서 선택된 토큰의 logit)
            logits = []
            for step_idx, token_id in enumerate(generated_ids):
                if step_idx < len(outputs.scores):
                    token_logit = outputs.scores[step_idx][0, token_id].item()
                    logits.append(token_logit)

            responses.append(Response(text=text.strip(), logits=logits, probability=1.0 / k))

        return responses


# ============================================================
# 메트릭 계산
# ============================================================
def compute_semantic_entropy(clusters: list[list[int]], k: int) -> float:
    """클러스터 분포의 Shannon Entropy 계산"""
    if len(clusters) == 1:
        return 0.0
    probs = [len(c) / k for c in clusters]
    return -sum(p * np.log(p) for p in probs if p > 0)


def compute_semantic_energy(responses: list[Response]) -> float:
    """응답들의 평균 negative logit (Energy) 계산"""
    all_logits = []
    for r in responses:
        all_logits.extend(r.logits)
    if not all_logits:
        return 0.0
    return -np.mean(all_logits)


def compute_cascade(se: float, energy: float, num_clusters: int) -> tuple[float, str]:
    """SE-gated Cascade: |C|=1이면 Energy, 아니면 SE"""
    if num_clusters == 1:
        return energy, "energy"
    else:
        return se, "se"


# ============================================================
# 환각 판정
# ============================================================
def is_hallucination(response_texts: list[str], correct_answers: list[str]) -> int:
    """응답이 정답과 일치하면 0(정상), 아니면 1(환각)"""
    for resp in response_texts:
        resp_lower = resp.lower().strip()
        for ans in correct_answers:
            if ans.lower() in resp_lower or resp_lower in ans.lower():
                return 0
    return 1


# ============================================================
# 메인 실험
# ============================================================
def run_experiment(n_samples: int, seed: int) -> list[ExperimentResult]:
    """실험 실행"""
    print("=" * 60)
    print("SE-gated Cascade 환각 탐지 실험")
    print("=" * 60)

    # 시드 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 데이터셋 로드
    print(f"\n[1/4] TruthfulQA 데이터셋 로딩...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    print(f"  전체: {len(dataset)}개, 사용: {len(indices)}개")

    # 모델 로드
    print(f"\n[2/4] 모델 로딩...")
    sampler = LLMSampler(CONFIG["llm_model"])
    clusterer = NLIClusterer(model_name=CONFIG["nli_model"], device=sampler.device)

    # 실험 실행
    print(f"\n[3/4] 실험 실행...")
    results = []

    for idx in tqdm(indices, desc="Processing"):
        item = dataset[int(idx)]
        question = item["question"]
        correct_answers = item.get("correct_answers", [])

        # K개 응답 샘플링
        responses = sampler.sample(
            question,
            k=CONFIG["num_samples_k"],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_new_tokens"],
        )

        # NLI 클러스터링
        clusters = clusterer.cluster(responses)
        cluster_indices = [[responses.index(r) for r in c.responses] for c in clusters]

        # 메트릭 계산
        se = compute_semantic_entropy(cluster_indices, CONFIG["num_samples_k"])
        energy = compute_semantic_energy(responses)
        cascade_score, cascade_method = compute_cascade(se, energy, len(clusters))

        # 환각 판정
        hall = is_hallucination([r.text for r in responses], correct_answers)

        results.append(ExperimentResult(
            question=question,
            responses=[r.text for r in responses],
            logits=[r.logits for r in responses],
            clusters=cluster_indices,
            num_clusters=len(clusters),
            semantic_entropy=se,
            semantic_energy=energy,
            is_hallucination=hall,
            cascade_score=cascade_score,
            cascade_method=cascade_method,
        ))

    return results


def analyze_results(results: list[ExperimentResult]) -> dict:
    """결과 분석"""
    print(f"\n[4/4] 결과 분석...")

    labels = [r.is_hallucination for r in results]
    se_scores = [r.semantic_entropy for r in results]
    energy_scores = [r.semantic_energy for r in results]
    cascade_scores = [r.cascade_score for r in results]

    n_hall = sum(labels)
    n_norm = len(labels) - n_hall

    # AUROC 계산
    se_auroc = roc_auc_score(labels, se_scores) if len(set(labels)) > 1 else 0.5
    energy_auroc = roc_auc_score(labels, energy_scores) if len(set(labels)) > 1 else 0.5
    cascade_auroc = roc_auc_score(labels, cascade_scores) if len(set(labels)) > 1 else 0.5

    # Zero-SE 분석
    zero_se = [r for r in results if r.num_clusters == 1]
    zero_se_labels = [r.is_hallucination for r in zero_se]
    zero_se_energy = [r.semantic_energy for r in zero_se]
    zero_se_auroc = roc_auc_score(zero_se_labels, zero_se_energy) if len(set(zero_se_labels)) > 1 and len(zero_se) > 10 else None

    analysis = {
        "n_samples": len(results),
        "n_hallucination": n_hall,
        "n_normal": n_norm,
        "hallucination_rate": n_hall / len(results) * 100,
        "se_auroc": se_auroc,
        "energy_auroc": energy_auroc,
        "cascade_auroc": cascade_auroc,
        "cascade_improvement": cascade_auroc - se_auroc,
        "zero_se": {
            "n": len(zero_se),
            "percentage": len(zero_se) / len(results) * 100,
            "n_hallucination": sum(zero_se_labels),
            "hallucination_rate": sum(zero_se_labels) / len(zero_se) * 100 if zero_se else 0,
            "energy_auroc": zero_se_auroc,
        },
    }

    return analysis


def print_results(analysis: dict):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("실험 결과")
    print("=" * 60)

    print(f"\n[데이터셋]")
    print(f"  샘플 수: {analysis['n_samples']}")
    print(f"  환각: {analysis['n_hallucination']} ({analysis['hallucination_rate']:.1f}%)")
    print(f"  정상: {analysis['n_normal']}")

    print(f"\n[전체 성능 - AUROC]")
    print(f"  SE-only:     {analysis['se_auroc']:.3f}")
    print(f"  Energy-only: {analysis['energy_auroc']:.3f}")
    print(f"  Cascade:     {analysis['cascade_auroc']:.3f} (+{analysis['cascade_improvement']:.3f})")

    zs = analysis['zero_se']
    print(f"\n[Zero-SE 분석]")
    print(f"  비율: {zs['n']}/{analysis['n_samples']} ({zs['percentage']:.1f}%)")
    print(f"  환각률: {zs['hallucination_rate']:.1f}%")
    if zs['energy_auroc']:
        print(f"  Energy AUROC: {zs['energy_auroc']:.3f}")


def save_results(results: list[ExperimentResult], analysis: dict, output_dir: Path):
    """결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 상세 결과
    results_data = [
        {
            "question": r.question,
            "responses": r.responses,
            "num_clusters": r.num_clusters,
            "semantic_entropy": r.semantic_entropy,
            "semantic_energy": r.semantic_energy,
            "is_hallucination": r.is_hallucination,
            "cascade_score": r.cascade_score,
            "cascade_method": r.cascade_method,
        }
        for r in results
    ]

    with open(output_dir / "results.json", "w") as f:
        json.dump({"config": CONFIG, "results": results_data}, f, indent=2, ensure_ascii=False)

    # 분석 요약
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n결과 저장: {output_dir}/")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SE-gated Cascade 환각 탐지 실험")
    parser.add_argument("--samples", type=int, default=CONFIG["default_n_samples"], help="샘플 수 (기본: 200)")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="랜덤 시드 (기본: 42)")
    parser.add_argument("--output", type=str, default="output", help="출력 디렉토리")
    args = parser.parse_args()

    results = run_experiment(n_samples=args.samples, seed=args.seed)
    analysis = analyze_results(results)
    print_results(analysis)
    save_results(results, analysis, Path(args.output))

    print("\n완료!")


if __name__ == "__main__":
    main()
