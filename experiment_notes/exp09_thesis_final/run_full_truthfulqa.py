#!/usr/bin/env python
"""
Exp09: Full TruthfulQA 실험 (졸프 논문용)

목적:
- TruthfulQA 전체 817개 샘플에서 SE/Energy/Cascade 성능 측정
- Adaptive threshold (SE-gated cascade) 검증
- 논문용 그래프 생성

출력:
- results_full.json: 전체 결과
- figures/: 논문용 그래프
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Path 설정
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "hfe-core" / "src"))

from hfe_core.nli_clusterer import NLIClusterer, Response
from hfe_core.semantic_entropy import SemanticEntropyCalculator
from hfe_core.semantic_energy import SemanticEnergyCalculator


class LLMSampler:
    """Transformers 기반 다중 샘플링 + logit 추출"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
    ):
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
    def sample(
        self,
        question: str,
        num_samples: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> list[Response]:
        """다중 응답 샘플링 + raw logit 추출"""

        messages = [
            {
                "role": "system",
                "content": "Answer the question concisely in one sentence.",
            },
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

            # 각 토큰의 raw logit 추출
            logits = []
            for step, score in enumerate(outputs.scores):
                token_id = generated_ids[step] if step < len(generated_ids) else None
                if token_id is not None:
                    token_logit = score[0, token_id].item()
                    logits.append(token_logit)

            responses.append(Response(text=text.strip(), logits=logits))

        return responses


def is_correct(response_text: str, correct_answers: list[str], incorrect_answers: list[str]) -> bool:
    """응답이 정답인지 확인"""
    response_lower = response_text.lower().strip()
    
    for correct in correct_answers:
        if correct.lower() in response_lower:
            return True
    
    return False


def run_experiment(
    max_samples: Optional[int] = None,
    num_responses: int = 5,
    output_dir: Optional[Path] = None,
):
    """전체 TruthfulQA 실험 실행"""
    
    output_dir = output_dir or Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Exp09: Full TruthfulQA 실험")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"  샘플 수: {len(dataset)}")
    
    # 모델 로드
    print("\n[2/4] 모델 로드...")
    sampler = LLMSampler()
    clusterer = NLIClusterer()
    
    # 실험 실행
    print("\n[3/4] 실험 실행...")
    results = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        question = item["question"]
        correct_answers = item["correct_answers"]
        incorrect_answers = item["incorrect_answers"]
        
        # 응답 샘플링
        responses = sampler.sample(question, num_samples=num_responses)
        
        # NLI 클러스터링
        clusters = clusterer.cluster(responses)
        
        # SE 계산 (static method 사용)
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))
        
        # Energy 계산 (static method 사용)
        energy = SemanticEnergyCalculator.compute_energy_only(responses)
        
        # 정답 여부 확인 (하나라도 맞으면 정상)
        any_correct = any(
            is_correct(r.text, correct_answers, incorrect_answers) 
            for r in responses
        )
        is_hallucination = 0 if any_correct else 1
        
        results.append({
            "idx": idx,
            "question": question,
            "responses": [r.text for r in responses],
            "semantic_entropy": se,
            "semantic_energy": energy,
            "num_clusters": len(clusters),
            "is_hallucination": is_hallucination,
            "correct_answers": correct_answers[:3],  # 처음 3개만 저장
        })
        
        # 중간 저장 (100개마다)
        if (idx + 1) % 100 == 0:
            _save_results(results, output_dir / "results_checkpoint.json")
    
    # 최종 저장
    print("\n[4/4] 결과 저장...")
    _save_results(results, output_dir / "results_full.json")
    
    # 통계 출력
    _print_stats(results)
    
    return results


def _save_results(results: list, path: Path):
    """결과 저장"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": "Qwen/Qwen2.5-3B-Instruct",
            "nli_model": "microsoft/deberta-large-mnli",
            "dataset": "TruthfulQA",
            "num_samples": len(results),
            "num_responses": 5,
        },
        "samples": results,
    }
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"  저장됨: {path}")


def _print_stats(results: list):
    """통계 출력"""
    se_scores = [r["semantic_entropy"] for r in results]
    energy_scores = [r["semantic_energy"] for r in results]
    labels = [r["is_hallucination"] for r in results]
    
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)
    
    print(f"\n총 샘플: {len(results)}")
    print(f"환각 수: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    
    # AUROC 계산
    if len(set(labels)) > 1:
        se_auroc = roc_auc_score(labels, se_scores)
        energy_auroc = roc_auc_score(labels, energy_scores)
        
        print(f"\nSE AUROC: {se_auroc:.3f}")
        print(f"Energy AUROC: {energy_auroc:.3f}")
        
        # Zero-SE 분석
        zero_se_mask = np.array(se_scores) < 0.1
        zero_se_count = zero_se_mask.sum()
        zero_se_hall = sum(1 for i, m in enumerate(zero_se_mask) if m and labels[i])
        
        print(f"\nZero-SE 샘플: {zero_se_count} ({100*zero_se_count/len(results):.1f}%)")
        if zero_se_count > 0:
            print(f"Zero-SE 내 환각률: {100*zero_se_hall/zero_se_count:.1f}%")
            
            # Zero-SE 내 Energy AUROC
            zero_se_labels = [labels[i] for i, m in enumerate(zero_se_mask) if m]
            zero_se_energy = [energy_scores[i] for i, m in enumerate(zero_se_mask) if m]
            
            if len(set(zero_se_labels)) > 1:
                zero_se_energy_auroc = roc_auc_score(zero_se_labels, zero_se_energy)
                print(f"Zero-SE Energy AUROC: {zero_se_energy_auroc:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-responses", type=int, default=5)
    args = parser.parse_args()
    
    run_experiment(
        max_samples=args.max_samples,
        num_responses=args.num_responses,
    )
