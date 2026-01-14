#!/usr/bin/env python
"""
Exp02: HaluEval QA - Semantic Entropy / Semantic Energy AUROC 실험

목적:
- HaluEval QA 데이터셋에서 SE와 Energy의 환각 탐지 성능 비교
- 가중치 조합 없이 개별 지표의 AUROC 측정

HaluEval 데이터 구조:
- knowledge: 배경 지식
- question: 질문
- right_answer: 정답
- hallucinated_answer: 환각 답변

방법:
1. LLM으로 질문에 대해 K개 응답 샘플링
2. NLI 클러스터링으로 의미 그룹화
3. Semantic Entropy 계산
4. Semantic Energy 계산
5. 응답이 right_answer와 일치하면 정상(0), 아니면 환각(1)
6. AUROC 계산
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

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
        knowledge: str = None,
        num_samples: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> list[Response]:
        """다중 응답 샘플링 + raw logit 추출"""

        # knowledge가 있으면 포함
        if knowledge:
            user_content = f"Based on the following knowledge, answer the question concisely.\n\nKnowledge: {knowledge}\n\nQuestion: {question}"
        else:
            user_content = question

        messages = [
            {
                "role": "system",
                "content": "Answer the question concisely in one sentence.",
            },
            {"role": "user", "content": user_content},
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
            token_logits = []
            log_prob_sum = 0.0

            for step_idx, token_id in enumerate(generated_ids):
                if step_idx >= len(outputs.scores):
                    break

                logits = outputs.scores[step_idx][0]
                raw_logit = logits[token_id].item()
                token_logits.append(raw_logit)

                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_sum += log_probs[token_id].item()

                if token_id == self.tokenizer.eos_token_id:
                    break

            responses.append(
                Response(
                    text=text.strip(),
                    probability=1.0 / num_samples,
                    log_probability=log_prob_sum,
                    logits=token_logits,
                )
            )

        return responses


def is_correct_halueval(response_text: str, right_answer: str) -> bool:
    """HaluEval 정답 판정"""
    response_lower = response_text.lower().strip()
    right_lower = right_answer.lower().strip()

    # 정확 일치
    if right_lower in response_lower:
        return True

    # 부분 일치 (정답이 짧은 경우)
    if len(right_lower) <= 30 and right_lower in response_lower:
        return True

    # 응답이 정답에 포함되는 경우
    if len(response_lower) >= 3 and response_lower in right_lower:
        return True

    return False


def run_experiment(
    max_samples: int = 100,
    num_responses: int = 5,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    nli_model: str = "microsoft/deberta-large-mnli",
    output_dir: Path = None,
):
    """실험 실행"""

    if output_dir is None:
        output_dir = Path(__file__).parent

    print("=" * 70)
    print("Exp02: HaluEval QA - Semantic Entropy / Energy AUROC 실험")
    print("=" * 70)

    # 1. LLM 샘플러 초기화
    print(f"\n[1/4] LLM 샘플러 초기화")
    sampler = LLMSampler(model_name)

    # 2. NLI 클러스터러 초기화
    print(f"\n[2/4] NLI 클러스터러 초기화: {nli_model}")
    clusterer = NLIClusterer(model_name=nli_model, device=sampler.device)

    # 3. 데이터셋 로드
    print(f"\n[3/4] HaluEval QA 데이터셋 로딩...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    print(f"  총 {len(ds)} 샘플, {max_samples}개 사용")

    # 4. 실험 실행
    print(f"\n[4/4] 실험 실행 (각 질문당 {num_responses}개 응답)")

    results = []

    for idx, item in enumerate(ds):
        if idx >= max_samples:
            break

        if idx % 10 == 0:
            print(f"  진행: {idx}/{max_samples}")

        question = item["question"]
        knowledge = item.get("knowledge", None)
        right_answer = item["right_answer"]

        # 다중 응답 샘플링 (knowledge 포함)
        responses = sampler.sample(
            question=question,
            knowledge=knowledge,
            num_samples=num_responses,
            temperature=0.7,
        )

        # NLI 클러스터링
        clusters = clusterer.cluster(responses)

        # Semantic Entropy
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))

        # Semantic Energy
        energy = SemanticEnergyCalculator.compute_energy_only(responses)

        # 환각 레이블: 모든 응답이 오답이면 환각(1)
        any_correct = any(is_correct_halueval(r.text, right_answer) for r in responses)
        is_hallucination = 0 if any_correct else 1

        results.append(
            {
                "question": question,
                "knowledge": knowledge[:100] + "..."
                if knowledge and len(knowledge) > 100
                else knowledge,
                "right_answer": right_answer,
                "responses": [r.text for r in responses],
                "num_clusters": len(clusters),
                "semantic_entropy": se,
                "semantic_energy": energy,
                "is_hallucination": is_hallucination,
            }
        )

    # === 분석 ===
    print("\n" + "=" * 70)
    print("결과 분석")
    print("=" * 70)

    labels = [r["is_hallucination"] for r in results]
    entropies = [r["semantic_entropy"] for r in results]
    energies = [r["semantic_energy"] for r in results]

    n_hall = sum(labels)
    n_norm = len(labels) - n_hall
    print(f"\n샘플 분포: 환각 {n_hall}, 정상 {n_norm}")

    metrics = {}

    if n_hall > 0 and n_norm > 0:
        # AUROC
        se_auroc = roc_auc_score(labels, entropies)
        energy_auroc = roc_auc_score(labels, energies)

        # AUPRC
        se_auprc = average_precision_score(labels, entropies)
        energy_auprc = average_precision_score(labels, energies)

        print(f"\n[Semantic Entropy]")
        print(f"  AUROC: {se_auroc:.4f}")
        print(f"  AUPRC: {se_auprc:.4f}")

        print(f"\n[Semantic Energy]")
        print(f"  AUROC: {energy_auroc:.4f}")
        print(f"  AUPRC: {energy_auprc:.4f}")

        metrics = {
            "semantic_entropy": {"auroc": se_auroc, "auprc": se_auprc},
            "semantic_energy": {"auroc": energy_auroc, "auprc": energy_auprc},
        }

        # 통계
        hall_se = [r["semantic_entropy"] for r in results if r["is_hallucination"]]
        norm_se = [r["semantic_entropy"] for r in results if not r["is_hallucination"]]
        hall_e = [r["semantic_energy"] for r in results if r["is_hallucination"]]
        norm_e = [r["semantic_energy"] for r in results if not r["is_hallucination"]]

        print(f"\n[통계]")
        if hall_se:
            print(f"  SE (환각):     {np.mean(hall_se):.4f} +/- {np.std(hall_se):.4f}")
        if norm_se:
            print(f"  SE (정상):     {np.mean(norm_se):.4f} +/- {np.std(norm_se):.4f}")
        if hall_e:
            print(f"  Energy (환각): {np.mean(hall_e):.4f} +/- {np.std(hall_e):.4f}")
        if norm_e:
            print(f"  Energy (정상): {np.mean(norm_e):.4f} +/- {np.std(norm_e):.4f}")

        # 제로-엔트로피 분석
        zero_se_hall = [
            r for r in results if r["semantic_entropy"] < 0.1 and r["is_hallucination"]
        ]
        zero_se_norm = [
            r
            for r in results
            if r["semantic_entropy"] < 0.1 and not r["is_hallucination"]
        ]

        print(f"\n[제로-엔트로피 케이스 (SE < 0.1)]")
        print(f"  환각: {len(zero_se_hall)}개")
        print(f"  정상: {len(zero_se_norm)}개")

        if zero_se_hall and zero_se_norm:
            ze_labels = [1] * len(zero_se_hall) + [0] * len(zero_se_norm)
            ze_energies = [r["semantic_energy"] for r in zero_se_hall] + [
                r["semantic_energy"] for r in zero_se_norm
            ]
            ze_auroc = roc_auc_score(ze_labels, ze_energies)
            print(f"  Energy AUROC (제로-SE 케이스): {ze_auroc:.4f}")
            metrics["zero_entropy_energy_auroc"] = ze_auroc

    else:
        print("한 클래스만 존재하여 AUROC 계산 불가")

    # 저장
    output = {
        "experiment": "exp02_halueval",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": model_name,
            "nli_model": nli_model,
            "dataset": "HaluEval QA",
            "max_samples": max_samples,
            "num_responses": num_responses,
        },
        "metrics": metrics,
        "statistics": {
            "n_hallucination": n_hall,
            "n_normal": n_norm,
        },
        "samples": results,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp02: HaluEval QA SE/Energy AUROC")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-responses", type=int, default=5)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--nli-model", type=str, default="microsoft/deberta-large-mnli")
    args = parser.parse_args()

    run_experiment(
        max_samples=args.max_samples,
        num_responses=args.num_responses,
        model_name=args.model,
        nli_model=args.nli_model,
    )
