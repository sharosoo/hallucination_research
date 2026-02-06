#!/usr/bin/env python
"""
E2: 새로운 데이터셋에서 SE/Energy 측정

TriviaQA, NaturalQuestions, HaluEval-dialogue 데이터셋 추가.
기존 exp01/exp02과 동일한 파이프라인(Qwen2.5-3B + DeBERTa-large-mnli)으로
SE, Energy, is_hallucination을 계산하여 results_*.json으로 저장.

GPU 필요.
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "hfe-core" / "src"))

from hfe_core.nli_clusterer import NLIClusterer, Response
from hfe_core.semantic_entropy import SemanticEntropyCalculator
from hfe_core.semantic_energy import SemanticEnergyCalculator


class LLMSampler:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"  Loading: {model_name} on {self.device}")

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
        if knowledge:
            user_content = (
                f"Based on the following knowledge, answer the question concisely.\n\n"
                f"Knowledge: {knowledge}\n\nQuestion: {question}"
            )
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

            responses.append(
                Response(
                    text=text.strip(),
                    probability=1.0 / num_samples,
                    log_probability=log_prob_sum,
                    logits=token_logits,
                )
            )

        return responses


def is_correct_string_match(response_text: str, gold_answers: list[str]) -> bool:
    resp = response_text.lower().strip()
    for gold in gold_answers:
        gold_l = gold.lower().strip()
        if gold_l in resp or resp in gold_l:
            return True
        if len(gold_l) > 3 and gold_l[:20] in resp:
            return True
    return False


# ─── Dataset loaders ───


def load_triviaqa(max_samples=200):
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        gold_answers = row["answer"]["aliases"] + [row["answer"]["value"]]
        gold_answers = list(set(a for a in gold_answers if a.strip()))
        items.append(
            {
                "question": row["question"],
                "gold_answers": gold_answers,
                "knowledge": None,
            }
        )
    return items


def load_natural_questions(max_samples=200):
    ds = load_dataset("nq_open", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        items.append(
            {
                "question": row["question"],
                "gold_answers": answers,
                "knowledge": None,
            }
        )
    return items


def load_halueval_dialogue(max_samples=200):
    ds = load_dataset("pminervini/HaluEval", "dialogue", split="data")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        items.append(
            {
                "question": row["dialogue_history"],
                "gold_answers": [row["right_response"]],
                "knowledge": row.get("knowledge", None),
            }
        )
    return items


DATASET_LOADERS = {
    "TriviaQA": load_triviaqa,
    "NaturalQuestions": load_natural_questions,
    "HaluEval-dialogue": load_halueval_dialogue,
}


def run_dataset(
    dataset_name: str,
    max_samples: int = 200,
    num_responses: int = 5,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    nli_model: str = "microsoft/deberta-large-mnli",
):
    print(f"\n{'=' * 70}")
    print(
        f"Running: {dataset_name} ({max_samples} samples, {num_responses} responses each)"
    )
    print(f"{'=' * 70}")

    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_LOADERS.keys())}")
        return None

    print(f"\n[1/4] Loading dataset...")
    items = loader(max_samples)
    print(f"  Loaded {len(items)} items")

    print(f"\n[2/4] Initializing LLM sampler...")
    sampler = LLMSampler(model_name)

    print(f"\n[3/4] Initializing NLI clusterer: {nli_model}")
    clusterer = NLIClusterer(model_name=nli_model, device=sampler.device)

    print(f"\n[4/4] Running experiment...")
    results = []
    for idx, item in enumerate(items):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(items)}")

        question = item["question"]
        gold_answers = item["gold_answers"]
        knowledge = item.get("knowledge")

        responses = sampler.sample(
            question=question,
            knowledge=knowledge,
            num_samples=num_responses,
            temperature=0.7,
        )

        clusters = clusterer.cluster(responses)
        se = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))
        energy = SemanticEnergyCalculator.compute_energy_only(responses)

        any_correct = any(
            is_correct_string_match(r.text, gold_answers) for r in responses
        )
        is_hallucination = 0 if any_correct else 1

        results.append(
            {
                "question": question[:200],
                "gold_answers": gold_answers[:5],
                "responses": [r.text for r in responses],
                "num_clusters": len(clusters),
                "semantic_entropy": float(se),
                "semantic_energy": float(energy),
                "is_hallucination": int(is_hallucination),
            }
        )

    labels = [r["is_hallucination"] for r in results]
    entropies = [r["semantic_entropy"] for r in results]
    energies = [r["semantic_energy"] for r in results]

    n_hall = sum(labels)
    n_norm = len(labels) - n_hall
    print(f"\n  Distribution: hallucination={n_hall}, normal={n_norm}")

    metrics = {}
    if n_hall > 0 and n_norm > 0:
        se_auroc = roc_auc_score(labels, entropies)
        energy_auroc = roc_auc_score(labels, energies)
        metrics["se_auroc"] = se_auroc
        metrics["energy_auroc"] = energy_auroc
        print(f"  SE AUROC: {se_auroc:.4f}")
        print(f"  Energy AUROC: {energy_auroc:.4f}")

        zero_se_mask = np.array(entropies) < 0.1
        zero_labels = np.array(labels)[zero_se_mask]
        zero_energies = np.array(energies)[zero_se_mask]
        if len(np.unique(zero_labels)) == 2:
            ze_auroc = roc_auc_score(zero_labels, zero_energies)
            metrics["zero_se_energy_auroc"] = ze_auroc
            print(f"  Zero-SE Energy AUROC: {ze_auroc:.4f}")
        print(
            f"  Zero-SE count: {zero_se_mask.sum()} ({zero_se_mask.sum() / len(labels) * 100:.1f}%)"
        )

    output = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": model_name,
            "nli_model": nli_model,
            "max_samples": max_samples,
            "num_responses": num_responses,
        },
        "metrics": metrics,
        "statistics": {"n_hallucination": n_hall, "n_normal": n_norm},
        "samples": results,
    }

    output_path = (
        Path(__file__).parent / f"results_{dataset_name.lower().replace('-', '_')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="E2: Run new datasets for Zero-SE analysis"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["TriviaQA", "NaturalQuestions", "HaluEval-dialogue"],
        choices=list(DATASET_LOADERS.keys()),
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-responses", type=int, default=5)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--nli-model", type=str, default="microsoft/deberta-large-mnli")
    args = parser.parse_args()

    for ds_name in args.datasets:
        run_dataset(
            dataset_name=ds_name,
            max_samples=args.max_samples,
            num_responses=args.num_responses,
            model_name=args.model,
            nli_model=args.nli_model,
        )


if __name__ == "__main__":
    main()
