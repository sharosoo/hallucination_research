#!/usr/bin/env python
"""
LLM-as-judge relabeling for TriviaQA and NaturalQuestions.

Uses Azure OpenAI GPT-5.2 as an external judge to replace string-matching
hallucination labels. For each of the 5 sampled responses, asks the judge
whether the response correctly answers the question given the gold answer(s).
If ANY response is judged correct -> not hallucination.

Requires .env file at project root with:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL
"""

import json
import os
import time
import copy
from pathlib import Path

from openai import OpenAI

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


load_env()

MODEL = os.environ["AZURE_OPENAI_MODEL"]

JUDGE_PROMPT = """You are an answer correctness judge. Given a question and a reference (gold) answer, determine whether the candidate response contains the correct answer.

Rules:
- The candidate does NOT need to match the gold answer word-for-word
- It is correct if it conveys the same factual information as the gold answer
- Ignore differences in phrasing, extra context, or formatting
- If the candidate contains the correct answer among other text, it is still correct
- If the candidate is factually wrong or gives a different answer, it is incorrect

Question: {question}
Gold answer(s): {gold_answers}
Candidate response: {response}

Is the candidate response correct? Reply with ONLY "CORRECT" or "INCORRECT"."""


def create_client():
    return OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/").rsplit("/openai", 1)[0]
        + "/openai/v1/",
    )


def judge_single(client, question, gold_answers, response, max_retries=3):
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answers=", ".join(gold_answers),
        response=response,
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
                temperature=0.0,
            )
            generated = resp.choices[0].message.content.strip().upper()
            is_correct = "CORRECT" in generated and "INCORRECT" not in generated
            return is_correct
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API failed after {max_retries} retries: {e}")
                return False


def relabel_dataset(client, results_path):
    with open(results_path) as f:
        data = json.load(f)

    dataset_name = data["dataset_name"]
    samples = data["samples"]
    n_total = len(samples)

    print(f"\n{'=' * 60}")
    print(f"Relabeling: {dataset_name} ({n_total} samples x 5 responses)")
    print(f"Judge: {MODEL} via Azure OpenAI")
    print(f"{'=' * 60}")

    old_labels = [s["is_hallucination"] for s in samples]
    new_samples = copy.deepcopy(samples)

    for idx, sample in enumerate(new_samples):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{n_total}")

        question = sample["question"]
        gold_answers = sample["gold_answers"]
        responses = sample["responses"]

        any_correct = False
        for resp in responses:
            if judge_single(client, question, gold_answers, resp):
                any_correct = True
                break

        sample["is_hallucination_string_match"] = sample["is_hallucination"]
        sample["is_hallucination"] = 0 if any_correct else 1

    new_labels = [s["is_hallucination"] for s in new_samples]
    n_changed = sum(a != b for a, b in zip(old_labels, new_labels))
    n_hall_old = sum(old_labels)
    n_hall_new = sum(new_labels)

    print(f"\n  Results:")
    print(
        f"    Old (string match): {n_hall_old} hall ({n_hall_old / n_total * 100:.1f}%)"
    )
    print(
        f"    New ({MODEL} judge): {n_hall_new} hall ({n_hall_new / n_total * 100:.1f}%)"
    )
    print(f"    Changed: {n_changed} labels ({n_changed / n_total * 100:.1f}%)")
    print(
        f"    0->1 (was normal, now hall): "
        f"{sum(a == 0 and b == 1 for a, b in zip(old_labels, new_labels))}"
    )
    print(
        f"    1->0 (was hall, now normal): "
        f"{sum(a == 1 and b == 0 for a, b in zip(old_labels, new_labels))}"
    )

    output_data = copy.deepcopy(data)
    output_data["samples"] = new_samples
    output_data["labeling_method"] = "llm_judge"
    output_data["labeling_model"] = MODEL
    output_data["statistics"] = {
        "n_hallucination": n_hall_new,
        "n_normal": n_total - n_hall_new,
    }
    output_data["relabeling_stats"] = {
        "old_n_hallucination": n_hall_old,
        "new_n_hallucination": n_hall_new,
        "n_changed": n_changed,
        "n_0_to_1": sum(a == 0 and b == 1 for a, b in zip(old_labels, new_labels)),
        "n_1_to_0": sum(a == 1 and b == 0 for a, b in zip(old_labels, new_labels)),
    }

    output_path = results_path.parent / results_path.name.replace(
        ".json", "_llm_judge.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    return output_data


def main():
    print(f"Initializing Azure OpenAI client ({MODEL})...")
    client = create_client()

    targets = [
        EXP_DIR / "results_triviaqa.json",
        EXP_DIR / "results_naturalquestions.json",
    ]

    for path in targets:
        if not path.exists():
            print(f"  Skipping {path} -- not found")
            continue
        relabel_dataset(client, path)

    print("\nDone!")


if __name__ == "__main__":
    main()
