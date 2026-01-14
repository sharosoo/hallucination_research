"""TruthfulQA dataset loader for hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset


@dataclass
class TruthfulQASample:
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str
    source: str


class TruthfulQALoader:
    """
    Load TruthfulQA dataset (Lin et al., 2022).

    ~817 questions designed to test truthfulness.
    Includes questions where common misconceptions lead to wrong answers.
    """

    DATASET_NAME = "truthful_qa"
    SUBSET = "generation"

    def __init__(self, split: str = "validation"):
        self.split = split
        self._dataset = None

    def load(self) -> None:
        self._dataset = load_dataset(self.DATASET_NAME, self.SUBSET, split=self.split)

    def __len__(self) -> int:
        if self._dataset is None:
            self.load()
        return len(self._dataset)

    def __iter__(self) -> Iterator[TruthfulQASample]:
        if self._dataset is None:
            self.load()

        for item in self._dataset:
            yield TruthfulQASample(
                question=item["question"],
                best_answer=item["best_answer"],
                correct_answers=item["correct_answers"],
                incorrect_answers=item["incorrect_answers"],
                category=item["category"],
                source=item["source"],
            )

    def __getitem__(self, idx: int) -> TruthfulQASample:
        if self._dataset is None:
            self.load()

        item = self._dataset[idx]
        return TruthfulQASample(
            question=item["question"],
            best_answer=item["best_answer"],
            correct_answers=item["correct_answers"],
            incorrect_answers=item["incorrect_answers"],
            category=item["category"],
            source=item["source"],
        )

    def get_prompt_format(self, sample: TruthfulQASample) -> str:
        return f"Question: {sample.question}\nAnswer:"

    def is_correct(self, sample: TruthfulQASample, response: str) -> bool:
        response_lower = response.lower().strip()
        for correct in sample.correct_answers:
            if correct.lower() in response_lower:
                return True
        return False
