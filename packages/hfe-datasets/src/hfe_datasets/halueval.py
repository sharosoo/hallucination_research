"""HaluEval dataset loader - Hallucination evaluation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset


@dataclass
class HaluEvalSample:
    question: str
    answer: str
    hallucination: int
    task: str
    knowledge: str | None = None
    right_answer: str | None = None


class HaluEvalLoader:
    """
    Load HaluEval dataset (Li et al., 2023).

    Contains QA, summarization, and dialogue samples with hallucination labels.
    Hallucination=1 means the answer contains hallucinated content.
    """

    DATASET_NAME = "pminervini/HaluEval"

    def __init__(self, task: str = "qa", split: str = "data"):
        self.task = task
        self.split = split
        self._dataset = None

    def load(self) -> None:
        self._dataset = load_dataset(self.DATASET_NAME, self.task, split=self.split)

    def __len__(self) -> int:
        if self._dataset is None:
            self.load()
        return len(self._dataset)

    def __iter__(self) -> Iterator[HaluEvalSample]:
        if self._dataset is None:
            self.load()

        for item in self._dataset:
            yield self._parse_item(item)

    def __getitem__(self, idx: int) -> HaluEvalSample:
        if self._dataset is None:
            self.load()

        return self._parse_item(self._dataset[idx])

    def _parse_item(self, item: dict) -> HaluEvalSample:
        if self.task == "qa":
            return HaluEvalSample(
                question=item.get("question", ""),
                answer=item.get("hallucinated_answer", item.get("answer", "")),
                hallucination=1
                if "hallucinated" in str(item.get("hallucinated_answer", ""))
                else 0,
                task="qa",
                knowledge=item.get("knowledge", None),
                right_answer=item.get("right_answer", None),
            )
        elif self.task == "summarization":
            return HaluEvalSample(
                question=item.get("document", ""),
                answer=item.get("hallucinated_summary", item.get("summary", "")),
                hallucination=1,
                task="summarization",
            )
        else:
            return HaluEvalSample(
                question=item.get("dialogue_history", ""),
                answer=item.get("hallucinated_response", item.get("response", "")),
                hallucination=1,
                task="dialogue",
            )

    def get_prompt_format(self, sample: HaluEvalSample) -> str:
        if sample.task == "qa":
            if sample.knowledge:
                return f"Knowledge: {sample.knowledge}\nQuestion: {sample.question}\nAnswer:"
            return f"Question: {sample.question}\nAnswer:"
        elif sample.task == "summarization":
            return f"Document: {sample.question}\nSummary:"
        else:
            return f"Dialogue: {sample.question}\nResponse:"
