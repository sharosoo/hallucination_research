"""vLLM wrapper for fast inference with multiple response sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


@dataclass
class GenerationResult:
    text: str
    log_prob: float
    tokens: list[str]
    token_log_probs: list[float]


class VLLMWrapper:
    """
    Wrapper for vLLM engine optimized for hallucination detection.

    Supports:
    - Fast batched inference
    - Multiple response sampling (for semantic entropy)
    - Log probability extraction (for internal energy)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm: LLM | None = None

    def load(self) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self.load()
        return self._llm

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        n: int = 1,
    ) -> list[list[GenerationResult]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            logprobs=1,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            prompt_results = []
            for completion in output.outputs:
                token_log_probs = []
                tokens = []

                if completion.logprobs:
                    for logprob_dict in completion.logprobs:
                        if logprob_dict:
                            for token, logprob_obj in logprob_dict.items():
                                tokens.append(token)
                                token_log_probs.append(logprob_obj.logprob)
                                break

                total_log_prob = sum(token_log_probs) if token_log_probs else 0.0
                avg_log_prob = (
                    total_log_prob / len(token_log_probs) if token_log_probs else 0.0
                )

                prompt_results.append(
                    GenerationResult(
                        text=completion.text,
                        log_prob=avg_log_prob,
                        tokens=tokens,
                        token_log_probs=token_log_probs,
                    )
                )
            results.append(prompt_results)

        return results

    def generate_with_multiple_samples(
        self,
        prompt: str,
        num_samples: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Generate multiple samples for semantic entropy calculation."""
        results = self.generate([prompt], n=num_samples, **kwargs)

        return [{"text": r.text, "log_prob": r.log_prob} for r in results[0]]
