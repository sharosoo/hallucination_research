"""Qwen model wrapper for hallucination detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch


class QwenWrapper:
    """
    Wrapper for Qwen2.5 models optimized for hallucination detection.

    Supports various sizes: 0.5B, 1.5B, 3B, 7B
    All provide access to internal states for HFE analysis.
    """

    MODEL_SIZES = {
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "3b": "Qwen/Qwen2.5-3B-Instruct",
        "7b": "Qwen/Qwen2.5-7B-Instruct",
    }

    def __init__(
        self,
        size: str = "3b",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        self.size = size
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.model_name = self.MODEL_SIZES.get(size, size)
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        do_sample: bool = True,
        return_scores: bool = False,
    ) -> dict:
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=return_scores,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        result = {"text": text, "generated_ids": generated_ids}

        if return_scores:
            result["scores"] = outputs.scores

        return result

    def get_hidden_states(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, ...]:
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        return outputs.hidden_states
