"""
HuggingFace Transformers 기반 다중 샘플링 + Logit 추출

Semantic Entropy / Energy 계산을 위한 응답 샘플러.
각 토큰의 raw logit을 추출하여 Semantic Energy 계산에 사용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from hfe_core.nli_clusterer import Response


class HFSampler:
    """
    HuggingFace Transformers 기반 다중 샘플링 + Logit 추출

    Semantic Entropy / Energy 계산을 위해:
    1. 동일 질문에 대해 여러 응답 샘플링
    2. 각 토큰의 raw logit 추출 (Semantic Energy용)
    3. log probability 계산 (continuous Semantic Entropy용)

    Usage:
        from hfe_models import HFSampler
        from hfe_core import NLIClusterer, SemanticEntropyCalculator, SemanticEnergyCalculator

        sampler = HFSampler("Qwen/Qwen2.5-0.5B-Instruct")
        responses = sampler.sample("What is the capital of France?", num_samples=5)

        # 클러스터링
        clusterer = NLIClusterer()
        clusters = clusterer.cluster(responses)

        # Semantic Entropy
        entropy = SemanticEntropyCalculator.compute_from_clusters(clusters, len(responses))

        # Semantic Energy
        energy = SemanticEnergyCalculator.compute_energy_only(responses)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_name: HuggingFace 모델 이름
            device: 'cuda' 또는 'cpu' (None이면 자동 감지)
            torch_dtype: 모델 dtype (float16 권장)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        self._model = None
        self._tokenizer = None

    def load(self):
        """모델 로드 (lazy loading)"""
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self._model.eval()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @torch.no_grad()
    def sample(
        self,
        question: str,
        num_samples: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        system_prompt: str = "Answer the question concisely.",
    ) -> list["Response"]:
        """
        다중 응답 샘플링 + raw logit 추출

        Args:
            question: 입력 질문
            num_samples: 샘플링할 응답 수
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도 (높을수록 다양)
            system_prompt: 시스템 프롬프트

        Returns:
            Response 객체 리스트 (text, logits, log_probability 포함)
        """
        # Lazy import to avoid circular dependency
        from hfe_core.nli_clusterer import Response

        # Chat template 적용
        messages = [
            {"role": "system", "content": system_prompt},
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

            # 생성된 토큰
            generated_ids = outputs.sequences[0, input_len:].tolist()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 각 토큰의 raw logit 추출 (Semantic Energy용)
            token_logits = []
            log_prob_sum = 0.0

            for step_idx, token_id in enumerate(generated_ids):
                if step_idx >= len(outputs.scores):
                    break

                logits = outputs.scores[step_idx][0]  # (vocab_size,)

                # 선택된 토큰의 raw logit (softmax 전!)
                raw_logit = logits[token_id].item()
                token_logits.append(raw_logit)

                # log probability (continuous SE용)
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

    @torch.no_grad()
    def sample_raw(
        self,
        prompt: str,
        num_samples: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> list["Response"]:
        """
        Raw 프롬프트로 샘플링 (chat template 없이)

        Args:
            prompt: 입력 프롬프트 (chat template 미적용)
            num_samples: 샘플링할 응답 수
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            Response 객체 리스트
        """
        from hfe_core.nli_clusterer import Response

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
