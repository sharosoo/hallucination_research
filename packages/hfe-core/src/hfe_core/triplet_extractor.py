"""
Triplet Extractor using LLM (QuCo-RAG style)

Prompt templates and parser from QuCo-RAG (commit 0bf02a4):
https://github.com/ZhishanQ/QuCo-RAG/blob/0bf02a4943348635107490f429c873355981dc73/src/prompt_templates.json
https://github.com/ZhishanQ/QuCo-RAG/blob/0bf02a4943348635107490f429c873355981dc73/src/generate_quco.py#L302

Paper: "QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic RAG"
https://arxiv.org/abs/2512.19134
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"

# fmt: off
# Source: https://github.com/ZhishanQ/QuCo-RAG/blob/0bf02a4/src/prompt_templates.json
ENTITY_EXTRACTION_PROMPT = """Extract the knowledge triples that convey core semantic information from the given sentence, strictly following the output format used in the examples.
The sentence may be a question or a declarative sentence.
* **If the sentence is a question**, return [[entity, relation]], because you don't know the answer.
  * This question could be a multi-hop question. In that case, **only consider the first relation**.
  * You don't need to include any query about entities **not explicitly mentioned** in the question.
* **If the sentence contains one knowledge triple**, return a list in the format:
  [[head_entity_1, relation_1, tail_entity_1]]
* **If the sentence contains no factual semantic information involving entities**, return an empty list: `[]`
* **If the sentence contains more than one knowledge triple**, return a list in the format:
  [[head_entity_1, relation_1, tail_entity_1], [head_entity_2, relation_2, tail_entity_2], ...]
---
**Example 1:**
Sentence:
Which film came out first, Kumbasaram or Mystery Of The 13Th Guest?
entities:
[["Kumbasaram", "came out"], ["Mystery of the 13th Guest", "came out"]]

**Example 2:**
Sentence:
Kumbasaram was released in 2017.
entities:
[["Kumbasaram", "released in", "2017"]]

**Example 3:**
Sentence:
Thus, Kumbasaram came out first.
entities:
[]

**Example 4:**
Sentence:
Where did Diane Meyer Simon's husband graduate from?
entities:
[["Diane Meyer Simon", "husband"]]

**Example 5:**
Sentence:
Diane Meyer Simon's husband is George F. Simon.
entities:
[["Diane Meyer Simon", "husband", "George F. Simon"]]

**Example 6:**
Sentence:
Rajiv Rai was born in Mumbai, Maharashtra, India.
entities:
[["Rajiv Rai", "place of birth", "Mumbai, Maharashtra, India"]]

**Example 7:**
Sentence:
Beowulf & Grendel was directed by Sturla Gunnarsson.
entities:
[["Beowulf & Grendel", "directed by", "Sturla Gunnarsson"]]

---
Sentence:
{}
entities:
"""

# Source: https://github.com/ZhishanQ/QuCo-RAG/blob/0bf02a4/src/prompt_templates.json
ENTITY_EXTRACTION_PROMPT_FOR_QUESTION = """Extract the key entities that convey core semantic information from the given question, strictly following the output format used in the examples.
---
**Example 1:**
Sentence:
Which film came out first, Kumbasaram or Mystery Of The 13Th Guest?
entities:
[["Kumbasaram"], ["Mystery of the 13th Guest"]]

**Example 2:**
Sentence:
Who is the mother of the director of film Polish-Russian War (Film)?
entities:
[["Polish-Russian War (Film)"]]

**Example 3:**
Sentence:
When did John V, Prince Of Anhalt-Zerbst's father die?
entities:
[["John V, Prince of Anhalt-Zerbst"]]

**Example 4:**
Sentence:
Blackfin is a family of processors developed by the company that is headquartered in what city?
entities:
[["Blackfin"]]

**Example 5:**
Sentence:
Were both of the following rock groups formed in California: Dig and Thinking Fellers Union Local 282?
entities:
[["Dig rock group"], ["Thinking Fellers Union Local 282 rock group"]]

**Example 6:**
Sentence:
Are Billy and Barak both breeds of scenthound? (Barak is also known as a Bosnian Coarse-haired Hound)
entities:
[["Billy"], ["Barak"]]

**Example 7:**
Sentence:
Are both Volvic and Canfield's Diet Chocolate Fudge natural spring waters?
entities:
[["Volvic"], ["Canfield's Diet Chocolate Fudge"]]

**Example 8:**
Sentence:
Has The Baseball Project released any albums together?
entities:
[["The Baseball Project"]]

**Example 9:**
Sentence:
How old was Heard when he co-produced with Buck?
entities:
[["Heard"], ["Buck"]]

**Example 10:**
Sentence:
Are more people today related to Genghis Khan than Julius Caesar?
entities:
[["Genghis Khan"], ["Julius Caesar"]]

---
Sentence:
{}
entities:
"""
# fmt: on


@dataclass
class Triplet:
    head: str
    relation: str | None
    tail: str | None

    @property
    def is_binary(self) -> bool:
        return self.tail is None

    @property
    def is_ternary(self) -> bool:
        return self.tail is not None

    @property
    def entities(self) -> list[str]:
        entities = [self.head]
        if self.tail:
            entities.append(self.tail)
        return entities

    def __repr__(self) -> str:
        if self.is_ternary:
            return f"({self.head}, {self.relation}, {self.tail})"
        elif self.relation:
            return f"({self.head}, {self.relation})"
        else:
            return f"({self.head})"


@dataclass
class ExtractionResult:
    triplets: list[Triplet]
    raw_response: str
    source_text: str

    @property
    def entities(self) -> list[str]:
        entities = []
        for t in self.triplets:
            entities.extend(t.entities)
        return list(dict.fromkeys(entities))

    @property
    def entity_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for t in self.triplets:
            if t.is_ternary:
                pairs.append((t.head, t.tail))
        return pairs


# Source: https://github.com/ZhishanQ/QuCo-RAG/blob/0bf02a4/src/generate_quco.py#L302
def _parse_case(case_str: str) -> list:
    case_str = case_str.strip()
    if not case_str:
        return []

    stack = []
    current_token = []
    in_quotes = False
    escape_next = False
    expected_quote = None
    root = None

    quote_pairs = {
        "\u201c": "\u201d",
        '"': '"',
        "'": "'",
    }
    closing_quotes = {closing: opening for opening, closing in quote_pairs.items()}

    for ch in case_str:
        if in_quotes:
            if escape_next:
                current_token.append(ch)
                escape_next = False
                continue

            if ch == "\\":
                escape_next = True
                continue

            if expected_quote and ch == expected_quote:
                in_quotes = False
                expected_quote = None
                continue

            current_token.append(ch)
            continue

        if ch in quote_pairs:
            in_quotes = True
            expected_quote = quote_pairs[ch]
            continue

        if ch in closing_quotes:
            current_token.append(ch)
            continue

        if ch == "[":
            new_list = []
            if stack:
                stack[-1].append(new_list)
            stack.append(new_list)
            if root is None:
                root = stack[0]
            continue

        if ch == "]":
            token = "".join(current_token).strip()
            if token:
                stack[-1].append(token)
            current_token = []
            if stack:
                finished = stack.pop()
                if not stack:
                    root = finished
            continue

        if ch == ",":
            if stack:
                current_list = stack[-1]
                if len(stack) >= 2 and len(current_list) >= 2 and current_token:
                    current_token.append(ch)
                    continue

            token = "".join(current_token).strip()
            if token and stack:
                stack[-1].append(token)
            current_token = []
            continue

        if ch.isspace():
            if current_token:
                current_token.append(" ")
            continue

        current_token.append(ch)

    if stack:
        token = "".join(current_token).strip()
        if token:
            stack[-1].append(token)

    return root or []


def _parse_triplet_response(response: str) -> list[Triplet]:
    response = response.strip()

    if "assistantfinal" in response:
        response = response.split("assistantfinal")[-1].strip()

    if response.lower().startswith("entities:"):
        response = response[9:].strip()

    if not response or response == "[]":
        return []

    start_idx = response.find("[")
    if start_idx == -1:
        return []
    response = response[start_idx:]

    parsed = _parse_case(response)

    if not isinstance(parsed, list):
        return []

    def strip_quotes(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        return s

    triplets = []
    for item in parsed:
        if not isinstance(item, list) or len(item) == 0:
            continue

        if len(item) == 1:
            triplets.append(
                Triplet(head=strip_quotes(str(item[0])), relation=None, tail=None)
            )
        elif len(item) == 2:
            triplets.append(
                Triplet(
                    head=strip_quotes(str(item[0])),
                    relation=strip_quotes(str(item[1])),
                    tail=None,
                )
            )
        elif len(item) >= 3:
            triplets.append(
                Triplet(
                    head=strip_quotes(str(item[0])),
                    relation=strip_quotes(str(item[1])),
                    tail=strip_quotes(str(item[2])),
                )
            )

    return triplets


class TripletExtractor:
    """
    로컬 LLM 기반 Triplet Extractor (QuCo-RAG 방식)

    Usage:
        extractor = TripletExtractor(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        result = extractor.extract("Marie Curie was born in Warsaw.")
    """

    _shared_model = None
    _shared_tokenizer = None
    _shared_model_name = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        cache_enabled: bool = True,
    ):
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._cache: dict[str, list[Triplet]] = {}

        if (
            TripletExtractor._shared_model is None
            or TripletExtractor._shared_model_name != model_name
        ):
            self._init_model(device_map)
        else:
            self.model = TripletExtractor._shared_model
            self.tokenizer = TripletExtractor._shared_tokenizer

    def _init_model(self, device_map: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Run: pip install transformers"
            )

        logger.info(f"Loading local model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=device_map
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        TripletExtractor._shared_model = self.model
        TripletExtractor._shared_tokenizer = self.tokenizer
        TripletExtractor._shared_model_name = self.model_name

        logger.info(f"TripletExtractor initialized with {self.model_name}")

    def _generate(self, prompt: str) -> str:
        try:
            from openai_harmony import (
                SystemContent,
                ReasoningEffort,
                HarmonyEncodingName,
                load_harmony_encoding,
                Conversation,
                Message,
                Role,
                TextContent,
            )

            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            system_content = SystemContent.new().with_reasoning_effort(
                ReasoningEffort.LOW
            )
            system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)
            user_msg = Message.from_role_and_content(
                Role.USER, TextContent(text=prompt)
            )

            convo = Conversation.from_messages([system_msg, user_msg])
            input_ids = encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )
            stop_token_ids = encoding.stop_tokens_for_assistant_actions()

            import torch

            inputs = torch.tensor([input_ids], device=self.model.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                do_sample=False,
            )

            generated = outputs[:, len(input_ids) :]
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return ""

    def extract(self, text: str, is_question: bool = False) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(triplets=[], raw_response="", source_text=text)

        cache_key = f"{text}::{is_question}"
        if self.cache_enabled and cache_key in self._cache:
            return ExtractionResult(
                triplets=self._cache[cache_key],
                raw_response="[cached]",
                source_text=text,
            )

        if is_question:
            prompt = ENTITY_EXTRACTION_PROMPT_FOR_QUESTION.format(text)
        else:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text)

        response = self._generate(prompt)
        triplets = _parse_triplet_response(response)

        if self.cache_enabled:
            self._cache[cache_key] = triplets

        return ExtractionResult(
            triplets=triplets, raw_response=response, source_text=text
        )

    def extract_from_qa(
        self, question: str, answer: str
    ) -> tuple[ExtractionResult, ExtractionResult]:
        q_result = self.extract(question, is_question=True)
        a_result = self.extract(answer, is_question=False)
        return q_result, a_result


@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy

    return spacy.load("en_core_web_sm")


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    nlp = _load_spacy_model()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return [s for s in sentences if s]


def extract_triplets(
    text: str,
    is_question: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> list[Triplet]:
    extractor = TripletExtractor(model_name=model_name)
    result = extractor.extract(text, is_question=is_question)
    return result.triplets


def extract_entities_from_triplets(triplets: list[Triplet]) -> list[str]:
    entities = []
    for t in triplets:
        entities.extend(t.entities)
    return list(dict.fromkeys(entities))
