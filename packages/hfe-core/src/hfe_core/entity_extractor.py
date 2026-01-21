"""
Entity Extractor using spaCy NER.

DEPRECATED: 이 모듈은 QuCo-RAG의 Triplet Extractor로 대체되었습니다.
새로운 코드에서는 triplet_extractor.py를 사용하세요.

from hfe_core.triplet_extractor import TripletExtractor
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache

logger = logging.getLogger(__name__)

warnings.warn(
    "entity_extractor is deprecated. Use triplet_extractor instead.",
    DeprecationWarning,
    stacklevel=2,
)

SPACY_MODEL = "en_core_web_sm"
ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "EVENT",
    "WORK_OF_ART",
    "PRODUCT",
    "NORP",
}


@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy

    return spacy.load(SPACY_MODEL)


class EntityExtractor:
    def __init__(self, entity_types: set[str] | None = None):
        self.entity_types = entity_types or ENTITY_TYPES
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = _load_spacy_model()
        return self._nlp

    def extract(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip()
                if entity_text and len(entity_text) > 1:
                    entities.append(entity_text)

        return list(dict.fromkeys(entities))

    def extract_with_labels(self, text: str) -> list[tuple[str, str]]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip()
                if entity_text and len(entity_text) > 1:
                    entities.append((entity_text, ent.label_))

        return entities

    def extract_from_qa(
        self, question: str, answer: str
    ) -> tuple[list[str], list[str]]:
        return self.extract(question), self.extract(answer)


def extract_entities(text: str) -> list[str]:
    extractor = EntityExtractor()
    return extractor.extract(text)
