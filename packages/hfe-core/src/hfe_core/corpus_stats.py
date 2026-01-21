"""
Corpus Statistics Module

Infini-gram API를 사용하여 pre-training corpus 통계를 쿼리하는 모듈.
QuCo-RAG 논문의 방법론을 따라 구현.

주요 기능:
- Entity 빈도 쿼리 (frequency)
- Entity 동시출현 쿼리 (co-occurrence)
- Corpus coverage 계산 (Triplet 기반)

Reference:
- QuCo-RAG: https://github.com/ZhishanQ/QuCo-RAG
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from hfe_core.triplet_extractor import TripletExtractor, Triplet

logger = logging.getLogger(__name__)


DEFAULT_INDEX = "v4_dolma-v1_7_llama"
API_ENDPOINT = "https://api.infini-gram.io/"

ENTITY_FREQ_THRESHOLD = 1000  # τ_entity (QuCo-RAG)
COOC_THRESHOLD = 1  # τ_cooc (QuCo-RAG)
COOC_WINDOW_SIZE = 1000
MAX_QUERY_CHARS = 1000
MAX_QUERY_TOKENS = 500
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30


@dataclass
class FrequencyResult:
    """Entity 빈도 쿼리 결과"""

    query: str
    count: int
    is_approximate: bool
    latency_ms: float
    token_ids: list[int]
    tokens: list[str]

    @property
    def is_low_frequency(self) -> bool:
        """QuCo-RAG 기준 저빈도 여부"""
        return self.count < ENTITY_FREQ_THRESHOLD

    @property
    def log_frequency(self) -> float:
        """Log-scaled frequency (0~1 정규화용)"""
        import math

        if self.count == 0:
            return 0.0
        max_log = math.log1p(1e12)
        return math.log1p(self.count) / max_log


@dataclass
class CooccurrenceResult:
    """동시출현 쿼리 결과"""

    entity1: str
    entity2: str
    count: int
    is_approximate: bool
    latency_ms: float
    window_size: int

    @property
    def has_cooccurrence(self) -> bool:
        """동시출현 존재 여부"""
        return self.count >= COOC_THRESHOLD

    @property
    def is_hallucination_risk(self) -> bool:
        """환각 위험 여부 (동시출현 0)"""
        return self.count == 0


@dataclass
class CorpusCoverage:
    """Corpus Coverage 계산 결과"""

    freq_score: float
    cooc_score: float
    coverage: float
    entity_frequencies: dict[str, int]
    cooc_results: list[CooccurrenceResult]
    num_entities: int
    num_cooc_pairs: int
    total_latency_ms: float


class InfiniGramClient:
    """
    Infini-gram API 클라이언트

    Usage:
        client = InfiniGramClient()
        freq = client.count("Paris")
        cooc = client.count_cooccurrence("Paris", "France")
    """

    def __init__(
        self,
        index: str = DEFAULT_INDEX,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Args:
            index: Corpus index 이름 (기본: Dolma v1.7)
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.index = index
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _query(self, payload: dict) -> dict:
        """API 쿼리 실행 (재시도 로직 포함)"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    API_ENDPOINT, json=payload, timeout=self.timeout
                )
                result = response.json()

                if "error" in result:
                    logger.warning(f"API Error: {result['error']}")
                    return result

                return result

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Request failed, retrying in {wait_time}s... "
                        f"(Attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retries failed: {e}")
                    return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    def count(self, query: str) -> FrequencyResult | None:
        """
        Entity/n-gram 빈도 쿼리

        Args:
            query: 검색할 텍스트 (entity 또는 n-gram)

        Returns:
            FrequencyResult 또는 None (에러 시)
        """
        if len(query) > MAX_QUERY_CHARS:
            logger.warning(f"Query too long ({len(query)} chars), truncating...")
            query = query[:MAX_QUERY_CHARS]

        payload = {"index": self.index, "query_type": "count", "query": query}

        result = self._query(payload)

        if "error" in result:
            return None

        return FrequencyResult(
            query=query,
            count=result.get("count", 0),
            is_approximate=result.get("approx", False),
            latency_ms=result.get("latency", 0.0),
            token_ids=result.get("token_ids", []),
            tokens=result.get("tokens", []),
        )

    def count_cooccurrence(
        self,
        entity1: str,
        entity2: str,
        max_diff_tokens: int = COOC_WINDOW_SIZE,
    ) -> CooccurrenceResult | None:
        """
        두 entity의 동시출현 빈도 쿼리 (CNF 형식)

        Args:
            entity1: 첫 번째 entity
            entity2: 두 번째 entity
            max_diff_tokens: 최대 토큰 거리 (기본 1000)

        Returns:
            CooccurrenceResult 또는 None (에러 시)
        """
        cnf_query = f"({entity1}) AND ({entity2})"

        payload = {
            "index": self.index,
            "query_type": "count",
            "query": cnf_query,
            "max_diff_tokens": max_diff_tokens,
        }

        result = self._query(payload)

        if "error" in result:
            return None

        return CooccurrenceResult(
            entity1=entity1,
            entity2=entity2,
            count=result.get("count", 0),
            is_approximate=result.get("approx", False),
            latency_ms=result.get("latency", 0.0),
            window_size=max_diff_tokens,
        )

    def batch_count(self, queries: list[str]) -> list[FrequencyResult | None]:
        """
        여러 entity의 빈도를 일괄 쿼리

        Args:
            queries: 검색할 텍스트 리스트

        Returns:
            FrequencyResult 리스트 (에러 시 해당 위치는 None)
        """
        results = []
        for query in queries:
            result = self.count(query)
            results.append(result)
            time.sleep(0.05)
        return results

    def batch_count_cooccurrence(
        self,
        pairs: list[tuple[str, str]],
        max_diff_tokens: int = COOC_WINDOW_SIZE,
    ) -> list[CooccurrenceResult | None]:
        """
        여러 entity 쌍의 동시출현을 일괄 쿼리

        Args:
            pairs: (entity1, entity2) 튜플 리스트
            max_diff_tokens: 최대 토큰 거리

        Returns:
            CooccurrenceResult 리스트 (에러 시 해당 위치는 None)
        """
        results = []
        for entity1, entity2 in pairs:
            result = self.count_cooccurrence(entity1, entity2, max_diff_tokens)
            results.append(result)
            time.sleep(0.05)
        return results


class CorpusCoverageCalculator:
    """
    Corpus Coverage 계산기

    QuCo-RAG 논문의 방법론을 기반으로 corpus coverage 계산:
    - FreqScore: entity 빈도의 정규화된 평균
    - CoocScore: 동시출현이 존재하는 entity 쌍의 비율

    Usage:
        calculator = CorpusCoverageCalculator()
        coverage = calculator.compute(
            entities_q=["Paris", "France"],
            entities_a=["Eiffel Tower"]
        )
    """

    def __init__(
        self,
        client: InfiniGramClient | None = None,
        freq_weight: float = 0.5,
        cooc_weight: float = 0.5,
    ):
        """
        Args:
            client: InfiniGramClient 인스턴스 (없으면 새로 생성)
            freq_weight: FreqScore 가중치 (기본 0.5)
            cooc_weight: CoocScore 가중치 (기본 0.5)
        """
        self.client = client or InfiniGramClient()
        self.freq_weight = freq_weight
        self.cooc_weight = cooc_weight

        total = self.freq_weight + self.cooc_weight
        self.freq_weight /= total
        self.cooc_weight /= total

    def compute(
        self,
        entities_q: list[str],
        entities_a: list[str],
    ) -> CorpusCoverage:
        """
        Corpus coverage 계산

        Args:
            entities_q: 질문에서 추출한 entity 리스트
            entities_a: 답변에서 추출한 entity 리스트

        Returns:
            CorpusCoverage 결과
        """
        start_time = time.time()

        all_entities = list(set(entities_q + entities_a))

        freq_results = self.client.batch_count(all_entities)
        entity_frequencies = {}
        valid_freq_scores = []

        for entity, result in zip(all_entities, freq_results):
            if result is not None:
                entity_frequencies[entity] = result.count
                valid_freq_scores.append(result.log_frequency)
            else:
                entity_frequencies[entity] = 0
                valid_freq_scores.append(0.0)

        freq_score = (
            sum(valid_freq_scores) / len(valid_freq_scores)
            if valid_freq_scores
            else 0.0
        )

        cooc_pairs = [(eq, ea) for eq in entities_q for ea in entities_a if eq != ea]
        cooc_results = []
        cooc_exists_count = 0

        if cooc_pairs:
            cooc_query_results = self.client.batch_count_cooccurrence(cooc_pairs)
            for result in cooc_query_results:
                if result is not None:
                    cooc_results.append(result)
                    if result.has_cooccurrence:
                        cooc_exists_count += 1

        cooc_score = cooc_exists_count / len(cooc_pairs) if cooc_pairs else 1.0

        coverage = self.freq_weight * freq_score + self.cooc_weight * cooc_score

        total_latency = (time.time() - start_time) * 1000

        return CorpusCoverage(
            freq_score=freq_score,
            cooc_score=cooc_score,
            coverage=coverage,
            entity_frequencies=entity_frequencies,
            cooc_results=cooc_results,
            num_entities=len(all_entities),
            num_cooc_pairs=len(cooc_pairs),
            total_latency_ms=total_latency,
        )

    def compute_from_text(
        self,
        question: str,
        answer: str,
        triplet_extractor: "TripletExtractor | None" = None,
    ) -> CorpusCoverage:
        """
        텍스트에서 직접 corpus coverage 계산 (QuCo-RAG Triplet Extractor 사용)
        """
        if triplet_extractor is None:
            from hfe_core.triplet_extractor import TripletExtractor

            triplet_extractor = TripletExtractor()

        q_result = triplet_extractor.extract(question, is_question=True)
        a_result = triplet_extractor.extract(answer, is_question=False)

        entities_q = q_result.entities
        entities_a = a_result.entities

        return self.compute(entities_q, entities_a)

    def compute_from_triplets(
        self,
        triplets_q: list["Triplet"],
        triplets_a: list["Triplet"],
    ) -> CorpusCoverage:
        """
        추출된 triplet에서 corpus coverage 계산 (QuCo-RAG Stage 2 방식)

        QuCo-RAG에서는 ternary triplet (head, relation, tail)의 head-tail 쌍으로
        동시출현을 확인합니다.
        """
        start_time = time.time()

        entities_q = []
        for t in triplets_q:
            entities_q.extend(t.entities)
        entities_q = list(dict.fromkeys(entities_q))

        entities_a = []
        for t in triplets_a:
            entities_a.extend(t.entities)
        entities_a = list(dict.fromkeys(entities_a))

        all_entities = list(set(entities_q + entities_a))

        freq_results = self.client.batch_count(all_entities)
        entity_frequencies = {}
        valid_freq_scores = []

        for entity, result in zip(all_entities, freq_results):
            if result is not None:
                entity_frequencies[entity] = result.count
                valid_freq_scores.append(result.log_frequency)
            else:
                entity_frequencies[entity] = 0
                valid_freq_scores.append(0.0)

        freq_score = (
            sum(valid_freq_scores) / len(valid_freq_scores)
            if valid_freq_scores
            else 0.0
        )

        cooc_pairs = []
        for t in triplets_a:
            if t.is_ternary:
                cooc_pairs.append((t.head, t.tail))

        if not cooc_pairs:
            cooc_pairs = [
                (eq, ea) for eq in entities_q for ea in entities_a if eq != ea
            ]

        cooc_results = []
        cooc_exists_count = 0

        if cooc_pairs:
            cooc_query_results = self.client.batch_count_cooccurrence(cooc_pairs)
            for result in cooc_query_results:
                if result is not None:
                    cooc_results.append(result)
                    if result.has_cooccurrence:
                        cooc_exists_count += 1

        cooc_score = cooc_exists_count / len(cooc_pairs) if cooc_pairs else 1.0

        coverage = self.freq_weight * freq_score + self.cooc_weight * cooc_score

        total_latency = (time.time() - start_time) * 1000

        return CorpusCoverage(
            freq_score=freq_score,
            cooc_score=cooc_score,
            coverage=coverage,
            entity_frequencies=entity_frequencies,
            cooc_results=cooc_results,
            num_entities=len(all_entities),
            num_cooc_pairs=len(cooc_pairs),
            total_latency_ms=total_latency,
        )


def get_entity_frequency(entity: str, client: InfiniGramClient | None = None) -> int:
    """단일 entity의 corpus 빈도 조회"""
    if client is None:
        client = InfiniGramClient()
    result = client.count(entity)
    return result.count if result else 0


def check_cooccurrence(
    entity1: str, entity2: str, client: InfiniGramClient | None = None
) -> bool:
    """두 entity의 동시출현 여부 확인"""
    if client is None:
        client = InfiniGramClient()
    result = client.count_cooccurrence(entity1, entity2)
    return result.has_cooccurrence if result else False


def is_low_frequency_entity(
    entity: str,
    threshold: int = ENTITY_FREQ_THRESHOLD,
    client: InfiniGramClient | None = None,
) -> bool:
    """Entity가 저빈도인지 확인 (QuCo-RAG 기준)"""
    freq = get_entity_frequency(entity, client)
    return freq < threshold
