# Corpus 기반 적응형 환각 탐지

> **"혼란은 다양성을 만들고, 지어냄은 일관성을 만든다. Corpus 통계가 어떤 경우인지 알려준다."**

## 핵심 통찰

LLM이 환각을 일으킬 때, 모델이 해당 주제를 *아는지 여부*에 따라 근본적으로 다른 방식으로 환각한다:

| 유형 | 이름 | 원인 | 행동적 신호 | 최적 탐지기 |
|------|------|------|-------------|-------------|
| **A** | 혼란 (Confusion) | 모델이 알지만 헷갈림 | **다양한** 오답 | Semantic Entropy |
| **B** | 지어냄 (Confabulation) | 모델이 모르고 지어냄 | **일관된** 오답 | Semantic Energy |

**우리의 핵심 기여**: **사전학습 corpus 통계**를 사용하여 어떤 유형의 오류인지 판단하고, 이에 따라 SE와 Energy의 가중치를 적응적으로 조정한다.

---

## 연구 동기

### 왜 이 연구가 중요한가

1. **시급한 문제**: LLM 환각은 의료, 법률, 금융 분야에서 심각한 위험 초래
2. **현재 연구의 공백**: 기존 방법들은 불확실성을 *어떻게* 측정할지에 집중하지만, *언제* 어떤 방법이 효과적인지는 설명 못함
3. **실용적 영향**: RAG 없이 작동하는 실시간 환각 탐지 솔루션

### 핵심 관찰

| 데이터셋 | SE AUROC | Energy AUROC | 승자 |
|---------|----------|--------------|------|
| TruthfulQA | **0.619** | 0.538 | SE |
| HaluEval QA | 0.506 | **0.604** | Energy |

**질문**: 왜 같은 방법이 데이터셋마다 다르게 작동하는가?

**답**: 모델이 사전학습 중 관련 지식을 *학습했는지* 여부에 따라 다르다.

---

## 방법론 개요

### 적응형 가중치 공식

```
Score = w(corpus) × Energy + (1 - w(corpus)) × SE

여기서 w(corpus) = f(entity_frequency, co-occurrence)
```

### Corpus Coverage

사전학습 corpus 통계를 쿼리하여 모델이 해당 내용을 "아는지" 추정:

```python
# 높은 빈도 → 모델이 알 가능성 높음 → SE 사용
freq("Paris", "France") = 10,000,000  →  w = 0.3 (SE 가중치 높음)

# 낮은 빈도 → 모델이 모를 가능성 높음 → Energy 사용
freq("Silas Hardy") = 258  →  w = 0.8 (Energy 가중치 높음)

# 동시출현 0 → 지어냄 가능성 높음
cooc("Il Seduttore", "Mario Camerini") = 0  →  w = 0.9 (Energy 가중치 높음)
```

### 이론적 근거

**Physics of Language Models** (Allen-Zhu & Li, 2023)에서:
> "지식이 안정적으로 추출되려면, 사전학습 중 충분히 증강(paraphrasing, shuffling 등)되어야 한다. 이러한 증강 없이는 지식이 암기되더라도 추출 불가능할 수 있다."

**우리의 종합**:
- 높은 corpus 빈도 → 잘 학습됨 → 혼란형 오류 → SE 효과적
- 낮은 corpus 빈도 → 잘 학습 안됨 → 지어냄형 오류 → Energy 효과적

---

## 기존 연구 대비 차별점

| 방법 | 신호 | 출력 | RAG 필요 | 우리와의 차이 |
|------|------|------|----------|---------------|
| **SE** (Farquhar, 2024) | 내부 | 불확실성 점수 | 아니오 | 우리는 *언제* SE가 작동하는지 설명 |
| **Energy** (Ma, 2025) | 내부 | 확신도 점수 | 아니오 | 우리는 *언제* Energy가 작동하는지 설명 |
| **KLE/SNNE** | 내부 | 개선된 SE | 아니오 | 우리는 *외부* corpus 신호 추가 |
| **QuCo-RAG** (Min, 2025) | Corpus | 검색 트리거 | **예** | 우리는 *점수 자체* 개선, RAG 불필요 |

**우리의 고유한 위치**: Corpus coverage와 SE/Energy 효과성 간의 연결을 최초로 규명하여 적응형 결합의 이론적 근거 제공

---

## 실험 결과 (Baseline)

### 제로-엔트로피 문제

TruthfulQA에서 SE < 0.1인 샘플:
- 41개 샘플 중 31개 (75.6%)가 환각
- SE로는 구분 불가 (모두 ≈ 0)
- **Energy AUROC: 0.768** → Energy가 성공적으로 구분

이는 우리 가설을 검증: SE가 실패할 때 (일관된 지어냄), Energy가 탐지 가능.

---

## 프로젝트 구조

```
hallucination_lfe/
├── README.md
├── plan.md
├── 260116_plan.md
├── references/
│   ├── quco_rag_summary_korean.md
│   ├── KLE_summary_korean.md
│   └── snne-paper-summary-kr.md
├── packages/
│   └── hfe-core/
│       └── src/hfe_core/
│           ├── triplet_extractor.py   # QuCo-RAG 방식 LLM 기반
│           ├── corpus_stats.py        # Infini-gram API
│           ├── adaptive_weights.py    # 적응형 가중치
│           ├── semantic_entropy.py
│           └── semantic_energy.py
└── experiment_notes/
    ├── exp01_truthfulqa/
    ├── exp02_halueval/
    └── exp04_corpus_adaptive/         # Corpus 기반 실험
```

## 설치

```bash
uv sync
```

## Corpus 접근

**OLMo/Dolma corpus를 Infini-gram API**를 통해 proxy corpus로 사용.

**왜 proxy corpus가 작동하는가** (QuCo-RAG에서 검증):
- Web-scale corpus들은 상당 부분 겹침 (Common Crawl, Wikipedia 등)
- Cross-model transfer 검증됨: OLMo 통계가 Llama, Qwen, GPT-4에도 유효

### Triplet Extractor (QuCo-RAG 방식)

QuCo-RAG 논문의 방법론을 따라 **로컬 LLM 기반 Triplet Extractor** 사용:

```python
from hfe_core.triplet_extractor import TripletExtractor

# 로컬 모델 사용 (Qwen2.5-0.5B-Instruct)
extractor = TripletExtractor()

# 문장에서 triplet 추출
result = extractor.extract("Marie Curie was born in Warsaw, Poland.")
# → [("Marie Curie", "born in", "Warsaw, Poland")]

# 질문에서 entity 추출
result = extractor.extract("Who discovered radium?", is_question=True)
# → [("radium", "discovered")]
```

### Corpus Coverage 계산

```python
from hfe_core.corpus_stats import InfiniGramClient, CorpusCoverageCalculator

client = InfiniGramClient()
calculator = CorpusCoverageCalculator(client)

# Entity 빈도 쿼리
freq = client.count("Silas Hardy")  # → 258 (저빈도)
freq = client.count("Paris")        # → 10,000,000+ (고빈도)

# 동시출현 쿼리 (head-tail entity 쌍)
cooc = client.count_cooccurrence("Marie Curie", "Warsaw")  # → 존재
cooc = client.count_cooccurrence("Il Seduttore", "Mario Camerini")  # → 0 (환각 위험)
```

---

## 참고문헌

### 핵심 방법론
1. **Semantic Entropy**: Farquhar et al., Nature 2024
2. **Semantic Energy**: Ma et al., arXiv:2412.07965, 2025
3. **QuCo-RAG**: Min et al., arXiv:2512.19134, 2025

### 불확실성 정량화
4. **KLE**: Nikitin et al., arXiv:2405.20003, 2024
5. **SNNE**: Nguyen et al., arXiv:2506.00245, 2025
6. **Semantic Volume**: Li et al., arXiv:2502, 2025
7. **SE Probes**: Kossen et al., arXiv:2406, 2024

### 지식 경계
8. **Physics of LLMs Part 3.1**: Allen-Zhu & Li, arXiv:2309, 2023
9. **Knowledge Boundary Survey**: Ren et al., ACL 2025
10. **Dated Data**: Cheng et al., arXiv:2403, 2024

### 환각 탐지
11. **Cleanse**: Joo & Cho, arXiv:2507.14649, 2025
12. **BTProp**: Hou et al., arXiv:2406, 2024
13. **HaloScope**: Du et al., arXiv:2409, 2024
