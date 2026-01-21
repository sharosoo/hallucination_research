# QuCo-RAG 논문 요약 (한국어)

> **논문**: QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation  
> **저자**: Dehai Min, Kailin Zhang, Tongtong Wu, Lu Cheng  
> **소속**: University of Illinois at Chicago, NYU, Monash University  
> **arXiv**: 2512.19134 (2025년 12월)  
> **코드**: https://github.com/ZhishanQ/QuCo-RAG

---

## 1. 핵심 아이디어

### 1.1 문제 제기

기존 Dynamic RAG 방법들은 **모델 내부 신호** (logits, entropy 등)에 의존하여 retrieval 타이밍을 결정한다. 그러나:

- LLM은 **poorly calibrated** (잘못된 답에도 높은 확신도)
- **Confident hallucination**: 틀린 내용을 자신있게 생성
- 내부 신호는 근본적으로 신뢰할 수 없음

### 1.2 해결책: Corpus 통계 기반 불확실성

QuCo-RAG는 **주관적 내부 확신도** 대신 **객관적 corpus 통계**를 사용:

```
내부 신호 (unreliable)     →    외부 corpus 통계 (reliable)
- logits                        - entity frequency
- entropy                       - co-occurrence
- attention                     
```

### 1.3 핵심 통찰

> "LLM의 사실적 지식은 근본적으로 사전학습 corpus에 의해 형성된다"

- **저빈도 entity** → long-tail 지식, 환각 위험 높음
- **동시출현 0** → 모델이 두 entity 관계를 학습한 적 없음 → 환각 위험 높음

---

## 2. 방법론

### 2.1 두 단계 검증 (Two-Stage Detection)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Pre-Generation Knowledge Assessment (생성 전 지식 평가)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 질문에서 entity 추출: E_Q = {e1, e2, ..., em}                          │
│  2. 각 entity의 corpus 빈도 쿼리: freq(e; P)                               │
│  3. 평균 빈도가 임계값 미만이면 retrieval 트리거:                          │
│                                                                             │
│     δ_pre = 1  if  Avg(freq(e)) < τ_entity                                 │
│                                                                             │
│  기본값: τ_entity = 10^3 (1,000회)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Runtime Claim Verification (런타임 주장 검증)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 생성된 문장에서 knowledge triplet 추출: T = {(h, r, t)}                │
│     - h: head entity, r: relation, t: tail entity                          │
│                                                                             │
│  2. head-tail entity 동시출현 빈도 계산:                                   │
│                                                                             │
│     cooc(h, t; P) = |{ω ∈ P : h ∈ ω ∧ t ∈ ω}|                             │
│                                                                             │
│     ω = 윈도우 (기본 1,000 토큰, passage 수준)                             │
│                                                                             │
│  3. 동시출현이 임계값 미만이면 retrieval 트리거:                           │
│                                                                             │
│     δ_i = 1  if  min cooc(h, t) < τ_cooc                                   │
│                                                                             │
│  기본값: τ_cooc = 1 (즉, 동시출현 0이면 트리거)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 왜 동시출현 0이 중요한가?

```
동시출현 > 0  →  정확성 보장 안 됨 (다른 맥락일 수 있음)
동시출현 = 0  →  환각 위험 강하게 시사 (evidential support 없음)
```

> **비대칭적 특성**: 동시출현이 있다고 정답은 아니지만, 동시출현이 없으면 환각일 가능성 높음

### 2.3 왜 cooc(h, t)만 계산하는가? (relation r 제외)

- **relation은 어휘적 변이가 큼**: "employed by" vs "worked at" vs "was hired by"
- **entity는 어휘적으로 안정적**: "Barack Obama", "Paris" 등 고유명사

---

## 3. 구현 세부사항

### 3.1 Infini-gram API

**Infini-gram**: suffix array 기반 corpus 통계 엔진, 4조 토큰에 밀리초 지연시간 쿼리

```python
# API 엔드포인트
# https://infini-gram.readthedocs.io/en/latest/api.html

# Entity 빈도 쿼리
freq = infini_gram.count("Silas Hardy")  # → 258

# 동시출현 쿼리 (윈도우 1000 토큰 내)
cooc = infini_gram.count("Il Seduttore", "Franco Rossi", window=1000)  # → 0
```

**Corpus**: OLMo-2 사전학습 corpus (Dolma), 약 4조 토큰

### 3.2 Entity/Triplet 추출

**Lightweight Triplet Extractor**:
- GPT-4o-mini로 40K 예제 생성 (in-context learning)
- Qwen2.5-0.5B-Instruct로 distillation (full fine-tuning)
- 결과: 0.5B 파라미터의 경량 추출기

**추출 형식**:
```
입력: "Marie Curie was born in Warsaw, Poland."
출력: [("Marie Curie", "born in", "Warsaw"), ("Warsaw", "located in", "Poland")]
```

### 3.3 임계값 설정

| 파라미터 | 기본값 | 의미 | 민감도 |
|---------|--------|------|--------|
| `τ_entity` | 10^3 | entity 빈도 임계값 | 10^3 ~ 10^7 범위에서 안정적 |
| `τ_cooc` | 1 | 동시출현 임계값 | 0 vs 1+ 이진 판단 |
| `window` | 1000 토큰 | 동시출현 윈도우 크기 | passage 수준 |

---

## 4. 실험 결과

### 4.1 주요 결과 (OLMo-2)

| 모델 | 데이터셋 | QuCo-RAG EM | 최고 baseline | 향상 |
|------|---------|-------------|---------------|------|
| OLMo-2-7B | 2WikiMQA | 32.7 | 25.3 | **+7.4** |
| OLMo-2-7B | HotpotQA | 35.3 | 29.7 | **+5.6** |
| OLMo-2-13B | 2WikiMQA | 41.7 | 29.7 | **+12.0** |
| OLMo-2-13B | HotpotQA | 35.0 | 29.7 | **+5.3** |
| OLMo-2-32B | 2WikiMQA | 46.8 | 37.4 | **+9.4** |
| OLMo-2-32B | HotpotQA | 41.6 | 30.8 | **+10.8** |

### 4.2 Cross-Model Transfer (Proxy Corpus)

**핵심 발견**: OLMo corpus를 다른 모델의 proxy로 사용해도 효과적!

| 모델 | 데이터셋 | QuCo-RAG EM | 향상 |
|------|---------|-------------|------|
| Qwen2.5-32B | 2WikiMQA | 50.0 | **+14.1** |
| Llama-3-8B | 2WikiMQA | 38.4 | **+4.9** |
| GPT-4.1 | 2WikiMQA | 64.6 | **+4.6** |
| GPT-5-chat | 2WikiMQA | 59.7 | **+8.7** |

**왜 작동하는가?**
> Web-scale 사전학습 corpus들은 상당 부분 겹침 (Common Crawl, Wikipedia 등)

### 4.3 효율성

- **평균 retrieval 횟수**: 1.70회/질문 (다른 방법들보다 적음)
- **토큰 소비**: 87 토큰/질문 (FS-RAG의 2-4배 적음)
- **LLM 호출**: 1.84회/질문

---

## 5. 우리 연구에의 적용

### 5.1 핵심 차용 포인트

| QuCo-RAG 요소 | 우리 연구 적용 |
|---------------|----------------|
| Entity frequency | Corpus coverage 계산 |
| Co-occurrence = 0 → 환각 | Zero-SE 문제와 연결 |
| Proxy corpus (OLMo) | 동일하게 사용 |
| Infini-gram API | 동일하게 사용 |

### 5.2 차별점

| QuCo-RAG | 우리 연구 |
|----------|-----------|
| Retrieval 트리거 결정 | SE/Energy 가중치 결정 |
| RAG 필요 | RAG 불필요 |
| 이진 결정 (retrieve/not) | 연속 가중치 (0~1) |

### 5.3 구현 계획

```python
# 우리의 Corpus Coverage 계산 (QuCo-RAG 참고)
def compute_corpus_coverage(question: str, answer: str) -> float:
    """
    QuCo-RAG의 entity frequency + co-occurrence를 결합
    """
    # 1. Entity 추출
    entities_q = extract_entities(question)
    entities_a = extract_entities(answer)
    
    # 2. Frequency Score (QuCo-RAG Stage 1)
    freq_scores = [infini_gram.count(e) for e in entities_q | entities_a]
    freq_score = np.mean([np.log1p(f) / np.log1p(MAX_FREQ) for f in freq_scores])
    
    # 3. Co-occurrence Score (QuCo-RAG Stage 2)
    cooc_pairs = [(eq, ea) for eq in entities_q for ea in entities_a]
    cooc_scores = [1 if infini_gram.cooc(h, t) > 0 else 0 for h, t in cooc_pairs]
    cooc_score = np.mean(cooc_scores) if cooc_scores else 0
    
    # 4. 결합
    coverage = 0.5 * freq_score + 0.5 * cooc_score
    return coverage
```

---

## 6. 참고 코드 (GitHub)

```bash
git clone https://github.com/ZhishanQ/QuCo-RAG.git
```

주요 파일:
- `src/corpus_stats.py`: Infini-gram 쿼리 로직
- `src/triplet_extractor.py`: Entity/triplet 추출
- `tools/`: 유틸리티 함수들

---

## 7. 핵심 인용

### Entity 빈도와 환각의 관계
> "Low-frequency entities correspond to long-tail knowledge that models struggle to memorize reliably"

### 동시출현 0의 의미
> "Zero co-occurrence between entity pairs indicates the model has no evidential basis for claims relating them"

### Proxy corpus의 유효성
> "Web-scale pre-training corpora share substantial overlap... statistics derived from a transparent and comprehensive corpus can serve as effective proxies for other models"

---

## 8. 요약

| 항목 | 내용 |
|------|------|
| **핵심 기여** | 내부 신호 대신 corpus 통계로 불확실성 정량화 |
| **두 단계** | (1) 생성 전: entity 빈도 확인, (2) 생성 중: 동시출현 검증 |
| **임계값** | entity < 10^3, cooc < 1 |
| **Corpus** | OLMo-2/Dolma via Infini-gram API |
| **Cross-model** | OLMo corpus가 Llama, Qwen, GPT에도 유효 |
| **성능** | EM +5~14 향상 |
