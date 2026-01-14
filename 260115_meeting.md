# 260115 조교님 미팅 논의사항

## 1. 이번 주 수행한 작업

### 1.1 실험 환경 구축
- 실제 LLM 샘플러 구현 (Qwen2.5-3B-Instruct)
- 각 토큰의 raw logit 추출 (Semantic Energy용)
- NLI 클러스터링 (DeBERTa-large-mnli)

### 1.2 SE/Energy 베이스라인 실험
- TruthfulQA, HaluEval 두 데이터셋에서 AUROC 측정
- 제로-엔트로피 문제 실험적 확인

---

## 2. 실험 구조

### 2.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                        입력: 질문 (Question)                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: LLM 다중 샘플링 (K=5, temp=0.7)              │
│  - 같은 질문에 5번 응답 생성                                      │
│  - 각 응답의 토큰별 raw logit 저장 (softmax 전)                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: NLI 클러스터링 (DeBERTa-large-mnli)          │
│  - 양방향 함의(Bidirectional Entailment) 기반                    │
│  - "Paris", "파리", "Paris is the capital" → 같은 클러스터        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌────────────────────────────┐  ┌────────────────────────────┐
│   Semantic Entropy (SE)    │  │   Semantic Energy          │
│                            │  │                            │
│  SE = -Σ p(C_k) log p(C_k) │  │  U = (1/nT) ΣΣ -z_θ(x_t)   │
│  p(C_k) = |C_k| / K        │  │  z_θ = raw logit           │
│                            │  │                            │
│  클러스터 1개 → SE=0       │  │  logit 낮음 → U 높음       │
│  클러스터 많음 → SE 높음   │  │  (확신 낮음)               │
└────────────────────────────┘  └────────────────────────────┘
```

### 2.2 Semantic Entropy 계산 예시

```
질문: "프랑스의 수도는?"
5개 응답: ["Paris", "파리", "Paris", "런던", "베를린"]

NLI 클러스터링:
- 클러스터 1: ["Paris", "파리", "Paris"] (3개)
- 클러스터 2: ["런던"] (1개)
- 클러스터 3: ["베를린"] (1개)

SE = -(0.6×log(0.6) + 0.2×log(0.2) + 0.2×log(0.2)) = 0.95

→ SE 높음 = 응답 다양 = 불확실 = 환각 가능성
```

### 2.3 Semantic Energy 계산 예시

```
응답: "Paris"
토큰별 raw logit: [12.5, 8.3, 10.1]

E(응답) = -mean([12.5, 8.3, 10.1]) = -10.3

→ Energy 낮음(음수 큼) = logit 높음 = 확신 높음 = 신뢰 가능
→ Energy 높음(음수 작음) = logit 낮음 = 확신 낮음 = 환각 가능
```

---

## 3. 실험 결과

### 3.1 실험 설정
| 항목 | 값 |
|------|-----|
| LLM | Qwen2.5-3B-Instruct |
| NLI 모델 | DeBERTa-large-mnli |
| 샘플링 | 5개/질문, temperature=0.7 |
| 평가 샘플 | 각 200개 |

### 3.2 결과 요약

| 데이터셋 | SE AUROC | Energy AUROC | 승자 |
|----------|----------|--------------|------|
| TruthfulQA | **0.6190** | 0.5380 | SE |
| HaluEval | 0.5057 | **0.6036** | Energy |

**→ 데이터셋마다 최적 지표가 다름 = 고정 가중치 한계**

---

## 4. 제로-엔트로피 문제 (핵심)

### 4.1 정의

**제로-엔트로피**: 모델이 **일관되게 같은 (틀린) 답변**을 생성하여 SE ≈ 0이 되는 현상

```
질문: "What happens if you crack your knuckles?"
5개 응답: ["관절염", "관절염", "관절염", "관절염", "관절염"]

→ 클러스터 1개 → SE = 0 → "신뢰 가능"으로 판단
→ 그러나 "관절염"은 오답! (실제로는 아무 해 없음)
```

### 4.2 실험에서 확인

**TruthfulQA에서 SE < 0.1인 케이스:**

| 항목 | 값 |
|------|-----|
| 총 샘플 | 41개 |
| 환각 | 31개 (75.6%) |
| 정상 | 10개 (24.4%) |

**→ SE가 낮은데 75%가 환각!**

### 4.3 Energy가 일부 해결

| 지표 | 제로-SE 케이스 AUROC |
|------|---------------------|
| SE | 0 (구분 불가) |
| **Energy** | **0.7677** |

**→ Energy가 제로-엔트로피 케이스에서 효과적**

### 4.4 문제의 본질

```
┌─────────────────────────────────────────────────────────────┐
│  SE와 Energy 모두 "모델 내부 신호"에 의존                     │
│                                                              │
│  - SE: 응답 다양성 (모델이 다르게 답하는지)                   │
│  - Energy: 모델 확신도 (logit 크기)                          │
│                                                              │
│  문제: 모델이 "확신을 갖고 틀리면" 둘 다 속음                 │
│        (LLM은 poorly calibrated)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 기존 SE/Energy 결합 연구 조사

### 5.0 연구 배경: 왜 선형 결합인가?

| 문제 | SE의 한계 | Energy의 한계 |
|------|----------|---------------|
| 측정 대상 | 응답 다양성 | 모델 확신도 |
| 장점 | 의미적 불확실성 | 제로-엔트로피 해결 |
| 한계 | 일관된 오답 탐지 실패 | 다양성 정보 부족 |

**→ 둘을 결합하면 상호 보완 가능하다는 가설**

---

### 5.1 KLE (Kernel Language Entropy) - Nikitin et al., 2024

**arXiv:2405.20003**

#### 핵심 아이디어
- SE의 **hard clustering**을 **soft clustering**으로 일반화
- **von Neumann Entropy** + **Semantic Kernel** 사용

#### 수식
```
KLE(x) = VNE(K_sem) = -Σ λ_i log λ_i

여기서 λ_i는 의미 커널 K_sem의 고유값
```

#### SE와의 차이점
| 항목 | SE | KLE |
|------|-----|-----|
| 의미적 관계 | 동치 관계 (equivalence) | 유사도 관계 (similarity) |
| 클러스터링 | Hard (이진) | Soft (연속) |
| 거리 개념 | 없음 | 커널 기반 거리 |
| 표현력 | 제한적 | **SE를 일반화** (Theorem 3.5) |

#### 실험 결과
- 60개 시나리오 (12 모델 × 5 데이터셋)에서 SE 대비 우수
- BioASQ: KLE AUROC 0.92 vs SE 0.85

#### 한계
- **O(N³) 계산 복잡도** (고유값 분해)
- NLI 모델 의존성

---

### 5.2 SNNE (Semantic Nearest Neighbor Entropy) - Nguyen et al., 2025

**arXiv:2506.00245**

#### 핵심 아이디어
- **클러스터링 없이** 직접 유사도 기반 불확실성 측정
- Nearest Neighbor 엔트로피 추정에서 영감
- **LogSumExp**로 이상치 영향 완화

#### 수식
```
SNNE(q) = -(1/n) Σ_i log Σ_j exp(f(a_i, a_j)/τ)
```

#### SE/KLE와의 차이점
| 항목 | SE | KLE | SNNE |
|------|-----|-----|------|
| 클러스터링 | 필요 | 필요 | **불필요** |
| 복잡도 | O(N²) | O(N³) | **O(N²)** |
| 긴 응답 | 성능 저하 | 보통 | **강건** |

#### 이론적 기여
> **Theorem 4.1-4.2**: SNNE/WSNNE는 DSE/SE를 각각 일반화

#### 실험 결과
| 모델 | SE | KLE | SNNE |
|------|-----|------|------|
| Llama-3.1-8B | 0.79 | 0.80 | **0.83** |
| Phi-3-mini | 0.80 | 0.81 | **0.84** |

#### 한계
- 다중 문장 생성 미탐구
- 수학/코드 등 특수 형식 미지원

---

### 5.3 Cleanse (Clustering-based Semantic Consistency) - Joo & Cho, 2025

**arXiv:2507.14649**

#### 핵심 아이디어
- **Hidden Embedding** + **클러스터 내/간 유사도 비율** 활용
- Intra-cluster 유사도 = 일관성
- Inter-cluster 유사도 = 페널티

#### 수식
```
Cleanse Score = intra-cluster sim. / total sim.
             = 1 - (inter-cluster sim. / total sim.)
```

#### SE와의 차이점
| 항목 | SE | Cleanse |
|------|-----|---------|
| 측정 방식 | 엔트로피 | 유사도 비율 |
| 임베딩 | 미사용 | **Hidden Embedding 활용** |
| 페널티 | 없음 | **Inter-cluster 명시적 페널티** |

#### 실험 결과
| 모델 | Lexical Sim. | Cosine Score | **Cleanse** |
|------|--------------|--------------|-------------|
| LLaMA-7B (SQuAD) | 76.9 | 79.6 | **81.7** |
| Mistral-7B (SQuAD) | 69.0 | 65.9 | **75.9** |

#### 한계
- **White-box only**: Hidden Embedding 접근 필요
- QA 태스크 특화

---

### 5.4 기존 연구들이 이미 해결한 것들

| 문제 | 해결한 연구 |
|------|-----------|
| Hard → Soft clustering | KLE (von Neumann Entropy) |
| 클러스터 내/간 유사도 무시 | SNNE, Cleanse |
| SE 일반화 | KLE, SNNE (이론적 증명) |
| 제로-엔트로피 | Semantic Energy |
| 긴 응답에서 성능 저하 | SNNE |

---

### 5.5 AHSFE와의 비교 및 차별점

| 방법 | 핵심 | AHSFE와의 차이 |
|------|------|----------------|
| KLE | Soft clustering via kernel | AHSFE: 가중치 결정에 corpus 사용 |
| SNNE | 클러스터링 제거 | AHSFE: 클러스터링 유지, 가중치 적응 |
| Cleanse | Hidden embedding 활용 | AHSFE: 외부 corpus stats 활용 |

**AHSFE의 차별점 (가설)**:
- 기존 연구들: **모델 내부 신호**의 결합 방식 개선
- AHSFE: **외부 신호(corpus)**로 결합 가중치 결정

**솔직한 우려**:
- 기존 연구들이 이미 SE를 다양하게 확장
- AHSFE가 "또 다른 결합 방식"으로 보일 수 있음
- corpus 기반 가중치가 실질적으로 얼마나 도움이 되는지 검증 필요

---

## 6. QuCo-RAG 논문 분석 (Min et al., 2025)

### 5.1 논문 배경

**제목**: QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic RAG

**핵심 문제 인식**:
- 기존 Dynamic RAG 방법들 (DRAGIN, ETC 등)은 **모델 내부 신호** (entropy, logit)로 retrieval 필요성 판단
- 그러나 LLM은 **poorly calibrated** → 틀린 답에도 높은 확신
- 결과: **"확신에 찬 오답" (Confident Hallucination)** 탐지 실패

### 5.2 논문의 핵심 관찰

```
┌─────────────────────────────────────────────────────────────┐
│  기존 방법 (DRAGIN)의 실패 사례                              │
├─────────────────────────────────────────────────────────────┤
│  질문: "Il Seduttore와 Joan of Arc 재판의 감독이 같은 나라?" │
│                                                              │
│  LLM 출력: "Il Seduttore was directed by Mario Camerini..." │
│            (실제 감독: Franco Rossi → 오답!)                 │
│                                                              │
│  DRAGIN 판단:                                                │
│  - "Il" 토큰: 높은 불확실성 (질문에서 온 토큰인데...)        │
│  - "Mario Camerini": 낮은 불확실성 ← 오답인데 확신!          │
│                                                              │
│  → 모델 내부 신호가 정확성과 상관없음                        │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 QuCo-RAG의 해결책: Corpus Statistics

**핵심 아이디어**: 
> "모델이 확신한다" ≠ "모델이 실제로 학습했다"
> → Pre-training corpus에서 **외부 증거**를 찾자

**두 단계 검증**:

#### Stage 1: Pre-Generation Knowledge Assessment (생성 전)

```
질문에서 entity 추출 → corpus frequency 확인

예: "Silas Hardy와 Lee Mantle 중 누가 먼저 태어났나?"
    
    Entity: "Silas Hardy" → corpus frequency: 258
    Entity: "Lee Mantle"  → corpus frequency: 180
    
    평균 frequency < threshold → long-tail knowledge
    → retrieval 트리거
```

**의미**: 
- corpus에 적게 등장한 entity = 모델이 잘 학습 못했을 가능성
- → retrieval로 외부 지식 보충

#### Stage 2: Runtime Claim Verification (생성 중)

```
생성된 문장에서 entity pair 추출 → co-occurrence 확인

예: LLM 출력 "Il Seduttore was directed by Mario Camerini"

    Entity pair: ("Il Seduttore", "Mario Camerini")
    Corpus co-occurrence: 0  ← 한 번도 같이 등장 안 함!
    
    → 환각 위험 → retrieval 트리거 → 재생성
```

**의미**:
- corpus에서 두 entity가 함께 등장한 적 없음 = 모델이 학습한 적 없는 관계
- = 모델이 지어낸 것 (hallucination)

### 5.4 Infini-gram 사용

**Infini-gram**: 4조 토큰 corpus에서 millisecond 단위 n-gram 쿼리 가능

```python
# Entity frequency 쿼리
freq = infini_gram.count("Silas Hardy")  # → 258

# Co-occurrence 쿼리  
co_occur = infini_gram.count("Il Seduttore * Mario Camerini")  # → 0
```

**장점**: 실시간 검증 가능 (latency < 100ms)

### 5.5 QuCo-RAG 실험 결과

| 모델 | 방법 | 2WikiMQA EM | HotpotQA EM |
|------|------|-------------|-------------|
| OLMo-2-7B | No RAG | 32.0 | 25.5 |
| | DRAGIN | 35.5 | 28.0 |
| | **QuCo-RAG** | **44.5** | **37.5** |

**→ +9~12 EM 개선**

**Cross-model transfer** (corpus 없는 모델에서도):
- Llama-3, Qwen, GPT에서도 효과 (OLMo corpus로 검증해도 유효)
- 이유: web-scale corpus들은 상당 부분 겹침

---

## 7. 제안 아이디어: AHSFE with Corpus-based Weights

### 6.1 핵심 아이디어

```
기존 HSFE: score = α × Energy + β × SE  (α, β 고정)

제안 AHSFE: score = w(corpus) × Energy + (1-w(corpus)) × SE
            
            w(corpus) = corpus statistics 기반 가중치
```

**SE와 Energy를 어떤 비율로 믿을지를 corpus statistics로 결정**

### 6.2 가중치 설계 아이디어

```
IF entity_frequency 낮음 (long-tail knowledge):
    → w ↑ (Energy 가중치 증가)
    → 이유: 모델이 잘 모르는 지식 = SE보다 Energy가 더 신뢰할 만함
    
IF entity_cooccurrence = 0 (학습 안 된 관계):
    → w ↑ (Energy 가중치 증가)
    → 이유: corpus에 없음 = 모델이 지어낸 것 = Energy로 확신 수준 체크
    
IF corpus_coverage 높음 (잘 학습된 지식):
    → w ↓ (SE 가중치 증가)
    → 이유: 모델이 아는 지식 = SE로 다양성 체크가 의미 있음
```

### 6.3 예상 시나리오

**시나리오 1: 확신에 찬 오답**
```
질문: "손가락 관절 꺾으면?"
응답: ["관절염"] × 5 (일관됨)

기존 신호:
- SE = 0, Energy = 낮음 → "정상"으로 오판

Corpus 신호:
- ("손가락 관절", "관절염") co-occurrence = 낮음
- → w ↑ (Energy 가중치 증가)
- → Energy로 재평가 → "환각" 탐지
```

**시나리오 2: 확신에 찬 정답**
```
질문: "프랑스 수도?"
응답: ["파리"] × 5 (일관됨)

Corpus 신호:
- ("프랑스", "파리") co-occurrence = 매우 높음
- → w 유지
- → "정상"으로 정확히 판단
```

### 6.4 기존 방법과의 차이

| 방법 | 신호 | 용도 |
|------|------|------|
| QuCo-RAG | Corpus stats | Retrieval 결정 |
| **AHSFE (제안)** | **Corpus stats** | **SE/Energy 가중치 결정** |

**차별점**: QuCo-RAG는 retrieval 여부만 결정, 우리는 불확실성 측정 자체를 개선

---

## 8. 핵심 질문: 연구의 근본적 타당성

### 7.0 AHSFE의 한계에 대한 의문

**문제 제기**:
```
AHSFE = w(corpus) × Energy + (1-w(corpus)) × SE
              ↑                        ↑
         내부 신호                  내부 신호

→ corpus는 가중치만 결정
→ 결국 SE와 Energy의 조합일 뿐
→ 둘 다 내부 신호라 "확신에 찬 오답"은 못 잡는 거 아닌가?
```

**핵심 질문**:

| # | 질문 | 왜 중요한가 |
|---|------|-------------|
| Q1 | corpus가 "Energy를 더 봐라"라고 해도, Energy 자체가 확신에 찬 오답을 못 잡으면 의미 없지 않나? | 연구의 근본 가정 |
| Q2 | 실험에서 제로-SE 케이스 Energy AUROC 0.768인데, 이게 "corpus에 없는 지식"일 때 더 높아지는지 확인 필요 | 가설 검증 가능성 |
| Q3 | 차라리 QuCo-RAG처럼 corpus stats를 **직접** 환각 점수로 쓰는 게 더 효과적이지 않나? | 대안과의 비교 |
| Q4 | AHSFE의 가치가 "고정 가중치보다 나음" 수준이면 논문으로 부족하지 않나? | 기여도 수준 |

**실험 결과 재해석**:
```
제로-SE 케이스 (SE < 0.1):
- SE = 0 (일관된 응답) ≠ Energy = 낮음 (완전한 확신)
- Energy AUROC 0.768 → Energy가 어느 정도 구분 가능

하지만:
- "corpus에 없을 때 Energy가 더 잘 작동한다"는 증거는 아직 없음
- 이 가설이 틀리면 corpus 기반 가중치의 의미가 없어짐
```

**평가**:
```
강점:
✓ 데이터셋마다 최적 가중치가 다름 → 적응형으로 해결 가능
✓ 제로-SE 문제 → Energy로 일부 해결 (실험 결과 있음)

약점:
✗ "확신에 찬 오답"의 근본적 해결은 아님 (내부 신호 한계)
✗ corpus 가중치의 추가적 가치 아직 불명확
✗ QuCo-RAG 대비 차별점이 "RAG 불필요" 정도?
```

**조교님께 질문드릴 것**:
1. 이 한계를 인정하고 "적응형 가중치"에 집중하는 방향이 맞는지?
2. 아니면 QuCo-RAG처럼 corpus stats를 직접 환각 점수로 쓰는 방향으로 전환?
3. 또는 QuCo-RAG + AHSFE 결합으로 시너지 보여주는 방향?

---

### 8.1 QuCo-RAG와의 차별점 (내 생각)

| 비교 항목 | QuCo-RAG | AHSFE (제안) |
|----------|----------|--------------|
| **목적** | Retrieval 트리거 결정 | 불확실성 측정 자체 개선 |
| **출력** | "retrieval 할지 말지" (binary) | "환각 확률" (continuous) |
| **사용 시점** | 생성 중 (online) | 생성 후 평가 (offline도 가능) |
| **RAG 의존성** | RAG 시스템 필수 | RAG 없이도 독립 사용 가능 |

**차별점 요약**:
- QuCo-RAG: "이 지식 모르니까 검색해서 가져와"
- AHSFE: "이 지식 모르니까 SE 말고 Energy를 더 믿어"

**솔직한 우려**:
- 차별점이 "용도"의 차이일 뿐, 핵심 아이디어(corpus stats 활용)는 동일
- QuCo-RAG가 이미 corpus stats의 효과를 증명함
- 우리는 "같은 신호를 다른 목적에 적용"하는 것
- 이게 충분한 기여인지는 실험 결과에 달림

**기여가 되려면**:
- SE/Energy 가중치 조정이 retrieval보다 효과적인 케이스 제시
- 또는 RAG 없이도 작동하는 경량 솔루션으로 포지셔닝
- 또는 QuCo-RAG와 결합했을 때 추가 성능 향상 보여주기

### 8.2 기술적 구현
1. Infini-gram 사용 가능한가? (OLMo corpus)
2. 다른 모델 (Qwen, Llama)에서는? (transfer 가능성)
3. Entity 추출 방법? (NER vs LLM)

### 8.3 실험 계획
1. 먼저 간단한 규칙 기반으로 가중치 적용해서 효과 확인?
2. 어떤 데이터셋으로 검증?
3. QuCo-RAG 재현 후 비교 실험?

---

## 9. 다음 단계

**조교님 피드백 후 결정**

1. Corpus statistics 기반 가중치 규칙 설계
2. Infini-gram 연동 (또는 대안)
3. 간단한 실험으로 효과 확인
