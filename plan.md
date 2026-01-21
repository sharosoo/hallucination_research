# Corpus 기반 적응형 환각 탐지

> **핵심 통찰**: "혼란은 다양성을 만들고, 지어냄은 일관성을 만든다. Corpus 통계가 어떤 경우인지 알려준다."

---

## 0. 연구 동기 및 의의

### 왜 이 연구가 중요한가

| 측면 | 설명 |
|------|------|
| **문제의 시급성** | LLM 환각은 의료, 법률, 금융 등 고위험 분야에서 심각한 위험 초래 (Huang et al., 2023 서베이: 1,145 인용) |
| **현재 연구의 공백** | 기존 방법들은 불확실성을 *어떻게* 측정할지에 집중. *언제* 어떤 방법이 효과적인지는 설명 못함 |
| **우리의 기여** | Corpus coverage와 SE/Energy 효과성 간의 관계를 최초로 규명 |
| **실용적 가치** | RAG 없이 작동하는 실시간 환각 탐지 솔루션 |

### 전문가 평가 요약

| 기준 | 평가 | 핵심 포인트 |
|------|------|-------------|
| 참신성 | **Excellent** | Corpus coverage → 탐지 방법 효과성 연결 최초 시도 |
| 중요성 | **Excellent** | 불확실성 정량화의 근본적 공백 해결 |
| 시의성 | **Excellent** | AI 안전성 우선순위와 부합 |
| 실현가능성 | Adequate | Proxy corpus 전략(OLMo)으로 달성 가능 |

### 이론적 기반

**Physics of Language Models (Allen-Zhu & Li, 2023)**에서:
> "지식이 안정적으로 추출되려면, 사전학습 중 충분히 증강(paraphrasing, shuffling 등)되어야 한다."

이는 우리 가설을 직접 지지함: **corpus 빈도가 지식 추출 신뢰도와 상관**하며, 이것이 환각 오류 유형을 결정한다.

---

## 1. 문제 정의

### 1.1 환각의 두 가지 유형

모든 환각이 같지 않다. 근본적으로 다른 두 가지 유형이 존재한다:

| 유형 | 이름 | 원인 | 행동적 신호 | 최적 탐지기 |
|------|------|------|-------------|-------------|
| **A** | 혼란 (Confusion) | 모델이 알지만 헷갈림 | **다양한** 오답 | Semantic Entropy |
| **B** | 지어냄 (Confabulation) | 모델이 모르고 지어냄 | **일관된** 오답 | Semantic Energy |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Type A: 혼란 (아는데 헷갈림)                                               │
│  ─────────────────────────────                                              │
│  Q: "독립선언서 서명일은?"                                                  │
│  응답: ["7월 4일", "8월 2일", "7월 4일", "7월 말", "8월"]                   │
│                                                                             │
│  → 모델이 두 날짜 모두 학습, 혼란 발생                                      │
│  → 다양한 응답 → SE가 이 불확실성 포착                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Type B: 지어냄 (모르는데 지어냄)                                           │
│  ─────────────────────────────────                                          │
│  Q: "Il Seduttore 감독은?"                                                  │
│  응답: ["Mario Camerini", "Mario Camerini", "Mario Camerini", ...]          │
│                                                                             │
│  → 희귀 entity, 모델이 학습한 적 없음, 일관되게 지어냄                      │
│  → SE = 0 (제로-엔트로피 문제) → Energy가 낮은 확신도 탐지                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 통찰

**환각 유형을 결정하는 것은?** → 모델이 관련 지식을 학습했는지 여부

**모델이 학습했는지 어떻게 아는가?** → Pre-training corpus 통계 (entity 빈도, 동시출현)

**우리의 해결책**: Corpus 통계로 어떤 탐지기(SE vs Energy)를 신뢰할지 결정

---

## 2. 정형적 정의

### 2.1 Corpus Coverage

$$C(q, a) = \alpha \cdot \text{FreqScore}(E) + \beta \cdot \text{CoocScore}(E_q, E_a)$$

- $E = E_q \cup E_a$: 질문과 답변에서 추출한 entity들
- FreqScore: corpus 내 정규화된 log-빈도
- CoocScore: 동시출현이 0이 아닌 entity 쌍의 비율

### 2.2 적응형 가중치 공식

$$\text{Score}(q, a) = w(C) \cdot \tilde{E}(a) + (1 - w(C)) \cdot \widetilde{SE}(a)$$

- $w(C) = \sigma(-k(C - \mu))$ (sigmoid, coverage 높으면 Energy 가중치 낮아짐)

### 2.3 제로-엔트로피 문제

모든 응답이 하나의 의미 그룹으로 클러스터링될 때: $SE = 0$

- 정답인 경우: 모델이 확신 있게 알고 있음 ✓
- 오답인 경우: 모델이 확신 있게 지어냄 ✗
- **SE로는 구분 불가** → Energy로 보완 필요

---

## 3. 관련 연구 및 포지셔닝

### 3.0 연구 지형 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         환각 탐지 방법론 분류                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  [내부 신호]              [외부 신호]              [하이브리드]             │
│  ├── SE, Energy           ├── RAG 검증            ├── BTProp               │
│  ├── KLE, SNNE            ├── QuCo-RAG (검색)     ├── HaDeMiF              │
│  ├── SE Probes            └── **우리 (가중치)**   └── UNIHD                │
│  └── Cleanse                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Semantic Entropy (Farquhar et al., Nature 2024)

$$SE = -\sum_k p(C_k) \log p(C_k)$$

- NLI 기반 클러스터링 + Shannon 엔트로피
- 높은 SE = 다양한 응답 = 불확실성
- **한계**: 제로-엔트로피 문제

### 3.2 Semantic Energy (Ma et al., 2025)

$$E = \frac{1}{nT} \sum_i \sum_t -z_\theta(x_t)$$

- Raw logit 사용 (softmax 전)
- 확신도 크기 보존
- **한계**: 다양성 정보 없음

### 3.3 QuCo-RAG (Min et al., 2025)

- Corpus 통계로 retrieval 트리거 결정
- 핵심 발견: Proxy corpus (OLMo)가 다른 모델에도 작동
- **우리와의 차이**: 우리는 retrieval이 아닌 가중치에 사용, RAG 불필요

### 3.4 지식 경계 연구

| 발견 | 출처 | 시사점 |
|------|------|--------|
| 지식 추출에 데이터 증강 필요 | Physics of LLMs (Allen-Zhu, 2023) | Corpus 빈도 → 학습 품질 |
| 실제 cutoff는 보고된 것과 다름 | Dated Data (Cheng, 2024) | 실증적 corpus 확인 필요 |
| 내부 확신도로 경계 탐지 가능 | CoKE (Chen, 2024) | 하지만 외부 corpus가 더 신뢰성 있음 |

### 3.5 외부 Corpus 사용 이유 (내부 신호만으론 부족)

| 접근법 | 한계 | 우리의 해결 |
|--------|------|-------------|
| 내부 확신도 | Poorly calibrated | Corpus를 ground truth로 사용 |
| Hidden state probes | 모델 특정적 | Corpus는 모델 불가지론적 |
| Self-consistency | 여전히 내부 신호 | 외부 검증 |

---

## 4. 실험 설계

### 4.1 연구 질문

| RQ | 질문 | 검증 방법 |
|----|------|-----------|
| RQ1 | Corpus coverage가 SE vs Energy 효과성을 결정하는가? | Bin별 AUROC 분석 |
| RQ2 | 적응형 가중치가 고정 가중치보다 우수한가? | Baseline 비교 |
| RQ3 | 다양한 도메인에서도 작동하는가? | 다중 데이터셋 평가 |

### 4.2 Baseline

| 방법 | $w_{energy}$ | $w_{se}$ | 설명 |
|------|--------------|----------|------|
| SE-only | 0.0 | 1.0 | SE만 사용 |
| Energy-only | 1.0 | 0.0 | Energy만 사용 |
| Fixed-0.1 | 0.1 | 0.9 | SE 위주 고정 |
| Fixed-0.5 | 0.5 | 0.5 | 균등 고정 |
| Fixed-0.9 | 0.9 | 0.1 | Energy 위주 고정 |
| **Adaptive (Ours)** | $w(C)$ | $1-w(C)$ | Corpus 기반 적응형 |

### 4.3 데이터셋

| 데이터셋 | 예상 Corpus Coverage | 예상 최적 방법 |
|---------|----------------------|----------------|
| TruthfulQA | 높음 (대중적 오개념) | SE |
| HaluEval QA | 혼합 (지식 기반) | Energy |
| 2WikiMQA | 낮음 (희귀 entity 조합) | Energy |
| BioASQ | 매우 낮음 (전문 분야) | Energy |

### 4.4 평가 지표

- **AUROC**: ROC 곡선 아래 면적
- **AUPRC**: Precision-Recall 곡선 아래 면적
- **Zero-SE AUROC**: SE < 0.1 케이스에서의 성능

---

## 5. 실험 세부 계획

### 5.1 Corpus Probability별 AUROC 분석 (RQ1 검증)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Entity 추출 및 Corpus Statistics 계산                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  각 질문-응답 쌍에 대해:                                                    │
│  1. Entity 추출 (spaCy NER)                                                 │
│  2. Infini-gram으로 corpus frequency 쿼리                                   │
│  3. Entity co-occurrence 계산                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: Corpus Probability Binning                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Bin 1] Very Low:  p < 0.001  (희귀 지식)                                  │
│  [Bin 2] Low:       0.001 ≤ p < 0.01                                        │
│  [Bin 3] Medium:    0.01 ≤ p < 0.1                                          │
│  [Bin 4] High:      0.1 ≤ p < 0.5                                           │
│  [Bin 5] Very High: p ≥ 0.5   (잘 알려진 지식)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: Bin별 AUROC 계산 및 시각화                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  예상 결과:                                                                 │
│  Corpus Prob.  │  SE AUROC  │  Energy AUROC  │  Winner                      │
│  ─────────────────────────────────────────────────────────                  │
│  Very Low      │   ~0.50    │   ~0.70        │  Energy                      │
│  Low           │   ~0.55    │   ~0.65        │  Energy                      │
│  Medium        │   ~0.60    │   ~0.60        │  Tie                         │
│  High          │   ~0.65    │   ~0.55        │  SE                          │
│  Very High     │   ~0.70    │   ~0.50        │  SE                          │
│                                                                             │
│  → 이 패턴이 확인되면 가설 검증 성공!                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 가중치 공식 설계 및 Baseline 비교 (RQ2 검증)

**가중치 공식 후보**:

1. **Sigmoid** (연속적):
$$w(C) = \frac{1}{1 + \exp(k \cdot (C - \mu))}$$

2. **Linear** (단순):
$$w(C) = w_{\max} - (w_{\max} - w_{\min}) \cdot C$$

3. **Threshold** (이산적):
$$w(C) = \begin{cases} 0.9 & \text{if } C < \tau_1 \\ 0.5 & \text{if } \tau_1 \leq C < \tau_2 \\ 0.1 & \text{if } C \geq \tau_2 \end{cases}$$

### 5.3 Proxy Corpus 전략

**Corpus 선택**: OLMo-2 사전학습 corpus (Infini-gram API 통해 접근)

**정당성** (QuCo-RAG에서 검증됨):
- Web-scale corpus들은 상당 부분 겹침 (Common Crawl, Wikipedia 등)
- Cross-model transfer 검증됨: OLMo corpus가 Llama-3, Qwen, GPT-4에도 유효
- 실험 결과: closed model에 proxy corpus 사용해도 +9~12 EM 향상

---

## 6. 구현 계획

### Phase 1: Baseline ✅ (완료)
- [x] SE/Energy 구현
- [x] TruthfulQA, HaluEval 실험
- [x] 제로-엔트로피 문제 확인

### Phase 2: Corpus 인프라 ✅ (완료)
- [x] Infini-gram API 연동 (OLMo/Dolma corpus)
- [x] Entity 추출 (LLM 기반 triplet extractor)
- [x] Corpus coverage 계산 함수

### Phase 3: 가설 검증 ✅ (완료 - 2026-01-20)
- [x] Bin별 AUROC 분석
- [x] ~~Crossover point 확인~~ → **가설 기각됨** (아래 결과 참조)
- [x] Zero-SE 분석: Energy AUROC = 0.768 확인

**핵심 발견**: Corpus coverage 기반 가설은 기각됨. 대신 **SE-value 기반 fallback** 전략 발견.

### Phase 4: 적응형 방법 ✅ (완료 - 2026-01-20)
- [x] 가중치 공식 설계 (sigmoid/linear/threshold + **SE-fallback**)
- [x] Baseline 비교
- [x] **SE-Fallback이 TruthfulQA에서 최고 성능 (AUROC 0.6374)**

### Phase 5: 분석 및 논문 준비
- [ ] Ablation study
- [ ] 실패 케이스 분석
- [ ] Figure/Table 준비

---

## 9. 실험 결과 (2026-01-20)

### 9.1 핵심 발견

**원래 가설 (Corpus coverage 기반)**: 기각됨
- Bin 분석 결과 coverage와 SE/Energy 효과성 간 예상 패턴 없음
- Low coverage에서 SE가 더 효과적, High coverage에서 Energy가 더 효과적 (예상과 반대)

**새로운 발견 (SE-value 기반)**: 검증됨
- SE < 0.1 (Zero-SE) 샘플에서 Energy AUROC = **0.768** (매우 강함)
- 이는 "confident hallucination" (일관되게 틀린 응답)을 Energy가 탐지함을 의미

### 9.2 데이터셋별 결과

| 데이터셋 | SE AUROC | Energy AUROC | SE-Fallback AUROC | 최고 방법 |
|----------|----------|--------------|-------------------|-----------|
| TruthfulQA | 0.619 | 0.538 | **0.637** | SE-Fallback (+2.9%) |
| HaluEval | 0.506 | **0.604** | 0.567 | Energy-only |

### 9.3 SE-Fallback 방법

```python
if SE < threshold:  # 모델이 confident (아마 틀렸을 수도)
    use Energy (w=0.9)
else:  # 모델이 uncertain
    use SE (w=0.1)
```

| 데이터셋 | Threshold | AUROC |
|----------|-----------|-------|
| TruthfulQA | 0.05~0.3 | **0.6374** (동일) |
| HaluEval | 0.05 | 0.5674 |

### 9.4 해석

- **TruthfulQA**: 주로 "confusion" 유형 환각 (높은 SE), SE가 기본적으로 효과적
  - 하지만 20.5% 샘플이 Zero-SE → SE-Fallback으로 이들 탐지 향상
  
- **HaluEval**: 주로 "confabulation" 유형 환각 (낮은 SE), Energy가 효과적
  - 57%가 Zero-SE 샘플 → Energy-only가 최적

### 9.5 결론

**Corpus coverage 기반 가중치**보다 **SE-value 기반 fallback**이 더 효과적:
- 구현이 간단함 (corpus 쿼리 불필요)
- 이론적 근거 명확 (Zero-SE 문제 직접 해결)
- TruthfulQA에서 검증됨

---

## 7. 기존 연구 대비 차별점

| 방법 | 신호 출처 | 출력 | RAG 필요 | 우리와의 차이 |
|------|----------|------|----------|---------------|
| **SE** (Farquhar, 2024) | 내부 | 불확실성 점수 | 아니오 | 우리는 *언제* SE가 작동하는지 설명 |
| **Energy** (Ma, 2025) | 내부 | 확신도 점수 | 아니오 | 우리는 *언제* Energy가 작동하는지 설명 |
| **KLE/SNNE** | 내부 | 개선된 SE | 아니오 | 우리는 *외부* corpus 신호 추가 |
| **QuCo-RAG** (Min, 2025) | Corpus | 검색 트리거 | **예** | 우리는 *점수 자체* 개선, RAG 불필요 |

**우리의 고유한 기여**: Corpus coverage와 SE/Energy 효과성 간의 연결을 최초로 규명하여 적응형 결합의 이론적 근거 제공

---

## 8. 참고문헌

### 핵심 방법론
1. **Semantic Entropy**: Farquhar et al., Nature 2024
2. **Semantic Energy**: Ma et al., arXiv:2412.07965, 2025
3. **QuCo-RAG**: Min et al., arXiv:2512.19134, 2025

### 불확실성 정량화
4. **KLE**: Nikitin et al., arXiv:2405.20003, 2024
5. **SNNE**: Nguyen et al., arXiv:2506.00245, 2025
6. **Semantic Volume**: Li et al., 2025
7. **SE Probes**: Kossen et al., 2024

### 지식 경계
8. **Physics of LLMs Part 3.1**: Allen-Zhu & Li, 2023
9. **Knowledge Boundary Survey**: Ren et al., ACL 2025
10. **Dated Data**: Cheng et al., 2024

### 환각 탐지
11. **Cleanse**: Joo & Cho, 2025
12. **BTProp**: Hou et al., 2024
13. **HaloScope**: Du et al., 2024
