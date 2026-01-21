# KLE (Kernel Language Entropy) 논문 요약

> **논문 제목**: Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity  
> **저자**: Dang Nguyen (UCLA CS), Ali Payani (Cisco Systems Inc.), Baharan Mirzasoleiman (UCLA CS)  
> **출처**: arXiv:2506.00245v1 [cs.LG] 30 May 2025

---

## 1. 문제 정의 (Problem Definition)

### 1.1 배경
- **LLM의 환각(Hallucination) 문제**: LLM이 그럴듯해 보이지만 실제로는 틀리거나 조작된 정보를 생성하는 현상
- **불확실성 정량화(Uncertainty Quantification, UQ)**: 환각 탐지를 위해 모델 출력의 불확실성을 측정하는 접근법

### 1.2 기존 Semantic Entropy의 한계
- **Semantic Entropy (SE)**: NLI 모델을 사용한 양방향 함의(bidirectional entailment) 예측으로 의미적으로 유사한 출력을 클러스터링한 후 엔트로피 계산
- **문제점**: 최신 LLM들이 더 긴 한 문장 응답을 생성하면서 SE의 효과가 감소
  - Llama-3.1-8B: 평균 4.1 단어
  - Phi-3-mini: 평균 4.9 단어
  - Llama2-7B: 평균 2.3 단어 (이전 모델)

### 1.3 SE가 간과하는 두 가지 핵심 요소
1. **클러스터 내 유사성(Intra-cluster similarity)**: 클러스터 내부의 분포/퍼짐 정도
2. **클러스터 간 유사성(Inter-cluster similarity)**: 클러스터들 사이의 거리

### 1.4 구체적 문제 상황
- 응답 길이가 길어질수록 의미적 클러스터 수가 증가 (Spearman 상관계수 = 0.83)
- 클러스터 수 M이 샘플 수 n에 근접하면 **DSE(Discrete SE)가 상수값**을 출력
- 클러스터 수가 적어도 클러스터 내 퍼짐 정도를 고려하지 않아 **SE 성능 저하**

---

## 2. 제안 방법: SNNE (Semantic Nearest Neighbor Entropy)

### 2.1 핵심 아이디어
- **클러스터링 없이** 클러스터 내/클러스터 간 유사성을 모두 활용
- **Nearest Neighbor 기반 엔트로피 추정**에서 영감을 받음
- **LogSumExp 연산**으로 이상치(outlier)의 영향을 완화

### 2.2 Black-box 버전: SNNE

$$\text{SNNE}(q) = -\frac{1}{n} \sum_{i=1}^{n} \log \sum_{j=1}^{n} \exp\left(\frac{f(a_i, a_j | q)}{\tau}\right)$$

**변수 설명**:
- $q$: 질문
- $a_i, a_j$: 생성된 답변들
- $n$: 생성된 답변 수
- $f(a_i, a_j | q)$: 두 답변 간 유사도 함수
- $\tau$: 스케일 팩터 (기본값 1)

### 2.3 White-box 버전: WSNNE

$$\text{WSNNE}(q) = -\sum_{i=1}^{n} \bar{P}(a^i | q) \log \sum_{j=1}^{n} \exp\left(\frac{f(a_i, a_j | q)}{\tau}\right)$$

**추가 변수**:
- $\bar{P}(a^i | q) = \frac{\tilde{P}(a^i | q)}{\sum_{j=1}^{n} \tilde{P}(a^j | q)}$: 정규화된 시퀀스 확률
- $\tilde{P}(a | q) = P(a | q) / \text{len}(a)$: 길이 정규화된 시퀀스 확률

### 2.4 유사도 함수 선택지
1. **ROUGE-L**: 어휘적 중복 측정 (가장 좋은 성능)
2. **NLI 점수(entail)**: DeBERTa 모델 기반 함의 예측 점수
3. **문장 임베딩 코사인 유사도(embed)**: Sentence Transformer 사용

---

## 3. 핵심 수식 (Key Formulas)

### 3.1 기존 Semantic Entropy

$$\text{SE}(q) = -\sum_{k=1}^{M} \bar{P}(C_k) \log \bar{P}(C_k)$$

- $\bar{P}(C_k) = \frac{P(C_k)}{\sum_{j=1}^{M} P(C_j)}$: 정규화된 의미 클래스 확률
- $P(C_k) = \sum_{i, a_i \in C_k} \tilde{P}(a_i | q)$

### 3.2 Discrete Semantic Entropy

$$\text{DSE}(q) = -\sum_{k=1}^{M} \frac{|C_k|}{n} \log \frac{|C_k|}{n}$$

### 3.3 이론적 결과 (일반화 증명)

**Theorem 4.1**: SNNE → DSE 일반화
- $f(a_i, a_j | q) = \tau \log(1/n)$ (같은 클러스터), $-\infty$ (다른 클러스터)일 때 SNNE = DSE

**Theorem 4.2**: WSNNE → SE 일반화
- $f(a_i, a_j | q) = \tau \log(\tilde{P}(a^j | q) / Q)$ (같은 클러스터), $-\infty$ (다른 클러스터)일 때 WSNNE = SE
- 여기서 $Q = \sum_{i=1}^{n} \tilde{P}(a^i | q)$

---

## 4. 기존 Semantic Entropy와의 차이점

| 측면 | Semantic Entropy (SE) | SNNE/WSNNE |
|------|----------------------|------------|
| **클러스터링** | NLI 기반 양방향 함의 클러스터링 필요 | 클러스터링 불필요 |
| **클러스터 내 유사성** | 고려하지 않음 | 자연스럽게 통합 |
| **클러스터 간 유사성** | 고려하지 않음 | 자연스럽게 통합 |
| **긴 응답 처리** | 성능 저하 (상수값 출력 가능) | 강건한 성능 유지 |
| **이상치 민감도** | 해당 없음 | LogSumExp로 완화 |
| **설정** | White-box | Black-box(SNNE) / White-box(WSNNE) |
| **계산 복잡도** | $O(N^2)$ | $O(N^2)$ |

### 핵심 차별점
1. **클러스터링 제거**: SE는 discrete한 클러스터 할당에 의존하지만, SNNE는 연속적인 유사도 사용
2. **정보 손실 방지**: SE는 같은 클러스터 내 답변들을 동등하게 취급하여 정보 손실 발생
3. **긴 응답에서의 강건성**: 클러스터 수가 증가해도 안정적인 불확실성 추정

---

## 5. 실험 결과 요약

### 5.1 실험 설정
- **모델**: Llama-3.1-8B, Phi-3-mini, Llama2-7B/13B, gemma-2-2b, Mistral-Nemo
- **태스크**: 질의응답(QA), 텍스트 요약(TS), 기계번역(MT)
- **데이터셋**: 
  - QA: SQuAD, TriviaQA, NaturalQuestion, Svamp, BioASQ
  - TS: XSUM, AESLC
  - MT: WMT-14 de-en, WMT-14 fr-en
- **평가 지표**: AUROC, AUARC (QA), PRR (TS/MT)

### 5.2 QA 태스크 결과 (AUROC)

| 모델 | SE | DSE | KLE_full | SNNE | WSNNE |
|------|-----|-----|----------|------|-------|
| Llama-3.1-8B | 0.79 | 0.79 | 0.80 | **0.83** | **0.83** |
| Phi-3-mini | 0.80 | 0.80 | 0.81 | **0.84** | **0.84** |
| Llama2-7B | 0.79 | 0.79 | 0.80 | **0.80** | **0.81** |

### 5.3 요약/번역 태스크 결과 (PRR)

| 태스크 | SE | DSE | LexSim | SNNE | WSNNE |
|--------|-----|-----|--------|------|-------|
| Text Summarization (ROUGE-L) | 0.20 | 0.14 | 0.23 | **0.26** | **0.27** |
| Machine Translation (ROUGE-L) | 0.58 | 0.57 | 0.61 | **0.63** | **0.63** |

### 5.4 주요 발견
1. **SNNE/WSNNE가 모든 태스크와 모델에서 일관되게 최고 성능**
2. **클러스터 수가 많을수록 SE 대비 SNNE의 우위가 커짐** (Figure 1 right)
3. **긴 응답 생성 시나리오에서 특히 효과적**
4. **ROUGE-L 유사도가 가장 좋은 성능**을 보임
5. **스케일 팩터 τ = 1**이 대부분의 경우에서 최적

### 5.5 하이퍼파라미터 분석
- **생성 샘플 수**: 증가하면 성능 향상, 10개에서 포화
- **온도(Temperature)**: 너무 낮거나(0.5) 높으면(2.0) 성능 저하, 1.0이 최적

---

## 6. 한계점 (Limitations)

### 6.1 논문에서 명시한 한계
1. **다중 문장/문단 생성 미탐구**: 현재 한 문장 출력에 초점, 여러 문장의 경우 추가 연구 필요
2. **특수 데이터 형식**: 수학적 표현, LaTeX 수식, 코드 등에 대해서는 적절한 유사도 함수 설계 필요
3. **추론 비용**: 다른 UQ 방법들과 마찬가지로 불확실성 추정을 위해 여러 답변을 샘플링해야 함

### 6.2 추가적으로 고려할 한계점
4. **유사도 함수 의존성**: 성능이 유사도 함수 선택에 민감할 수 있음
5. **도메인 특화**: 특정 도메인(의료, 법률 등)에서의 성능 검증 부족
6. **실시간 적용성**: 여러 샘플 생성 필요로 인한 지연 시간

### 6.3 향후 연구 방향 (저자 제안)
- LUQ와의 통합: SNNE를 atomic score로 사용하여 다중 문장 시나리오로 확장
- 특수 데이터 형식에 맞는 유사도 함수 개발

---

## 참고: KLE(Kernel Language Entropy)와의 비교

> **주의**: 이 논문의 제안 방법은 **SNNE/WSNNE**이며, **KLE**는 비교 대상 baseline 중 하나입니다.

| 방법 | 특징 | 단점 |
|------|------|------|
| **KLE (Nikitin et al., 2024)** | von Neumann entropy + semantic kernel (heat/Matern) | 해석 어려움, 정보 손실 가능, $O(N^3)$ 복잡도 |
| **SNNE (본 논문)** | Nearest neighbor 기반, 직접적 유사도 사용 | $O(N^2)$ 복잡도, 해석 용이 |

---

## 코드 및 재현성

- **코드**: https://github.com/BigML-CS-UCLA/SNNE
- **하드웨어**: NVIDIA RTX A6000 GPUs
- **실험 반복**: 각 실험 3회 수행
