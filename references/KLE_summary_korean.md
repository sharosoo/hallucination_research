# Kernel Language Entropy (KLE) 논문 요약

> **논문 제목**: Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities
>
> **저자**: Alexander Nikitin (Aalto University), Jannik Kossen, Yarin Gal (University of Oxford), Pekka Marttinen (Aalto University)
>
> **arXiv**: 2405.20003v1 [cs.LG] 30 May 2024

---

## 1. 문제 정의 (Problem Definition)

### 핵심 문제
- **LLM의 환각(Hallucination) 탐지**: LLM이 생성하는 "사실과 다르거나 무의미한" 응답을 탐지하는 것이 핵심 과제
- **의미적 불확실성(Semantic Uncertainty) 측정 필요성**: LLM 출력의 불확실성을 측정할 때, 단어 선택(lexical)이나 문장 구조(syntactic) 변화가 아닌 **의미(semantic)**에 대한 불확실성을 포착해야 함

### 기존 방법의 한계
1. **Token-level Predictive Entropy**: 토큰 확률 기반 엔트로피는 의미적, 어휘적, 구문적 불확실성을 모두 혼합
2. **Semantic Entropy (SE)의 한계**: 
   - 의미적 관계를 **동치 관계(equivalence relation)**로만 포착
   - "apple"과 "house", "apple"과 "granny smith"를 동일하게 구분 (실제로는 후자가 더 유사함에도)
   - **거리 개념(distance metric)**이 없어 세밀한 의미적 유사도 반영 불가

---

## 2. 제안 방법: KLE (Kernel Language Entropy)

### 핵심 아이디어
- **의미 공간에서의 거리 측정**: 생성된 답변들 간의 **의미적 유사도(semantic similarity)**를 커널(kernel)로 인코딩
- **von Neumann Entropy 활용**: 의미 커널의 von Neumann 엔트로피를 계산하여 불확실성 정량화
- **세밀한 의미적 관계 포착**: 동치 관계 대신 **유사도 기반 관계**를 사용하여 더 정교한 불확실성 추정

### 방법론 구조

```
입력 질문 → LLM에서 N개 답변 샘플링 → 의미 그래프 구성 → 그래프 커널 계산 → von Neumann Entropy 계산
```

### 두 가지 변형
1. **KLE**: 개별 생성된 텍스트에 직접 적용
2. **KLE-c**: 의미적 클러스터(semantic clusters)에 적용 (계산 비용 절감, 해석 용이)

### 의미 그래프(Semantic Graph) 구성
- NLI(Natural Language Inference) 모델(예: DeBERTa)을 사용하여 답변 쌍의 관계 예측
- 엣지 가중치: `W_ij = f(NLI(S_i, S_j), NLI(S_j, S_i))`
- entailment, neutral, contradiction 확률을 가중 합산

### 그래프 커널 유형
1. **Heat Kernel**: `K_t = e^{-tL}` (L: 그래프 라플라시안)
2. **Matern Kernel**: `K_{νκ} = (2ν/κ² I + L)^{-ν}`

---

## 3. 핵심 수식 (Key Formulas)

### 3.1 Predictive Entropy (기존 방법)
시퀀스 모델의 예측 엔트로피:
```
U(x) = H(S|x) = -Σ_s p(s|x) log(p(s|x))
```

### 3.2 Semantic Entropy (Kuhn et al., 2023)
의미적 클러스터 기반 엔트로피:
```
SE(x) = -Σ_{C∈Ω} p(C|x) log p(C|x)
      = -Σ_{C∈Ω} [Σ_{s∈C} p(s|x)] log[Σ_{s∈C} p(s|x)]
```

Monte Carlo 추정:
```
SE(x) ≈ -Σ_{i=1}^{M} p'(C_i|x) log p'(C_i|x)
```

### 3.3 von Neumann Entropy (VNE)
단위 트레이스 양의 준정부호(unit trace PSD) 행렬 A에 대해:
```
VNE(A) = -Tr[A log A] = -Σ_i λ_i log λ_i
```
여기서 λ_i는 A의 고유값(eigenvalues)

### 3.4 Kernel Language Entropy (KLE) - 제안 방법
```
KLE(x) = VNE(K_sem)
```
여기서 K_sem은 생성된 답변들에 대한 의미 커널

### 3.5 Heat Kernel (그래프 상)
```
K_t = e^{-tL}
```
- t: lengthscale 하이퍼파라미터
- L: 그래프 라플라시안 (L = D - W)

### 3.6 Matern Kernel (그래프 상)
```
K_{νκ} = (2ν/κ² I + L)^{-ν}
```
- κ: lengthscale
- ν: smoothness 파라미터

### 3.7 커널 조합
```
K_FULL = αK_HEAT + (1-α)K_SE
```
여기서 α ∈ [0,1], K_SE는 Semantic Entropy 커널

---

## 4. 기존 Semantic Entropy와의 차이점

| 항목 | Semantic Entropy (SE) | Kernel Language Entropy (KLE) |
|------|----------------------|------------------------------|
| **의미적 관계** | 동치 관계 (equivalence) | 유사도 관계 (similarity) |
| **클러스터링** | Hard clustering (이진 분류) | Soft clustering (연속적 유사도) |
| **거리 개념** | 없음 | 커널 기반 거리 측정 |
| **표현력** | 제한적 | SE를 일반화 (Theorem 3.5) |
| **정보 활용** | 클러스터 소속 여부만 | 클러스터 간 유사도까지 활용 |

### 이론적 관계 (Theorem 3.5)
> **KLE는 Semantic Entropy를 일반화**: 임의의 의미적 클러스터링에 대해, KLE가 SE와 동일한 값을 반환하는 의미 커널이 존재함

### 직관적 예시
- 두 LLM이 동일한 수의 클러스터와 클러스터 확률을 가질 때:
  - SE: 두 LLM에 동일한 불확실성 할당
  - KLE: 클러스터 간 의미적 유사도가 높은 LLM에 더 낮은 불확실성 할당

---

## 5. 실험 결과 요약

### 실험 설정
- **데이터셋**: TriviaQA, SQuAD, Natural Questions (NQ), BioASQ, SVAMP
- **모델**: Llama-2 (7B, 13B, 70B), Falcon (7B, 40B), Mistral 7B (일반 및 instruction-tuned 버전)
- **NLI 모델**: DeBERTa-Large-MNLI
- **평가 지표**: AUROC, AUARC (Area Under Accuracy-Rejection Curve)
- **총 60개 시나리오** (12 모델 × 5 데이터셋)

### 주요 결과

#### 5.1 전체 성능 비교
| 방법 | AUROC 기준 승률 | AUARC 기준 승률 |
|------|----------------|----------------|
| KLE(K_HEAT) | **최고 성능** | **최고 성능** |
| KLE(K_FULL) | 우수 | 우수 |
| SE | 기준선 | 기준선 |

#### 5.2 대형 모델 결과 (Llama 2 70B Chat, Falcon 40B Instruct)
- **BioASQ**: KLE(K_HEAT) AUROC 0.92 vs SE 0.85
- **NQ**: KLE(K_HEAT) AUROC 0.76-0.78 vs SE 0.71-0.78
- **SQuAD**: KLE(K_HEAT) AUROC 0.70-0.71 vs SE 0.66
- **SVAMP**: KLE(K_HEAT) AUROC 0.76-0.77 vs SE 0.62-0.66

#### 5.3 핵심 발견
1. **KLE(K_HEAT)가 일관되게 최고 성능** 달성
2. **Black-box 설정에서도 작동**: 토큰 확률 불필요
3. **Instruction-tuned 모델에서 특히 큰 성능 향상**
4. **하이퍼파라미터 선택**: Entropy Convergence Plot을 통해 검증 세트 없이도 합리적 선택 가능

---

## 6. 한계점 (Limitations)

### 6.1 계산 비용
- LLM에서 **다중 샘플링 필요** (일반적으로 N=10개)
- 생성 비용 증가 (단, 안전-중요 태스크에서는 환각 비용이 샘플링 비용보다 큼)

### 6.2 의미 커널 설계
- 현재는 **NLI 기반 의미 그래프** 중심으로 연구
- 임베딩 기반 커널 등 다른 의미 커널은 향후 연구 필요

### 6.3 적용 범위
- NLG(자연어 생성) 태스크에서 검증됨
- **코드 생성** 등 다른 LLM 응용 분야에 대한 평가 필요

### 6.4 클러스터링 의존성
- 의미적 클러스터링의 품질에 여전히 의존
- NLI 모델의 성능에 영향받음

---

## 7. 기여 및 의의

1. **새로운 불확실성 정량화 방법 KLE 제안**: 의미적 유사도를 활용한 세밀한 불확실성 추정
2. **이론적 기여**: KLE가 SE를 일반화함을 증명 (Theorem 3.5)
3. **실용적 설계 제안**: Heat/Matern 커널, 하이퍼파라미터 선택 방법
4. **광범위한 실험적 검증**: 60개 시나리오에서 SOTA 달성
5. **White-box/Black-box 모두 지원**: 토큰 확률 없이도 작동

---

## 8. 참고 자료

- **코드**: https://github.com/AlexanderVNikitin/kernel-language-entropy
- **관련 연구**: Kuhn et al. (2023) - Semantic Entropy
- **핵심 개념**: von Neumann Entropy, Graph Kernels, NLI (Natural Language Inference)
