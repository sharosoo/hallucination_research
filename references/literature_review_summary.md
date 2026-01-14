# 문헌 조사 요약: LLM 환각 탐지를 위한 불확실성 추정 방법론

> **작성일**: 2025-01-15  
> **목적**: 기존 방법론들의 비교 분석 및 새로운 연구 방향 도출

---

## 1. 기존 방법론 비교표

| 방법 | 논문 | 핵심 아이디어 | 장점 | 한계 |
|------|------|--------------|------|------|
| **Semantic Entropy (SE)** | Farquhar et al., Nature 2024 | NLI 기반 클러스터링 + Shannon Entropy | 의미적 불확실성 측정의 시초 | 제로-엔트로피 문제, Hard clustering |
| **Kernel Language Entropy (KLE)** | Nikitin et al., 2024 | von Neumann Entropy + Semantic Kernel | SE를 이론적으로 일반화 | O(N³) 계산 복잡도 |
| **SNNE** | Nguyen et al., 2025 | Nearest Neighbor 기반 LogSumExp | 클러스터링 불필요, O(N²) | 유사도 함수 의존성 |
| **Cleanse** | Joo & Cho, 2025 | Hidden Embedding + Intra/Inter cluster 비율 | 클러스터 간 유사도 활용 | White-box only |
| **Semantic Energy** | Ma et al., 2025 | Raw Logit 기반 에너지 함수 | 제로-엔트로피 해결 | 다양성 정보 부족 |
| **BTProp** | 2024 | Belief Tree Propagation | 확률론적 신념 통합 | 복잡한 구조 |

---

## 2. 핵심 방법론 상세 분석

### 2.1 Kernel Language Entropy (KLE)

**arXiv:2405.20003** - Nikitin et al., 2024

#### 핵심 아이디어
- Semantic Entropy의 **hard clustering**을 **soft clustering**으로 일반화
- von Neumann Entropy를 사용하여 의미 커널의 엔트로피 계산

#### 수식
```
KLE(x) = VNE(K_sem) = -Σ λ_i log λ_i
```
여기서 λ_i는 의미 커널 K_sem의 고유값

#### SE와의 차이점
| 항목 | SE | KLE |
|------|-----|-----|
| 의미적 관계 | 동치 관계 (equivalence) | 유사도 관계 (similarity) |
| 클러스터링 | Hard (이진) | Soft (연속) |
| 거리 개념 | 없음 | 커널 기반 거리 |
| 표현력 | 제한적 | **SE를 일반화** (Theorem 3.5) |

#### 이론적 기여
> **Theorem 3.5**: KLE는 Semantic Entropy를 일반화. 임의의 의미적 클러스터링에 대해 KLE = SE가 되는 의미 커널이 존재.

#### 한계점
- O(N³) 계산 복잡도 (고유값 분해)
- NLI 모델 의존성
- 코드 생성 등 다른 도메인 미검증

---

### 2.2 Semantic Nearest Neighbor Entropy (SNNE)

**arXiv:2506.00245** - Nguyen et al., 2025

#### 핵심 아이디어
- **클러스터링 없이** 직접 유사도 기반 불확실성 측정
- Nearest Neighbor 엔트로피 추정에서 영감
- LogSumExp로 이상치 영향 완화

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

#### 한계점
- 다중 문장 생성 미탐구
- 수학/코드 등 특수 형식 미지원
- 여전히 다중 샘플링 필요

---

### 2.3 Cleanse (Clustering-based Semantic Consistency)

**arXiv:2507.14649** - Joo & Cho, 2025

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

#### 한계점
- **White-box only**: Hidden Embedding 접근 필요
- NLI 모델 의존성
- QA 태스크 특화

---

### 2.4 Semantic Energy

**arXiv:2412.07965** - Ma et al., 2025

#### 핵심 아이디어
- **Softmax 전 raw logit** 사용
- Energy = negative logit → 낮은 에너지 = 높은 확신

#### 수식
```
U = (1/nT) ΣΣ -z_θ(x_t)
```

#### SE와의 차이점
| 항목 | SE | Semantic Energy |
|------|-----|-----------------|
| 사용 값 | softmax 후 확률 | **softmax 전 logit** |
| 정보 손실 | 정규화로 손실 | **logit 크기 보존** |
| 제로-엔트로피 | 해결 못함 | **해결** |

#### 장점
- **제로-엔트로피 문제 해결**: 일관된 오답도 logit 크기로 구분
- softmax 정규화로 인한 정보 손실 방지

#### 한계점
- 다양성 정보 부족 (엔트로피 개념 없음)
- 단독으로는 불완전

---

## 3. 기존 HSFE의 문제점

### 3.1 원래 제안
```python
HSFE = Semantic_Energy - T × Semantic_Entropy
```

### 3.2 문제점

1. **단순 선형 결합**: 이론적 깊이 부족
2. **기존 연구와 중복**: KLE, SNNE 등이 이미 유사한 통합 시도
3. **참신성 제한**: 단순 가중합은 새로운 기여로 보기 어려움

### 3.3 기존 연구들이 이미 해결한 것들

| 문제 | 해결한 연구 |
|------|-----------|
| Hard → Soft clustering | KLE (von Neumann Entropy) |
| 클러스터 내/간 유사도 무시 | SNNE, Cleanse |
| SE 일반화 | KLE, SNNE (이론적 증명) |
| 제로-엔트로피 | Semantic Energy |

---

## 4. 새로운 연구 방향 제안

### 4.1 Option A: KESFE (Kernel-Enhanced Semantic Free Energy)

**핵심 아이디어**: von Neumann Entropy 프레임워크 내에서 Energy-based confidence와 Semantic clustering 결합

```python
# 의미 커널 구성
K_sem = semantic_kernel(responses)  # NLI 기반

# Energy 정보를 커널에 통합
K_energy = energy_weighted_kernel(responses)

# 통합 커널
K_combined = α * K_sem + (1-α) * K_energy

# KESFE = von Neumann Entropy of combined kernel
KESFE = VNE(K_combined) = -Σ λ_i log λ_i
```

**차별점**:
- KLE의 이론적 프레임워크 활용
- Energy 정보를 커널 수준에서 통합
- 제로-엔트로피 문제를 커널 수준에서 해결

---

### 4.2 Option B: AHSFE (Adaptive Hybrid Semantic Free Energy)

**핵심 아이디어**: 동적 가중치 학습을 통해 입력별로 Energy/Entropy 기여도 자동 조정

```python
# 입력 특성 추출
features = extract_features(question, responses)

# 동적 가중치 예측 (학습된 네트워크)
w1, w2 = weight_predictor(features)  # w1 + w2 = 1

# AHSFE
AHSFE = w1 * Semantic_Energy + w2 * Semantic_Entropy
```

**차별점**:
- 고정 가중치 → **입력별 적응형 가중치**
- 메타러닝 관점에서 태스크/도메인 적응
- 제로-엔트로피 케이스 자동 감지 (w1 ↑)

---

### 4.3 Option C: HSFE + Hidden State (확장)

**핵심 아이디어**: Cleanse의 Hidden Embedding 아이디어를 Semantic Energy와 결합

```python
# Hidden Embedding 기반 유사도
emb_sim = hidden_embedding_similarity(responses)

# Logit 기반 Energy
energy = semantic_energy(responses)

# Cluster-aware 통합
# Intra-cluster: 높은 유사도 + 높은 Energy → 신뢰
# Inter-cluster: 낮은 유사도 + 다양한 Energy → 불확실

HSFE_extended = f(emb_sim, energy, clusters)
```

**차별점**:
- White-box 정보 (Hidden State) 활용
- 클러스터 수준의 세밀한 분석

---

## 5. 권장 연구 방향

### 5.1 최종 권장: Option B (AHSFE)

**이유**:
1. **이론적 참신성**: 적응형 가중치는 기존 연구에서 미탐구
2. **실용성**: 태스크/도메인별 자동 최적화
3. **구현 용이성**: 기존 SE/Energy 구현 재활용 가능
4. **확장성**: 다른 불확실성 지표 추가 용이

### 5.2 연구 질문 (재정립)

| RQ | 질문 |
|----|------|
| RQ1 | 적응형 가중치가 고정 가중치보다 환각 탐지에 효과적인가? |
| RQ2 | 어떤 입력 특성이 가중치 결정에 중요한가? |
| RQ3 | 제로-엔트로피 케이스에서 가중치가 Energy 쪽으로 이동하는가? |
| RQ4 | 도메인/태스크 간 전이학습이 가능한가? |

---

## 6. 참고문헌

1. **Semantic Entropy**: Farquhar et al., Nature 2024
2. **KLE**: Nikitin et al., arXiv:2405.20003, 2024
3. **SNNE**: Nguyen et al., arXiv:2506.00245, 2025
4. **Cleanse**: Joo & Cho, arXiv:2507.14649, 2025
5. **Semantic Energy**: Ma et al., arXiv:2412.07965, 2025
6. **BTProp**: arXiv:2404.18930, 2024
