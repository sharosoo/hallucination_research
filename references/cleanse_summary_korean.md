# Cleanse: 클러스터링 기반 의미적 일관성을 활용한 LLM 불확실성 추정

> **논문**: Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs  
> **저자**: Minsuh Joo, Hyunsoo Cho (이화여자대학교)  
> **출처**: arXiv:2507.14649v1 [cs.CL] 19 Jul 2025

---

## 1. 문제 정의 (Problem Definition)

### 핵심 문제
- **LLM 환각(Hallucination)**: LLM이 유창하고 그럴듯해 보이지만, 사실적으로 부정확하거나 근거 없는 응답을 생성하는 문제
- 특히 **QA(Question-Answering) 태스크**에서 정확성을 객관적으로 검증할 수 있어 환각 문제가 더욱 심각함
- 비전문가 사용자가 환각을 탐지하기 어려워 고위험 응용 분야에서 심각한 결과 초래 가능

### 기존 접근법의 한계
1. **데이터셋 정제**: 노동 집약적이고 확장성 부족
2. **RAG (Retrieval-Augmented Generation)**: 복잡한 파이프라인, 높은 계산 비용
3. **토큰 레벨 불확실성 추정** (Perplexity, LN-Entropy): 의미적 동등성을 고려하지 못함
4. **문장 레벨 유사도** (Rabinovich et al., 2023): 모든 쌍별 유사도의 단순 평균 사용 → 의미적으로 상이한 출력이 마스킹될 수 있음

---

## 2. 제안 방법: Cleanse (Clustering-based Semantic Consistency)

### 핵심 아이디어
> "클러스터 내(intra-cluster) 유사도의 비율을 일관성의 척도로, 클러스터 간(inter-cluster) 유사도를 페널티로 사용하여 불확실성을 정량화"

### 방법론 파이프라인

```
[입력 질문] → [다중 출력 생성] → [Hidden Embedding 추출] → [양방향 NLI 클러스터링] → [Cleanse Score 계산]
```

### 2.1 Hidden Embeddings
- LLM의 **중간 레이어(middle layer)**에서 **마지막 토큰의 임베딩**을 사용
- 이전 연구에서 의미적 정보를 효과적으로 포착한다고 검증됨 (Azaria and Mitchell, 2023)
- 임베딩 간 일관성은 **코사인 유사도**로 측정

### 2.2 클러스터링 기법
- **양방향 NLI(Bi-directional Natural Language Inference)** 기반 클러스터링
- 두 출력이 **서로를 함의(entail)**해야만 동일 클러스터로 분류 → 진정한 의미적 동등성 보장
- 입력 형식: `<Question + Answer>` 연결 (문맥 이해 향상)
- 클러스터링 모델: **nli-deberta-v3-base** (184M 파라미터, 효율적)

### 2.3 클러스터링 직관
- **좋은 클러스터링**: 클러스터 간 거리 최대화, 클러스터 내 거리 최소화 (Dunn's Index)
- **유사도 관점 변환**: 
  - 높은 intra-cluster 유사도 = 동일한 의미를 공유하는 임베딩 다수
  - 높은 inter-cluster 유사도 = 다양한 의미의 임베딩 존재 (불확실성)

---

## 3. 핵심 수식 (Key Formulas)

### 3.1 Intra-cluster Similarity (클러스터 내 유사도)

$$\text{intra-cluster sim.} = \sum_{k=1}^{C} \sum_{i=1}^{N_k-1} \sum_{j=i+1}^{N_k} \text{cosine}(e_i, e_j)$$

- $C$: 클러스터 수
- $N_k$: k번째 클러스터의 임베딩 수
- $\text{cosine}(e_i, e_j)$: i번째와 j번째 임베딩 간 코사인 유사도

### 3.2 Total Similarity (전체 유사도)

$$\text{total sim.} = \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} \text{cosine}(e_i, e_j)$$

- $K$: 전체 출력 수

### 3.3 Cleanse Score

$$\text{Cleanse Score} = 1 - \frac{\text{inter-cluster sim.}}{\text{total sim.}} = \frac{\text{intra-cluster sim.}}{\text{total sim.}}$$

### 해석
| 상황 | 클러스터 수 | Intra/Total 비율 | Cleanse Score | 불확실성 |
|------|-------------|------------------|---------------|----------|
| 일관된 응답 | 적음 | 높음 | 높음 (예: 0.947) | 낮음 |
| 불일관 응답 | 많음 | 낮음 | 낮음 (예: 0.409) | 높음 |

### 3.4 비교 기준: Cosine Score (Baseline)

$$\text{cosine score} = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} \text{cosine}(e_i, e_j)$$

---

## 4. 기존 Semantic Entropy와의 차이점

| 구분 | Semantic Entropy (Kuhn et al., 2023) | Cleanse |
|------|--------------------------------------|---------|
| **클러스터링 기반** | 의미적 동등 그룹에 대한 엔트로피 계산 | 클러스터 내/간 유사도 비율로 일관성 측정 |
| **측정 방식** | 그룹별 엔트로피의 합 | Intra-cluster 유사도 / Total 유사도 |
| **임베딩 활용** | NLI 기반 클러스터링만 사용 | Hidden embedding의 코사인 유사도 적극 활용 |
| **페널티 개념** | 없음 | Inter-cluster 유사도를 명시적 페널티로 사용 |
| **핵심 차별점** | 단순 클러스터 수 기반 | 클러스터 간 연결 강도까지 고려 |

### Cleanse의 개선점
1. **문장 레벨 임베딩 활용**: 토큰 확률 대신 의미적 임베딩 사용
2. **Inter-cluster 유사도 페널티**: 단순 평균이 아닌, 의미적 불일치를 명시적으로 반영
3. **더 정밀한 경계 구분**: 확신/불확신 응답 간 경계를 더 명확하게 분리

---

## 5. 실험 결과 요약

### 5.1 실험 설정
- **데이터셋**: SQuAD, CoQA (QA 벤치마크)
- **모델**: LLaMA-7B, LLaMA-13B, LLaMA2-7B, Mistral-7B
- **평가 지표**: AUROC, PCC (Pearson Correlation Coefficient)
- **정확성 기준**: Rouge-L > 0.7

### 5.2 주요 결과

| 방법 | 레벨 | 평균 성능 순위 |
|------|------|----------------|
| Perplexity | 토큰 | 5위 |
| LN-Entropy | 토큰 | 4위 |
| Lexical Similarity | 토큰 | 3위 |
| Cosine Score | 문장 | 2위 |
| **Cleanse Score** | 문장 | **1위** |

### 5.3 대표적 AUROC 결과 (Rouge-L threshold = 0.7)

| 모델 | 데이터셋 | Lexical Sim. | Cosine Score | **Cleanse** |
|------|----------|--------------|--------------|-------------|
| LLaMA-7B | SQuAD | 76.9 | 79.6 | **81.7** |
| LLaMA-7B | CoQA | 76.1 | 78.5 | **79.4** |
| LLaMA-13B | SQuAD | 78.9 | 81.1 | **82.8** |
| LLaMA2-7B | SQuAD | 80.4 | 82.1 | **83.0** |
| Mistral-7B | SQuAD | 69.0 | 65.9 | **75.9** |
| Mistral-7B | CoQA | 74.9 | 74.1 | **80.2** |

### 5.4 엄격한 조건에서의 우수성
- Rouge-L threshold가 0.5 → 0.9로 증가할수록 (더 엄격한 정확성 기준)
- Cleanse와 Lexical Similarity 간 성능 차이가 **더욱 확대**
- Mistral-7B에서 최대 **8.4% (SQuAD), 8.0% (CoQA)** 차이

### 5.5 클러스터링 모델 비교
| 클러스터링 모델 | 평균 AUROC | 클러스터 수 차이 |
|----------------|------------|------------------|
| deberta-large-mnli | 79.9 | 2.56 |
| roberta-large-mnli | 79.4 | 2.41 |
| **nli-deberta-v3-base** | **80.3** | **2.63** |
| nli-deberta-v3-large | 80.0 | 2.53 |

---

## 6. 한계점 (Limitations)

### 주요 한계
1. **White-box LLM에만 적용 가능**
   - 모델 내부의 Hidden Embedding에 직접 접근해야 함
   - Black-box API 기반 LLM (GPT-4, Claude 등)에는 적용 불가

### 저자가 제안한 극복 방안
- 모델 내부 임베딩 대신 **다른 벡터 임베딩** 사용 가능성 언급
- 외부 임베딩 모델로 출력 텍스트를 인코딩하여 유사한 접근 가능

### 추가적 한계 (논문에서 직접 언급되지 않음)
2. **다중 출력 생성 비용**: K개의 출력을 생성해야 하므로 추론 비용 증가
3. **NLI 모델 의존성**: 클러스터링 품질이 NLI 모델 성능에 좌우됨
4. **QA 태스크 특화**: 정답이 명확한 QA에서 검증, 개방형 생성 태스크에서의 효과 미검증
5. **언어 제한**: 영어 데이터셋에서만 실험 수행

---

## 핵심 요약

| 항목 | 내용 |
|------|------|
| **문제** | LLM의 환각 탐지를 위한 불확실성 추정 |
| **핵심 아이디어** | 클러스터 내 유사도를 일관성으로, 클러스터 간 유사도를 페널티로 활용 |
| **주요 기여** | 의미적 클러스터링 + Hidden Embedding 유사도 결합 |
| **성능** | 모든 베이스라인 대비 AUROC, PCC 최고 성능 |
| **한계** | White-box 모델에만 적용 가능 |

---

## 참고: 핵심 알고리즘 (Bi-directional Entailment)

```
입력: 문맥 x, 출력 집합 {s(2), ..., s(M)}, NLI 분류기 M
초기화: 클러스터 집합 C = {{s(1)}}

for m = 2 to M:
    for each cluster c in C:
        s(c) ← c의 대표 출력
        left ← M(cat(x, s(c), "</g>", x, s(m)))
        right ← M(cat(x, s(m), "</g>", x, s(c)))
        
        if left와 right 모두 entailment:
            c에 s(m) 추가  # 기존 클러스터에 합류
            break
    
    if s(m)이 어떤 클러스터에도 추가되지 않음:
        C에 {s(m)} 추가  # 새 클러스터 생성

return C
```
