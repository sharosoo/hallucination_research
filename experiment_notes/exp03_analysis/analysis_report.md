# Exp03: 데이터셋별 결과 분석 및 비교

> 작성일: 2025-01-15
> 실험: Semantic Entropy vs Semantic Energy AUROC 비교

## 1. 실험 개요

### 1.1 목적
- TruthfulQA와 HaluEval 두 데이터셋에서 SE와 Energy의 환각 탐지 성능 비교
- 데이터셋 특성에 따른 최적 지표 파악
- AHSFE 적응형 가중치의 필요성 검증

### 1.2 실험 설정
| 항목 | 값 |
|------|-----|
| LLM | Qwen2.5-3B-Instruct |
| NLI 모델 | DeBERTa-large-mnli |
| 샘플링 수 | 5개/질문 |
| Temperature | 0.7 |
| 평가 샘플 | 각 200개 |

---

## 2. 실험 결과

### 2.1 TruthfulQA

| 지표 | AUROC | AUPRC |
|------|-------|-------|
| **Semantic Entropy** | **0.6190** | 0.8727 |
| Semantic Energy | 0.5380 | 0.8445 |

**클래스 분포:** 환각 165개 (82.5%), 정상 35개 (17.5%)

**통계:**
```
SE (환각):     0.9625 ± 0.5689
SE (정상):     0.7277 ± 0.5597
Energy (환각): -44.1163 ± 2.9685
Energy (정상): -45.1367 ± 4.1497
```

**제로-엔트로피 케이스 (SE < 0.1):**
- 환각: 31개, 정상: 10개
- Energy AUROC: **0.7677**

### 2.2 HaluEval QA

| 지표 | AUROC | AUPRC |
|------|-------|-------|
| Semantic Entropy | 0.5057 | 0.1150 |
| **Semantic Energy** | **0.6036** | 0.2002 |

**클래스 분포:** 환각 21개 (10.5%), 정상 179개 (89.5%)

**통계:**
```
SE (환각):     0.3893 ± 0.5222
SE (정상):     0.3627 ± 0.4765
Energy (환각): -49.8967 ± 3.3266
Energy (정상): -51.4426 ± 3.0023
```

**제로-엔트로피 케이스 (SE < 0.1):**
- 환각: 12개, 정상: 102개
- Energy AUROC: 0.5940

---

## 3. 비교 분석

### 3.1 데이터셋별 최적 지표

| 데이터셋 | 승자 | AUROC 차이 | 특성 |
|----------|------|------------|------|
| TruthfulQA | SE | +0.081 | 대중적 오개념 → 다양한 오답 생성 |
| HaluEval | Energy | +0.098 | knowledge 기반 → 일관된 답변 |

### 3.2 왜 다른 결과가 나왔는가?

**TruthfulQA:**
- 대중적 오개념을 테스트하는 질문들
- 모델이 "그럴듯하지만 틀린" 다양한 답변 생성
- 응답 다양성이 높음 → SE가 효과적

**HaluEval:**
- knowledge 컨텍스트가 주어짐
- 모델이 knowledge 기반으로 일관된 답변 생성
- 응답 다양성이 낮음 → SE가 구분 못함 → Energy가 효과적

### 3.3 제로-엔트로피 문제

TruthfulQA에서 명확히 확인됨:
- SE < 0.1인 케이스: 환각 31개, 정상 10개
- **SE만으로는 75%가 환각인데도 "신뢰 가능"으로 오판**
- Energy AUROC 0.7677 → Energy가 이를 보완

---

## 4. 핵심 인사이트

### 4.1 고정 가중치의 한계
```
TruthfulQA 최적: SE 위주 (w_se ↑)
HaluEval 최적: Energy 위주 (w_energy ↑)
```
→ 하나의 고정 가중치로 두 데이터셋 모두 최적화 불가능

### 4.2 AHSFE 필요성 확인
- 입력/태스크 특성에 따라 최적 가중치가 다름
- 적응형 가중치가 필요한 이유가 실험적으로 검증됨

### 4.3 Weight Predictor 설계 힌트
입력 특성 → 가중치 예측에 유용한 신호:
1. `num_clusters`: 클러스터 수 (많으면 SE 가중치 ↑)
2. `largest_cluster_ratio`: 최대 클러스터 비율 (높으면 Energy 가중치 ↑)
3. `raw_entropy`: SE 값 자체 (낮으면 Energy로 전환)
4. `has_knowledge`: knowledge 유무

---

## 5. 결론

1. **SE와 Energy는 상호 보완적**
   - SE: 응답 다양성이 높을 때 효과적
   - Energy: 일관된 오답 케이스에서 효과적

2. **데이터셋/태스크별 최적 전략이 다름**
   - 고정 가중치로는 범용 성능 달성 불가
   - AHSFE의 적응형 가중치가 해답

3. **제로-엔트로피 문제는 실제로 존재**
   - TruthfulQA에서 31개 케이스 확인
   - Energy가 이를 효과적으로 보완 (AUROC 0.77)

---

## 6. 다음 단계

1. FeatureExtractor 구현
2. WeightPredictor 신경망 구현
3. AHSFE 클래스 구현
4. TruthfulQA로 학습, HaluEval로 검증 (cross-domain 전이)
