# 2026-01-21 실험 요약

## 1. 연구 목표

LLM 환각(hallucination) 탐지를 위해 **Semantic Entropy (SE)**와 **Semantic Energy**를 적응적으로 결합하는 방법 연구.

**핵심 질문**: 언제 SE를 쓰고, 언제 Energy를 쓸지 미리 예측할 수 있는가?

---

## 2. 실험 1: Corpus Coverage 기반 적응적 가중치

### 가설
> "Corpus coverage가 낮으면 모델이 모르는 영역 → Energy 효과적"
> "Corpus coverage가 높으면 모델이 아는 영역 → SE 효과적"

### 방법
- QuCo-RAG 논문의 triplet extractor로 질문/응답에서 entity 추출
- Infini-gram API로 corpus 내 빈도 조회
- Coverage 백분위별로 SE/Energy AUROC 비교

### 결과: ❌ 실패

| 데이터셋 | 모든 백분위에서 우세한 방법 |
|---------|------------------------|
| TruthfulQA | SE |
| HaluEval | Energy |

**Coverage 값과 무관하게 데이터셋 전체 특성이 결정적**

### 문제점

1. **Corpus coverage가 의미 없음**: 낮은 coverage든 높은 coverage든 같은 방법이 우세
2. **데이터셋 간 패턴이 완전히 다름**: TruthfulQA는 항상 SE, HaluEval은 항상 Energy
3. **개별 샘플 수준 예측 불가**: 데이터셋 특성을 모르면 방법 선택 불가

---

## 3. 실험 2: 학습 기반 분류기

### 가설
> "SE, Energy 값과 추가 features로 최적 가중치를 학습할 수 있다"

### 방법
- Features: SE, Energy, 응답 길이, 클러스터 수, SE=0 여부 등
- 모델: Logistic Regression, MLP
- 평가: 같은 데이터셋 test + cross-dataset 일반화

### 결과

**같은 데이터셋 내 (✅ 성공)**:
```
Combined 데이터셋 → LogisticRegression_ext AUROC 0.91
```

**Cross-dataset (❌ 실패)**:
```
TruthfulQA로 학습 → HaluEval 평가: Energy_only(0.60) > 학습모델(0.52)
HaluEval로 학습 → TruthfulQA 평가: SE_only(0.62) > 학습모델(0.51)
```

### 문제점

1. **데이터셋 특성 차이가 너무 큼**
   - TruthfulQA: SE 평균 0.92, Zero-SE 20%
   - HaluEval: SE 평균 0.37, Zero-SE 57%

2. **학습된 패턴이 전이되지 않음**: 한 데이터셋에서 배운 "SE vs Energy 선택 기준"이 다른 데이터셋에서 무효

3. **데이터 부족**: 각 200 샘플로는 robust한 학습 어려움

---

## 4. 핵심 발견

### 왜 SE와 Energy가 데이터셋마다 다르게 작동하는가?

| 상황 | 모델 행동 | SE | Energy |
|-----|---------|-----|--------|
| **혼란 (TruthfulQA)** | 다양한 응답 생성 | ✅ 높음 (탐지 가능) | 보통 |
| **확신 있는 오답 (HaluEval)** | 일관된 오답 생성 | ❌ 낮음 (탐지 실패) | ✅ 높음 (탐지 가능) |

- **TruthfulQA**: 모델이 헷갈리는 질문 → 응답이 다양함 → SE 작동
- **HaluEval**: 모델이 확신하며 틀림 → 응답이 일관됨 → SE ≈ 0, Energy 필요

### 근본적 한계

**"어떤 방법이 효과적일지 미리 예측할 수 없다"**

- Corpus coverage: ❌ 의미 없음
- SE/Energy 값 자체: 계산해야 알 수 있음 (닭과 달걀)
- 학습 기반: 같은 분포 내에서만 효과적

---

## 5. 다음 단계

### Option A: SelfCheckGPT 방식 도입

SelfCheckGPT (1,177 citations, EMNLP 2023)의 아이디어 활용:
- 여러 샘플 간 **일관성**으로 환각 판단
- 일관되면 → 알고 있음, 불일관하면 → 환각

```
SelfCheck(s) = (1/N) Σ 1[contradict(s, s_i)]
```

**장점**: SE와 유사하지만 black-box로도 적용 가능

### Option B: Verbalized Confidence 활용

"Models Know What They Know" (880 citations) 기반:
- 모델에게 직접 확신도를 물어봄
- P(True) scoring

```
P(True) = p("True" | question, proposed answer)
```

**장점**: 별도 계산 없이 모델 내부 지식 활용

### Option C: 더 다양한 데이터셋으로 검증

현재 TruthfulQA, HaluEval 두 개만 사용 → 일반화 판단 어려움

추가 데이터셋 후보:
- WikiBio (FActScore에서 사용)
- CNN/DailyMail
- Natural Questions

### Option D: Ensemble 전략 변경

현재: `Score = w × Energy + (1-w) × SE`

대안:
- **Cascading**: SE 먼저 → SE ≈ 0이면 Energy 사용
- **Max voting**: SE와 Energy 중 더 극단적인 값 선택
- **Threshold-based**: SE > τ면 SE 사용, 아니면 Energy

---

## 6. 결론

1. **Corpus coverage 기반 적응적 가중치**: 의미 없음 (실험으로 확인)

2. **학습 기반 분류기**: 같은 분포 내에서만 효과적, cross-dataset 일반화 실패

3. **근본 원인**: 데이터셋마다 "모델이 혼란 vs 확신"의 비율이 다름. 이 비율을 미리 알 수 없음.

4. **다음 방향**: SelfCheckGPT 방식 또는 verbalized confidence 등 다른 신호 탐색 필요

---

## 7. 파일 구조

```
hallucination_lfe/
├── 260120_plan.md              # 상세 계획 + 참고문헌
├── 260121_summary.md           # 이 문서
├── experiment_notes/
│   ├── exp04_corpus_adaptive/  # Corpus 실험
│   ├── exp05_quintile_analysis/# 백분위 분석
│   └── exp06_weight_classifier/# 분류기 실험
│       ├── RESULTS.md
│       └── evaluation_results.csv
```
