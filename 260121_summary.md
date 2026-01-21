# 2026-01-21 연구 진행 보고

## 0. 요약 (TL;DR)

| 조교님 제기 문제 | 제안한 답 | 검증 결과 | 상태 |
|-----------------|----------|----------|------|
| 논문 스토리 부재 | "혼란 vs 지어냄" 두 유형 구분, corpus로 예측 | 실험 결과 불일치 | **재검토 필요** |
| Corpus별 SE/Energy AUROC 분석 | Coverage 백분위별 AUROC 비교 | Coverage 무의미, 데이터셋 특성이 결정적 | **가설 기각** |
| 가중치 도출 공식 + baseline | 학습 기반 분류기 시도 | 같은 분포 내에서만 작동, cross-dataset 실패 | **일반화 실패** |

---

## 1. 조교님이 제기한 문제 (260115 미팅)

### 문제 1: 논문 스토리 부재
> "논문 전체를 끌고갈 Introduction 및 직관적인 스토리가 없다"

### 문제 2: Corpus probability별 분석 필요
> "Corpus probability에 따라 SE와 Energy의 AUROC가 어떻게 변하는지 분석해야 한다"

### 문제 3: 가중치 도출 + Baseline 비교
> "Corpus statistics로부터 가중치를 도출하는 공식이 필요하고, baseline과 비교해야 한다"

---

## 2. 제안한 답 (260116 계획)

### 2.1 논문 스토리 제안

**핵심 아이디어**:
```
환각에는 두 종류가 있다:

[Type A: 혼란 (Confusion)]
- 모델이 알지만 헷갈림
- 다양한 (틀린) 답변 생성
- Semantic Entropy로 탐지

[Type B: 지어냄 (Confabulation)]  
- 모델이 모르고 지어냄
- 일관된 (틀린) 답변 생성
- Semantic Energy로 탐지
```

**One-liner**: "혼란은 다양성을 만들고, 지어냄은 일관성을 만든다. Corpus 통계가 어떤 경우인지 알려준다."

### 2.2 Corpus 기반 가중치 가설

```
가설:
- Corpus coverage 낮음 → 모델이 모름 → Energy 효과적
- Corpus coverage 높음 → 모델이 앎 → SE 효과적

공식:
Score = w(corpus) x Energy + (1-w(corpus)) x SE
```

---

## 3. 검증 과정 및 결과

### 3.1 실험 1: Corpus Coverage별 AUROC 분석

**방법**:
- QuCo-RAG 방식 triplet extractor로 entity 추출
- Infini-gram API로 corpus 빈도 조회
- Coverage 백분위(20%, 40%, 60%, 80%, 100%)별 SE/Energy AUROC 비교

**결과**: **가설 기각**

| 데이터셋 | 예상 | 실제 결과 |
|---------|------|----------|
| TruthfulQA | Coverage별로 SE/Energy 우세 교차 | **모든 백분위에서 SE 우세** |
| HaluEval | Coverage별로 SE/Energy 우세 교차 | **모든 백분위에서 Energy 우세** |

**문제점**:
1. Corpus coverage 값이 SE/Energy 선택에 영향 없음
2. 데이터셋 전체 특성이 결정적 (개별 샘플 수준 예측 불가)

---

### 3.2 실험 2: 학습 기반 분류기

**방법**:
- Features: SE, Energy, 응답 길이, 클러스터 수, SE=0 여부
- 모델: Logistic Regression, MLP
- 평가: 같은 데이터셋 test + cross-dataset 일반화

**결과**: **일반화 실패**

| 조건 | AUROC | 비교 |
|-----|-------|------|
| **같은 분포 (Combined)** | 0.91 | 성공 |
| TruthfulQA 학습 → HaluEval 평가 | 0.52 | Energy_only(0.60)보다 낮음 |
| HaluEval 학습 → TruthfulQA 평가 | 0.51 | SE_only(0.62)보다 낮음 |

**문제점**:
- 학습된 패턴이 다른 데이터셋으로 전이 안 됨
- 결국 "어떤 방법이 효과적일지 미리 예측 불가"

---

## 4. 왜 실패했는가? (분석)

### 데이터셋 특성이 근본적으로 다름

| 지표 | TruthfulQA | HaluEval |
|-----|------------|----------|
| SE 평균 | **0.92** (높음) | **0.37** (낮음) |
| Zero-SE 비율 | 20% | **57%** |
| Hallucination rate | 82.5% | 10.5% |
| 모델 행동 | 혼란 (다양한 응답) | 확신 (일관된 응답) |
| 효과적 방법 | SE | Energy |

**핵심 발견**:
- TruthfulQA: 모델이 헷갈려함 → 응답 다양 → SE 작동
- HaluEval: 모델이 확신하며 틀림 → 응답 일관 → SE 실패, Energy 필요

**근본적 한계**:
> "어떤 방법이 효과적일지 미리 예측할 수 없다"
> - Corpus coverage: 의미 없음 (실험으로 확인)
> - SE/Energy 값: 계산해야 알 수 있음 (닭과 달걀)
> - 학습 기반: 같은 분포 내에서만 효과적

---

## 5. 앞으로 어떻게 해야 할까?

### Option A: 다른 신호 탐색 (추천)

| 방법 | 설명 | 인용수 |
|-----|------|-------|
| **SelfCheckGPT** | 샘플 간 모순으로 환각 판단 | 1,177 |
| **Verbalized Confidence** | 모델에게 직접 확신도 질문 | 880 |
| **P(True) Scoring** | "이 답이 맞나요?" 확률 측정 | 880 |

### Option B: Ensemble 전략 변경

현재 시도한 방식:
```
Score = w x Energy + (1-w) x SE   (w를 예측하려 했으나 실패)
```

대안:
```
[Cascading] SE 먼저 계산 → SE ≈ 0이면 Energy 사용
[Max voting] SE와 Energy 중 더 극단적인 값 선택
[Threshold] SE > τ면 SE 사용, 아니면 Energy
```

### Option C: 더 다양한 데이터셋 검증

현재 2개(TruthfulQA, HaluEval)만 사용 → 일반화 판단 어려움

추가 후보:
- WikiBio (FActScore)
- Natural Questions
- TriviaQA

### Option D: 스토리 방향 전환

**현재 스토리 (실패)**:
> "Corpus coverage로 어떤 방법이 효과적일지 예측할 수 있다"

**대안 스토리 후보**:
1. "SE와 Energy는 상호보완적이며, cascading 방식으로 결합하면 효과적"
2. "Zero-SE 문제를 Energy가 해결한다" (이건 이미 실험으로 확인됨: AUROC 0.768)
3. 완전히 새로운 방향 (SelfCheckGPT 확장 등)

---

## 6. 조교님께 여쭤보고 싶은 것

1. **방향 전환 여부**: Corpus 기반 가중치가 실패했는데, 다른 접근법으로 전환해야 할까요?

2. **스토리 수정**: "혼란 vs 지어냄" 프레임워크는 유지하되, 예측 방법만 바꿔야 할까요?

3. **Cascading 방식**: `SE → Energy fallback` 전략이 더 실용적일 수 있을까요?

4. **데이터셋 확장**: 더 다양한 데이터셋에서 패턴을 찾아야 할까요?

---

## 7. 파일 구조

```
hallucination_lfe/
├── 260115_meeting.md           # 조교님 미팅 원본
├── 260116_plan.md              # 스토리 제안 + 실험 계획
├── 260120_plan.md              # 상세 결과 + 참고문헌
├── 260121_summary.md           # 이 문서
└── experiment_notes/
    ├── exp04_corpus_adaptive/  # Corpus 실험
    ├── exp05_quintile_analysis/# 백분위 분석 (가설 기각)
    └── exp06_weight_classifier/# 분류기 실험 (일반화 실패)
```
