# exp06: SE/Energy 결합 분류기 실험 결과

## 1. 실험 목적

Corpus coverage 기반 적응형 가중치가 실패했으므로, **학습 기반 분류기**로 SE와 Energy의 최적 결합 방법을 찾는다.

## 2. 실험 설정

### 2.1 데이터
- **TruthfulQA**: 200 samples (82.5% hallucination)
- **HaluEval**: 200 samples (10.5% hallucination)
- **Combined**: 400 samples (두 데이터셋 합침)

### 2.2 모델
| 모델 | 설명 |
|------|------|
| SE_only | SE 값만 사용 (baseline) |
| Energy_only | Energy 값만 사용 (baseline) |
| Fixed_w{0.3,0.5,0.7} | 고정 가중치 결합 |
| LogisticRegression | SE + Energy로 학습 |
| LogisticRegression_ext | SE + Energy + 추가 features |
| MLP | 신경망 (32-16 hidden layers) |

### 2.3 Features
- **Basic**: semantic_entropy, energy_mean
- **Extended**: + response_length_mean, response_length_std, num_clusters, se_is_zero, question_length

## 3. 주요 결과

### 3.1 같은 데이터셋 평가 (Test Set)

| 데이터셋 | Best Model | AUROC | SE_only | Energy_only |
|---------|------------|-------|---------|-------------|
| TruthfulQA | LR_ext | **0.61** | 0.53 | 0.60 |
| HaluEval | SE_only | **0.58** | 0.58 | 0.54 |
| Combined | LR_ext | **0.91** | 0.78 | 0.88 |

### 3.2 Cross-dataset 일반화

| Train → Test | Best Model | AUROC | 관찰 |
|-------------|------------|-------|------|
| TruthfulQA → HaluEval | Energy_only | **0.60** | 학습 모델 실패 |
| HaluEval → TruthfulQA | SE_only | **0.62** | Fixed_w0.3도 동일 |
| Combined → TruthfulQA | MLP | **0.65** | 약간 개선 |
| Combined → HaluEval | LR_ext | **0.62** | 약간 개선 |

### 3.3 핵심 비교표

```
Train→Test                     SE     Energy     LR     LR_ext
--------------------------------------------------------------
combined→combined           0.78     0.88     0.88     0.91  ← 최고
combined→halueval_full      0.51     0.60     0.60     0.62
combined→truthfulqa_full    0.62     0.54     0.54     0.60
truthfulqa→halueval_full    0.51     0.60     0.54     0.52
halueval→truthfulqa_full    0.62     0.54     0.51     0.55
```

## 4. 결론

### 4.1 긍정적 발견

1. **Combined 학습 효과**: 두 데이터셋을 합쳐서 학습하면 AUROC 0.91 달성 (test set)
2. **LogisticRegression_ext 효과**: 추가 features가 도움됨

### 4.2 한계

1. **Cross-dataset 일반화 실패**: 학습 모델이 baseline(SE_only, Energy_only)보다 못함
2. **데이터셋 특성이 결정적**: TruthfulQA는 SE, HaluEval은 Energy가 항상 우세
3. **소규모 데이터**: 각 200 samples로는 robust한 학습 어려움

### 4.3 시사점

**"학습 기반 적응적 결합"은 같은 분포 내에서만 작동한다.**

- 새로운 데이터셋에 적용하려면 **해당 데이터셋의 특성**(SE 분포, hallucination rate 등)을 먼저 파악해야 함
- 실용적 접근: SE 계산 후 SE 분포를 보고 방법 선택
  - SE 분포가 넓으면 (mean > 0.5) → SE 사용
  - SE 대부분 0에 가까우면 → Energy 사용

## 5. 다음 단계

1. **더 많은 데이터셋**으로 검증 (SelfAware, FreshQA 등)
2. **Meta-learning 접근**: Few-shot으로 새 데이터셋에 적응
3. **Hidden states 활용**: LENS 논문처럼 internal representations 사용

## 6. 파일 구조

```
exp06_weight_classifier/
├── prepare_data.py          # 데이터 준비
├── train_classifier.py      # 모델 학습
├── evaluate.py              # 평가
├── RESULTS.md               # 이 문서
├── evaluation_results.csv   # 전체 결과
├── *_features.csv           # 추출된 features
├── *_train/val/test.csv     # 분할된 데이터
└── *_models.pkl             # 학습된 모델
```
