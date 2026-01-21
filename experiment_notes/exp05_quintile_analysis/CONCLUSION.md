# Exp05: Coverage 백분위별 SE vs Energy AUROC 분석 결론

## 실험 목적

QuCo-RAG 논문 기반 corpus coverage를 활용하여, coverage 수준에 따라 SE와 Energy의 효과성이 달라지는지 검증.

**원래 가설**: 
- Low coverage (모델이 모르는 지식) → Energy 효과적
- High coverage (모델이 아는 지식) → SE 효과적

---

## 실험 결과

### TruthfulQA

| 백분위 | n | SE AUROC | Energy AUROC | 우세 |
|--------|---|----------|--------------|------|
| 하위 20% | 40 | 0.639 | 0.556 | **SE** |
| 20-40% | 40 | 0.639 | 0.556 | **SE** |
| 40-60% | 40 | 0.469 | 0.310 | **SE** |
| 60-80% | 40 | 0.678 | 0.680 | Tie |
| 상위 20% | 40 | 0.678 | 0.680 | Tie |

→ **전 구간에서 SE 우세 또는 동등**

### HaluEval

| 백분위 | n | SE AUROC | Energy AUROC | 우세 |
|--------|---|----------|--------------|------|
| 하위 20% | 40 | 0.458 | 0.598 | **Energy** |
| 20-40% | 40 | 0.458 | 0.598 | **Energy** |
| 40-60% | 40 | 0.458 | 0.598 | **Energy** |
| 60-80% | 40 | 0.544 | 0.598 | **Energy** |
| 상위 20% | 40 | 0.544 | 0.598 | **Energy** |

→ **전 구간에서 Energy 우세**

---

## 결론

### 1. 원래 가설 기각

Coverage 백분위에 따라 SE/Energy의 효과성이 변하지 않음.
- TruthfulQA: 모든 백분위에서 SE 우세
- HaluEval: 모든 백분위에서 Energy 우세

### 2. 실제 원인: 데이터셋 특성 차이

| 데이터셋 | 특성 | 최적 방법 |
|----------|------|-----------|
| TruthfulQA | 대중적 오개념, 모델이 혼란스러워하는 질문 | SE |
| HaluEval | 지식 기반 QA, 모델이 모르면서 지어내는 답변 | Energy |

### 3. Corpus Coverage의 한계

- Corpus coverage는 SE/Energy 선택에 **통계적으로 유의미한 신호가 아님**
- 데이터셋 자체의 환각 유형(confusion vs confabulation)이 결정적 요인
- Coverage 분포가 매우 skewed하여 실질적 변별력 없음

---

## 향후 방향

1. **단순화**: 데이터셋/도메인 특성에 따라 SE 또는 Energy 선택
2. **Zero-SE Fallback**: SE가 0에 가까울 때만 Energy 사용 (이전 실험에서 효과 확인)
3. **Corpus 활용 포기**: Coverage 기반 adaptive weighting은 효과 없음

---

## 파일

- `quintile_analysis.ipynb`: 분석 노트북 (Plotly 그래프 포함)
- `truthfulqa_with_corpus.json`: TruthfulQA 데이터 (symlink)
- `halueval_with_corpus.json`: HaluEval 데이터 (symlink)
