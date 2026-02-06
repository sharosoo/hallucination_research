# 260206 조교님 미팅 보고

## 0. 요약 (TL;DR)

| 항목 | 내용 | 상태 |
|------|------|------|
| 이전 문제 | Corpus 기반 가중치 예측 실패, 학습 기반 분류기 일반화 실패 | 해결 방향 전환 |
| 현재 스토리 | "Zero-SE 문제 → Energy가 해결" (SE-gated cascade) | **5개 데이터셋으로 검증** |
| 핵심 결과 | Zero-SE에서 Energy AUROC 0.60~0.74 (3/5), Cascade는 SE-only와 통계적 동등 (5/5) | **스토리 일부 확인** |

---

## 1. 배경: 왜 방향을 전환했는가

### 1.1 이전 접근법 (실패)

[260121 미팅 보고서](https://github.com/sharosoo/hallucination_research/blob/master/260121_summary.md)에서 보고한 결과:
- **Corpus coverage 기반 가중치 예측**: Coverage가 SE/Energy 선택에 영향 없음 → **가설 기각** ([exp05](https://github.com/sharosoo/hallucination_research/tree/master/experiment_notes/exp05_combined_analysis))
- **학습 기반 분류기**: 같은 분포 내에서만 작동 (AUROC 0.91), cross-dataset에서 실패 (AUROC 0.51) → **일반화 실패** ([exp06](https://github.com/sharosoo/hallucination_research/tree/master/experiment_notes/exp06_weight_classifier))

### 1.2 260121 미팅에서 조교님 피드백

> "Cascading 방식 (SE → Energy fallback)이 더 실용적"
> "더 다양한 데이터셋에서 검증 필요"

### 1.3 새로운 스토리

```
핵심 관찰: SE는 "자기 일관성"을 측정하지, "진실"을 측정하지 않는다.

문제: 모델이 일관되게 틀리면 (Zero-SE) SE는 무력하다.

해결: SE가 낮을 때 (모델이 과확신) → Energy로 fallback

      ┌─────────────────────────────────────────────┐
      │  SE 높음 → 혼란(Confusion) → SE가 탐지       │
      │  SE 낮음 → 지어냄(Confabulation) → Energy가 탐지│
      └─────────────────────────────────────────────┘
```

---

## 2. 실험 설정

### 2.1 공통 파이프라인

| 항목 | 값 |
|------|-----|
| LLM | Qwen2.5-3B-Instruct |
| NLI 모델 | DeBERTa-large-mnli |
| 샘플링 | K=5, temperature=0.7 |
| 샘플 수 | 각 200개 |

### 2.2 데이터셋 (기존 2개 + 신규 3개)

| 데이터셋 | 소스 | 환각 수 | 정상 수 | 환각률 | 특성 | 원본 데이터 |
|----------|------|---------|---------|--------|------|------------|
| **TruthfulQA** | 기존 exp01 | 164 | 36 | 82.0% | 대중적 오개념 | [results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp01_truthfulqa/results.json) |
| **HaluEval-QA** | 기존 exp02 | 21 | 179 | 10.5% | knowledge 기반 QA | [results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp02_halueval/results.json) |
| **TriviaQA** | 신규 | 108 | 92 | 54.0% | trivia 지식 | [results_triviaqa.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_triviaqa.json) |
| **NaturalQuestions** | 신규 | 134 | 66 | 67.0% | 일반 QA | [results_naturalquestions.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_naturalquestions.json) |
| **HaluEval-dialogue** | 신규 | 188 | 12 | 94.0% | dialogue 기반 | [results_halueval_dialogue.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_halueval_dialogue.json) |

---

> **exp07 분석 코드**: [analyze_existing.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/analyze_existing.py) | **신규 데이터셋 실험 코드**: [run_new_datasets.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/run_new_datasets.py) | **전체 분석 결과 JSON**: [analysis_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/analysis_results.json)
>
> **exp08 보강 분석**: [analyze_robustness.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/analyze_robustness.py) | [robustness_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/robustness_results.json)
>
> **상세 결과 문서**: [RESULTS.md](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/RESULTS.md) | **Figure 생성 코드**: [generate_all.py](https://github.com/sharosoo/hallucination_research/blob/master/figures/generate_all.py)

## 3. 실험 결과

### 3.1 E1: Zero-SE 현상 정량화

> **Zero-SE 정의**: K=5 응답이 모두 **단일 NLI 클러스터**에 속하는 경우 (SE=0).
> exp08 검증 결과 SE≤0.001과 num_clusters==1이 **5/5 데이터셋에서 완전 일치** 확인.

**핵심 질문**: "SE가 0에 가까운 샘플이 얼마나 있고, 그 중 환각은 몇 %인가?"

![Figure 1. Zero-SE Phenomenon Across 5 Datasets](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig1_zero_se_overview.png)

| 데이터셋 | Zero-SE 비율 | Zero-SE 내 환각률 | **Energy AUROC** | 95% CI |
|----------|-------------|------------------|-----------------|--------|
| **TruthfulQA** | 19.0% (38/200) | **73.7%** | **0.736** | [0.52, 0.93] |
| **HaluEval-QA** | 53.5% (107/200) | 9.3% | **0.602** | [0.39, 0.79] |
| **TriviaQA** | 29.0% (58/200) | 29.3% | **0.689** | [0.55, 0.82] |
| NaturalQuestions | 17.0% (34/200) | 50.0% | 0.457 | [0.26, 0.66] |
| HaluEval-dialogue | 24.5% (49/200) | 89.8% | 0.486 | [0.13, 0.85] |

**발견**:
1. Zero-SE 현상은 **모든 데이터셋에서 존재** (17~53.5%)
2. Zero-SE 영역에서 Energy가 효과적인 경우: **3/5 데이터셋** (AUROC > 0.6)
   - **TruthfulQA** (0.736), **TriviaQA** (0.689), **HaluEval-QA** (0.602)
3. Energy가 비효과적인 경우: **NaturalQuestions** (0.457), **HaluEval-dialogue** (0.486)

> **ε = 0.001~0.1 에서 결과 동일** — SE 값이 정확히 0.0인 샘플이 대부분이므로 ε 선택에 robust

---

### 3.2 E3: SE 구간별 Crossover 분석

**핵심 질문**: "SE 값이 낮을 때는 Energy가 우세하고, 높을 때는 SE가 우세한 crossover가 존재하는가?"

![Figure 2. SE vs Energy AUROC by Semantic Entropy Bin](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig2_se_bin_crossover.png)

**TruthfulQA — 가장 명확한 crossover**:

| SE 구간 | n | 환각률 | SE AUROC | Energy AUROC | **승자** |
|---------|---|--------|----------|-------------|---------|
| **Zero-SE [0, 0.05]** | 38 | 73.7% | N/A | **0.736** | **Energy** |
| Medium (0.3, 0.6] | 23 | 78.3% | N/A | 0.578 | Energy |
| High (0.6, 1.0] | 44 | 84.1% | **0.664** | 0.517 | **SE** |
| Very High (1.0, ∞) | 95 | 85.3% | **0.658** | 0.422 | **SE** |

**Crossover 패턴 요약**:

| 데이터셋 | Zero-SE: Energy 우세? | High-SE: SE 우세? | Crossover? |
|----------|----------------------|-------------------|-----------|
| TruthfulQA | ✅ 0.736 | ✅ 0.664 | ✅ |
| HaluEval-QA | ✅ 0.602 | ✅ 0.343 | ✅ |
| TriviaQA | ✅ 0.689 | ✅ 0.636 | ✅ |
| NaturalQuestions | ❌ 0.457 | ✅ 0.570 | ❌ |
| HaluEval-dialogue | ❌ 0.486 | ✅ 0.574 | ❌ |

> **5개 중 3개 데이터셋에서 crossover 확인** — Factoid QA 데이터셋에서 패턴이 일관됨

---

### 3.3 E4: SE-Gated Cascade 성능

**핵심 질문**: "SE < τ면 Energy 사용, 아니면 SE 사용하는 cascade가 SE-only보다 나은가?"

![Figure 4. Cascade Threshold Sweep](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig4_cascade_sweep.png)

#### 3.3.1 최적 τ 결과

| 데이터셋 | Best τ | Cascade AUROC | SE-only | Energy-only | Δ vs SE | Δ vs Energy |
|----------|--------|-------------|---------|-------------|---------|-------------|
| **TruthfulQA** | 0.526 | **0.643** | 0.613 | 0.550 | **+0.030** | +0.093 |
| **HaluEval-QA** | 1.332 | **0.614** | 0.540 | 0.616 | **+0.074** | -0.002 |
| TriviaQA | 0.000 | 0.663 | **0.676** | 0.628 | -0.013 | +0.035 |
| NaturalQuestions | 1.609 | 0.619 | 0.615 | **0.619** | +0.004 | +0.000 |
| HaluEval-dialogue | 0.526 | 0.596 | **0.599** | 0.566 | -0.002 | +0.030 |

#### 3.3.2 Cross-Dataset τ Transfer

TruthfulQA에서 학습한 **τ=0.526을 고정**하여 다른 데이터셋에 적용:

| 적용 데이터셋 | Cascade | SE-only | Δ vs SE | 판정 |
|-------------|---------|---------|---------|------|
| TruthfulQA (in-domain) | 0.643 | 0.613 | **+0.030** | ✅ 개선 |
| HaluEval-QA | 0.594 | 0.540 | **+0.054** | ✅ 개선 |
| TriviaQA | 0.660 | 0.676 | -0.016 | ≈ 동등 |
| NaturalQuestions | 0.604 | 0.615 | -0.011 | ≈ 동등 |
| HaluEval-dialogue | 0.596 | 0.599 | -0.002 | ≈ 동등 |

#### 3.3.3 통계 검정 (exp08 보강: Paired Bootstrap 5000회)

| 데이터셋 | Δ AUROC (τ=0.526) | 95% CI | p-value |
|----------|-------------------|--------|---------|
| TruthfulQA | +0.030 | [-0.016, +0.074] | 0.095 |
| HaluEval-QA | +0.054 | [-0.090, +0.204] | 0.234 |
| TriviaQA | -0.016 | [-0.066, +0.031] | 0.252 |
| NaturalQuestions | -0.011 | [-0.046, +0.021] | 0.250 |
| HaluEval-dialogue | -0.002 | [-0.130, +0.100] | 0.506 |

![Figure 8. Bootstrap 95% CI](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig8_bootstrap_ci.png)

> **어떤 데이터셋도 p<0.05에서 유의하지 않음** — n=200에서 Cascade는 SE-only와 **통계적으로 동등**.
> 모든 음의 delta가 ≤1.6% 수준이며, TruthfulQA는 p=0.095로 개선 경향.

---

### 3.4 상보성 분석

**핵심 질문**: "SE와 Energy가 각각 잡는 환각 영역이 얼마나 다른가?"

![Figure 3. Complementarity: Who Catches What?](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig3_complementarity.png)

| 데이터셋 | SE만 탐지 | Energy만 탐지 | 둘 다 | 못 잡음 | Oracle 탐지율 |
|----------|----------|-------------|-------|---------|-------------|
| TruthfulQA | 9.8% | **17.7%** | 62.2% | 10.4% | **89.6%** |
| HaluEval-QA | 0.0% | **23.8%** | 52.4% | 23.8% | 76.2% |
| TriviaQA | 9.3% | **16.7%** | 63.0% | 11.1% | **88.9%** |
| NaturalQuestions | 7.5% | **13.4%** | 66.4% | 12.7% | **87.3%** |
| HaluEval-dialogue | 9.0% | **12.2%** | 67.6% | 11.2% | **88.8%** |

#### 3.4.1 Threshold Sensitivity (exp08 보강)

80th percentile 선택에 의존하지 않는지 확인 (60/70/80/90th에서 반복):

![Figure 7. Complementarity Threshold Sensitivity](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig7_complementarity_sensitivity.png)

| 데이터셋 | Energy-only 범위 (60~90th) |
|----------|--------------------------|
| TruthfulQA | 11.6% ~ 22.6% |
| HaluEval-QA | 19.0% ~ 33.3% |
| TriviaQA | 12.0% ~ 18.5% |
| NaturalQuestions | 7.5% ~ 15.7% |
| HaluEval-dialogue | 12.2% ~ 19.7% |

> **모든 threshold에서 Energy-only 비율이 7~33% 범위로 일관되게 존재** — threshold 선택에 과도하게 의존하지 않음

---

### 3.5 전체 비교

![Figure 5. Overall Comparison: SE vs Energy vs Cascade](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig5_overall_comparison.png)

---

## 4. 제안하는 논문 스토리

![Figure 6. SE-Gated Cascade: Regime-Aware Hallucination Detection](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig6_story_diagram.png)

### 4.1 One-liner

> **"SE는 혼란을 측정하고, Energy는 지어냄을 측정한다. SE-gated cascade가 두 유형 모두를 탐지한다."**

### 4.2 세 가지 주장

#### 주장 1 (5/5 데이터셋 근거)
> "SE와 Energy는 서로 다른 환각 패턴을 탐지한다. 80th percentile threshold 기준 **Energy만 탐지하는 환각이 12~24%** 이며, threshold를 60~90th로 변화시켜도 **7~33% 범위에서 일관되게 존재**한다."

#### 주장 2 (3/5 데이터셋, factoid QA 한정)
> "Factoid QA 데이터셋에서 **Zero-SE(단일 NLI 클러스터) 영역의 환각을 Energy가 AUROC 0.60~0.74로** 효과적으로 탐지한다."

#### 주장 3 (5/5 데이터셋)
> "SE-gated cascade (SE < τ → Energy)는 cross-dataset τ=0.526 적용 시, **SE-only 대비 최대 -1.6% 감소로 통계적으로 동등 수준**이며 (paired bootstrap 모두 p>0.05), Zero-SE 환각이 많은 데이터셋에서 개선 경향을 보인다 (TruthfulQA: +3.0%, p=0.095)."

---

## 5. 한계점 및 논의

### 5.1 Energy가 안 되는 2개 데이터셋

| 데이터셋 | Energy AUROC | 추정 원인 |
|----------|-------------|----------|
| NaturalQuestions | 0.457 | 답변이 길고 다양 → logit 평균의 정보량 적음 |
| HaluEval-dialogue | 0.486 | dialogue 맥락 → 단일 Energy 점수로 부족 |

**공통점**: 응답이 길거나 복잡한 형태 → token-level logit 평균이 semantic-level 정보를 반영 못함

### 5.2 통계적 한계

- 각 200개 샘플 — Zero-SE 영역은 34~107개로 더 적음
- **Paired bootstrap 검정에서 어떤 cascade delta도 p<0.05를 달성 못함** — 통계적 유의성 부족
- Bootstrap CI가 넓은 경우 존재 (특히 HaluEval-dialogue)
- HaluEval-QA의 base rate가 10.5%로 매우 낮아 AUROC 해석 주의 필요
- TruthfulQA cascade 개선이 **min-max 정규화에서만 관측되고 rank 정규화에서는 소멸** — 정규화 방법에 민감

### 5.3 실험 범위

- 단일 모델 (Qwen2.5-3B-Instruct) — 더 큰 모델에서도 패턴이 유지되는지 미확인
- temperature 고정 (0.7) — 다른 온도에서 Zero-SE 비율 변화 미탐구

---

## 6. 조교님께 여쭤보고 싶은 것

### Q1: 스토리 방향
현재 "Zero-SE → Energy fallback" 스토리로 충분한지, 아니면 더 강한 주장이 필요한지 궁금합니다.

### Q2: 한계 데이터셋 (NQ, dialogue)
- Energy가 안 되는 2개 데이터셋을 한계로 인정하고 넘어가려고 하는데 괜찮을지

### Q3: 통계적 보강
- AUPRC 추가 보고 필요할지
- CI가 넓은 결과에 대한 추가로 검증이 필요할지

### Q4: 모델 확장
- 더 큰 모델 (7B)에서 재현 실험 필요한지
- Zero-SE 비율이 모델 크기에 따라 어떻게 변하는지?

### Q5: τ 설정
- 현재 절대값 기반 τ (SE < 0.526)으로 했는데 데이터셋마다 다르게 나올 것 같아서 조금 더 엄밀한 정의가 필요하진 않을지?

---

## 7. 관련 문서 링크

| 문서 | 링크 |
|------|------|
| 1차 미팅 보고 (260115) | [260115_meeting.md](https://github.com/sharosoo/hallucination_research/blob/master/260115_meeting.md) |
| 2차 보고 — 가설 기각 (260121) | [260121_summary.md](https://github.com/sharosoo/hallucination_research/blob/master/260121_summary.md) |
| exp07 실험 계획 (260203) | [260203.md](https://github.com/sharosoo/hallucination_research/blob/master/260203.md) |
| exp07 상세 결과 | [RESULTS.md](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/RESULTS.md) |
| 전체 분석 JSON | [analysis_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/analysis_results.json) |
| SE 구현 코드 | [semantic_entropy.py](https://github.com/sharosoo/hallucination_research/blob/master/packages/hfe-core/src/hfe_core/semantic_entropy.py) |
| Energy 구현 코드 | [semantic_energy.py](https://github.com/sharosoo/hallucination_research/blob/master/packages/hfe-core/src/hfe_core/semantic_energy.py) |
| NLI 클러스터링 코드 | [nli_clusterer.py](https://github.com/sharosoo/hallucination_research/blob/master/packages/hfe-core/src/hfe_core/nli_clusterer.py) |
| exp08 보강 분석 코드 | [analyze_robustness.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/analyze_robustness.py) |
| exp08 보강 결과 JSON | [robustness_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/robustness_results.json) |

---

## 8. 예상 논문 작성 단계

- [ ] Introduction: "SE의 blind spot" 문제 제기
- [ ] Method: SE-gated cascade 정의
- [ ] Experiments: 5개 데이터셋 결과
- [ ] Discussion: 한계 (2/5 데이터셋)와 향후 연구

---

## 9. 파일 구조

```
hallucination_lfe/
├── 260115_meeting.md                            # 1차 미팅 (실험 환경 구축)
├── 260121_summary.md                            # 2차 보고 (가설 기각)
├── 260203.md                                    # 실험 계획
├── 260206_meeting.md                            # 이 문서
├── figures/
│   ├── generate_all.py                          # Figure 생성 스크립트
│   ├── fig1~fig6*.png                           # exp07 기본 figures
│   ├── fig7_complementarity_sensitivity.png     # exp08: threshold sensitivity
│   └── fig8_bootstrap_ci.png                    # exp08: bootstrap CI
└── experiment_notes/
    ├── exp07_zero_se_analysis/
    │   ├── analyze_existing.py                  # E1/E3/E4 분석 (GPU 불필요)
    │   ├── run_new_datasets.py                  # 새 데이터셋 실험 (GPU 필요)
    │   ├── analysis_results.json                # 전체 분석 결과
    │   ├── results_*.json                       # 각 데이터셋 원본 결과
    │   └── RESULTS.md                           # 상세 결과 문서
    └── exp08_robustness/
        ├── analyze_robustness.py                # 보강 분석 (GPU 불필요)
        └── robustness_results.json              # 보강 결과
```
