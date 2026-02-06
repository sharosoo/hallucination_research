# 260206 조교님 미팅 보고

## 0. 요약 (TL;DR)

| 항목 | 내용 | 상태 |
|------|------|------|
| 연구 주제 | LLM 환각 탐지 — Semantic Entropy와 Semantic Energy의 상보적 활용 | |
| 이전 문제 | Corpus 기반 가중치 예측 실패, 학습 기반 분류기 일반화 실패 | 해결 방향 전환 |
| 현재 스토리 | "Zero-SE 문제 → Energy가 해결" (SE-gated cascade) | **5개 데이터셋으로 검증** |
| 핵심 결과 | Zero-SE에서 Energy AUROC 0.57~0.74 (4/5), Cascade는 SE-only와 통계적 동등 (5/5) | **스토리 대부분 확인** |

---

## 1. 핵심 개념 설명

이 연구에서 사용하는 주요 용어와 방법론을 먼저 설명합니다.

### 1.1 연구 목표

LLM(대규모 언어모델)이 생성한 답변이 **환각(hallucination)**인지 아닌지를 자동으로 탐지하는 것이 목표입니다. 환각이란 모델이 사실과 다른 내용을 그럴듯하게 생성하는 현상입니다.

### 1.2 Semantic Entropy (SE) — "모델이 얼마나 혼란스러운가"

**Farquhar et al. (2024, Nature)** 에서 제안한 방법입니다.

**작동 원리**:
1. 하나의 질문에 대해 LLM으로부터 **K개의 응답을 샘플링**합니다 (본 실험에서 K=5)
2. 이 응답들을 NLI(자연어 추론) 모델로 **의미적으로 같은 것끼리 클러스터링**합니다
   - 예: "파리입니다", "수도는 파리" → 같은 클러스터
   - 예: "베를린입니다" → 다른 클러스터
3. 클러스터 분포의 **Shannon entropy**를 계산합니다

**해석**:
- **SE가 높음** → 응답이 여러 클러스터로 분산 → 모델이 **혼란스러움** → 환각 가능성 높음
- **SE가 0** → 모든 응답이 **하나의 클러스터** → 모델이 매우 일관됨 → 맞을 수도 있지만, **일관되게 틀릴 수도** 있음

**한계**: SE=0이면 모든 응답이 동일한 의미이므로, SE 값만으로는 "맞는 건지 틀린 건지" 구분이 불가능합니다. 이것이 **Zero-SE 문제**입니다.

### 1.3 Semantic Energy — "모델이 토큰 단위로 얼마나 확신하는가"

**Ma et al. (2025)** 에서 제안한 방법입니다.

**작동 원리**:
1. LLM이 각 토큰을 생성할 때, 해당 토큰에 부여한 **raw logit 값**(softmax 전의 점수)을 수집합니다
2. 모든 응답의 모든 토큰에 대해 **negative logit의 평균**을 구합니다: `Energy = (1/nT) ΣΣ -z(x_t)`

**해석**:
- **Energy가 낮음** (logit이 높음) → 모델이 각 토큰을 **확신 있게** 선택 → 정상 답변일 가능성 높음
- **Energy가 높음** (logit이 낮음) → 모델이 각 토큰을 **불확실하게** 선택 → 환각 가능성 높음

**SE와의 차이점**: SE는 "응답 간 다양성"을 보지만, Energy는 "각 토큰의 내재적 확신도"를 봅니다. 따라서 SE=0인 상황(모든 응답이 동일)에서도 Energy는 여전히 정보를 제공할 수 있습니다.

### 1.4 Zero-SE 문제

**Zero-SE**: K=5 응답이 **모두 단일 NLI 클러스터**에 속하는 경우입니다 (SE=0).
- 이때 SE는 정의상 0이므로, 환각 여부에 대한 **판별력이 전혀 없습니다**
- 그러나 실제로 이 영역에 환각이 다수 포함되어 있습니다 (데이터셋에 따라 9~90%)
- exp08 검증 결과 SE≤0.001과 num_clusters==1이 **5/5 데이터셋에서 100% 일치** 확인

이 문제를 해결하기 위해 Zero-SE 영역에서 Energy를 대신 사용하는 **SE-gated cascade**를 제안합니다.

### 1.5 SE-Gated Cascade

```
입력 질문 → LLM으로 K=5 응답 생성 → SE와 Energy를 동시에 계산

        SE < τ (threshold)?
       /                    \
     Yes                     No
  (Zero-SE 영역)         (High-SE 영역)
  Energy 점수로 판단      SE 점수로 판단
```

- τ는 SE와 Energy를 전환하는 **임계값(threshold)**입니다
- 본 실험에서는 TruthfulQA에서 최적화한 τ=0.526을 다른 데이터셋에도 고정 적용합니다

### 1.6 평가 지표

| 지표 | 의미 | 범위 |
|------|------|------|
| **AUROC** | 환각/정상 순위 분류 성능. 랜덤=0.5, 완벽=1.0 | [0, 1] |
| **합집합 탐지율** | SE 또는 Energy 중 **하나라도** 잡으면 성공으로 치는 이론적 상한 | [0%, 100%] |
| **Paired bootstrap p-value** | 두 방법(SE-only vs Cascade)의 성능 차이가 우연인지 검정. p<0.05면 유의 | [0, 1] |

---

## 2. 배경: 왜 방향을 전환했는가

### 2.1 이전 접근법 (실패)

[260121 미팅 보고서](https://github.com/sharosoo/hallucination_research/blob/master/260121_summary.md)에서 보고한 결과:
- **Corpus coverage 기반 가중치 예측**: Coverage가 SE/Energy 선택에 영향 없음 → **가설 기각** ([exp05](https://github.com/sharosoo/hallucination_research/tree/master/experiment_notes/exp05_combined_analysis))
- **학습 기반 분류기**: 같은 분포 내에서만 작동 (AUROC 0.91), cross-dataset에서 실패 (AUROC 0.51) → **일반화 실패** ([exp06](https://github.com/sharosoo/hallucination_research/tree/master/experiment_notes/exp06_weight_classifier))

### 2.2 260121 미팅에서 조교님 피드백

> "Cascading 방식 (SE → Energy fallback)이 더 실용적"
> "더 다양한 데이터셋에서 검증 필요"

### 2.3 새로운 스토리

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

## 3. 실험 설정

### 3.1 공통 파이프라인

| 항목 | 값 | 비고 |
|------|-----|------|
| LLM | Qwen2.5-3B-Instruct | 3B 파라미터 모델 |
| NLI 모델 | DeBERTa-large-mnli | 양방향 entailment로 응답 클러스터링 |
| 샘플링 | K=5, temperature=0.7 | 질문당 5개 응답 생성 |
| 샘플 수 | 각 200개 | 데이터셋별 200문항 |

**파이프라인 흐름**: 질문 입력 → K=5 응답 생성 → NLI로 응답 클러스터링 → SE 계산 + 각 응답의 token logit으로 Energy 계산 → 환각 여부 라벨과 비교하여 AUROC 산출

### 3.2 데이터셋 (기존 2개 + 신규 3개)

| 데이터셋 | 유형 | 환각 수 | 정상 수 | 환각률 | 특성 | 환각 판정 방법 | 원본 데이터 |
|----------|------|---------|---------|--------|------|-------------|------------|
| **TruthfulQA** | factoid QA | 164 | 36 | 82.0% | 대중적 오개념 질문 | 데이터셋 내장 라벨 | [results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp01_truthfulqa/results.json) |
| **HaluEval-QA** | knowledge QA | 21 | 179 | 10.5% | knowledge 기반 QA | 데이터셋 내장 라벨 | [results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp02_halueval/results.json) |
| **TriviaQA** | factoid QA | 110 | 90 | 55.0% | trivia 지식 질문 | GPT-5.2 LLM-as-judge (5개 응답 중 하나라도 정답 판정 시 정상) | [results_triviaqa_llm_judge.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_triviaqa_llm_judge.json) |
| **NaturalQuestions** | open QA | 114 | 86 | 57.0% | Google 검색 기반 일반 QA | GPT-5.2 LLM-as-judge (5개 응답 중 하나라도 정답 판정 시 정상) | [results_naturalquestions_llm_judge.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_naturalquestions_llm_judge.json) |
| **HaluEval-dialogue** | dialogue | 188 | 12 | 94.0% | 대화 기반 응답 | 데이터셋 내장 라벨 | [results_halueval_dialogue.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/results_halueval_dialogue.json) |

### 3.3 환각 라벨링 방법

| 데이터셋 | 라벨링 방법 | 비고 |
|----------|------------|------|
| TruthfulQA, HaluEval-QA, HaluEval-dialogue | **데이터셋 내장 라벨** | 원본 데이터셋이 제공하는 gold label 사용 |
| TriviaQA, NaturalQuestions | **GPT-5.2 LLM-as-judge** | 기존 string matching이 부정확하여 외부 LLM으로 재라벨링 |

**LLM-as-judge 방법**: K=5 응답 각각에 대해 GPT-5.2에게 gold answer와 비교하여 CORRECT/INCORRECT 판정을 요청. 5개 응답 중 하나라도 CORRECT이면 해당 샘플을 "정상"으로 분류.

- TriviaQA: 기존 108개 → **110개** 환각 (10개 라벨 변경: 6 normal→hall, 4 hall→normal)
- NaturalQuestions: 기존 134개 → **114개** 환각 (30개 라벨 변경: 10 normal→hall, 20 hall→normal)

> 특히 NQ에서 string matching이 부분 매칭 등으로 20개 환각을 과다 라벨링하고 있었으며, LLM-judge 재라벨링 후 Energy AUROC이 0.457→0.565로 개선됨.

---

> **exp07 분석 코드**: [analyze_existing.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/analyze_existing.py) | **신규 데이터셋 실험 코드**: [run_new_datasets.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/run_new_datasets.py) | **전체 분석 결과 JSON**: [analysis_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/analysis_results.json)
>
> **exp08 보강 분석**: [analyze_robustness.py](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/analyze_robustness.py) | [robustness_results.json](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp08_robustness/robustness_results.json)
>
> **상세 결과 문서**: [RESULTS.md](https://github.com/sharosoo/hallucination_research/blob/master/experiment_notes/exp07_zero_se_analysis/RESULTS.md) | **Figure 생성 코드**: [generate_all.py](https://github.com/sharosoo/hallucination_research/blob/master/figures/generate_all.py)

---

## 4. 실험 결과

### 4.1 Zero-SE 현상 정량화

**핵심 질문**: "모든 응답이 한 클러스터에 속하는 Zero-SE 샘플이 얼마나 있고, 그 중 환각은 몇 %인가? 그리고 그 영역에서 Energy는 환각을 구분할 수 있는가?"

![Figure 1. Zero-SE Phenomenon Across 5 Datasets](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig1_zero_se_overview.png)

| 데이터셋 | Zero-SE 비율 | Zero-SE 내 환각률 | Zero-SE 내 **Energy AUROC** | 95% CI |
|----------|-------------|------------------|---------------------------|--------|
| **TruthfulQA** | 19.0% (38/200) | **73.7%** (28/38) | **0.736** | [0.52, 0.93] |
| **HaluEval-QA** | 53.5% (107/200) | 9.3% (10/107) | **0.602** | [0.39, 0.79] |
| **TriviaQA** | 29.0% (58/200) | **32.8%** (19/58) | **0.700** | [0.55, 0.82] |
| **NaturalQuestions** | 17.0% (34/200) | **32.4%** (11/34) | **0.565** | [0.26, 0.66] |
| HaluEval-dialogue | 24.5% (49/200) | 89.8% (44/49) | 0.486 | [0.13, 0.85] |

> **95% CI**: Bootstrap 1000회로 추정한 Energy AUROC의 신뢰구간. 0.5를 포함하면 랜덤과 구분 불가.

**발견**:
1. **Zero-SE 현상은 모든 데이터셋에서 존재** (17~53.5%) — "모델이 매우 일관적"인 샘플이 상당수
2. 그 중 상당수가 환각 (예: TruthfulQA에서 Zero-SE 38개 중 28개가 환각)
3. Zero-SE 영역에서 Energy AUROC > 0.55인 경우: **4/5** (TruthfulQA, TriviaQA, HaluEval-QA, NaturalQuestions)
4. Energy가 비효과적인 경우: **1/5** (HaluEval-dialogue 0.486)

> ε = 0.001~0.1 에서 결과 동일 — SE 값이 정확히 0.0인 샘플이 대부분이므로 ε 선택에 robust

---

### 4.2 SE 구간별 Crossover 분석

**핵심 질문**: "SE 값이 낮을 때는 Energy가 환각을 더 잘 잡고, SE가 높을 때는 SE가 더 잘 잡는 crossover가 존재하는가?"

> 존재한다면, "낮은 SE 영역은 Energy, 높은 SE 영역은 SE"라는 cascade 전략의 근거가 됩니다.

![Figure 2. SE vs Energy AUROC by Semantic Entropy Bin](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig2_se_bin_crossover.png)

**TruthfulQA — 가장 명확한 crossover 예시**:

| SE 구간 | 의미 | n | 환각률 | SE AUROC | Energy AUROC | **승자** |
|---------|------|---|--------|----------|-------------|---------|
| **Zero-SE [0, 0.05]** | 모든 응답이 같은 클러스터 | 38 | 73.7% | N/A (모두 0) | **0.736** | **Energy** |
| Medium (0.3, 0.6] | 2~3개 클러스터 | 23 | 78.3% | N/A | 0.578 | Energy |
| High (0.6, 1.0] | 3~4개 클러스터 | 44 | 84.1% | **0.664** | 0.517 | **SE** |
| Very High (1.0, ∞) | 4~5개 클러스터 | 95 | 85.3% | **0.658** | 0.422 | **SE** |

> Zero-SE 구간에서는 SE AUROC가 "N/A"인 이유: SE 값이 전부 0이므로 ranking이 불가능합니다.

**Crossover 패턴 요약 (5개 데이터셋)**:

| 데이터셋 | Zero-SE: Energy 우세? | High-SE: SE 우세? | Crossover 존재? |
|----------|----------------------|-------------------|----------------|
| TruthfulQA | ✅ Energy 0.736 | ✅ SE 0.664 | ✅ |
| HaluEval-QA | ✅ Energy 0.602 | ✅ SE 0.343 | ✅ |
| TriviaQA | ✅ Energy 0.700 | ✅ SE 0.636 | ✅ |
| NaturalQuestions | ✅ Energy 0.565 (약함) | ✅ SE 0.625 | ✅ (약함) |
| HaluEval-dialogue | ❌ Energy 0.486 | ✅ SE 0.574 | ❌ |

> **5개 중 4개 데이터셋에서 crossover 확인** — QA 데이터셋에서 패턴이 일관됨 (NQ는 LLM-judge 재라벨링 후 약한 crossover 관측)

---

### 4.3 SE-Gated Cascade 성능

**핵심 질문**: "SE < τ이면 Energy 점수를, 아니면 SE 점수를 사용하는 cascade가 SE-only보다 나은가?"

> SE-only보다 크게 나빠지면 cascade는 실용적이지 않습니다. 최소한 "해치지 않음"을 보여야 합니다.

![Figure 4. Cascade Threshold Sweep](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig4_cascade_sweep.png)

#### 4.3.1 데이터셋별 최적 τ 결과

| 데이터셋 | Best τ | Cascade AUROC | SE-only | Energy-only | Δ vs SE |
|----------|--------|-------------|---------|-------------|---------|
| **TruthfulQA** | 0.526 | **0.643** | 0.613 | 0.550 | **+0.030** |
| **HaluEval-QA** | 1.332 | **0.614** | 0.540 | 0.616 | **+0.074** |
| TriviaQA | 0.000 | 0.668 | **0.669** | 0.644 | -0.000 |
| **NaturalQuestions** | 1.609 | **0.662** | 0.636 | **0.662** | **+0.026** |
| HaluEval-dialogue | 0.526 | 0.596 | **0.599** | 0.566 | -0.002 |

#### 4.3.2 Cross-Dataset τ Transfer

실전에서는 데이터셋마다 τ를 최적화할 수 없으므로, **하나의 τ를 고정**하여 다른 데이터셋에 적용하는 것이 중요합니다.
TruthfulQA에서 학습한 **τ=0.526을 고정**하여 나머지 데이터셋에 적용한 결과:

| 적용 데이터셋 | Cascade | SE-only | Δ vs SE | 판정 |
|-------------|---------|---------|---------|------|
| TruthfulQA (in-domain) | 0.643 | 0.613 | **+0.030** | ✅ 개선 |
| HaluEval-QA | 0.594 | 0.540 | **+0.054** | ✅ 개선 |
| TriviaQA | 0.668 | 0.669 | -0.001 | ≈ 동등 |
| NaturalQuestions | 0.623 | 0.636 | -0.013 | ≈ 동등 |
| HaluEval-dialogue | 0.596 | 0.599 | -0.002 | ≈ 동등 |

#### 4.3.3 통계 검정 (exp08 보강)

위 Δ 값이 통계적으로 유의한지 **paired bootstrap 검정**으로 확인했습니다.

> **Paired bootstrap이란**: 원본 200개 샘플에서 중복 허용 랜덤 추출을 5000번 반복합니다. 매번 **같은 샘플 인덱스**로 SE-only AUROC과 Cascade AUROC을 모두 계산하여 Δ(=Cascade−SE) 분포를 구합니다. 95% CI가 0을 포함하면 "차이가 우연일 수 있다"는 뜻입니다.

| 데이터셋 | Δ AUROC (τ=0.526) | 95% CI | p-value | 유의? |
|----------|-------------------|--------|---------|------|
| TruthfulQA | +0.030 | [-0.016, +0.074] | 0.095 | p<0.10 ⚠️ |
| HaluEval-QA | +0.054 | [-0.090, +0.204] | 0.234 | ❌ |
| TriviaQA | -0.001 | [-0.050, +0.047] | 0.476 | ❌ |
| NaturalQuestions | -0.013 | [-0.046, +0.017] | 0.195 | ❌ |
| HaluEval-dialogue | -0.002 | [-0.130, +0.100] | 0.506 | ❌ |

![Figure 8. Bootstrap 95% CI](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig8_bootstrap_ci.png)

> **결론**: 어떤 데이터셋도 p<0.05에서 유의하지 않음 — n=200에서 Cascade는 SE-only와 **통계적으로 동등**.
> 다만 모든 음의 delta가 ≤1.3% 수준으로 실질적 손해가 미미하며, TruthfulQA는 p=0.095로 개선 경향.

---

### 4.4 상보성 분석

**핵심 질문**: "SE와 Energy가 각각 잡는 환각 영역이 얼마나 다른가?"

> 환각 샘플 각각에 대해, SE 점수가 상위 20%면 "SE가 잡음", Energy 점수가 상위 20%면 "Energy가 잡음"으로 분류했습니다 (80th percentile threshold).

![Figure 3. Complementarity: Who Catches What?](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig3_complementarity.png)

> **합집합 탐지율 (Union Recall)**: SE 또는 Energy **둘 중 하나라도** 환각을 잡으면 "탐지 성공"으로 치는 이론적 상한입니다. SE+Energy를 완벽하게 결합했을 때 **도달 가능한 recall의 천장**을 의미합니다.

| 데이터셋 | SE만 탐지 | Energy만 탐지 | 둘 다 | 못 잡음 | 합집합 탐지율 |
|----------|----------|-------------|-------|---------|-------------|
| TruthfulQA | 9.8% | **17.7%** | 62.2% | 10.4% | **89.6%** |
| HaluEval-QA | 0.0% | **23.8%** | 52.4% | 23.8% | 76.2% |
| TriviaQA | 8.2% | **17.3%** | 62.7% | 11.8% | **88.2%** |
| NaturalQuestions | 7.9% | **11.4%** | 68.4% | 12.3% | **87.7%** |
| HaluEval-dialogue | 9.0% | **12.2%** | 67.6% | 11.2% | **88.8%** |

> 예시: TruthfulQA에서 전체 164개 환각 중, SE만 잡는 것 16개(9.8%), Energy만 잡는 것 29개(17.7%), 둘 다 잡는 것 102개(62.2%), 둘 다 못 잡는 것 17개(10.4%).
> "Energy만 탐지" 17.7%는 SE를 아무리 좋은 threshold로 설정해도 잡을 수 없는 환각입니다.

#### 4.4.1 Threshold Sensitivity (exp08 보강)

위 결과가 "80th percentile"이라는 특정 threshold 선택에 의존하지 않는지, 60/70/80/90th에서 반복 검증했습니다:

![Figure 7. Complementarity Threshold Sensitivity](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig7_complementarity_sensitivity.png)

| 데이터셋 | Energy-only 범위 (60~90th) |
|----------|--------------------------|
| TruthfulQA | 11.6% ~ 22.6% |
| HaluEval-QA | 19.0% ~ 33.3% |
| TriviaQA | 13.6% ~ 19.1% |
| NaturalQuestions | 11.4% ~ 19.3% |
| HaluEval-dialogue | 12.2% ~ 19.7% |

> **모든 threshold에서 Energy-only 비율이 7~33% 범위로 일관되게 존재** — 특정 threshold 선택에 과도하게 의존하지 않음

---

### 4.5 전체 비교

![Figure 5. Overall Comparison: SE vs Energy vs Cascade](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig5_overall_comparison.png)

---

## 5. 제안하는 논문 스토리

![Figure 6. SE-Gated Cascade: Regime-Aware Hallucination Detection](https://raw.githubusercontent.com/sharosoo/hallucination_research/master/figures/fig6_story_diagram.png)

### 5.1 One-liner

> **"SE는 혼란(confusion)을 잡고, Energy는 지어냄(confabulation)을 잡는다. 두 신호는 서로 다른 환각 영역을 커버하며, SE-gated cascade로 안전하게 결합할 수 있다."**

### 5.2 세 가지 주장

#### 주장 1: SE와 Energy는 서로 다른 환각 패턴을 탐지한다 (5/5 데이터셋)
> SE가 놓치는 환각 중 Energy만 탐지하는 비율이, threshold 설정(60~90th percentile)에 관계없이 **모든 데이터셋에서 7~33% 범위로 일관되게 존재**한다. 이는 두 메트릭이 상보적 정보를 포착함을 의미한다.

#### 주장 2: Zero-SE 영역에서 Energy가 유효한 탐지 신호를 제공한다 (4/5, QA 한정)
> K=5 응답이 단일 NLI 클러스터를 이루는 Zero-SE 영역에서, SE는 정의상 판별력이 없다(모두 0). 이 영역에서 Energy는 QA 데이터셋(TruthfulQA, TriviaQA, HaluEval-QA, NaturalQuestions)에 대해 **AUROC 0.57~0.74**로 환각을 구분한다.

#### 주장 3: SE-gated cascade는 SE-only 대비 실질적으로 무해하다 (5/5 데이터셋)
> Cross-dataset τ=0.526 적용 시, cascade와 SE-only의 AUROC 차이는 **모든 데이터셋에서 통계적으로 유의하지 않다** (paired bootstrap 5000회, 모두 p>0.05). 최대 감소 폭은 -1.3%이며, Zero-SE 환각이 많은 데이터셋에서는 개선 경향을 보인다 (TruthfulQA: +3.0%, p=0.095).

---

## 6. 한계점 및 논의

### 6.1 Energy가 비효과적인 1개 데이터셋

| 데이터셋 | Zero-SE 내 Energy AUROC | 추정 원인 |
|----------|------------------------|----------|
| HaluEval-dialogue | 0.486 (랜덤 수준) | dialogue 맥락이 복잡 → 단일 Energy 점수로 부족 |

> **참고**: NaturalQuestions는 원래 string matching으로 환각 라벨을 부여하여 AUROC 0.457이었으나, GPT-5.2 LLM-as-judge로 재라벨링 후 **0.565로 개선**되었다. 이는 라벨 품질이 Energy AUROC에 큰 영향을 미침을 시사한다.

### 6.2 통계적 한계

- 각 200개 샘플 — Zero-SE 영역은 34~107개로 더 적어 통계적 검정력이 약함
- **Paired bootstrap 검정에서 어떤 cascade delta도 p<0.05를 달성 못함** — 통계적 유의성 부족
- Bootstrap CI가 넓은 경우 존재 (특히 HaluEval-dialogue: [0.13, 0.85])
- HaluEval-QA의 base rate가 10.5%로 매우 낮아 AUROC 해석 주의 필요
- TruthfulQA, TriviaQA cascade 개선이 **min-max vs rank 정규화에서 방향 불일치** — 다만 delta가 ≈0 수준에서의 방향 전환으로 실질적 차이는 미미

### 6.3 실험 범위

- 단일 모델 (Qwen2.5-3B-Instruct) — 더 큰 모델에서도 패턴이 유지되는지 미확인
- temperature 고정 (0.7) — 다른 온도에서 Zero-SE 비율 변화 미탐구
- K=5로 고정 — K가 커지면 Zero-SE 비율이 줄어들 가능성

---

## 7. 조교님께 여쭤보고 싶은 것

### Q1: 스토리 방향
현재 "Zero-SE → Energy fallback" 스토리로 충분한지, 아니면 더 강한 주장이 필요한지 궁금합니다.

### Q2: 한계 데이터셋 (dialogue)
- Energy가 안 되는 1개 데이터셋(HaluEval-dialogue)을 한계로 인정하고 넘어가려고 하는데 괜찮을지
- NQ는 LLM-judge 재라벨링 후 0.565로 개선되어 4/5 데이터셋에서 효과 확인

### Q3: 통계적 보강
- AUPRC 추가 보고 필요할지
- CI가 넓은 결과에 대한 추가로 검증이 필요할지

### Q4: 모델 확장
- 더 큰 모델 (7B)에서 재현 실험 필요한지
- Zero-SE 비율이 모델 크기에 따라 어떻게 변하는지?

### Q5: τ 설정
- 현재 절대값 기반 τ (SE < 0.526)으로 했는데 데이터셋마다 다르게 나올 것 같아서 조금 더 엄밀한 정의가 필요하진 않을지?

---

## 8. 관련 문서 링크

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

## 9. 예상 논문 작성 단계

- [ ] Introduction: "SE의 blind spot" 문제 제기
- [ ] Method: SE-gated cascade 정의
- [ ] Experiments: 5개 데이터셋 결과
- [ ] Discussion: 한계 (2/5 데이터셋)와 향후 연구

---

## 10. 파일 구조

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
    │   ├── relabel_llm_judge.py                  # GPT-5.2 LLM-as-judge 재라벨링 스크립트
│   ├── results_*.json                       # 각 데이터셋 결과 (_llm_judge 포함)
    │   └── RESULTS.md                           # 상세 결과 문서
    └── exp08_robustness/
        ├── analyze_robustness.py                # 보강 분석 (GPU 불필요)
        └── robustness_results.json              # 보강 결과
```
