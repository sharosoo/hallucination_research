# AHSFE: Adaptive Hybrid Semantic Free Energy

> **Corpus Statistics 기반 적응형 가중치를 사용한 LLM 환각 탐지**

## 프로젝트 개요

### 문제: 확신에 찬 오답 (Confident Hallucination)

LLM이 **높은 확신으로 틀린 답변**을 생성하는 현상. 기존 방법들의 한계:

| 방법 | 측정 대상 | 한계 |
|------|----------|------|
| Semantic Entropy (SE) | 응답 다양성 | 일관된 오답 탐지 실패 (SE=0) |
| Semantic Energy | 모델 확신도 (logit) | 다양성 정보 부족 |
| HSFE (고정 가중치) | 둘의 조합 | 데이터셋마다 최적 가중치 다름 |

**공통 문제**: 모두 **모델 내부 신호**에 의존 → poorly calibrated LLM에서 실패

### 해결: AHSFE (Corpus-based Weights)

```
AHSFE = w(corpus) × Energy + (1 - w(corpus)) × SE

w(corpus): Pre-training corpus statistics 기반 가중치
```

**핵심 아이디어**: 
- Corpus에 없는 지식 → 모델이 지어낸 것 → Energy 가중치 ↑
- Corpus에 있는 지식 → 모델이 아는 것 → SE 가중치 유지

## 방법론

### 1. Semantic Entropy (SE)

응답들의 **의미적 다양성** 측정 (Farquhar et al., Nature 2024)

```
1. 같은 질문에 K번 응답 생성
2. NLI 클러스터링으로 의미 그룹화
3. SE = -Σ p(C_k) log p(C_k)
```

- SE 높음 → 응답 다양 → 불확실 → 환각 가능성
- SE 낮음 → 응답 일관 → **정답 또는 일관된 오답**

### 2. Semantic Energy

모델의 **내재적 확신도** 측정 (Ma et al., 2025)

```
Energy = (1/nT) ΣΣ -z_θ(x_t)

z_θ = raw logit (softmax 전)
```

- Energy 낮음 → logit 높음 → 확신 높음
- Energy 높음 → logit 낮음 → 확신 낮음 → 환각 가능성

### 3. Corpus-based Weight (QuCo-RAG 영감)

**외부 증거**로 가중치 결정 (Min et al., 2025 아이디어 활용)

```python
# Entity frequency
freq = corpus.count("Silas Hardy")  # 낮으면 long-tail

# Entity co-occurrence  
co_occur = corpus.count("Il Seduttore * Mario Camerini")  # 0이면 환각 위험
```

**가중치 규칙**:
```
IF entity_frequency 낮음 OR co-occurrence = 0:
    w ↑ (Energy 가중치 증가)
    → 모델이 모르는 지식 = Energy로 확신 체크
    
ELSE:
    w 유지
    → 모델이 아는 지식 = SE로 다양성 체크
```

## 실험 결과 (베이스라인)

### SE vs Energy AUROC

| 데이터셋 | SE | Energy | 최적 |
|----------|-----|--------|------|
| TruthfulQA | **0.619** | 0.538 | SE |
| HaluEval | 0.506 | **0.604** | Energy |

→ **데이터셋마다 최적 지표가 다름** = 고정 가중치 한계

### 제로-엔트로피 케이스

TruthfulQA에서 SE < 0.1인 샘플:
- 41개 중 31개(75.6%)가 환각
- SE로는 구분 불가 (모두 ~0)
- **Energy AUROC: 0.768** → Energy가 구분

## 프로젝트 구조

```
hallucination_lfe/
├── README.md
├── plan.md                    # 연구 계획
├── 260115_meeting.md          # 미팅 논의
├── study/
│   └── basic_concepts.md      # 기본 개념 설명
├── references/                # 참고 논문
├── packages/
│   ├── hfe-core/              # 핵심 알고리즘
│   │   └── src/hfe_core/
│   │       ├── nli_clusterer.py
│   │       ├── semantic_entropy.py
│   │       ├── semantic_energy.py
│   │       └── ahsfe.py
│   ├── hfe-datasets/          # 데이터셋 로더
│   ├── hfe-models/            # LLM 샘플러
│   └── hfe-eval/              # 평가 도구
└── experiment_notes/          # 실험 결과
    ├── exp01_truthfulqa/
    └── exp02_halueval/
```

## 설치

```bash
uv sync
```

## 참고문헌

1. **Semantic Entropy**: Farquhar et al., Nature 2024
2. **Semantic Energy**: Ma et al., arXiv:2412.07965, 2025
3. **QuCo-RAG**: Min et al., arXiv:2512.19134, 2025
