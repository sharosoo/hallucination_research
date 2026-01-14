# AHSFE: Adaptive Hybrid Semantic Free Energy

> Corpus Statistics 기반 적응형 가중치를 사용한 LLM 환각 탐지

## 1. 문제 정의

### 1.1 확신에 찬 오답 (Confident Hallucination)

LLM이 **높은 확신으로 틀린 답변**을 일관되게 생성하는 현상

```
질문: "손가락 관절을 꺾으면 어떻게 되나요?"
응답: ["관절염", "관절염", "관절염", "관절염", "관절염"]

- 일관된 응답 → SE = 0 → "신뢰 가능"으로 오판
- 하지만 "관절염"은 오답! (실제로는 아무 해 없음)
```

### 1.2 기존 방법들의 한계

| 방법 | 측정 대상 | 한계 |
|------|----------|------|
| Semantic Entropy | 응답 다양성 | 제로-엔트로피 문제 (일관된 오답 탐지 실패) |
| Semantic Energy | logit 기반 확신도 | 다양성 정보 부족 |
| HSFE (고정 가중치) | 둘의 조합 | 데이터셋마다 최적 가중치가 다름 |

**공통 한계**: 모두 **모델 내부 신호**에 의존 → LLM이 poorly calibrated면 실패

---

## 2. 기존 연구

### 2.1 Semantic Entropy (Farquhar et al., Nature 2024)

```
SE = -Σ p(C_k) log p(C_k)
```

- NLI 클러스터링 + Shannon Entropy
- 장점: 의미적 불확실성 측정
- 한계: 제로-엔트로피 문제

### 2.2 Semantic Energy (Ma et al., 2025)

```
U = (1/nT) ΣΣ -z_θ(x_t)
```

- Raw logit 기반 에너지
- 장점: 제로-엔트로피 문제 일부 해결
- 한계: 다양성 정보 없음

### 2.3 QuCo-RAG (Min et al., 2025)

**핵심 관찰**: 모델 내부 신호는 신뢰할 수 없음

**해결책**: Pre-training corpus statistics 사용
- Entity frequency: 낮으면 long-tail knowledge
- Entity co-occurrence: 0이면 환각 위험

**용도**: Retrieval 트리거 결정

---

## 3. 제안 방법: AHSFE

### 3.1 핵심 아이디어

```
AHSFE = w(corpus) × Energy + (1 - w(corpus)) × SE

w(corpus) = corpus statistics 기반 가중치
```

**QuCo-RAG와의 차이**:
- QuCo-RAG: corpus stats → retrieval 결정
- AHSFE: corpus stats → **SE/Energy 가중치 결정**

### 3.2 가중치 규칙

```python
def compute_weight(entity_freq, cooccurrence):
    """Corpus statistics 기반 Energy 가중치 계산"""
    
    if cooccurrence == 0:
        # 학습된 적 없는 관계 → Energy 위주
        return 0.9
    
    if entity_freq < THRESHOLD_LOW:
        # Long-tail knowledge → Energy 위주
        return 0.8
    
    if entity_freq > THRESHOLD_HIGH:
        # 잘 학습된 지식 → SE 위주
        return 0.3
    
    # 기본
    return 0.5
```

### 3.3 직관

| Corpus 상태 | 의미 | 가중치 |
|-------------|------|--------|
| co-occurrence = 0 | 모델이 학습 안 함 | w ↑ (Energy 위주) |
| frequency 낮음 | long-tail 지식 | w ↑ (Energy 위주) |
| frequency 높음 | 잘 아는 지식 | w ↓ (SE 위주) |

---

## 4. 실험 설계

### 4.1 연구 질문

| RQ | 질문 |
|----|------|
| RQ1 | Corpus 기반 가중치가 고정 가중치보다 AUROC가 높은가? |
| RQ2 | 제로-엔트로피 케이스에서 성능이 개선되는가? |
| RQ3 | QuCo-RAG와 결합 시 추가 성능 향상이 있는가? |

### 4.2 베이스라인

| 방법 | 설명 |
|------|------|
| SE | Semantic Entropy only |
| Energy | Semantic Energy only |
| HSFE-0.5 | 고정 가중치 (0.5, 0.5) |
| HSFE-best | 그리드 서치로 찾은 최적 고정 가중치 |
| **AHSFE** | Corpus 기반 적응형 가중치 |

### 4.3 데이터셋

| Dataset | 특징 |
|---------|------|
| TruthfulQA | 대중적 오개념, SE 우세 |
| HaluEval | Knowledge 기반, Energy 우세 |
| 2WikiMQA | Multi-hop QA |

### 4.4 평가 지표

- AUROC: 환각 탐지 성능
- AUPRC: 불균형 데이터 대응
- 제로-SE AUROC: 제로-엔트로피 케이스 성능

---

## 5. 구현 계획

### 5.1 Phase 1: 베이스라인 확인 (완료)

- [x] SE/Energy 구현
- [x] TruthfulQA, HaluEval 실험
- [x] 제로-엔트로피 문제 확인

### 5.2 Phase 2: Corpus 연동

- [ ] Infini-gram API 연동 (또는 대안)
- [ ] Entity 추출 구현
- [ ] Frequency, co-occurrence 쿼리

### 5.3 Phase 3: AHSFE 실험

- [ ] 가중치 규칙 설계
- [ ] AUROC 비교 실험
- [ ] 제로-엔트로피 케이스 분석

### 5.4 Phase 4: 분석

- [ ] QuCo-RAG 대비 차별점 검증
- [ ] Ablation study
- [ ] 실패 케이스 분석

---

## 6. 프로젝트 구조

```
hallucination_lfe/
├── README.md
├── plan.md                     # 이 문서
├── 260115_meeting.md           # 미팅 논의
├── study/
│   └── basic_concepts.md       # 기본 개념
├── references/                 # 참고 논문
├── packages/
│   ├── hfe-core/               # SE, Energy, AHSFE
│   ├── hfe-datasets/           # 데이터셋 로더
│   ├── hfe-models/             # LLM 샘플러
│   └── hfe-eval/               # 평가 도구
└── experiment_notes/
    ├── exp01_truthfulqa/       # 베이스라인 결과
    └── exp02_halueval/
```

---

## 7. 참고문헌

1. **Semantic Entropy**: Farquhar et al., Nature 2024
2. **Semantic Energy**: Ma et al., arXiv:2412.07965, 2025
3. **QuCo-RAG**: Min et al., arXiv:2512.19134, 2025
4. **KLE**: Nikitin et al., arXiv:2405.20003, 2024
5. **SNNE**: Nguyen et al., arXiv:2506.00245, 2025
