# SE-gated Cascade: LLM 환각 탐지

> **Semantic Entropy와 Semantic Energy의 상보성을 활용한 환각 탐지**

## 개요

LLM의 환각을 탐지하기 위한 두 가지 대표적 지표인 **Semantic Entropy (SE)**와 **Semantic Energy**는 서로 다른 유형의 환각에 효과적이다:

| 환각 유형 | 특성 | 효과적 지표 |
|----------|------|------------|
| **혼란 (Confusion)** | 모델이 알지만 헷갈림 → 다양한 오답 | Semantic Entropy |
| **지어냄 (Confabulation)** | 모델이 모르고 지어냄 → 일관된 오답 | Semantic Energy |

**핵심 발견**: SE=0인 영역(Zero-SE)에서는 SE가 판별 불가능하지만, Energy는 AUROC 0.736을 달성한다.

**제안 방법**: **SE-gated Cascade** - 클러스터 수 |C|=1이면 Energy, 그 외에는 SE 사용

## 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/hallucination_lfe.git
cd hallucination_lfe

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install torch transformers datasets scikit-learn tqdm

# 코어 라이브러리 설치
pip install -e packages/hfe-core
```

### 실험 재현

```bash
# 기본 실험 (200샘플, seed=42)
python run_experiment.py

# 샘플 수 지정
python run_experiment.py --samples 100

# 시드 변경
python run_experiment.py --seed 123

# 출력 디렉토리 지정
python run_experiment.py --output results/exp1
```

### 예상 결과 (200샘플 기준)

```
[데이터셋]
  샘플 수: 200
  환각: 164 (82.0%)
  정상: 36

[전체 성능 - AUROC]
  SE-only:     0.613
  Energy-only: 0.550
  Cascade:     0.642 (+0.029)

[Zero-SE 분석]
  비율: 38/200 (19.0%)
  환각률: 73.7%
  Energy AUROC: 0.736
```

## 프로젝트 구조

```
hallucination_lfe/
├── run_experiment.py          # 실험 재현 스크립트 ⭐
├── packages/
│   └── hfe-core/src/hfe_core/
│       ├── nli_clusterer.py       # NLI 기반 의미 클러스터링
│       ├── semantic_entropy.py    # SE 계산
│       └── semantic_energy.py     # Energy 계산
├── experiment_notes/          # 실험 기록 (개발용)
└── thesis/                    # 논문 LaTeX 소스
```

## 핵심 알고리즘

### SE-gated Cascade

```python
def cascade(se, energy, num_clusters):
    """
    |C| = 1 (Zero-SE) → Energy 사용
    |C| >= 2          → SE 사용
    """
    if num_clusters == 1:
        return energy, "energy"
    else:
        return se, "se"
```

### Semantic Entropy

```python
def semantic_entropy(clusters, k):
    """클러스터 분포의 Shannon Entropy"""
    probs = [len(c) / k for c in clusters]
    return -sum(p * log(p) for p in probs if p > 0)
```

### Semantic Energy

```python
def semantic_energy(responses):
    """응답들의 평균 negative logit"""
    all_logits = [logit for r in responses for logit in r.logits]
    return -mean(all_logits)
```

## 요구사항

- Python 3.10+
- PyTorch 2.0+
- GPU 권장 (NVIDIA RTX 3090 이상, VRAM 24GB+)
- transformers, datasets, scikit-learn

## 인용

```bibtex
@article{moon2026segated,
  title={Semantic Entropy와 Semantic Energy의 상보성을 활용한 LLM 환각 탐지},
  author={Moon, JeongHyeok},
  year={2026}
}
```

## 라이선스

MIT License
