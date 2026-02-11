# Exp10: TruthfulQA 논문용 완전 실험 결과

**실험 일시**: 2026-02-11 03:38

## 1. 실험 설정

| 항목 | 설정 |
|------|------|
| LLM | Qwen/Qwen2.5-3B-Instruct |
| NLI 모델 | microsoft/deberta-large-mnli |
| 데이터셋 | TruthfulQA (validation) |
| 샘플링 수 (K) | 5 |
| Temperature | 0.7 |
| Seed | 42 |

## 2. 데이터셋 통계

| 항목 | 값 |
|------|------|
| 전체 샘플 수 | 817 |
| 환각 샘플 수 | 613 (75.0%) |
| 정상 샘플 수 | 204 (25.0%) |

## 3. 전체 성능

| 방법 | AUROC | AUPRC |
|------|-------|-------|
| Semantic Entropy (SE) | 0.5685 | 0.7848 |
| Semantic Energy | 0.5690 | 0.7927 |
| **SE-gated Cascade (τ=0.85)** | **0.5744** | - |

**Cascade 개선**: +0.0059 vs SE-only

## 4. Zero-SE 분석 (ε=0.05)

| 지표 | 값 |
|------|------|
| Zero-SE 비율 | 15.1% (123/817) |
| Zero-SE 내 환각률 | 65.9% |
| Zero-SE 내 Energy AUROC | 0.5711346266901822 |

## 5. SE 구간별 성능

| 구간 | 범위 | n | 환각률 | SE AUROC | Energy AUROC |
|------|------|---|--------|----------|--------------|
| Zero-SE | [0, 0.05) | 123 | 65.9% | N/A | 0.571 |
| Medium | (0.3, 0.6] | 84 | 71.4% | N/A | 0.515 |
| High | (0.6, 1.0] | 167 | 76.6% | 0.548 | 0.538 |
| Very High | (1.0, inf] | 443 | 77.7% | 0.553 | 0.561 |

## 6. 상보성 분석 (80th percentile)

| 카테고리 | 개수 | 비율 |
|----------|------|------|
| SE-only | 0 | 0.0% |
| Energy-only | 136 | 22.2% |
| Both | 0 | 0.0% |
| Neither | 477 | 77.8% |

**합집합 탐지율**: 22.2%

## 7. 생성된 Figure

1. `fig1_zero_se_overview.png` - Zero-SE 현상 개요
2. `fig2_se_bin_crossover.png` - SE 구간별 Crossover 패턴
3. `fig3_cascade_sweep.png` - Cascade threshold sweep
4. `fig4_complementarity.png` - 상보성 분석
5. `fig5_overall_comparison.png` - 전체 방법 비교

## 8. 논문 초록/결론용 핵심 수치

```
- 전체 샘플: 817개
- 환각률: 75.0%
- Zero-SE 비율: 15.1%
- Zero-SE 내 환각률: 65.9%
- Zero-SE 내 Energy AUROC: 0.5711346266901822
- SE AUROC: 0.568
- Energy AUROC: 0.569
- Cascade AUROC: 0.574 (τ=0.85)
- Cascade 개선: +0.006
- 합집합 탐지율: 22.2%
```

## 9. 재현 방법

```bash
cd hallucination_lfe
source .venv/bin/activate
python experiment_notes/exp10_thesis_complete/run_experiment.py
```
