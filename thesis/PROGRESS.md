# 논문 작성 진행 상황

**최종 수정일**: 2026-02-11 11:32 KST

---

## 📊 실험 데이터 기준

| 항목 | 값 |
|------|-----|
| 데이터셋 | TruthfulQA (generation split) |
| 샘플 수 | **200개** (exp01 기준) |
| 환각 샘플 | 164 (82%) |
| 정상 샘플 | 36 (18%) |
| 결과 파일 | `experiment_notes/exp01_truthfulqa/results.json` |
| 분석 파일 | `experiment_notes/thesis_200_analysis.json` |

---

## 📈 핵심 수치 (200샘플 기준)

| 메트릭 | 값 |
|--------|-----|
| SE-only AUROC | 0.6132 |
| Energy-only AUROC | 0.5498 |
| **Cascade AUROC** | **0.6420 ~ 0.6427** |
| 개선 | +0.029 |

### Zero-SE 분석
| 항목 | 값 |
|------|-----|
| Zero-SE 샘플 수 | 38개 (19%) |
| Zero-SE 환각률 | 73.7% (28/38) |
| Zero-SE Energy AUROC | **0.7357** |

---

## ✅ 완료된 수정

- [x] GPU: RTX 4090 → **RTX 5090 (32GB)** (`sections/experiment_method.tex:168`)
- [x] **Cascade 수식**: τ 기반 → **클러스터 기반** (`|C|=1 → Energy`)
- [x] **알고리즘**: 임계값 τ 제거, 클러스터 수 조건으로 변경
- [x] **임계값 결정 섹션**: 수학적 정당화 추가 (SE=0 ⟺ |C|=1)
- [x] **Adaptive Cascade 섹션**: 전체 삭제 → "확장 가능성" 한 문단으로 축소
- [x] **결과 테이블**: `Cascade (τ=0.55)` → `Cascade (|C|=1 → Energy)`
- [x] **수치 업데이트**: AUROC 0.643 → 0.642, 개선 +0.030 → +0.029
- [x] **키워드**: "Adaptive Threshold" → "SE-gated Cascade"
- [x] **향후 연구**: Adaptive Threshold 항목 → 대규모 검증, 경계 영역 처리로 변경
- [x] **PDF 컴파일**: 32페이지, xelatex 사용

---

## 🔧 진행 중 / 예정된 수정

### 1. ~~Cascade 방식 단순화~~ ✅ 완료

**최종 방식**:
```
if |C| = 1:  use Energy  (Zero-SE)
else:        use SE
```

**결과**:
| 방법 | AUROC | 개선 |
|------|-------|------|
| SE-only | 0.613 | - |
| Cascade (\|C\|=1) | 0.642 | +0.029 |

### 2. ~~논문 수치 일관성~~ ✅ 확인됨
- 모든 수치 200샘플 (exp01) 기준
- Limitation에 817샘플 확장 가능성 언급됨

### 3. 그림 파일 확인 필요
- [ ] `fig4_cascade_sweep.png` 삭제됨 (더 이상 참조 안 함)
- [ ] 필요시 새 그림 생성

---

## 📁 주요 파일 위치

```
hallucination_lfe/
├── thesis/
│   ├── main.tex              # 메인 논문
│   ├── main.pdf              # 컴파일된 PDF
│   ├── sections/
│   │   └── experiment_method.tex  # 실험 방법 섹션
│   ├── images/               # 논문 그림
│   └── PROGRESS.md           # 이 파일
├── experiment_notes/
│   ├── exp01_truthfulqa/     # 200샘플 원본 실험
│   │   └── results.json
│   ├── exp10_thesis_complete/  # 817샘플 전체 실험
│   │   └── results.json
│   └── thesis_200_analysis.json  # 200샘플 분석 요약
```

---

## 💬 결정 사항 로그

### 2026-02-11 11:40
**main.tex 대폭 수정 완료**:
1. **GPU**: 5090 사용 중이므로 논문에서 4090→5090 수정
2. **Cascade 방식**: 클러스터 기반 (|C|=1 → Energy)으로 변경 - 수학적으로 엄밀
3. **Adaptive Cascade 삭제**: 엄밀하지 않은 Rule-based/Cluster-aware/Hybrid 섹션 제거
4. **데이터**: 200샘플(exp01) 기준, 결과 파일은 `experiment_notes/exp01_truthfulqa/results.json`
5. **PDF 컴파일 확인**: 32페이지, 에러 없음

---

## 📝 다음 단계

1. ~~main.tex에서 Adaptive Cascade 섹션 수정~~ ✅
2. ~~클러스터 기반 cascade 수식/설명으로 변경~~ ✅
3. ~~임계값 결정 섹션 수정~~ ✅
4. ~~PDF 재컴파일 후 확인~~ ✅
5. [ ] 그림 파일 정리 (불필요한 sweep 그림 등)
6. [ ] exp01 데이터로 새 그림 생성 (필요시)
7. [ ] 최종 검토 및 제출
