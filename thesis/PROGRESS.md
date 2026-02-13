# 논문 작성 진행 상황

**최종 수정일**: 2026-02-11 15:10 KST

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
| SE-only AUROC | 0.613 |
| Energy-only AUROC | 0.550 |
| **Cascade AUROC** | **0.642** |
| 개선 | +0.029 |

### Zero-SE 분석
| 항목 | 값 |
|------|-----|
| Zero-SE 샘플 수 | 38개 (19%) |
| Zero-SE 환각률 | 73.7% (28/38) |
| Zero-SE Energy AUROC | **0.736** |

### 상보성 분석
| 탐지 영역 | 개수 | 비율 |
|----------|------|------|
| SE만 탐지 | 22 | 13.4% |
| Energy만 탐지 | 22 | 13.4% |
| 둘 다 탐지 | 12 | 7.3% |
| 미탐지 | 108 | 65.9% |
| **합집합** | **56** | **34.1%** |

---

## ✅ 완료된 수정

### 2026-02-11 11:40 - Cascade 방식 변경
- [x] GPU: RTX 4090 → **RTX 5090 (32GB)**
- [x] Cascade 수식: τ 기반 → **클러스터 기반** (`|C|=1 → Energy`)
- [x] 알고리즘: 임계값 τ 제거, 클러스터 수 조건으로 변경
- [x] 임계값 결정 섹션: 수학적 정당화 추가 (SE=0 ⟺ |C|=1)
- [x] Adaptive Cascade 섹션: 전체 삭제 → "확장 가능성" 한 문단으로 축소
- [x] 결과 테이블: `Cascade (τ=0.55)` → `Cascade (|C|=1 → Energy)`
- [x] 수치 업데이트: AUROC 0.643 → 0.642, 개선 +0.030 → +0.029
- [x] 키워드: "Adaptive Threshold" → "SE-gated Cascade"

### 2026-02-11 12:30 - TikZ 다이어그램 추가
- [x] 모든 그림 TikZ로 교체 (5개)
- [x] figures_tikz.tex 생성

### 2026-02-11 15:00 - TikZ 디자인 개선
- [x] 학술 논문 스타일 색상 팔레트 적용
- [x] Fig 5.2 (Zero-SE 요약): 도넛 차트 + 수동 막대
- [x] Fig 5.4 (상보성 분석): 수동 막대 그래프로 단순화
- [x] pgfplots 일부 제거, 순수 TikZ로 변경

### 2026-02-11 15:10 - 논문 전체 검토
- [x] 부록 코드: `AdaptiveCascade` → `ClusterBasedCascade`로 변경
- [x] 재현성 경로: `exp10` → `exp01_truthfulqa`로 수정
- [x] 모든 수치 일관성 확인 완료

---

## 📁 파일 구조

```
hallucination_lfe/
├── thesis/
│   ├── main.tex              # 메인 논문 (32페이지)
│   ├── main.pdf              # 컴파일된 PDF
│   ├── figures_tikz.tex      # TikZ 다이어그램 정의
│   ├── sections/
│   │   └── experiment_method.tex  # 실험 방법 섹션
│   ├── images/               # (더 이상 사용 안 함)
│   └── PROGRESS.md           # 이 파일
├── experiment_notes/
│   ├── exp01_truthfulqa/     # 200샘플 원본 실험 ⭐
│   │   ├── run_experiment.py
│   │   └── results.json
│   └── thesis_200_analysis.json  # 200샘플 분석 요약
```

---

## 🎨 TikZ 다이어그램 색상 팔레트

```latex
\definecolor{primaryblue}{RGB}{31,119,180}     % SE 관련
\definecolor{secondaryorange}{RGB}{255,127,14} % Energy 관련
\definecolor{accentgreen}{RGB}{44,160,44}      % Cascade/성공
\definecolor{accentpurple}{RGB}{148,103,189}   % 강조
\definecolor{hallred}{RGB}{214,39,40}          % 환각
\definecolor{normgreen}{RGB}{44,160,44}        % 정상
```

---

## 💬 결정 사항 로그

### 2026-02-11
1. **데이터**: 200샘플 (exp01) 기준, Limitation에 817샘플 확장 가능성 언급
2. **Cascade 방식**: 클러스터 기반 (|C|=1 → Energy) - 수학적으로 엄밀
3. **GPU**: RTX 5090 (32GB) 사용
4. **TikZ**: 모든 그림 TikZ로 교체, 학술 논문 스타일 색상

---

## 📝 남은 작업

- [ ] 최종 PDF 검토
- [ ] 지도교수 피드백 반영
- [ ] 제출

---

## 🔄 2026-02-11 15:26 업데이트

### 재현성 개선
- [x] `run_experiment.py` 생성 (프로젝트 루트) - 완전 재현 가능한 단일 스크립트
- [x] `README.md` 업데이트 - 실험 재현 방법 안내
- [x] 논문에서 디렉토리 구조/bash 명령어 제거 → GitHub README로 이동
- [x] 부록 프로젝트 구조 리스팅 제거

---

## 🔗 참고

- **데이터 원본**: `experiment_notes/exp01_truthfulqa/results.json`
- **분석 요약**: `experiment_notes/thesis_200_analysis.json`
- **논문 브랜치**: `thesis-paper`
