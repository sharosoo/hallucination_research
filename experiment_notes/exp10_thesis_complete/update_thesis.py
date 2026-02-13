#!/usr/bin/env python
"""
실험 완료 후 논문 자동 업데이트 스크립트

기능:
1. analysis.json에서 수치 읽기
2. thesis/images/에 새 Figure 복사
3. main.tex의 수치들 자동 업데이트
4. 업데이트된 LaTeX 테이블 생성

실행:
  python experiment_notes/exp10_thesis_complete/update_thesis.py
"""

import json
import shutil
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent
THESIS_DIR = ROOT / "thesis"


def load_analysis():
    """분석 결과 로드"""
    path = EXP_DIR / "analysis.json"
    if not path.exists():
        print(f"Error: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def copy_figures():
    """새 Figure를 thesis/images/로 복사"""
    print("\n[1] Figure 복사...")
    
    src_dir = EXP_DIR / "figures"
    dst_dir = THESIS_DIR / "images"
    
    if not src_dir.exists():
        print(f"  Error: {src_dir} not found")
        return False
    
    # exp10 → thesis 매핑
    mapping = {
        "fig1_zero_se_overview.png": "fig1_zero_se_overview.png",
        "fig2_se_bin_crossover.png": "fig2_se_bin_crossover.png",
        "fig3_cascade_sweep.png": "fig4_cascade_sweep.png",  # thesis에서는 fig4
        "fig4_complementarity.png": "fig3_complementarity.png",  # thesis에서는 fig3
        "fig5_overall_comparison.png": "fig5_overall_comparison.png",
    }
    
    for src_name, dst_name in mapping.items():
        src = src_dir / src_name
        dst = dst_dir / dst_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✓ {src_name} → {dst_name}")
        else:
            print(f"  ✗ {src_name} not found")
    
    return True


def update_abstract(content: str, analysis: dict) -> str:
    """초록의 수치 업데이트"""
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    best = analysis["best_cascade"]
    comp = analysis["complementarity"]
    overall = analysis["overall"]
    
    # Zero-SE 비율
    content = re.sub(
        r'Zero-SE 영역이 전체의 \d+(\.\d+)?\\%',
        f'Zero-SE 영역이 전체의 {zero_se.get("percentage", 0):.0f}\\%',
        content
    )
    
    # Zero-SE 환각률
    content = re.sub(
        r'이 중 \d+(\.\d+)?\\%가 환각',
        f'이 중 {zero_se.get("hallucination_rate", 0):.1f}\\%가 환각',
        content
    )
    
    # Energy AUROC in Zero-SE
    if zero_se.get("energy_auroc"):
        content = re.sub(
            r'Energy AUROC \d+\.\d+ 달성',
            f'Energy AUROC {zero_se["energy_auroc"]:.3f} 달성',
            content
        )
    
    # Cascade improvement
    se_auroc = overall["se_auroc"]
    cascade_auroc = best["auroc"]
    delta = cascade_auroc - se_auroc
    
    content = re.sub(
        r'AUROC \+\d+\.\d+ 개선 \(\d+\.\d+ → \d+\.\d+\)',
        f'AUROC +{delta:.3f} 개선 ({se_auroc:.3f} → {cascade_auroc:.3f})',
        content
    )
    
    # 합집합 탐지율
    content = re.sub(
        r'합집합 탐지율 \d+(\.\d+)?\\% 달성',
        f'합집합 탐지율 {comp["union_catch_rate"]:.1f}\\% 달성',
        content
    )
    
    return content


def update_tables(content: str, analysis: dict) -> str:
    """테이블 수치 업데이트"""
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    best = analysis["best_cascade"]
    overall = analysis["overall"]
    
    # Zero-SE 테이블 (tab:zero_se)
    if zero_se:
        n_total = analysis["n_samples"]
        n_zero = zero_se.get("n", 0)
        n_hall = zero_se.get("n_hallucinations", 0)
        
        # Zero-SE 비율
        content = re.sub(
            r'Zero-SE 비율 & \d+\.\d+\\% \(\d+/\d+\)',
            f'Zero-SE 비율 & {zero_se["percentage"]:.1f}\\% ({n_zero}/{n_total})',
            content
        )
        
        # Zero-SE 내 환각률
        content = re.sub(
            r'Zero-SE 내 환각률 & \d+\.\d+\\% \(\d+/\d+\)',
            f'Zero-SE 내 환각률 & {zero_se["hallucination_rate"]:.1f}\\% ({n_hall}/{n_zero})',
            content
        )
        
        # Energy AUROC
        if zero_se.get("energy_auroc"):
            content = re.sub(
                r'Zero-SE 내 Energy AUROC & \d+\.\d+',
                f'Zero-SE 내 Energy AUROC & {zero_se["energy_auroc"]:.3f}',
                content
            )
    
    # Cascade 결과 테이블 (tab:cascade_result)
    se_auroc = overall["se_auroc"]
    en_auroc = overall["energy_auroc"]
    cascade_auroc = best["auroc"]
    tau = best["tau"]
    
    # SE-only
    content = re.sub(
        r'SE-only & \d+\.\d+ & -',
        f'SE-only & {se_auroc:.3f} & -',
        content
    )
    
    # Energy-only
    content = re.sub(
        r'Energy-only & \d+\.\d+ & -?\d+\.\d+',
        f'Energy-only & {en_auroc:.3f} & {en_auroc - se_auroc:+.3f}',
        content
    )
    
    # Cascade
    content = re.sub(
        r'Cascade \(\$\\tau\$=\d+\.\d+\)\} & \\textbf\{\d+\.\d+\} & \\textbf\{\+\d+\.\d+\}',
        f'Cascade ($\\tau$={tau:.2f})}} & \\textbf{{{cascade_auroc:.3f}}} & \\textbf{{+{cascade_auroc - se_auroc:.3f}}}',
        content
    )
    
    return content


def update_body_text(content: str, analysis: dict) -> str:
    """본문 텍스트의 수치 업데이트"""
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    
    # "전체 샘플의 X%가 Zero-SE"
    if zero_se:
        content = re.sub(
            r'전체 샘플의 \d+\\%가 Zero-SE',
            f'전체 샘플의 {zero_se["percentage"]:.0f}\\%가 Zero-SE',
            content
        )
        
        # "Zero-SE 샘플 중 X%가 실제 환각"
        content = re.sub(
            r'Zero-SE 샘플 중 \d+\.\d+\\%가 실제 환각',
            f'Zero-SE 샘플 중 {zero_se["hallucination_rate"]:.1f}\\%가 실제 환각',
            content
        )
        
        # "Energy는 AUROC X.XXX으로"
        if zero_se.get("energy_auroc"):
            content = re.sub(
                r'Energy는 AUROC \d+\.\d+으로',
                f'Energy는 AUROC {zero_se["energy_auroc"]:.3f}으로',
                content
            )
    
    return content


def update_main_tex(analysis: dict):
    """main.tex 업데이트"""
    print("\n[2] main.tex 수치 업데이트...")
    
    main_path = THESIS_DIR / "main.tex"
    with open(main_path, "r") as f:
        content = f.read()
    
    original = content
    
    content = update_abstract(content, analysis)
    content = update_tables(content, analysis)
    content = update_body_text(content, analysis)
    
    # 변경 사항 확인
    if content != original:
        with open(main_path, "w") as f:
            f.write(content)
        print("  ✓ main.tex 업데이트됨")
    else:
        print("  - 변경 사항 없음")


def generate_latex_tables(analysis: dict):
    """LaTeX 테이블 파일 생성"""
    print("\n[3] LaTeX 테이블 생성...")
    
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    best = analysis["best_cascade"]
    overall = analysis["overall"]
    comp = analysis["complementarity"]
    
    tables = f"""% 자동 생성됨: exp10_thesis_complete/update_thesis.py
% 실험 결과 기반 LaTeX 테이블

%% 데이터셋 통계
% 전체 샘플: {analysis['n_samples']}
% 환각: {analysis['n_hallucinations']} ({analysis['hallucination_rate']:.1f}%)
% 정상: {analysis['n_normal']} ({100-analysis['hallucination_rate']:.1f}%)

%% 전체 성능
% SE AUROC: {overall['se_auroc']:.4f}
% Energy AUROC: {overall['energy_auroc']:.4f}
% Cascade AUROC: {best['auroc']:.4f} (τ={best['tau']:.2f})
% Cascade 개선: +{best['auroc'] - overall['se_auroc']:.4f}

%% Zero-SE (ε=0.05)
% 비율: {zero_se.get('percentage', 0):.1f}%
% 환각률: {zero_se.get('hallucination_rate', 0):.1f}%
% Energy AUROC: {zero_se.get('energy_auroc', 'N/A')}

%% 상보성
% 합집합 탐지율: {comp['union_catch_rate']:.1f}%

%% SE 구간별 성능
\\begin{{table}}[htbp]
\\centering
\\caption{{SE 구간별 탐지 성능 (실험 결과)}}
\\label{{tab:crossover_exp10}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
SE 구간 & n & 환각률 & SE AUROC & Energy AUROC \\\\
\\midrule
"""
    
    for b in analysis["se_bins"]:
        se_str = f"{b['se_auroc']:.3f}" if b["se_auroc"] else "N/A"
        en_str = f"\\textbf{{{b['energy_auroc']:.3f}}}" if b["energy_auroc"] and b["bin"] == "Zero-SE" else (f"{b['energy_auroc']:.3f}" if b["energy_auroc"] else "N/A")
        tables += f"{b['bin']} {b['range']} & {b['n']} & {b['hallucination_rate']:.1f}\\% & {se_str} & {en_str} \\\\\n"
    
    tables += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    tables_path = EXP_DIR / "latex_tables.tex"
    with open(tables_path, "w") as f:
        f.write(tables)
    print(f"  ✓ {tables_path}")


def print_summary(analysis: dict):
    """논문에 넣을 핵심 수치 출력"""
    print("\n" + "=" * 60)
    print("논문 핵심 수치 (초록/결론용)")
    print("=" * 60)
    
    zero_se = next((z for z in analysis["zero_se"] if z["epsilon"] == 0.05), {})
    best = analysis["best_cascade"]
    overall = analysis["overall"]
    comp = analysis["complementarity"]
    
    print(f"""
  ■ 데이터셋
    - 전체 샘플: {analysis['n_samples']}개
    - 환각률: {analysis['hallucination_rate']:.1f}%

  ■ Zero-SE 현상
    - 비율: {zero_se.get('percentage', 0):.1f}%
    - 환각률: {zero_se.get('hallucination_rate', 0):.1f}%
    - Energy AUROC: {zero_se.get('energy_auroc', 'N/A')}

  ■ 전체 성능
    - SE AUROC: {overall['se_auroc']:.3f}
    - Energy AUROC: {overall['energy_auroc']:.3f}
    - Cascade AUROC: {best['auroc']:.3f} (τ={best['tau']:.2f})
    - Cascade 개선: +{best['auroc'] - overall['se_auroc']:.3f}

  ■ 상보성
    - 합집합 탐지율: {comp['union_catch_rate']:.1f}%
    - SE-only: {comp['se_only']}개 ({comp['se_only']/comp['n_hallucinations']*100:.1f}%)
    - Energy-only: {comp['energy_only']}개 ({comp['energy_only']/comp['n_hallucinations']*100:.1f}%)
""")


def main():
    print("=" * 60)
    print("논문 자동 업데이트")
    print("=" * 60)
    
    # 분석 결과 로드
    analysis = load_analysis()
    if not analysis:
        print("\nError: 분석 결과 없음. 실험을 먼저 완료하세요.")
        return
    
    # Figure 복사
    copy_figures()
    
    # main.tex 업데이트
    update_main_tex(analysis)
    
    # LaTeX 테이블 생성
    generate_latex_tables(analysis)
    
    # 요약 출력
    print_summary(analysis)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print("""
다음 단계:
  cd thesis
  pdflatex main.tex
  pdflatex main.tex  # 목차 업데이트용
""")


if __name__ == "__main__":
    main()
