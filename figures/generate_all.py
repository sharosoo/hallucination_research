#!/usr/bin/env python
"""
260206 미팅 보고용 Figure 생성 스크립트.
figures/ 디렉토리에 PNG 파일로 저장.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
FIG_DIR = Path(__file__).parent

with open(
    ROOT / "experiment_notes" / "exp07_zero_se_analysis" / "analysis_results.json"
) as f:
    DATA = json.load(f)

COLORS = {
    "se": "#4A90D9",
    "energy": "#E8553A",
    "cascade": "#2ECC71",
    "hall": "#E74C3C",
    "normal": "#95A5A6",
    "both": "#8E44AD",
    "neither": "#BDC3C7",
}
DS_ORDER = [
    "TruthfulQA",
    "HaluEval",
    "TriviaQA",
    "NaturalQuestions",
    "HaluEval-dialogue",
]
DS_SHORT = [
    "TruthfulQA",
    "HaluEval\n(QA)",
    "TriviaQA",
    "Natural\nQuestions",
    "HaluEval\n(Dialogue)",
]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def fig1_zero_se_overview():
    """Fig 1: Zero-SE 현상 개요 — 비율 + 환각률 + Energy AUROC"""
    zero_se = {
        r["dataset"]: r for r in DATA["zero_se_analysis"] if r["epsilon"] == 0.001
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    zero_pcts = [zero_se[d]["zero_se_pct"] for d in DS_ORDER]
    hall_rates = [zero_se[d]["hall_rate_in_zero_se"] for d in DS_ORDER]
    energy_aurocs = [zero_se[d]["energy_auroc_in_zero_se"] or 0 for d in DS_ORDER]

    bar_colors_energy = [
        COLORS["energy"] if v > 0.55 else COLORS["normal"] for v in energy_aurocs
    ]

    ax = axes[0]
    bars = ax.bar(
        range(len(DS_ORDER)),
        zero_pcts,
        color=COLORS["se"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_SHORT, fontsize=8)
    ax.set_ylabel("Zero-SE Samples (%)")
    ax.set_title("(a) Zero-SE Prevalence")
    ax.set_ylim(0, 65)
    for bar, val in zip(bars, zero_pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax = axes[1]
    bars = ax.bar(
        range(len(DS_ORDER)),
        hall_rates,
        color=COLORS["hall"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_SHORT, fontsize=8)
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("(b) Hallucination Rate within Zero-SE")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, hall_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax = axes[2]
    bars = ax.bar(
        range(len(DS_ORDER)),
        energy_aurocs,
        color=bar_colors_energy,
        alpha=0.85,
        edgecolor="white",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_SHORT, fontsize=8)
    ax.set_ylabel("Energy AUROC")
    ax.set_title("(c) Energy AUROC within Zero-SE")
    ax.set_ylim(0.3, 0.85)
    ax.legend(fontsize=8, loc="upper right")
    for bar, val in zip(bars, energy_aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Figure 1. Zero-SE Phenomenon Across 5 Datasets",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_zero_se_overview.png")
    plt.close(fig)
    print("  Saved fig1_zero_se_overview.png")


def fig2_se_bin_crossover():
    """Fig 2: SE 구간별 SE vs Energy AUROC — crossover 패턴"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    target_ds = ["TruthfulQA", "TriviaQA", "HaluEval"]
    titles = ["(a) TruthfulQA", "(b) TriviaQA", "(c) HaluEval-QA"]

    for idx, (ds_name, title) in enumerate(zip(target_ds, titles)):
        ax = axes[idx]
        bins_data = [
            b for b in DATA["se_bin_analysis"] if b["dataset"] == ds_name and b["n"] > 0
        ]

        bin_names = [b["bin"] for b in bins_data]
        se_aurocs = [b["se_auroc"] if b["se_auroc"] else None for b in bins_data]
        en_aurocs = [
            b["energy_auroc"] if b["energy_auroc"] else None for b in bins_data
        ]
        ns = [b["n"] for b in bins_data]

        x = np.arange(len(bin_names))
        width = 0.35

        se_vals = [v if v is not None else 0 for v in se_aurocs]
        en_vals = [v if v is not None else 0 for v in en_aurocs]
        se_mask = [v is not None for v in se_aurocs]
        en_mask = [v is not None for v in en_aurocs]

        se_bars = ax.bar(
            x - width / 2,
            se_vals,
            width,
            label="SE",
            color=COLORS["se"],
            alpha=0.85,
            edgecolor="white",
        )
        en_bars = ax.bar(
            x + width / 2,
            en_vals,
            width,
            label="Energy",
            color=COLORS["energy"],
            alpha=0.85,
            edgecolor="white",
        )

        for i, (sv, sm) in enumerate(zip(se_vals, se_mask)):
            if not sm:
                se_bars[i].set_alpha(0.15)
                ax.text(
                    x[i] - width / 2, 0.05, "N/A", ha="center", fontsize=7, color="gray"
                )

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n(n={ns_})" for n, ns_ in zip(bin_names, ns)], fontsize=7.5
        )
        ax.set_ylabel("AUROC" if idx == 0 else "")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc="upper left" if idx != 2 else "upper right")

    fig.suptitle(
        "Figure 2. SE vs Energy AUROC by Semantic Entropy Bin (Crossover Pattern)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_se_bin_crossover.png")
    plt.close(fig)
    print("  Saved fig2_se_bin_crossover.png")


def fig3_complementarity():
    """Fig 3: 상보성 분석 Stacked Bar"""
    comp = {c["dataset"]: c for c in DATA["complementarity"]}

    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ["SE only", "Energy only", "Both", "Neither"]
    colors = [COLORS["se"], COLORS["energy"], COLORS["both"], COLORS["neither"]]

    x = np.arange(len(DS_ORDER))
    width = 0.6

    bottom = np.zeros(len(DS_ORDER))
    for cat_idx, cat in enumerate(
        ["se_only", "energy_only", "both_catch", "neither_catch"]
    ):
        vals = []
        for ds in DS_ORDER:
            c = comp[ds]
            total = c["n_hallucinations"]
            vals.append(c[cat] / total * 100 if total > 0 else 0)
        vals = np.array(vals)
        ax.bar(
            x,
            vals,
            width,
            bottom=bottom,
            label=categories[cat_idx],
            color=colors[cat_idx],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT, fontsize=9)
    ax.set_ylabel("Proportion of Hallucinations (%)")
    ax.set_title(
        "Figure 3. Complementarity: Who Catches What?", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    for i, ds in enumerate(DS_ORDER):
        c = comp[ds]
        ax.text(
            x[i],
            101,
            f"n={c['n_hallucinations']}",
            ha="center",
            fontsize=7.5,
            color="gray",
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_complementarity.png")
    plt.close(fig)
    print("  Saved fig3_complementarity.png")


def fig4_cascade_sweep():
    """Fig 4: Cascade τ sweep — TruthfulQA & TriviaQA"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, ds_name in enumerate(["TruthfulQA", "TriviaQA"]):
        ax = axes[idx]
        sweep = DATA["cascade_sweep"][ds_name]

        taus = [s["tau"] for s in sweep]
        cascade = [s["cascade_auroc"] for s in sweep]
        se_base = sweep[0]["se_auroc"]
        en_base = sweep[0]["energy_auroc"]

        ax.plot(
            taus,
            cascade,
            color=COLORS["cascade"],
            linewidth=2,
            label="Cascade",
            zorder=3,
        )
        ax.axhline(
            se_base,
            color=COLORS["se"],
            linestyle="--",
            linewidth=1.2,
            label=f"SE-only ({se_base:.3f})",
        )
        ax.axhline(
            en_base,
            color=COLORS["energy"],
            linestyle="--",
            linewidth=1.2,
            label=f"Energy-only ({en_base:.3f})",
        )

        best = max(sweep, key=lambda s: s["cascade_auroc"])
        ax.plot(
            best["tau"],
            best["cascade_auroc"],
            "o",
            color=COLORS["cascade"],
            markersize=8,
            zorder=4,
        )
        ax.annotate(
            f"Best: τ={best['tau']:.2f}\nAUROC={best['cascade_auroc']:.3f}",
            xy=(best["tau"], best["cascade_auroc"]),
            xytext=(best["tau"] + 0.15, best["cascade_auroc"] + 0.015),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        ax.set_xlabel("Threshold τ (SE value)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"({'a' if idx == 0 else 'b'}) {ds_name}")
        ax.legend(fontsize=8, loc="lower right" if idx == 0 else "lower left")
        ax.set_ylim(min(se_base, en_base) - 0.05, max(cascade) + 0.04)

    fig.suptitle(
        "Figure 4. Cascade Threshold Sweep: AUROC vs τ",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_cascade_sweep.png")
    plt.close(fig)
    print("  Saved fig4_cascade_sweep.png")


def fig5_overall_comparison():
    """Fig 5: 전체 데이터셋 SE vs Energy vs Cascade 비교"""
    sweep_data = DATA["cascade_sweep"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(DS_ORDER))
    width = 0.25

    se_vals = [sweep_data[d][0]["se_auroc"] for d in DS_ORDER]
    en_vals = [sweep_data[d][0]["energy_auroc"] for d in DS_ORDER]
    cas_vals = [max(s["cascade_auroc"] for s in sweep_data[d]) for d in DS_ORDER]

    ax.bar(
        x - width,
        se_vals,
        width,
        label="SE-only",
        color=COLORS["se"],
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x,
        en_vals,
        width,
        label="Energy-only",
        color=COLORS["energy"],
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x + width,
        cas_vals,
        width,
        label="Cascade (best τ)",
        color=COLORS["cascade"],
        alpha=0.85,
        edgecolor="white",
    )

    for i in range(len(DS_ORDER)):
        best_val = max(se_vals[i], en_vals[i], cas_vals[i])
        best_method = ["SE", "Energy", "Cascade"][
            [se_vals[i], en_vals[i], cas_vals[i]].index(best_val)
        ]
        ax.text(
            x[i],
            best_val + 0.008,
            f"Best: {best_method}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="green",
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title(
        "Figure 5. Overall Comparison: SE vs Energy vs Cascade",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0.4, 0.75)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_overall_comparison.png")
    plt.close(fig)
    print("  Saved fig5_overall_comparison.png")


def fig6_story_diagram():
    """Fig 6: 논문 스토리 다이어그램 — 두 가지 환각 유형 + 탐지 전략"""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    title_style = dict(fontsize=12, fontweight="bold", ha="center", va="center")
    body_style = dict(fontsize=9, ha="center", va="center", family="monospace")
    small_style = dict(fontsize=8, ha="center", va="center")

    # Question box
    q_box = mpatches.FancyBboxPatch(
        (4.5, 6.0),
        3,
        0.7,
        boxstyle="round,pad=0.1",
        facecolor="#F0F0F0",
        edgecolor="#333",
    )
    ax.add_patch(q_box)
    ax.text(6, 6.35, "Input Question", **title_style)

    # Arrow down
    ax.annotate(
        "",
        xy=(6, 5.5),
        xytext=(6, 6.0),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
    )
    ax.text(6, 5.7, "LLM generates K responses", fontsize=8, ha="center", color="gray")

    # SE computation
    se_box = mpatches.FancyBboxPatch(
        (3.5, 4.5),
        5,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#EBF5FB",
        edgecolor=COLORS["se"],
        linewidth=1.5,
    )
    ax.add_patch(se_box)
    ax.text(6, 4.95, "Compute Semantic Entropy (SE)", **title_style, color=COLORS["se"])
    ax.text(
        6, 4.7, "NLI clustering → cluster distribution → Shannon entropy", **small_style
    )

    # Decision diamond
    ax.annotate(
        "",
        xy=(6, 3.8),
        xytext=(6, 4.5),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
    )

    diamond = plt.Polygon(
        [(6, 4.05), (7.2, 3.5), (6, 2.95), (4.8, 3.5)],
        facecolor="#FEF9E7",
        edgecolor="#F39C12",
        linewidth=1.5,
    )
    ax.add_patch(diamond)
    ax.text(
        6,
        3.5,
        "SE < τ ?",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color="#E67E22",
    )

    # Left: High SE → Confusion
    ax.annotate(
        "",
        xy=(2.5, 3.5),
        xytext=(4.8, 3.5),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["se"], lw=1.5),
    )
    ax.text(
        3.65,
        3.7,
        "No (High SE)",
        fontsize=8,
        ha="center",
        color=COLORS["se"],
        fontweight="bold",
    )

    conf_box = mpatches.FancyBboxPatch(
        (0.3, 2.2),
        4.3,
        1.1,
        boxstyle="round,pad=0.1",
        facecolor="#D6EAF8",
        edgecolor=COLORS["se"],
        linewidth=1.5,
    )
    ax.add_patch(conf_box)
    ax.text(
        2.45,
        2.95,
        "Type A: Confusion",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color=COLORS["se"],
    )
    ax.text(
        2.45,
        2.6,
        "Model knows but is confused\n→ Diverse responses → SE detects",
        **small_style,
    )

    # Right: Low SE → Confabulation
    ax.annotate(
        "",
        xy=(9.5, 3.5),
        xytext=(7.2, 3.5),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["energy"], lw=1.5),
    )
    ax.text(
        8.35,
        3.7,
        "Yes (Zero-SE)",
        fontsize=8,
        ha="center",
        color=COLORS["energy"],
        fontweight="bold",
    )

    confab_box = mpatches.FancyBboxPatch(
        (7.4, 2.2),
        4.3,
        1.1,
        boxstyle="round,pad=0.1",
        facecolor="#FDEDEC",
        edgecolor=COLORS["energy"],
        linewidth=1.5,
    )
    ax.add_patch(confab_box)
    ax.text(
        9.55,
        2.95,
        "Type B: Confabulation",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color=COLORS["energy"],
    )
    ax.text(
        9.55,
        2.6,
        "Model doesn't know, fabricates\n→ Consistent responses → Energy detects",
        **small_style,
    )

    # Bottom results
    ax.annotate(
        "",
        xy=(2.45, 1.5),
        xytext=(2.45, 2.2),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["se"], lw=1.2),
    )
    se_res = mpatches.FancyBboxPatch(
        (0.8, 0.7),
        3.3,
        0.7,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["se"],
        edgecolor="white",
        alpha=0.15,
    )
    ax.add_patch(se_res)
    ax.text(
        2.45,
        1.15,
        "Use SE Score",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["se"],
    )
    ax.text(
        2.45,
        0.85,
        "High SE = likely hallucination",
        fontsize=8,
        ha="center",
        color="gray",
    )

    ax.annotate(
        "",
        xy=(9.55, 1.5),
        xytext=(9.55, 2.2),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["energy"], lw=1.2),
    )
    en_res = mpatches.FancyBboxPatch(
        (7.9, 0.7),
        3.3,
        0.7,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["energy"],
        edgecolor="white",
        alpha=0.15,
    )
    ax.add_patch(en_res)
    ax.text(
        9.55,
        1.15,
        "Use Energy Score",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["energy"],
    )
    ax.text(
        9.55,
        0.85,
        "High Energy = likely hallucination",
        fontsize=8,
        ha="center",
        color="gray",
    )

    # Title
    ax.text(
        6,
        0.15,
        "Figure 6.  SE-Gated Cascade: Regime-Aware Hallucination Detection",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color="#333",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_story_diagram.png")
    plt.close(fig)
    print("  Saved fig6_story_diagram.png")


def fig7_complementarity_sensitivity():
    """Fig 7: Threshold sensitivity — Energy-only % across percentile thresholds"""
    rob_path = (
        ROOT / "experiment_notes" / "exp08_robustness" / "robustness_results.json"
    )
    if not rob_path.exists():
        print("  Skipping fig7 — exp08 results not found")
        return
    with open(rob_path) as f:
        rob = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    percentiles = [60, 70, 80, 90]
    x = np.arange(len(percentiles))
    width = 0.15

    for i, ds_name in enumerate(DS_ORDER):
        ds_data = [
            r for r in rob["complementarity_sensitivity"] if r["dataset"] == ds_name
        ]
        if not ds_data:
            continue
        vals = [r["energy_only_pct"] for r in ds_data]
        offset = (i - len(DS_ORDER) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=DS_SHORT[i].replace("\n", " "),
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th pctl" for p in percentiles], fontsize=10)
    ax.set_ylabel("Energy-only Hallucinations (%)")
    ax.set_title(
        "Figure 7. Complementarity Threshold Sensitivity:\n"
        "Energy-only Detection Rate Across Percentile Thresholds",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 40)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.axhspan(7, 34, alpha=0.06, color="orange")
    ax.text(
        3.5,
        35,
        "Range: 7–33%",
        fontsize=9,
        ha="right",
        color="orange",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_complementarity_sensitivity.png")
    plt.close(fig)
    print("  Saved fig7_complementarity_sensitivity.png")


def fig8_bootstrap_ci():
    """Fig 8: Bootstrap CI for Cascade vs SE-only AUROC delta"""
    rob_path = (
        ROOT / "experiment_notes" / "exp08_robustness" / "robustness_results.json"
    )
    if not rob_path.exists():
        print("  Skipping fig8 — exp08 results not found")
        return
    with open(rob_path) as f:
        rob = json.load(f)

    cross_results = [
        r
        for r in rob["bootstrap_tests"]
        if r.get("tau_source") == "cross_dataset_0.526"
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    y_pos = np.arange(len(cross_results))
    ds_labels = []

    for i, r in enumerate(cross_results):
        delta = r["observed_delta"]
        ci_lo = r["ci_95_lower"]
        ci_hi = r["ci_95_upper"]
        p_val = r["p_value"]

        color = COLORS["cascade"] if delta > 0 else COLORS["normal"]
        ax.barh(i, delta, height=0.5, color=color, alpha=0.7, edgecolor="white")
        ax.plot([ci_lo, ci_hi], [i, i], color="#333", linewidth=2, zorder=3)
        ax.plot([ci_lo, ci_hi], [i, i], "|", color="#333", markersize=10, zorder=3)

        label = f"p={p_val:.3f}"
        ax.text(
            max(ci_hi + 0.005, delta + 0.005),
            i,
            label,
            va="center",
            fontsize=8,
            color="gray",
        )
        ds_labels.append(r["dataset"])

    ax.axvline(0, color="#333", linewidth=1, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_labels, fontsize=10)
    ax.set_xlabel("AUROC Delta (Cascade − SE-only)")
    ax.set_title(
        "Figure 8. Bootstrap 95% CI for Cascade Improvement (τ=0.526)",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_bootstrap_ci.png")
    plt.close(fig)
    print("  Saved fig8_bootstrap_ci.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_zero_se_overview()
    fig2_se_bin_crossover()
    fig3_complementarity()
    fig4_cascade_sweep()
    fig5_overall_comparison()
    fig6_story_diagram()
    fig7_complementarity_sensitivity()
    fig8_bootstrap_ci()
    print("Done! All figures saved to figures/")
