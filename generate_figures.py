"""
generate_figures.py — BioGemma-Guard Academic Visualization
============================================================

Generates two publication-quality figures from the LLM-as-a-Judge
evaluation results (quantitative_judged_evaluation_FIX.csv):

  Figure 1: Comparative Radar Chart — Average scores across 3 metrics
  Figure 2: CoT Discipline Distribution — Histogram + KDE overlay

Usage:
    python generate_figures.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "quantitative_judged_evaluation_FIX.csv"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette — accessible, print-friendly
COLOR_BASE = "#4A90D9"       # Steel blue
COLOR_BIO  = "#E8833A"       # Warm orange
COLOR_BASE_FILL = "#4A90D933"
COLOR_BIO_FILL  = "#E8833A33"

DPI = 300

# ──────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> dict:
    """Load evaluation CSV and return structured score arrays."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data = {
        "base_faithfulness": np.array([float(r["base_faithfulness"]) for r in rows]),
        "base_differential": np.array([float(r["base_differential"]) for r in rows]),
        "base_discipline":   np.array([float(r["base_discipline"]) for r in rows]),
        "bio_faithfulness":  np.array([float(r["bio_faithfulness"]) for r in rows]),
        "bio_differential":  np.array([float(r["bio_differential"]) for r in rows]),
        "bio_discipline":    np.array([float(r["bio_discipline"]) for r in rows]),
    }
    return data


# ──────────────────────────────────────────────────────────────────────
# 2.  FIGURE 1 — COMPARATIVE RADAR CHART
# ──────────────────────────────────────────────────────────────────────

def make_radar_chart(data: dict, save_path: Path) -> None:
    """
    Three-axis radar chart comparing average scores of
    Base Model vs BioGemma-Guard.
    """
    # ── Metric labels and averages ──────────────────────────────────
    labels = [
        "Clinical\nFaithfulness",
        "Differential\nQuality",
        "CoT\nDiscipline",
    ]
    base_means = np.array([
        data["base_faithfulness"].mean(),
        data["base_differential"].mean(),
        data["base_discipline"].mean(),
    ])
    bio_means = np.array([
        data["bio_faithfulness"].mean(),
        data["bio_differential"].mean(),
        data["bio_discipline"].mean(),
    ])

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the polygon
    base_vals = np.concatenate([base_means, [base_means[0]]])
    bio_vals  = np.concatenate([bio_means, [bio_means[0]]])
    angles += angles[:1]

    # ── Figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    # Grid styling
    ax.set_facecolor("#FAFAFA")
    ax.spines["polar"].set_visible(False)
    ax.set_rlabel_position(30)

    # Radial scale 0–10
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(
        ["2", "4", "6", "8", "10"],
        fontsize=9, color="#666666", fontfamily="serif",
    )

    # Soften gridlines
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.6, linestyle="--")
    ax.xaxis.grid(True, color="#CCCCCC", linewidth=0.6)

    # ── Plot data ───────────────────────────────────────────────────
    ax.plot(
        angles, base_vals,
        color=COLOR_BASE, linewidth=2.2, linestyle="-",
        label=f"Base Model (Gemma-2-2B)", marker="o", markersize=7,
    )
    ax.fill(angles, base_vals, color=COLOR_BASE, alpha=0.12)

    ax.plot(
        angles, bio_vals,
        color=COLOR_BIO, linewidth=2.2, linestyle="-",
        label=f"BioGemma-Guard (GRPO)", marker="s", markersize=7,
    )
    ax.fill(angles, bio_vals, color=COLOR_BIO, alpha=0.12)

    # ── Annotate exact values ───────────────────────────────────────
    for i in range(N):
        angle_rad = angles[i]
        gap = abs(bio_means[i] - base_means[i])

        # When scores are very close, spread labels apart
        if gap < 0.3:
            r_offset_base = base_means[i] + 0.85
            r_offset_bio  = bio_means[i] - 0.85
        else:
            r_offset_base = base_means[i] + 0.55
            r_offset_bio  = bio_means[i] - 0.65 if i == 2 else bio_means[i] + 0.55

        ax.text(
            angle_rad, r_offset_base, f"{base_means[i]:.2f}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLOR_BASE, fontfamily="serif",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLOR_BASE, alpha=0.85, linewidth=0.6),
        )
        ax.text(
            angle_rad, r_offset_bio, f"{bio_means[i]:.2f}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLOR_BIO, fontfamily="serif",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLOR_BIO, alpha=0.85, linewidth=0.6),
        )

    # ── Axis labels ─────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontfamily="serif", fontweight="bold")

    # ── Legend & Title ──────────────────────────────────────────────
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.32, 1.12),
        fontsize=10, frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC", fancybox=True,
        prop={"family": "serif"},
    )

    fig.suptitle(
        "Figure 1: Comparative Evaluation — Base Model vs. BioGemma-Guard",
        fontsize=12, fontfamily="serif", fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "LLM-as-a-Judge Scores (1–10 scale, n = 50 cases)\n"
        "Judge: LLaMA-4-Scout-17B-16E-Instruct",
        fontsize=9, fontfamily="serif", color="#555555",
        pad=20,
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.94])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 3.  FIGURE 2 — COT DISCIPLINE DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────

def make_discipline_distribution(data: dict, save_path: Path) -> None:
    """
    Side-by-side histogram with KDE overlay for CoT Discipline scores,
    highlighting the rightward shift of BioGemma-Guard.
    """
    base_scores = data["base_discipline"]
    bio_scores  = data["bio_discipline"]

    base_mean = base_scores.mean()
    bio_mean  = bio_scores.mean()

    # ── Style ───────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    # ── Histograms (discrete bins aligned to integer scores) ────────
    bins = np.arange(0.5, 11.5, 1)  # center bins on integers 1–10

    ax.hist(
        base_scores, bins=bins, alpha=0.45,
        color=COLOR_BASE, edgecolor=COLOR_BASE,
        linewidth=1.2, label=f"Base Model (μ = {base_mean:.2f})",
        rwidth=0.85,
    )
    ax.hist(
        bio_scores, bins=bins, alpha=0.45,
        color=COLOR_BIO, edgecolor=COLOR_BIO,
        linewidth=1.2, label=f"BioGemma-Guard (μ = {bio_mean:.2f})",
        rwidth=0.85,
    )

    # ── KDE overlays ────────────────────────────────────────────────
    from scipy.stats import gaussian_kde

    x_range = np.linspace(1, 10, 200)

    kde_base = gaussian_kde(base_scores, bw_method=0.5)
    kde_bio  = gaussian_kde(bio_scores, bw_method=0.5)

    # Scale KDE to match histogram counts
    scale_factor = len(base_scores) * 1.0   # bin width = 1
    ax.plot(x_range, kde_base(x_range) * scale_factor,
            color=COLOR_BASE, linewidth=2.2, linestyle="-", alpha=0.9)
    ax.plot(x_range, kde_bio(x_range) * scale_factor,
            color=COLOR_BIO, linewidth=2.2, linestyle="-", alpha=0.9)

    # ── Mean markers (dashed vertical lines) ────────────────────────
    ax.axvline(
        base_mean, color=COLOR_BASE, linewidth=1.8, linestyle="--",
        alpha=0.85, zorder=5,
    )
    ax.axvline(
        bio_mean, color=COLOR_BIO, linewidth=1.8, linestyle="--",
        alpha=0.85, zorder=5,
    )

    # ── Annotate means ──────────────────────────────────────────────
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"μ = {base_mean:.2f}",
        xy=(base_mean, y_top * 0.92),
        fontsize=10, fontweight="bold", color=COLOR_BASE,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_BASE, alpha=0.9),
    )
    ax.annotate(
        f"μ = {bio_mean:.2f}",
        xy=(bio_mean, y_top * 0.78),
        fontsize=10, fontweight="bold", color=COLOR_BIO,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_BIO, alpha=0.9),
    )

    # ── Annotate the delta ──────────────────────────────────────────
    mid_x = (base_mean + bio_mean) / 2
    ax.annotate(
        "",
        xy=(bio_mean, y_top * 0.70), xytext=(base_mean, y_top * 0.70),
        arrowprops=dict(
            arrowstyle="<->", color="#333333",
            lw=1.5, shrinkA=2, shrinkB=2,
        ),
    )
    ax.text(
        mid_x, y_top * 0.72, f"Δ = +{bio_mean - base_mean:.2f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#333333", fontfamily="serif",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFDD",
                  edgecolor="#CCCC88", alpha=0.95),
    )

    # ── Formatting ──────────────────────────────────────────────────
    ax.set_xlabel("CoT Discipline Score", fontweight="bold")
    ax.set_ylabel("Number of Cases (n = 50)", fontweight="bold")
    ax.set_xlim(0, 11)
    ax.set_xticks(range(1, 11))
    ax.xaxis.set_major_locator(mticker.FixedLocator(range(1, 11)))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.legend(
        loc="upper left", fontsize=10, frameon=True,
        framealpha=0.95, edgecolor="#CCCCCC", fancybox=True,
    )

    fig.suptitle(
        "Figure 2: Distribution of Chain-of-Thought (CoT) Discipline Scores",
        fontsize=13, fontfamily="serif", fontweight="bold", y=1.01,
    )
    ax.set_title(
        "GRPO training shifts score mass toward 9–10, reducing low-score variance\n"
        "Judge: LLaMA-4-Scout-17B-16E-Instruct  |  n = 50 noisy clinical cases",
        fontsize=9, color="#555555", pad=10,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BioGemma-Guard — Academic Figure Generation")
    print("=" * 60)

    data = load_data(DATA_PATH)
    n = len(data["base_faithfulness"])
    print(f"\n  📊 Loaded {n} evaluation cases from:")
    print(f"     {DATA_PATH.name}")

    # Print summary stats
    print(f"\n  {'Metric':<25s} {'Base':>8s} {'BioGemma':>10s} {'Δ':>8s}")
    print(f"  {'─' * 53}")
    for label, bk, gk in [
        ("Clinical Faithfulness", "base_faithfulness", "bio_faithfulness"),
        ("Differential Quality",  "base_differential", "bio_differential"),
        ("CoT Discipline",        "base_discipline",   "bio_discipline"),
    ]:
        bm = data[bk].mean()
        gm = data[gk].mean()
        delta = gm - bm
        marker = " ◀◀" if abs(delta) > 0.3 else ""
        print(f"  {label:<25s} {bm:>8.2f} {gm:>10.2f} {delta:>+8.2f}{marker}")

    print(f"\n  📈 Generating figures...\n")

    make_radar_chart(data, OUTPUT_DIR / "fig1_radar_comparison.png")
    make_discipline_distribution(data, OUTPUT_DIR / "fig2_cot_discipline_distribution.png")

    print(f"\n  📁 All figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)
