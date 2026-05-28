"""
Visualization module — generates 5 analysis charts.

Output (saved to output/figures/):
  1. label_distribution.png    — LLM label counts across all repos
  2. signals_by_category.png   — box plots of stars, contributors, repo age,
                                  and CI presence rate per LLM label
  3. confusion_matrix_bert.png — BERT confusion matrix heatmap from JSON
  4. model_comparison.png      — BERT vs baseline on Acc/P/R/F1
  5. feature_correlations.png  — Pearson correlation heatmap of numeric signals

Data sources:
  - data/processed/repositories_clean.csv   (z-scored numerics + flags)
  - data/labeled/repositories_labeled.csv   (LLM labels)
  - output/metrics/evaluation_results.json  (BERT metrics + confusion matrix)
  - output/metrics/model_comparison.csv     (head-to-head comparison table)
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEAN_PATH     = Path("data/processed/repositories_clean.csv")
LABELED_PATH   = Path("data/labeled/repositories_labeled.csv")
EVAL_JSON      = Path("output/metrics/evaluation_results.json")
COMPARISON_CSV = Path("output/metrics/model_comparison.csv")
FIGURES_PATH   = Path("output/figures")

LABEL_NAMES = ["intern", "junior", "senior", "lead", "template", "low_value"]
LABEL_PALETTE = {
    "intern":    "#4C72B0",
    "junior":    "#55A868",
    "senior":    "#C44E52",
    "lead":      "#8172B2",
    "template":  "#CCB974",
    "low_value": "#64B5CD",
}

DPI   = 150
STYLE = "whitegrid"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    """Merge clean features with LLM labels on full_name."""
    for p in (CLEAN_PATH, LABELED_PATH):
        if not p.exists():
            logger.error("Required file missing: %s", p)
            sys.exit(1)

    clean   = pd.read_csv(CLEAN_PATH)
    labeled = pd.read_csv(LABELED_PATH, usecols=["full_name", "label"])

    df = clean.merge(labeled, on="full_name", how="inner")
    df = df[df["label"].isin(LABEL_NAMES)].reset_index(drop=True)
    logger.info("Merged dataset: %d rows, %d cols", *df.shape)
    return df


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    path = FIGURES_PATH / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", path)


# ---------------------------------------------------------------------------
# 1. Label distribution
# ---------------------------------------------------------------------------

def plot_label_distribution(df: pd.DataFrame) -> None:
    sns.set_theme(style=STYLE, font_scale=1.1)

    counts = (
        df["label"]
        .value_counts()
        .reindex(LABEL_NAMES)
        .fillna(0)
        .astype(int)
    )
    total = counts.sum()
    pcts  = counts / total * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [LABEL_PALETTE[l] for l in counts.index]
    bars   = ax.barh(counts.index[::-1], counts.values[::-1],
                     color=colors[::-1], edgecolor="white", linewidth=0.6)

    for bar, n, pct in zip(bars, counts.values[::-1], pcts.values[::-1]):
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{n}  ({pct:.1f}%)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Number of repositories")
    ax.set_title("LLM Label Distribution across 800 Repositories",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    _save(fig, "label_distribution.png")


# ---------------------------------------------------------------------------
# 2. Signals by category
# ---------------------------------------------------------------------------

def plot_signals_by_category(df: pd.DataFrame) -> None:
    sns.set_theme(style=STYLE, font_scale=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Repository Signals by LLM Label  (numeric values are z-scores)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    order = LABEL_NAMES

    box_kws = dict(linewidth=0.8,
                   flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})

    # ── stars ──────────────────────────────────────────────────────────────
    sns.boxplot(
        data=df, x="label", y="stars", hue="label", order=order,
        palette=LABEL_PALETTE, legend=False, **box_kws, ax=axes[0, 0],
    )
    axes[0, 0].set_title("Stars")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("z-score")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # ── contributors ───────────────────────────────────────────────────────
    sns.boxplot(
        data=df, x="label", y="contributors_count", hue="label", order=order,
        palette=LABEL_PALETTE, legend=False, **box_kws, ax=axes[0, 1],
    )
    axes[0, 1].set_title("Contributors")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("z-score")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # ── repo age ───────────────────────────────────────────────────────────
    sns.boxplot(
        data=df, x="label", y="repo_age_days", hue="label", order=order,
        palette=LABEL_PALETTE, legend=False, **box_kws, ax=axes[1, 0],
    )
    axes[1, 0].set_title("Repository Age")
    axes[1, 0].set_xlabel("LLM label")
    axes[1, 0].set_ylabel("z-score")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # ── CI presence rate (binary → grouped bar) ────────────────────────────
    ci_rate = (
        df.groupby("label")["has_ci"]
        .mean()
        .reindex(order)
        .fillna(0)
        .mul(100)
    )
    bar_colors = [LABEL_PALETTE[l] for l in ci_rate.index]
    axes[1, 1].bar(ci_rate.index, ci_rate.values,
                   color=bar_colors, edgecolor="white", linewidth=0.6)
    for i, (label, val) in enumerate(ci_rate.items()):
        axes[1, 1].text(i, val + 1.5, f"{val:.0f}%", ha="center", fontsize=9)
    axes[1, 1].set_title("CI/CD Presence Rate")
    axes[1, 1].set_xlabel("LLM label")
    axes[1, 1].set_ylabel("% of repos with CI")
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, "signals_by_category.png")


# ---------------------------------------------------------------------------
# 3. BERT confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix_bert() -> None:
    if not EVAL_JSON.exists():
        logger.warning(
            "Skipping confusion_matrix_bert.png — %s not found. "
            "Run notebooks/bert_training.ipynb on Colab first.", EVAL_JSON,
        )
        return

    with open(EVAL_JSON) as f:
        data = json.load(f)

    if "confusion_matrix" not in data:
        logger.warning(
            "Skipping confusion_matrix_bert.png — 'confusion_matrix' key missing "
            "from %s. Re-run the Colab notebook (save-metrics cell now includes it).",
            EVAL_JSON,
        )
        return

    cm      = np.array(data["confusion_matrix"])
    labels  = data.get("label_names", LABEL_NAMES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.set_theme(style="white", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("DistilBERT — Test Set Confusion Matrix",
                 fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="white", ax=axes[0],
    )
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    # Row-normalised (diagonal = per-class recall)
    annot_norm = np.array([[f"{v:.2f}" for v in row] for row in cm_norm])
    sns.heatmap(
        cm_norm, annot=annot_norm, fmt="", cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="white", ax=axes[1],
    )
    axes[1].set_title("Row-normalised  (diagonal = recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, "confusion_matrix_bert.png")


# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison() -> None:
    if not COMPARISON_CSV.exists():
        logger.warning(
            "Skipping model_comparison.png — %s not found. "
            "Run src/evaluation.py first.", COMPARISON_CSV,
        )
        return

    sns.set_theme(style=STYLE, font_scale=1.1)

    comp    = pd.read_csv(COMPARISON_CSV)
    overall = comp[comp["scope"] == "overall"].copy()

    name_map = {
        "baseline_logistic_regression": "Baseline (LR)",
        "distilbert-base-uncased":      "DistilBERT",
    }
    metrics = ["accuracy", "precision", "recall", "f1"]
    records = []
    for _, row in overall.iterrows():
        model_label = name_map.get(row["model"], row["model"])
        for m in metrics:
            records.append({
                "Model":  model_label,
                "Metric": m.capitalize(),
                "Score":  float(row[m]),
            })
    long_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    model_palette = {"Baseline (LR)": "#A8C5E2", "DistilBERT": "#2B6CB0"}
    sns.barplot(
        data=long_df, x="Metric", y="Score", hue="Model",
        palette=model_palette, edgecolor="white", linewidth=0.6,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Model Comparison — Baseline vs DistilBERT  (weighted averages, test set)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Model", loc="lower right")
    sns.despine()
    fig.tight_layout()
    _save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# 5. Feature correlations
# ---------------------------------------------------------------------------

def plot_feature_correlations(df: pd.DataFrame) -> None:
    sns.set_theme(style="white", font_scale=1.05)

    signal_cols = {
        "stars":              "Stars",
        "forks":              "Forks",
        "contributors_count": "Contributors",
        "releases_count":     "Releases",
        "readme_length":      "README length",
        "open_issues_count":  "Open issues",
        "repo_age_days":      "Repo age (days)",
        "days_since_push":    "Days since push",
    }

    sub  = df[list(signal_cols.keys())].rename(columns=signal_cols)
    corr = sub.corr(method="pearson")

    # Lower-triangle only
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap  (z-scored numeric signals)",
                 fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "feature_correlations.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def visualize() -> None:
    df = _load_data()

    plot_label_distribution(df)
    plot_signals_by_category(df)
    plot_confusion_matrix_bert()
    plot_model_comparison()
    plot_feature_correlations(df)

    logger.info("All charts saved to %s", FIGURES_PATH)


if __name__ == "__main__":
    visualize()
