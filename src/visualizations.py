"""
visualizations.py  (lives in src/)
------------------------------------
Week 5 (Descriptive and Diagnostic Analytics): additional visualizations,
complementing correlation_analysis.py. Uses ONLY the 'train' split, for the
same reason correlation_analysis.py does -- keeps validation/test untouched
for final evaluation.

Produces, under data/analysis/:
    risky_vs_normal_boxplots.png   -- distribution of top features, split by elevated_risk
    trend_over_time.png            -- elevated-risk rate and bug counts by year, per repository
    repository_profile.png         -- side-by-side summary bar charts per repository

Usage:
    python src/visualizations.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Features worth visually comparing between risky and normal releases --
# chosen to cover each research question, not just whatever ranked highest
# in the correlation table, so RQ1-RQ4 are each represented even where the
# correlation was weak or inconclusive.
COMPARISON_FEATURES = [
    "commit_count",              # RQ1
    "pr_count",                  # RQ2
    "median_time_to_merge_hours",# RQ2
    "first_time_contributor_share",  # RQ3
    "top_contributor_share",     # RQ3
    "cycle_length_days",         # RQ4
    "release_sequence",          # historical / repository-drift
    "open_bugs_at_release",      # issue backlog
]


def plot_risky_vs_normal(train: pd.DataFrame):
    cols = [c for c in COMPARISON_FEATURES if c in train.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        normal = train.loc[train[TARGET_COL] == 0, col].dropna()
        risky = train.loc[train[TARGET_COL] == 1, col].dropna()
        ax.boxplot([normal, risky], tick_labels=["Not risky", "Elevated risk"], showfliers=False)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions: Elevated-Risk vs. Not-Risky Releases (train split)", fontsize=13)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "risky_vs_normal_boxplots.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_trend_over_time(train: pd.DataFrame):
    df = train.copy()
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["year"] = df["release_date"].dt.year

    yearly = df.groupby(["year", "repository_name"]).agg(
        elevated_risk_rate=(TARGET_COL, "mean"),
        avg_bugs_in_window=("bugs_in_window", "mean"),
        n_releases=(TARGET_COL, "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for repo_name, g in yearly.groupby("repository_name"):
        g = g.sort_values("year")
        axes[0].plot(g["year"], g["elevated_risk_rate"], marker="o", label=repo_name)
        axes[1].plot(g["year"], g["avg_bugs_in_window"], marker="o", label=repo_name)

    axes[0].set_title("Elevated-Risk Rate by Year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Share of releases labeled elevated risk")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Average Post-Release Bug Count by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mean bugs_in_window")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Trends Over Time, by Repository (train split)", fontsize=13)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "trend_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print("\n[NOTE] If a repository's elevated-risk rate or average bug count trends sharply "
          "up or down over time, that reflects the repository-level drift already identified "
          "during train/validation/test split analysis -- see the leakage/target-variable "
          "documentation for the fuller discussion.")


def plot_repository_profile(train: pd.DataFrame):
    profile = train.groupby("repository_name").agg(
        n_releases=(TARGET_COL, "count"),
        elevated_risk_rate=(TARGET_COL, "mean"),
        avg_commit_count=("commit_count", "mean"),
        avg_pr_count=("pr_count", "mean"),
        avg_distinct_contributors=("distinct_contributors", "mean"),
        avg_cycle_length_days=("cycle_length_days", "mean"),
    ).reset_index()

    metrics = ["elevated_risk_rate", "avg_commit_count", "avg_pr_count",
               "avg_distinct_contributors", "avg_cycle_length_days"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric in zip(axes, metrics):
        ax.bar(profile["repository_name"], profile[metric], color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Repository Profile Summary (train split)", fontsize=13)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "repository_profile.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    profile.to_csv(ANALYSIS_DIR / "repository_profile.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'repository_profile.csv'}")
    print("\nRepository profile summary:")
    print(profile.to_string(index=False))


def plot_open_bugs_sign_flip(train: pd.DataFrame):
    """
    Targeted follow-up chart for a specific finding from hypothesis testing:
    open_bugs_at_release predicts elevated_risk in OPPOSITE directions
    depending on repository (positive for Airflow and Kubernetes, negative
    for VS Code, all three statistically significant). This deserves its
    own chart rather than being buried in the general risky-vs-normal
    comparison, since it's one of the strongest pieces of evidence for RQ5
    in the whole analysis.
    """
    col = "open_bugs_at_release"
    if col not in train.columns:
        print(f"  [skip] {col} not found -- cannot build sign-flip chart")
        return

    repos = sorted(train["repository_name"].unique())
    fig, axes = plt.subplots(1, len(repos), figsize=(4.5 * len(repos), 4.5))
    if len(repos) == 1:
        axes = [axes]

    for ax, repo_name in zip(axes, repos):
        g = train[train["repository_name"] == repo_name]
        normal = g.loc[g[TARGET_COL] == 0, col].dropna()
        risky = g.loc[g[TARGET_COL] == 1, col].dropna()
        bp = ax.boxplot([normal, risky], tick_labels=["Not risky", "Elevated risk"], showfliers=False,
                         patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(repo_name, fontsize=11)
        ax.set_ylabel("open_bugs_at_release" if repo_name == repos[0] else "")

    fig.suptitle("Open Bug Backlog at Release, by Risk Status and Repository\n"
                  "(direction of effect reverses between VS Code and the other two repositories)",
                  fontsize=12)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "open_bugs_sign_flip.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found -- run run_features.py and preprocess_for_modeling.py first")
        return

    df = pd.read_csv(INPUT_PATH)
    if "split" not in df.columns:
        print("[error] No 'split' column found -- re-run run_features.py with the updated script.")
        return

    train = df[df["split"] == "train"].copy()
    print(f"Loaded {len(df)} total releases; using {len(train)} 'train' split releases for visualization.\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    plot_risky_vs_normal(train)
    plot_trend_over_time(train)
    plot_repository_profile(train)
    plot_open_bugs_sign_flip(train)


if __name__ == "__main__":
    main()
