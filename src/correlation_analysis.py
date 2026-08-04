"""
correlation_analysis.py  (lives in src/)
------------------------------------------
Week 5 (Descriptive and Diagnostic Analytics): correlation analysis.

Uses ONLY the 'train' split from model_ready_release_level.csv -- validation
and test are deliberately left untouched at this stage, since exploring
correlations on data you'll later evaluate against risks unconsciously
shaping modelling decisions around it (a soft form of leakage).

Produces:
    data/analysis/feature_feature_correlation.csv   -- full correlation matrix
    data/analysis/feature_target_correlation.csv    -- each feature's correlation with elevated_risk, sorted
    data/analysis/high_correlation_pairs.csv         -- feature pairs with |r| > 0.7 (multicollinearity flags)
    data/analysis/correlation_heatmap.png            -- visualization

Usage:
    python src/correlation_analysis.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Columns deliberately excluded from the feature set:
#   - identifiers/dates: release_id, repository_name, release_date
#   - the split label itself
#   - bugs_in_window / risk_threshold_used / window_complete: these are the
#     raw ingredients used to CONSTRUCT the target, not predictors -- including
#     them here would trivially "predict" the target from itself.
#   - is_prerelease: confirmed constant (0 for every row, since prereleases
#     are excluded from cycle-building) -- zero variance, correlation is
#     undefined/meaningless for a constant column.
#   - open_issues_at_release, prior_releases_count: redundant with features
#     already included (r=0.965 with open_bugs_at_release; r=0.999 with
#     release_sequence respectively) -- dropped from the modeling feature
#     set for the same reason in every modelling script.
#   - open_bugs_at_release, prior_releases_avg_bugs (RAW versions): superseded
#     by their repo-normalized _repo_z counterparts, which resolved a severe
#     confound with repository identity (see preprocess_for_modeling.py).
#     Excluded here so this report reflects the ACTUAL feature set used
#     downstream, rather than mixing superseded and corrected versions of
#     the same underlying signal (which would also trivially show up as a
#     top "highly correlated pair," since one is derived from the other).
EXCLUDE_COLS = {
    "release_id", "repository_name", "release_date", "split",
    "bugs_in_window", "risk_threshold_used", "window_complete",
    "is_prerelease", TARGET_COL,
    "open_issues_at_release", "prior_releases_count",
    "open_bugs_at_release", "prior_releases_avg_bugs",
}

HIGH_CORR_THRESHOLD = 0.7


def main():
    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found -- run run_features.py and preprocess_for_modeling.py first")
        return

    df = pd.read_csv(INPUT_PATH)
    if "split" not in df.columns:
        print("[error] No 'split' column found -- re-run run_features.py with the updated script that adds it.")
        return

    train = df[df["split"] == "train"].copy()
    print(f"Loaded {len(df)} total releases; using {len(train)} 'train' split releases for correlation analysis.")

    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    dropped = set(feature_cols) - set(numeric_cols)
    if dropped:
        print(f"Excluding non-numeric columns from correlation analysis: {sorted(dropped)}")

    # Drop any remaining zero-variance columns defensively (correlation is
    # undefined for a constant column and would show up as NaN otherwise).
    zero_var = [c for c in numeric_cols if train[c].nunique() <= 1]
    if zero_var:
        print(f"Excluding zero-variance columns: {zero_var}")
        numeric_cols = [c for c in numeric_cols if c not in zero_var]

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Feature-to-feature correlation matrix ---
    corr_matrix = train[numeric_cols].corr(method="pearson")
    corr_matrix.to_csv(ANALYSIS_DIR / "feature_feature_correlation.csv")
    print(f"\nWrote feature-to-feature correlation matrix ({len(numeric_cols)} features) to "
          f"{ANALYSIS_DIR / 'feature_feature_correlation.csv'}")

    # --- Feature-to-target correlation: Pearson AND Spearman, with p-values ---
    # Pearson captures linear relationships; Spearman (rank-based) also
    # catches monotonic-but-non-linear relationships, which matters here
    # since several features (open_issues_at_release, prior_releases_avg_bugs)
    # are right-skewed counts rather than roughly-normal measurements.
    # P-values flag which correlations are plausibly real signal vs. likely
    # noise given the sample size -- raw correlation magnitude alone doesn't
    # tell you that.
    target_rows = []
    for col in numeric_cols:
        x = train[col].values
        y = train[TARGET_COL].values
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        target_rows.append({
            "feature": col,
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "n": len(train),
        })
    target_corr_df = pd.DataFrame(target_rows).sort_values(
        "pearson_r", key=lambda s: s.abs(), ascending=False
    )
    target_corr_df["significant_at_0.05"] = target_corr_df["pearson_p"] < 0.05
    target_corr_df.to_csv(ANALYSIS_DIR / "feature_target_correlation.csv", index=False)
    print(f"Wrote feature-to-target correlations (Pearson + Spearman + p-values) to "
          f"{ANALYSIS_DIR / 'feature_target_correlation.csv'}")
    print("\nTop features by |Pearson correlation| with elevated_risk:")
    print(target_corr_df[["feature", "pearson_r", "pearson_p", "significant_at_0.05", "spearman_r"]]
          .head(10).to_string(index=False))

    n_sig = int(target_corr_df["significant_at_0.05"].sum())
    print(f"\n{n_sig} of {len(target_corr_df)} features show a statistically significant "
          f"(p < 0.05) Pearson correlation with elevated_risk in the train split.")

    # --- Per-repository breakdown: does each feature's relationship with the
    # target hold up, reverse, or vanish within each repository individually?
    # This is a direct empirical check relevant to RQ5 (generalization) --
    # pooled correlation can mask a relationship that only holds for one
    # repository, or one that reverses sign across repositories.
    by_repo_rows = []
    for repo_name, g in train.groupby("repository_name"):
        for col in numeric_cols:
            if col.startswith("repo_"):
                continue  # a repo dummy is constant within its own subgroup -- undefined correlation
            if g[col].nunique() <= 1 or g[TARGET_COL].nunique() <= 1:
                continue  # skip pairs with no variance within this repo
            r, p = stats.pearsonr(g[col].values, g[TARGET_COL].values)
            by_repo_rows.append({
                "feature": col, "repository_name": repo_name,
                "pearson_r": r, "pearson_p": p, "n": len(g),
            })
    by_repo_df = pd.DataFrame(by_repo_rows)
    by_repo_df.to_csv(ANALYSIS_DIR / "feature_target_correlation_by_repo.csv", index=False)
    print(f"\nWrote per-repository feature-target correlations to "
          f"{ANALYSIS_DIR / 'feature_target_correlation_by_repo.csv'}")

    # A quick-scan pivot: one row per feature, one column per repo, so sign
    # flips or vanishing effects are visible at a glance without filtering.
    pivot = by_repo_df.pivot(index="feature", columns="repository_name", values="pearson_r")
    pivot.to_csv(ANALYSIS_DIR / "feature_target_correlation_by_repo_pivot.csv")
    print(f"Wrote pivoted view (feature x repository) to "
          f"{ANALYSIS_DIR / 'feature_target_correlation_by_repo_pivot.csv'}")

    # Flag features whose correlation sign disagrees across repositories --
    # direct evidence against generalization for that specific feature.
    sign_flips = []
    for feat, row in pivot.iterrows():
        vals = row.dropna()
        if len(vals) >= 2 and (vals > 0).any() and (vals < 0).any():
            sign_flips.append(feat)
    if sign_flips:
        print(f"\n[NOTE] {len(sign_flips)} feature(s) show a DIFFERENT SIGN of correlation with "
              f"elevated_risk across repositories (relationship reverses direction depending on "
              f"repository): {sign_flips}")
        print("This is directly relevant to RQ5 -- these specific features' relationships with risk "
              "do not appear to generalize across repositories in the training data.")

    # --- High feature-feature correlation pairs (multicollinearity flags) ---
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                pairs.append({"feature_1": c1, "feature_2": c2, "correlation": r})
    high_corr_df = pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False) if pairs else pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    high_corr_df.to_csv(ANALYSIS_DIR / "high_correlation_pairs.csv", index=False)
    print(f"\nWrote {len(high_corr_df)} highly correlated feature pairs (|r| >= {HIGH_CORR_THRESHOLD}) to "
          f"{ANALYSIS_DIR / 'high_correlation_pairs.csv'}")
    if not high_corr_df.empty:
        print(high_corr_df.to_string(index=False))
        print("\n[NOTE] Highly correlated feature pairs can cause instability in some models "
              "(e.g. logistic regression coefficients). Consider dropping one of each pair, "
              "or use a regularized model, when you reach Week 6.")

    # --- Heatmap visualization ---
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.5), max(6, len(numeric_cols) * 0.45)))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(numeric_cols, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature Correlation Matrix (train split only)")
    fig.tight_layout()
    heatmap_path = ANALYSIS_DIR / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote correlation heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
