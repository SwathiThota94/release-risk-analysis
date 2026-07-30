"""
preprocess_for_modeling.py  (lives in src/)
--------------------------------------------
Final preprocessing step between feature engineering (Week 4) and modelling
(Week 5). Consumes data/features/all_repos_release_level.csv (produced by
run_features.py) and produces a model-ready table:

    data/features/model_ready_release_level.csv

Handles three things the feature-engineering step deliberately left for
this stage:

1. UNLABELED ROWS: releases where elevated_risk is NaN (their post-release
   window hasn't fully completed yet) are dropped -- you cannot train on a
   row with no label. Reported separately, not silently discarded.

2. MISSING RATIO/AVERAGE FEATURES: features like avg_churn_per_commit,
   avg_review_count, pct_prs_with_review, avg_time_to_merge_hours,
   first_time_contributor_share, and top_contributor_share are NaN whenever
   there was no underlying activity to average over (e.g. a release with 0
   PRs has no avg_review_count). Filling these with 0 would be WRONG -- "0%
   of PRs got reviewed" and "there were no PRs at all" are different
   situations that a model needs to distinguish. Instead:
     - An explicit had_commit_activity / had_pr_activity /
       had_contributor_activity / has_prior_release binary flag is added
       for each gated group, so the model can see whether the ratio is a
       real measurement or a filled-in placeholder.
     - The ratio itself is imputed using that REPOSITORY's OWN median
       (not blended across repos), computed only from rows where real
       activity existed.

3. CATEGORICAL ENCODING: repository_name is one-hot encoded (dropping one
   level, since you only have two primary repositories, to avoid perfect
   multicollinearity in logistic regression). is_prerelease is coerced to
   an explicit 0/1 integer.

4. MULTICOLLINEARITY FIX (added after this was found to corrupt scenario
   analysis / SHAP explanations downstream): open_bugs_at_release and
   prior_releases_avg_bugs are both correlated at r=0.85-0.98 with
   repository identity (repo_microsoft/vscode) and with each other,
   because VS Code's absolute backlog/bug-history scale is thousands of
   issues larger than the other two repos'. This was fixed in
   explanatory_regression.py by repo-normalizing these two features, but
   that fix was never carried over to the predictive-modelling feature
   set -- an oversight, since multicollinearity was treated as only an
   interpretability concern for the explanatory model, not something that
   would also affect single-feature scenario analysis on the predictive
   model. Fixed here by adding _repo_z (repository-wise z-scored) versions
   of both columns, computed using each repository's TRAINING-split mean/
   std only (to avoid leakage into validation/test), then applied to all
   three splits. The original raw columns are KEPT (not replaced) for
   human-readable display in scenario analysis; downstream modelling
   scripts should use the _repo_z versions instead of the raw columns.
   Also: prior_releases_count (r=0.999 with release_sequence) and
   open_issues_at_release (r=0.965 with open_bugs_at_release) are
   redundant with features already in the model and should be excluded
   from the feature set downstream (still present in this table for
   reference, just not used as model inputs).

Usage:
    python src/preprocess_for_modeling.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
INPUT_PATH = FEATURES_DIR / "all_repos_release_level.csv"
OUTPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

# Each gated group: (indicator column name, gating count column, [ratio columns to impute])
GATED_GROUPS = [
    # avg_churn_per_commit and distinct_files_changed were removed from
    # build_commit_features() -- they were silently fabricated zeros from
    # entirely-missing source columns (additions/deletions/files changed),
    # not real measurements. had_commit_activity itself is still a
    # meaningful standalone feature (did any human commit happen in this
    # release's cycle at all?), so the group stays with an empty ratio list.
    ("had_commit_activity", "commit_count", []),
    ("had_pr_activity", "pr_count", [
        "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    ]),
    ("had_contributor_activity", "distinct_contributors", [
        "first_time_contributor_share", "top_contributor_share",
    ]),
    ("has_prior_release_history", "prior_releases_count", ["prior_releases_avg_bugs"]),
]


def add_activity_flags_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for indicator_col, gate_col, ratio_cols in GATED_GROUPS:
        if gate_col not in df.columns:
            continue
        df[indicator_col] = (df[gate_col] > 0).astype(int)

        for ratio_col in ratio_cols:
            if ratio_col not in df.columns:
                continue
            # Per-repository median, computed only from rows with real activity
            medians = (
                df.loc[df[indicator_col] == 1]
                .groupby("repository_name")[ratio_col]
                .median()
            )
            fill_values = df["repository_name"].map(medians)
            df[ratio_col] = df[ratio_col].fillna(fill_values)
            # Fallback: if a repo has NO rows with real activity at all
            # (medians itself is NaN), fall back to the global median across
            # both repos rather than leaving NaN.
            df[ratio_col] = df[ratio_col].fillna(df[ratio_col].median())

    # cycle_length_days is a special case: NaN only for each repo's very
    # first eligible release (no prior release to measure a cycle from).
    if "cycle_length_days" in df.columns:
        df["has_prior_release"] = df["cycle_length_days"].notna().astype(int)
        medians = df.groupby("repository_name")["cycle_length_days"].transform("median")
        df["cycle_length_days"] = df["cycle_length_days"].fillna(medians)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # One-hot encode repository_name, dropping one level (only 2 repos here,
    # so this becomes a single binary column) to avoid multicollinearity.
    dummies = pd.get_dummies(df["repository_name"], prefix="repo", drop_first=True)
    dummies = dummies.astype(int)
    df = pd.concat([df, dummies], axis=1)

    if "is_prerelease" in df.columns:
        df["is_prerelease"] = df["is_prerelease"].astype(int)

    return df


def add_repo_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds repo-wise z-scored versions of the two backlog features found
    confounded with repository identity (see module docstring, item 4).
    Mean/std are computed using ONLY each repository's 'train' split rows,
    then applied to that repository's train/validation/test rows alike --
    consistent with how the main StandardScaler elsewhere in this project
    is fit on training data only, to avoid leakage.
    """
    df = df.copy()
    cols_to_normalize = ["open_bugs_at_release", "prior_releases_avg_bugs"]

    for col in cols_to_normalize:
        if col not in df.columns:
            continue
        new_col = f"{col}_repo_z"
        df[new_col] = np.nan
        for repo_name, repo_group in df.groupby("repository_name"):
            train_rows = repo_group[repo_group["split"] == "train"]
            mean = train_rows[col].mean()
            std = train_rows[col].std()
            if pd.isna(std) or std == 0:
                std = 1  # defensive fallback, avoids divide-by-zero for a degenerate repo/column
            z = (repo_group[col] - mean) / std
            df.loc[repo_group.index, new_col] = z
        print(f"  Added {new_col} (repo-wise z-score, train-split mean/std)")

    return df


def main():
    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found -- run run_features.py first")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} releases from {INPUT_PATH}")

    # --- 1. Drop unlabeled rows ---
    n_before = len(df)
    unlabeled = df["elevated_risk"].isna()
    n_unlabeled = int(unlabeled.sum())
    df = df[~unlabeled].copy()
    print(f"Dropped {n_unlabeled} unlabeled releases (window not yet complete); {len(df)} remain")

    if df.empty:
        print("[error] No labeled releases remain -- nothing to write")
        return

    # --- 2. Missing ratio-feature handling ---
    df = add_activity_flags_and_impute(df)
    remaining_na = df.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if not remaining_na.empty:
        print("\n[warning] Columns still containing NaN after imputation:")
        print(remaining_na.to_string())
    else:
        print("\nNo remaining NaNs after imputation.")

    # --- 3. Repo-wise normalization for the multicollinear backlog features ---
    print("\nAdding repo-normalized backlog features (multicollinearity fix):")
    df = add_repo_normalized_features(df)

    # --- 4. Categorical encoding ---
    df = encode_categoricals(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nModel-ready table written to {OUTPUT_PATH} ({len(df)} releases, {df.shape[1]} columns)")
    print("\nLabel balance by repository:")
    print(df.groupby("repository_name")["elevated_risk"].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()
