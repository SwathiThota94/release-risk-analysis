"""
features.py
-----------
Week 4: Feature engineering. Consumes the cleaned tables produced by
run_cleaning.py (data/clean/<owner>_<repo>/*.csv) and assembles the single
release-level analytical table: one row per eligible release, carrying
pre-release features (RQ1-RQ4), the post-release outcome, and the derived
target variable (Elevated_Risk).

Each feature-building function is independent and keyed on
(repository_name, release_id) -- they can be built, tested, and reviewed
separately, then merged together at the end by assemble_release_table().

Usage:
    from features import (
        build_commit_features, build_pr_features, build_timing_features,
        build_target_variable, assemble_release_table
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RQ1: Commit volume / code churn features
# ---------------------------------------------------------------------------

def build_commit_features(commits_matched: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Per-release commit volume and churn features, human commits only
    (bots excluded -- a bot-heavy commit spike shouldn't register as
    "developer activity").

    Requires commits_matched to have: release_id, is_bot, churn (or
    additions/deletions), author, commit_sha.

    Returns one row per release_id with:
        commit_count            -- number of human commits in this release's cycle
        total_churn             -- sum of additions+deletions across human commits
        avg_churn_per_commit    -- total_churn / commit_count
        distinct_files_changed  -- sum of 'files changed' if available, else NaN
    """
    df = commits_matched[
        (commits_matched["repository_name"] == repo) & (~commits_matched["is_bot"])
    ].dropna(subset=["release_id"])

    if df.empty:
        return pd.DataFrame(columns=[
            "release_id", "commit_count", "total_churn", "avg_churn_per_commit",
        ])

    agg = df.groupby("release_id").agg(
        commit_count=("commit_sha", "count"),
        total_churn=("churn", "sum") if "churn" in df.columns else ("commit_sha", "count"),
    ).reset_index()
    agg["avg_churn_per_commit"] = agg["total_churn"] / agg["commit_count"].replace(0, np.nan)

    if "files changed" in df.columns:
        files = df.groupby("release_id")["files changed"].sum().rename("distinct_files_changed")
        agg = agg.merge(files, on="release_id", how="left")

    return agg


# ---------------------------------------------------------------------------
# RQ2: Pull-request review-depth features
# ---------------------------------------------------------------------------

def build_pr_features(prs_matched: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Per-release PR review-depth features, human PRs only (bot-authored PRs,
    e.g. automated dependency bumps, are excluded so review-depth signals
    reflect genuine human-authored changes).

    Requires prs_matched to have: release_id, is_bot, review_count,
    comment_count, is_merged, created_at, merged_at, pull_request_id.

    Returns one row per release_id with:
        pr_count                 -- number of human PRs created in this release's cycle
        avg_review_count         -- mean review_count per PR (thoroughness proxy)
        pct_prs_with_review      -- share of PRs with at least 1 review
        avg_comment_count        -- mean comment_count per PR (discussion depth)
        pct_merged               -- share of PRs that were merged (vs closed unmerged)
        avg_time_to_merge_hours  -- mean hours between created_at and merged_at, merged PRs only
    """
    df = prs_matched[
        (prs_matched["repository_name"] == repo) & (~prs_matched["is_bot"])
    ].dropna(subset=["release_id"])

    if df.empty:
        return pd.DataFrame(columns=[
            "release_id", "pr_count", "avg_review_count", "pct_prs_with_review",
            "avg_comment_count", "pct_merged", "avg_time_to_merge_hours",
        ])

    def _agg(group):
        n = len(group)
        merged = group[group["is_merged"] == True] if "is_merged" in group.columns else group.iloc[0:0]  # noqa: E712
        time_to_merge_hours = np.nan
        if not merged.empty and "merged_at" in merged.columns:
            deltas = (merged["merged_at"] - merged["created_at"]).dt.total_seconds() / 3600
            time_to_merge_hours = deltas.mean()
        return pd.Series({
            "pr_count": n,
            "avg_review_count": group["review_count"].mean() if "review_count" in group.columns else np.nan,
            "pct_prs_with_review": (group["review_count"] > 0).mean() if "review_count" in group.columns else np.nan,
            "avg_comment_count": group["comment_count"].mean() if "comment_count" in group.columns else np.nan,
            "pct_merged": group["is_merged"].mean() if "is_merged" in group.columns else np.nan,
            "avg_time_to_merge_hours": time_to_merge_hours,
        })

    return df.groupby("release_id").apply(_agg).reset_index()


# ---------------------------------------------------------------------------
# RQ4: Release timing features
# ---------------------------------------------------------------------------

def build_timing_features(cycles_df: pd.DataFrame, releases_clean: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Per-release timing features.

    Returns one row per release_id with:
        cycle_length_days   -- from cycles_df, directly (RQ4's primary feature)
        is_prerelease       -- from releases_clean's prerelease_flag (kept as a
                                covariate; Kubernetes retains prereleases by
                                default per the project's inclusion criteria)
        release_sequence    -- this release's ordinal position in the repo's
                                release history (1st, 2nd, 3rd... eligible release)
    """
    cy = cycles_df[cycles_df["repository_name"] == repo][["release_id", "cycle_length_days"]].copy()

    rel = releases_clean[releases_clean["repository_name"] == repo].sort_values("release_date").copy()
    rel["release_sequence"] = range(1, len(rel) + 1)
    rel = rel[["release_id", "prerelease_flag", "release_sequence"]].rename(
        columns={"prerelease_flag": "is_prerelease"}
    )

    return cy.merge(rel, on="release_id", how="left")


# ---------------------------------------------------------------------------
# Target variable: Elevated_Risk
# ---------------------------------------------------------------------------

def build_target_variable(issues_clean: pd.DataFrame, releases_clean: pd.DataFrame, repo: str,
                            window_days: int = 30, quantile: float = 0.75,
                            train_cutoff_date: str = None) -> pd.DataFrame:
    """
    Builds the post-release outcome and the derived Elevated_Risk label, per
    the project's target-variable definition:

        Elevated_Risk(i, r) = 1 if Bugs_W(i, r) > Q_quantile(r), else 0

    where Bugs_W(i, r) is the count of qualifying bug issues (is_qualifying_bug
    == True) created within `window_days` after release i's release_date, and
    Q_quantile(r) is the `quantile`-th percentile of that same count computed
    ACROSS releases in the training portion only, to prevent leakage.

    train_cutoff_date: if given (e.g. '2025-01-01'), only releases with
        release_date before this cutoff are used to compute the threshold
        Q_quantile(r); the label is then applied to ALL releases (train and
        test) using that training-only threshold. If None, the threshold is
        computed across all releases passed in -- fine for initial
        exploration, but you should supply a real train_cutoff_date before
        final modelling to avoid leakage, per the project's stated
        methodology.

    A release is only eligible for labelling if it has fully completed its
    post-release window as of the most recent issue timestamp in the data
    (i.e. release_date + window_days <= max(issues_clean['created_at'])).
    Releases too recent to have a complete window get bugs_in_window = NaN
    and elevated_risk = NaN, and should be excluded from supervised
    modelling (but can still appear in the table for completeness).

    Returns one row per release_id with:
        release_date, bugs_in_window, elevated_risk, risk_threshold_used
    """
    issues = issues_clean[issues_clean["repository_name"] == repo]
    bug_issues = issues[issues["is_qualifying_bug"]]

    rel = releases_clean[releases_clean["repository_name"] == repo][["release_id", "release_date"]].copy()
    rel["window_end"] = rel["release_date"] + pd.Timedelta(days=window_days)

    max_issue_date = issues["created_at"].max()
    rel["window_complete"] = rel["window_end"] <= max_issue_date

    counts = []
    for _, row in rel.iterrows():
        n_bugs = bug_issues[
            (bug_issues["created_at"] >= row["release_date"]) &
            (bug_issues["created_at"] < row["window_end"])
        ].shape[0]
        counts.append(n_bugs)
    rel["bugs_in_window"] = counts
    rel.loc[~rel["window_complete"], "bugs_in_window"] = np.nan

    # Compute the risk threshold on the training portion only, to prevent leakage.
    if train_cutoff_date is not None:
        train_mask = (rel["release_date"] < pd.Timestamp(train_cutoff_date, tz="UTC")) & rel["window_complete"]
    else:
        train_mask = rel["window_complete"]

    threshold = rel.loc[train_mask, "bugs_in_window"].quantile(quantile)

    rel["risk_threshold_used"] = threshold
    rel["elevated_risk"] = np.where(
        rel["window_complete"],
        (rel["bugs_in_window"] > threshold).astype(float),
        np.nan,
    )

    return rel[["release_id", "release_date", "bugs_in_window", "elevated_risk",
                "risk_threshold_used", "window_complete"]]


# ---------------------------------------------------------------------------
# Assemble everything into the single release-level table
# ---------------------------------------------------------------------------

def assemble_release_table(repo: str, releases_clean: pd.DataFrame, cycles_df: pd.DataFrame,
                             commits_matched: pd.DataFrame, prs_matched: pd.DataFrame,
                             turnover_features: pd.DataFrame, issues_clean: pd.DataFrame,
                             window_days: int = 30, train_cutoff_date: str = None) -> pd.DataFrame:
    """
    Merges every feature domain (RQ1-RQ4) plus the target variable into one
    release-level table, keyed on release_id. Only releases with a valid
    cycle_start (i.e. present in cycles_df) are included, since releases
    without a prior release have no pre-release window to build features
    from.

    Returns one row per eligible release with:
        repository_name, release_id, release_date, cycle_length_days,
        is_prerelease, release_sequence,
        commit_count, total_churn, avg_churn_per_commit,
        pr_count, avg_review_count, pct_prs_with_review, avg_comment_count,
        pct_merged, avg_time_to_merge_hours,
        distinct_contributors, first_time_contributor_count,
        first_time_contributor_share, top_contributor_share,
        bugs_in_window, elevated_risk, risk_threshold_used
    """
    timing = build_timing_features(cycles_df, releases_clean, repo)
    commit_feats = build_commit_features(commits_matched, repo)
    pr_feats = build_pr_features(prs_matched, repo)
    target = build_target_variable(issues_clean, releases_clean, repo,
                                     window_days=window_days, train_cutoff_date=train_cutoff_date)

    table = timing.merge(commit_feats, on="release_id", how="left")
    table = table.merge(pr_feats, on="release_id", how="left")
    if turnover_features is not None and not turnover_features.empty and 'release_id' in turnover_features.columns:
        table = table.merge(turnover_features, on="release_id", how="left")
    table = table.merge(target.drop(columns=["release_date"]), on="release_id", how="left")

    table.insert(0, "repository_name", repo)

    # Releases with no matched commit/PR activity get 0, not NaN, for count
    # columns (0 is a meaningful value here -- "no activity observed" -- as
    # opposed to a missing measurement).
    for col in ["commit_count", "total_churn", "pr_count", "distinct_contributors",
                "first_time_contributor_count"]:
        if col in table.columns:
            table[col] = table[col].fillna(0)

    return table
