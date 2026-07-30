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
    Per-release commit volume features, human commits only (bots excluded --
    a bot-heavy commit spike shouldn't register as "developer activity").

    IMPORTANT SCHEMA NOTE: 'additions', 'deletions', 'total changes', and
    'files changed' were confirmed 100% missing at the collection source for
    Kubernetes (GitHub's list-commits endpoint does not include per-commit
    stats without an additional per-commit API call, which was not made
    during collection). This was NOT caught by earlier NaN checks, because
    the original code computed churn as
    additions.fillna(0) + deletions.fillna(0) -- when both inputs are
    entirely NaN, this silently evaluates to a real-looking 0 for every row,
    rather than surfacing as missing data. Verified directly: total_churn,
    avg_churn_per_commit, and distinct_files_changed were a constant 0
    across all 357 releases in the assembled feature table.

    Given this, RQ1 ("commit volume AND code churn") is reduced to a
    volume-only signal: commit_count is genuine (built from real commit_sha
    rows), but churn/size cannot currently be measured. This function no
    longer computes fabricated churn features. If additions/deletions/files
    changed are ever populated by a fixed collectors.py, re-add churn
    computation guarded by an actual non-null check (not just column
    presence) -- see the commented-out block below for reference.

    Requires commits_matched to have: release_id, is_bot, commit_sha.

    Returns one row per release_id with:
        commit_count -- number of human commits in this release's cycle
    """
    df = commits_matched[
        (commits_matched["repository_name"] == repo) & (~commits_matched["is_bot"])
    ].dropna(subset=["release_id"])

    if df.empty:
        return pd.DataFrame(columns=["release_id", "commit_count"])

    agg = df.groupby("release_id").agg(
        commit_count=("commit_sha", "count"),
    ).reset_index()

    # --- Reference for re-enabling churn, IF the raw data is ever fixed ---
    # for col in ("additions", "deletions", "total changes", "files changed"):
    #     if col in df.columns and df[col].notna().any():
    #         ... compute real churn/size features here ...
    # A column merely being present is not sufficient (that was the bug);
    # confirm .notna().any() is True before trusting it.

    return agg


# ---------------------------------------------------------------------------
# RQ2: Pull-request review-depth features
# ---------------------------------------------------------------------------

def build_pr_features(prs_matched: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Per-release PR features, human PRs only (bot-authored PRs excluded).

    IMPORTANT SCHEMA NOTE: 'review count', 'comment count', 'changed files',
    'additions', and 'deletions' were ALL confirmed 100% missing at the
    collection source for both primary repositories (GitHub's list-PRs
    endpoint does not include these -- populating any of them requires a
    separate per-PR API call, not feasible within this project's time
    budget). This was discovered in two stages: review/comment counts
    first, then PR size fields when an attempted size-adjusted proxy also
    came back entirely null.

    The ONLY genuinely populated PR fields across both repos are
    timestamps (created_at, merged_at, closed_at) and merge status/author.
    RQ2 (pull-request review depth) is therefore reduced to a timing-only
    signal: how many PRs were opened, what share merged, and how long
    merging took. This is a real, documented limitation -- there is
    currently no reliable way to measure review depth, discussion volume,
    or change size from this dataset. Report this explicitly rather than
    fabricating a proxy from fields that don't exist.

    Requires prs_matched to have: release_id, is_bot, is_merged, created_at,
    merged_at, pull_request_id.

    Returns one row per release_id with:
        pr_count                 -- number of human PRs created in this release's cycle
        pct_merged                -- share of PRs that were merged (vs closed unmerged)
        avg_time_to_merge_hours   -- mean hours between created_at and merged_at, merged PRs only
        median_time_to_merge_hours -- median version of the same, less sensitive to outlier PRs
    """
    df = prs_matched[
        (prs_matched["repository_name"] == repo) & (~prs_matched["is_bot"])
    ].dropna(subset=["release_id"])

    if df.empty:
        return pd.DataFrame(columns=[
            "release_id", "pr_count", "pct_merged",
            "avg_time_to_merge_hours", "median_time_to_merge_hours",
        ])

    def _agg(group):
        n = len(group)
        merged = group[group["is_merged"] == True] if "is_merged" in group.columns else group.iloc[0:0]  # noqa: E712
        avg_hours = np.nan
        median_hours = np.nan
        if not merged.empty and "merged_at" in merged.columns:
            deltas = (merged["merged_at"] - merged["created_at"]).dt.total_seconds() / 3600
            avg_hours = deltas.mean()
            median_hours = deltas.median()
        return pd.Series({
            "pr_count": n,
            "pct_merged": group["is_merged"].mean() if "is_merged" in group.columns else np.nan,
            "avg_time_to_merge_hours": avg_hours,
            "median_time_to_merge_hours": median_hours,
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


def build_historical_features(target_df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Historical variables (previously missing from Week 4): for each release,
    computes a rolling average of bugs_in_window using ONLY prior releases
    whose own post-release window had already fully completed as of the
    current release's date. This is deliberately leakage-safe: a release
    from early in a repository's history where no prior release yet
    qualifies gets prior_releases_avg_bugs = NaN and prior_releases_count = 0,
    rather than silently looking ahead at releases that hadn't happened yet
    or whose outcome wasn't yet knowable.

    A prior release j qualifies to inform release i's historical average if:
        j.release_date + window_days <= i.release_date
    i.e. release j's own 30-day post-release window had already closed
    before release i even happened -- so its bug count was genuinely
    "known" information at the time of release i, not a future peek.

    Returns one row per release_id with:
        prior_releases_avg_bugs   -- mean bugs_in_window across qualifying prior releases
        prior_releases_count      -- how many prior releases qualified (0 for early releases)
    """
    df = target_df.sort_values("release_date").reset_index(drop=True)
    hist_avg = []
    hist_count = []
    for i, row in df.iterrows():
        eligible = df[df["release_date"] + pd.Timedelta(days=window_days) <= row["release_date"]]
        if len(eligible) > 0:
            hist_avg.append(eligible["bugs_in_window"].mean())
            hist_count.append(len(eligible))
        else:
            hist_avg.append(np.nan)
            hist_count.append(0)
    df["prior_releases_avg_bugs"] = hist_avg
    df["prior_releases_count"] = hist_count
    return df[["release_id", "prior_releases_avg_bugs", "prior_releases_count"]]


# ---------------------------------------------------------------------------
# Assemble everything into the single release-level table
# ---------------------------------------------------------------------------

def assemble_release_table(repo: str, releases_clean: pd.DataFrame, cycles_df: pd.DataFrame,
                             commits_matched: pd.DataFrame, prs_matched: pd.DataFrame,
                             turnover_features: pd.DataFrame, issues_clean: pd.DataFrame,
                             issues_open_at_release_df: pd.DataFrame = None,
                             window_days: int = 30, train_cutoff_date: str = None) -> pd.DataFrame:
    """
    Merges every feature domain (RQ1-RQ4) plus the target variable into one
    release-level table, keyed on release_id. Only releases with a valid
    cycle_start (i.e. present in cycles_df) are included, since releases
    without a prior release have no pre-release window to build features
    from.

    issues_open_at_release_df: output of cleaning.issues_open_at_release()
        (open_issues_at_release, open_bugs_at_release per release_id). This
        was built during Week 3 cleaning but was NOT being merged into the
        final feature table -- fixed here. Pass None to skip (columns will
        simply be absent, not silently zero).

    Returns one row per eligible release with:
        repository_name, release_id, release_date, cycle_length_days,
        is_prerelease, release_sequence,
        commit_count,
        pr_count, pct_merged, avg_time_to_merge_hours, median_time_to_merge_hours,
        distinct_contributors, first_time_contributor_count,
        first_time_contributor_share, top_contributor_share,
        open_issues_at_release, open_bugs_at_release,
        prior_releases_avg_bugs, prior_releases_count,
        bugs_in_window, elevated_risk, risk_threshold_used
    """
    timing = build_timing_features(cycles_df, releases_clean, repo)
    commit_feats = build_commit_features(commits_matched, repo)
    pr_feats = build_pr_features(prs_matched, repo)
    target = build_target_variable(issues_clean, releases_clean, repo,
                                     window_days=window_days, train_cutoff_date=train_cutoff_date)
    historical_feats = build_historical_features(target, window_days=window_days)

    table = timing.merge(commit_feats, on="release_id", how="left")
    table = table.merge(pr_feats, on="release_id", how="left")
    table = table.merge(turnover_features, on="release_id", how="left")
    if issues_open_at_release_df is not None and not issues_open_at_release_df.empty:
        table = table.merge(issues_open_at_release_df, on="release_id", how="left")
    table = table.merge(target, on="release_id", how="left")
    table = table.merge(historical_feats, on="release_id", how="left")

    table.insert(0, "repository_name", repo)

    # Releases with no matched commit/PR activity, no open-issue backlog, or
    # no qualifying prior releases get 0, not NaN, for count columns (0 is a
    # meaningful value here -- "no activity/history observed" -- as opposed
    # to a missing measurement). prior_releases_avg_bugs is deliberately NOT
    # filled here -- "no prior releases yet" means the average is genuinely
    # unknown, not zero; it's handled by preprocess_for_modeling.py's
    # imputation step instead, gated on prior_releases_count > 0.
    for col in ["commit_count", "pr_count", "distinct_contributors",
                "first_time_contributor_count", "open_issues_at_release",
                "open_bugs_at_release", "prior_releases_count"]:
        if col in table.columns:
            table[col] = table[col].fillna(0)

    return table
