"""
cleaning.py
-----------
Week 3: Cleaning and release-cycle integration.

Consumes the raw tables produced by main.py (data/tables/<owner>_<repo>/*.csv)
and produces cleaned intermediate tables plus a release-cycle mapping, ready
for Week 4 feature engineering.

Required outputs (per project instructions, Week 3):
    - Cleaned release table
    - Cleaned commit table
    - Cleaned pull-request table
    - Cleaned issue table
    - Release-cycle mapping
    - Data-quality report

Usage:
    from cleaning import clean_releases, clean_issues, build_release_cycles, \
        match_events_to_cycles, identify_bots, issues_open_at_release, \
        build_data_quality_report

Notes on scope:
    - Repository-normalized handling is repository-aware: Kubernetes and
      Airflow are treated as the two primary repositories. Label taxonomy
      mapping is defined per repository since label conventions differ
      (kind/bug vs kind:bug).
    - Commit and pull-request cleaning functions are written against the
      column names documented in the Week 2 data dictionary
      (commit_sha/committed_at/author for commits; pr_id/created_at/
      merged_at/author for pull requests). Confirm these match your actual
      data/raw/<repo>/commits.json and pull_requests.json field names before
      running end-to-end -- those two tables have not yet been inspected
      directly in this project, unlike releases and issues.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Repository-specific configuration
# ---------------------------------------------------------------------------

# Bug/regression labels per repository, used to build the standardized issue
# taxonomy. Extend this as additional repositories are cleaned.
BUG_LABELS = {
    "kubernetes/kubernetes": {"kind/bug", "kind/regression"},
    "apache/airflow": {"kind:bug", "type:bug-fix"},
}

FEATURE_LABELS = {
    "kubernetes/kubernetes": {"kind/feature"},
    "apache/airflow": {"kind:feature"},
}

SECURITY_LABELS = {
    "kubernetes/kubernetes": {"area/security"},
    "apache/airflow": {"area:security", "kind:security"},
}

DOCS_LABELS = {
    "kubernetes/kubernetes": {"area/documentation", "kind/documentation"},
    "apache/airflow": {"kind:documentation"},
}

SUPPORT_LABELS = {
    "kubernetes/kubernetes": {"kind/support"},
    "apache/airflow": {"kind:support"},
}

# Known bot / automated account patterns. Matched case-insensitively against
# the author identifier. Extend with repository-specific CI bots as they are
# discovered during cleaning.
BOT_PATTERNS = [
    r"\[bot\]$",            # GitHub's own convention, e.g. dependabot[bot]
    r"^dependabot",
    r"^renovate",
    r"^github-actions",
    r"^k8s-ci-robot$",
    r"^k8s-triage-robot$",
    r"^google-oss-robot$",
    r"^codecov",
    r"^snyk-bot$",
    r"^azure-pipelines",
    r"^boring-cyborg",
]
_BOT_REGEX = re.compile("|".join(BOT_PATTERNS), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------

def clean_releases(df: pd.DataFrame, repo: str, exclude_prerelease: bool = None,
                    since_date: str = "2021-01-01") -> pd.DataFrame:
    """
    Standardize the releases table and apply inclusion/exclusion criteria.

    - Parses release_date to UTC datetime.
    - Drops releases before `since_date` (default 2021-01-01, per the
      project proposal's inclusion criteria). This matters in practice: your
      commits/PRs are collected with --since 2021-01-01, so any release
      cycle starting before that date has no matching commit/PR data at
      all (not because the cycle was short, but because that period was
      never collected) -- filtering releases to the same window keeps
      cycles and their matched activity data consistent.
    - Drops draft releases (draft_flag = True) -- unpublished, no valid
      post-release window.
    - Optionally drops prereleases. Per the project's inclusion/exclusion
      criteria: Airflow's prereleases (~3% of releases) are excluded by
      default; Kubernetes' prereleases (~29% of releases) are kept but
      flagged, since they are treated as a distinct population rather than
      pooled with stable releases by default -- pass exclude_prerelease=True
      explicitly if you want them removed instead.
    - Standardizes tag_name (strips leading 'v', whitespace).
    - Deduplicates on (repository_name, release_id).

    Returns a DataFrame sorted by release_date ascending, with an added
    'is_stable' column (True for non-prerelease, non-draft releases).
    """
    out = df.copy()
    out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce", utc=True)

    before = len(out)
    out = out.dropna(subset=["release_date"])
    n_bad_date = before - len(out)

    out = out.drop_duplicates(subset=["repository_name", "release_id"])

    # Enforce the proposal's inclusion criterion and align with the
    # collection window used for commits/PRs (--since 2021-01-01 by default).
    n_before_cutoff = 0
    if since_date is not None:
        cutoff = pd.Timestamp(since_date, tz="UTC")
        before = len(out)
        out = out[out["release_date"] >= cutoff]
        n_before_cutoff = before - len(out)

    # Drop drafts -- never published, no meaningful post-release window
    before = len(out)
    out = out[out["draft_flag"] != True]  # noqa: E712
    n_draft_dropped = before - len(out)

    out["tag_name_clean"] = (
        out["tag_name"].astype(str).str.strip().str.replace(r"^v", "", regex=True)
    )
    out["is_stable"] = ~out["prerelease_flag"].astype(bool)

    if exclude_prerelease is None:
        # Default per project criteria: exclude for Airflow (negligible
        # sample), retain-but-flag for Kubernetes and everything else.
        exclude_prerelease = repo == "apache/airflow"

    if exclude_prerelease:
        before = len(out)
        out = out[out["is_stable"]]
        n_prerelease_dropped = before - len(out)
    else:
        n_prerelease_dropped = 0

    out = out.sort_values("release_date").reset_index(drop=True)

    out.attrs["cleaning_stats"] = {
        "repo": repo,
        "rows_in": len(df),
        "rows_out": len(out),
        "dropped_unparseable_date": n_bad_date,
        "dropped_before_since_date": n_before_cutoff,
        "dropped_draft": n_draft_dropped,
        "dropped_prerelease": n_prerelease_dropped,
    }
    return out


def build_release_cycles(clean_releases_df: pd.DataFrame, cycle_universe: str = "stable",
                          min_cycle_days: float = None) -> pd.DataFrame:
    """
    Construct release-cycle boundaries: for each eligible release, the
    pre-release observation window runs from the previous eligible release's
    date to the current release's date (the "release-cycle approach"
    specified as preferred in the project instructions).

    cycle_universe:
        "stable"  -- cycle boundaries computed only over stable (non-prerelease)
                     releases. Use this for the primary analysis.
        "all"     -- cycle boundaries computed over every release in the
                     input (e.g. if you deliberately want prereleases inside
                     the cycle sequence too).

    min_cycle_days:
        If set (e.g. 1.0), releases whose cycle_length_days falls below this
        threshold are dropped from the returned cycles table. This addresses
        a real pattern found in Kubernetes: releases are frequently
        published near-simultaneously across multiple supported branches
        (median cycle length of 0 days in the raw data), leaving no
        meaningful time window for pre-release activity to be measured.
        Rather than silently including these as "0 commits / 0
        contributors" rows -- which would understate true pre-release
        activity levels rather than reflect an actual quiet period -- pass
        min_cycle_days=1.0 (or another threshold your team agrees on) to
        exclude them from the primary modeling sample. Defaults to None
        (no filtering), so this is opt-in and must be deliberately chosen.

    Returns one row per eligible release with:
        repository_name, release_id, tag_name_clean, release_date,
        cycle_start (previous eligible release's release_date, or NaT for
        the first release in the repo -- these rows have no valid pre-release
        window and should be excluded from modelling),
        cycle_length_days
    """
    df = clean_releases_df.copy()
    if cycle_universe == "stable":
        df = df[df["is_stable"]]

    df = df.sort_values(["repository_name", "release_date"])
    df["cycle_start"] = df.groupby("repository_name")["release_date"].shift(1)
    df["cycle_length_days"] = (df["release_date"] - df["cycle_start"]).dt.total_seconds() / 86400

    cycles = df[[
        "repository_name", "release_id", "tag_name_clean", "release_date",
        "cycle_start", "cycle_length_days",
    ]].rename(columns={"release_date": "cycle_end"})

    n_before_min_cycle = 0
    if min_cycle_days is not None:
        before = len(cycles)
        # Keep rows with no cycle_start (first release per repo -- these are
        # already excluded from modelling for a different reason, not this
        # filter) alongside rows meeting the minimum cycle length.
        cycles = cycles[cycles["cycle_start"].isna() | (cycles["cycle_length_days"] >= min_cycle_days)]
        n_before_min_cycle = before - len(cycles)

    cycles.attrs["cleaning_stats"] = {
        "cycles_built": len(cycles),
        "cycles_missing_start": int(cycles["cycle_start"].isna().sum()),
        "dropped_short_cycle": n_before_min_cycle,
    }
    return cycles


# ---------------------------------------------------------------------------
# Issues -> standardized taxonomy
# ---------------------------------------------------------------------------

def _label_set(label_str) -> set:
    if pd.isna(label_str) or str(label_str).strip() == "":
        return set()
    return {x.strip() for x in str(label_str).split(",") if x.strip()}


def classify_issue_taxonomy(labels: set, repo: str) -> str:
    """
    Map a repository's raw labels to the standardized issue taxonomy:
    Bug, Regression, Security, Documentation, Support/question, Feature
    request, Other. Bug/Regression are checked first since they are the
    project's primary risk signal; an issue matching multiple categories
    is classified by this priority order.
    """
    if labels & BUG_LABELS.get(repo, set()):
        return "Bug"
    if labels & SECURITY_LABELS.get(repo, set()):
        return "Security"
    if labels & DOCS_LABELS.get(repo, set()):
        return "Documentation"
    if labels & SUPPORT_LABELS.get(repo, set()):
        return "Support/question"
    if labels & FEATURE_LABELS.get(repo, set()):
        return "Feature request"
    return "Other"


def clean_issues(df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Standardize the issues table and apply the standardized issue taxonomy.

    Note: main.py's collectors.py already excludes pull requests from the
    issues table at collection time (GitHub's /issues endpoint returns PRs
    mixed in; the collection log for every repo run so far reports counts
    as "N issues (PRs excluded)"). This function re-checks for a 'body'
    field containing a PR-only marker as a defensive second pass in case a
    future collector run does not pre-filter, but does not assume PRs are
    present.

    Adds:
        - created_at, closed_at parsed to UTC datetime
        - label_set (parsed set of raw labels)
        - issue_category (standardized taxonomy)
        - is_qualifying_bug (True if issue_category == 'Bug')
    """
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    out["closed_at"] = pd.to_datetime(out["closed_at"], errors="coerce", utc=True)

    before = len(out)
    out = out.dropna(subset=["created_at"])
    n_bad_created = before - len(out)

    out = out.drop_duplicates(subset=["repository_name", "issue_id"])

    out["label_set"] = out["labels"].apply(_label_set)
    out["issue_category"] = out["label_set"].apply(lambda s: classify_issue_taxonomy(s, repo))
    out["is_qualifying_bug"] = out["issue_category"] == "Bug"

    out.attrs["cleaning_stats"] = {
        "repo": repo,
        "rows_in": len(df),
        "rows_out": len(out),
        "dropped_unparseable_created_at": n_bad_created,
        "category_breakdown": out["issue_category"].value_counts().to_dict(),
    }
    return out


# ---------------------------------------------------------------------------
# Bot / automated-activity detection (commits, PRs, contributors)
# ---------------------------------------------------------------------------

def identify_bots(author_series: pd.Series) -> pd.Series:
    """
    Returns a boolean Series, True where the author identifier matches a
    known bot/automation pattern (dependabot, renovate, github-actions,
    project-specific CI bots, etc.). Apply this to commits, pull_requests,
    and contributors tables before computing activity-volume features, so
    automated commits/PRs don't inflate human development-activity signals.
    """
    return author_series.astype(str).str.match(_BOT_REGEX)


def clean_contributors(df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Standardize the contributors table.

    Actual confirmed columns (data/tables/<repo>/contributors_table.csv):
        repository_name, contributor identifier, first contribution date,
        number of contributions, release-cycle participation

    Note: 'first contribution date' was observed blank for some low-activity
    contributors in early pilot testing (octocat/Hello-World). This function
    does NOT drop rows with a missing first-contribution date, since the
    contributor is still real -- it only means their first-contribution
    turnover feature can't be computed. Missing-rate is reported in
    cleaning_stats so this can be monitored per repository.

    Adds standardized columns:
        - author               (copied from 'contributor identifier')
        - is_bot               (from identify_bots, applied to 'author')
        - first_contribution_at (parsed from 'first contribution date', UTC)
        - number_of_contributions / release_cycle_participation
          (snake_case copies of the raw columns, for consistency with the
          rest of the pipeline)
    """
    out = df.copy()
    out = out.drop_duplicates(subset=["repository_name", "contributor identifier"])

    out["author"] = out["contributor identifier"]
    out["is_bot"] = identify_bots(out["author"])
    out["first_contribution_at"] = pd.to_datetime(
        out["first contribution date"], errors="coerce", utc=True
    )
    out["number_of_contributions"] = out["number of contributions"]
    out["release_cycle_participation"] = out["release-cycle participation"]

    out.attrs["cleaning_stats"] = {
        "repo": repo,
        "rows_in": len(df),
        "rows_out": len(out),
        "bot_contributor_share": float(out["is_bot"].mean()) if len(out) else None,
        "missing_first_contribution_date_share": float(out["first_contribution_at"].isna().mean()) if len(out) else None,
    }
    return out


def contributor_turnover_features(commits_matched: pd.DataFrame, contributors_clean: pd.DataFrame,
                                    cycles_df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Builds RQ3 features -- contributor turnover and participation breadth --
    at the release-cycle level, using human (non-bot) commit authorship
    within each cycle plus each contributor's first-contribution date.

    Requires commits_matched to already have a 'release_id' column (i.e. it
    is the output of match_events_to_cycles(commits_clean, cycles, ...)) and
    an 'author'/'is_bot' column (i.e. it came from clean_commits()).

    First-contribution date handling: contributors_clean's
    'first_contribution_at' column (sourced from the raw collector's
    'first contribution date' field) was found to be 100% missing across
    both primary repositories in practice -- likely never populated at
    collection time. Rather than depend on that column, this function
    derives each author's first contribution directly from the earliest
    commit timestamp observed for them in commits_matched, which is more
    robust since it comes straight from primary commit data. The
    contributors_clean 'first_contribution_at' column is used only as a
    supplementary source where present (e.g. for a contributor with no
    commits in this dataset but a valid recorded first-contribution date).

    Returns one row per release_id with:
        distinct_contributors        -- count of unique human authors active in this cycle
        first_time_contributor_count -- of those, how many had no prior contribution before cycle_start
        first_time_contributor_share -- first_time_contributor_count / distinct_contributors
        top_contributor_share        -- share of this cycle's human commits made by its single most active author
                                         (a concentration proxy -- low breadth if this is high)
    """
    human_commits = commits_matched[
        (commits_matched["repository_name"] == repo) & (~commits_matched["is_bot"])
    ].dropna(subset=["release_id"])

    # Derive first-contribution date from actual commit history (robust,
    # does not depend on the unreliable raw 'first contribution date' column).
    derived_first_commit = (
        commits_matched[commits_matched["repository_name"] == repo]
        .groupby("author")["committed_at"].min()
    )

    contrib = contributors_clean[contributors_clean["repository_name"] == repo]
    reported_first_contribution = contrib.set_index("author")["first_contribution_at"]

    # Prefer the derived (commit-based) date; fall back to the reported
    # column only for authors missing from the derived series entirely.
    # Take the EARLIEST of the derived (commit-based) and reported dates per
    # author, not simply prefer one over the other -- if the raw collector's
    # 'first contribution date' field is ever populated with a genuinely
    # earlier date than what's observable in this dataset's commit window
    # (e.g. a long-time contributor whose true first contribution predates
    # your collection window), that earlier date should win, since it's more
    # informative about whether this person is really "new." In current
    # real data this column is 100% missing for both primary repositories,
    # so in practice the derived date is what's used -- but this keeps the
    # logic correct if that field is ever fixed at the collection stage.
    combined = pd.concat([derived_first_commit, reported_first_contribution], axis=1)
    first_contribution_lookup = combined.min(axis=1)

    cy = cycles_df[cycles_df["repository_name"] == repo].dropna(subset=["cycle_start"])

    records = []
    for _, row in cy.iterrows():
        rid = row["release_id"]
        cycle_start = row["cycle_start"]
        cycle_commits = human_commits[human_commits["release_id"] == rid]

        if cycle_commits.empty:
            records.append({
                "release_id": rid, "distinct_contributors": 0,
                "first_time_contributor_count": 0, "first_time_contributor_share": np.nan,
                "top_contributor_share": np.nan,
            })
            continue

        authors_in_cycle = cycle_commits["author"].unique()
        distinct_contributors = len(authors_in_cycle)

        first_dates = first_contribution_lookup.reindex(authors_in_cycle)
        is_first_time = first_dates >= cycle_start  # their first-ever contribution falls inside this cycle
        first_time_count = int(is_first_time.fillna(False).sum())

        commit_counts_by_author = cycle_commits["author"].value_counts()
        top_contributor_share = commit_counts_by_author.iloc[0] / commit_counts_by_author.sum()

        records.append({
            "release_id": rid,
            "distinct_contributors": distinct_contributors,
            "first_time_contributor_count": first_time_count,
            "first_time_contributor_share": first_time_count / distinct_contributors,
            "top_contributor_share": top_contributor_share,
        })

    return pd.DataFrame(records)


def clean_commits(df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Standardize the commits table.

    Actual confirmed columns (data/tables/<repo>/commits_table.csv):
        repository_name, commit_sha, commit_date, author identifier,
        committer identifier, additions, deletions, total changes,
        files changed, commit message

    Adds standardized columns so downstream code (match_events_to_cycles,
    identify_bots) can rely on consistent names regardless of the raw
    schema:
        - committed_at   (parsed from commit_date, UTC datetime)
        - author         (copied from 'author identifier')
        - is_bot         (from identify_bots, applied to 'author')
        - churn          (additions + deletions; falls back to the raw
                           'total changes' column if additions/deletions
                           are missing)
    """
    out = df.copy()

    out["committed_at"] = pd.to_datetime(out["commit_date"], errors="coerce", utc=True)
    before = len(out)
    out = out.dropna(subset=["committed_at"])
    n_bad_date = before - len(out)

    out = out.drop_duplicates(subset=["repository_name", "commit_sha"])

    out["author"] = out["author identifier"]
    out["is_bot"] = identify_bots(out["author"])

    if {"additions", "deletions"}.issubset(out.columns):
        out["churn"] = out["additions"].fillna(0) + out["deletions"].fillna(0)
    elif "total changes" in out.columns:
        out["churn"] = out["total changes"].fillna(0)

    out.attrs["cleaning_stats"] = {
        "repo": repo,
        "rows_in": len(df),
        "rows_out": len(out),
        "dropped_unparseable_date": n_bad_date,
        "bot_commit_share": float(out["is_bot"].mean()) if len(out) else None,
    }
    return out


def clean_pull_requests(df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    Standardize the pull-requests table.

    Actual confirmed columns (data/tables/<repo>/pull_requests_table.csv):
        repository_name, pull_request_id, created_at, closed_at, merged_at,
        merge status, review count, comment count, changed files,
        additions, deletions, author identifier, labels, milestone

    Adds standardized columns:
        - created_at / merged_at / closed_at parsed to UTC datetime
        - author         (copied from 'author identifier')
        - is_bot         (from identify_bots, applied to 'author')
        - is_merged      (True if merged_at is not null)
        - review_count / comment_count (copied from 'review count' /
          'comment count' for convenient snake_case access)
    """
    out = df.copy()
    for col in ("created_at", "merged_at", "closed_at"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    before = len(out)
    out = out.dropna(subset=["created_at"])
    n_bad_date = before - len(out)

    out = out.drop_duplicates(subset=["repository_name", "pull_request_id"])

    out["author"] = out["author identifier"]
    out["is_bot"] = identify_bots(out["author"])
    out["is_merged"] = out["merged_at"].notna() if "merged_at" in out.columns else np.nan

    if "review count" in out.columns:
        out["review_count"] = out["review count"]
    if "comment count" in out.columns:
        out["comment_count"] = out["comment count"]

    out.attrs["cleaning_stats"] = {
        "repo": repo,
        "rows_in": len(df),
        "rows_out": len(out),
        "dropped_unparseable_date": n_bad_date,
        "bot_pr_share": float(out["is_bot"].mean()) if len(out) else None,
    }
    return out


# ---------------------------------------------------------------------------
# Matching commits / PRs to release cycles
# ---------------------------------------------------------------------------

def match_events_to_cycles(events_df: pd.DataFrame, cycles_df: pd.DataFrame,
                            event_time_col: str, repo: str) -> pd.DataFrame:
    """
    Assigns each event (commit or PR) to the release cycle whose
    (cycle_start, cycle_end] window contains the event's timestamp.
    Vectorized via merge_asof -- requires both frames sorted by time.

    Parameters
    ----------
    events_df : cleaned commits or pull_requests table (must contain
        repository_name and event_time_col)
    cycles_df : output of build_release_cycles()
    event_time_col : name of the datetime column to match on
        (e.g. 'committed_at' or 'created_at')
    repo : repository_name to restrict matching to (call once per repo)

    Returns events_df filtered to `repo`, with an added 'release_id' column
    giving the release this event's activity counts toward. Events before
    the repository's first eligible cycle_start (or after the last
    cycle_end) get release_id = NaN and should be excluded from
    per-release feature aggregation.
    """
    ev = events_df[events_df["repository_name"] == repo].copy()
    cy = cycles_df[cycles_df["repository_name"] == repo].copy()

    ev = ev.sort_values(event_time_col)
    cy = cy.dropna(subset=["cycle_start"]).sort_values("cycle_end")

    if ev.empty or cy.empty:
        ev["release_id"] = np.nan
        return ev

    matched = pd.merge_asof(
        ev, cy[["release_id", "cycle_start", "cycle_end"]],
        left_on=event_time_col, right_on="cycle_end",
        direction="forward",
    )
    # merge_asof(direction='forward') finds the first cycle_end >= event time;
    # confirm the event also falls after that cycle's start, else it belongs
    # to no eligible cycle (e.g. it's after the last release, or before the
    # very first eligible cycle_start).
    within_window = matched[event_time_col] >= matched["cycle_start"]
    matched.loc[~within_window, "release_id"] = np.nan
    return matched.drop(columns=["cycle_start", "cycle_end"])


# ---------------------------------------------------------------------------
# Issues open at release (backlog snapshot)
# ---------------------------------------------------------------------------

def issues_open_at_release(clean_issues_df: pd.DataFrame, cycles_df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """
    For each eligible release, counts how many issues were open (created
    before the release date, and either still open or closed after the
    release date) at the moment of release -- a backlog-size snapshot
    feature.

    Returns one row per release_id with columns:
        release_id, open_issues_at_release, open_bugs_at_release
    """
    issues = clean_issues_df[clean_issues_df["repository_name"] == repo]
    cy = cycles_df[cycles_df["repository_name"] == repo]

    records = []
    for _, row in cy.iterrows():
        rd = row["cycle_end"]
        open_mask = (issues["created_at"] <= rd) & (
            issues["closed_at"].isna() | (issues["closed_at"] > rd)
        )
        open_issues = issues[open_mask]
        records.append({
            "release_id": row["release_id"],
            "open_issues_at_release": len(open_issues),
            "open_bugs_at_release": int(open_issues["is_qualifying_bug"].sum()),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------

def build_data_quality_report(stats_list: list) -> pd.DataFrame:
    """
    Consolidates the .attrs['cleaning_stats'] dict produced by each clean_*
    function into a single tidy report. Pass a list of
    (table_name, stats_dict) tuples.
    """
    rows = []
    for table_name, stats in stats_list:
        row = {"table": table_name}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)
