"""
run_features.py  (lives in src/, alongside cleaning.py and features.py)
------------------------------------------------------------------------
Week 4 driver: reads each primary repository's CLEANED tables from
data/clean/<owner>_<repo>/ (produced by run_cleaning.py), builds the
release-level feature table for each repo via features.py, and writes:

    data/features/<owner>_<repo>/release_level_features.csv   -- per repo
    data/features/all_repos_release_level.csv                 -- both repos combined,
                                                                   ready for modelling

IMPORTANT -- train_cutoff_date:
    The target-variable threshold (75th percentile bug count) must be
    computed on the TRAINING portion of the data only, to avoid leakage.
    Set TRAIN_CUTOFF_DATE below to your team's actual train/test split
    boundary before final modelling. Left as None, the threshold is
    computed across ALL releases -- fine for initial exploration of the
    feature table, but this MUST be set to a real date before you report
    any model performance numbers.

Prerequisite: run_cleaning.py must already have been run successfully for
both repos (data/clean/kubernetes_kubernetes/ and data/clean/apache_airflow/
must both exist with all expected files).

Usage:
    python src/run_features.py
"""

from pathlib import Path
import pandas as pd

from features import assemble_release_table

REPOS = {
    "kubernetes_kubernetes": "kubernetes/kubernetes",
    "apache_airflow": "apache/airflow",
    "microsoft_vscode": "microsoft/vscode",
}

# Train/validation/test cutoffs, finalized as a 60/20/20 split by release
# date across the combined three-repository timeline (2021-01-13 to
# 2026-06-17):
#   - Releases before TRAIN_CUTOFF_DATE       -> "train"      (~60%)
#   - Releases from TRAIN_CUTOFF_DATE up to
#     (not including) VALIDATION_CUTOFF_DATE  -> "validation"  (~20%)
#   - Releases from VALIDATION_CUTOFF_DATE on -> "test"        (~20%)
# The Q75 risk threshold is computed using ONLY "train" releases (the
# genuinely pre-cutoff data), then applied to validation and test -- this is
# what build_target_variable's train_cutoff_date parameter does. See the
# leakage audit for why this boundary was chosen and what it revealed about
# repository-level drift in bug-reporting rates over time.
TRAIN_CUTOFF_DATE = "2024-06-11"
VALIDATION_CUTOFF_DATE = "2025-07-18"
WINDOW_DAYS = 30

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = REPO_ROOT / "data" / "clean"
FEATURES_DIR = REPO_ROOT / "data" / "features"

DATE_COLUMNS = {
    "releases_clean.csv": ["release_date"],
    "release_cycles.csv": ["cycle_start", "cycle_end"],
    "commits_matched_to_cycles.csv": ["committed_at"],
    "pull_requests_matched_to_cycles.csv": ["created_at", "merged_at", "closed_at"],
    "issues_clean.csv": ["created_at", "closed_at"],
}


def load_table(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        print(f"  [skip] {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in DATE_COLUMNS.get(filename, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def run_for_repo(folder_name: str, repo: str) -> pd.DataFrame:
    print(f"\n=== Building features for {repo} ===")
    clean_dir = CLEAN_DIR / folder_name

    if not clean_dir.exists():
        print(f"  [skip] {clean_dir} not found -- run run_cleaning.py first")
        return pd.DataFrame()

    releases_clean = load_table(clean_dir, "releases_clean.csv")
    cycles = load_table(clean_dir, "release_cycles.csv")
    commits_matched = load_table(clean_dir, "commits_matched_to_cycles.csv")
    prs_matched = load_table(clean_dir, "pull_requests_matched_to_cycles.csv")
    turnover = load_table(clean_dir, "contributor_turnover_features.csv")
    issues_clean = load_table(clean_dir, "issues_clean.csv")
    issues_open_at_release_df = load_table(clean_dir, "issues_open_at_release.csv")

    if releases_clean.empty or cycles.empty:
        print(f"  [skip] missing releases_clean or release_cycles for {repo}")
        return pd.DataFrame()

    # issues_clean's boolean columns can come back as strings after a CSV
    # round-trip; coerce is_qualifying_bug back to boolean defensively.
    if "is_qualifying_bug" in issues_clean.columns and issues_clean["is_qualifying_bug"].dtype != bool:
        issues_clean["is_qualifying_bug"] = issues_clean["is_qualifying_bug"].astype(str).str.lower() == "true"

    table = assemble_release_table(
        repo, releases_clean, cycles, commits_matched, prs_matched, turnover, issues_clean,
        issues_open_at_release_df=issues_open_at_release_df,
        window_days=WINDOW_DAYS, train_cutoff_date=TRAIN_CUTOFF_DATE,
    )

    out_dir = FEATURES_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "release_level_features.csv"
    table.to_csv(out_path, index=False)

    n_labeled = table["elevated_risk"].notna().sum()
    n_risky = (table["elevated_risk"] == 1).sum()
    print(f"Wrote {out_path} ({len(table)} releases, {n_labeled} labeled, {n_risky} elevated_risk=1)")
    if TRAIN_CUTOFF_DATE is None:
        print("  [WARNING] TRAIN_CUTOFF_DATE is None -- risk threshold computed across ALL releases. "
              "Set a real train/test cutoff before reporting model performance.")

    return table


def main():
    all_tables = []
    for folder_name, repo in REPOS.items():
        t = run_for_repo(folder_name, repo)
        if not t.empty:
            all_tables.append(t)

    if not all_tables:
        print("\nNo repos produced a feature table -- nothing to combine.")
        return

    combined = pd.concat(all_tables, ignore_index=True)

    # Add an explicit split column so downstream modelling doesn't need to
    # recompute this from raw dates every time -- one source of truth for
    # which releases are train/validation/test.
    combined["release_date"] = pd.to_datetime(combined["release_date"], utc=True)
    train_cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE, tz="UTC") if TRAIN_CUTOFF_DATE else None
    val_cutoff_ts = pd.Timestamp(VALIDATION_CUTOFF_DATE, tz="UTC")

    def _assign_split(release_date):
        if train_cutoff_ts is not None and release_date < train_cutoff_ts:
            return "train"
        elif release_date < val_cutoff_ts:
            return "validation"
        else:
            return "test"

    combined["split"] = combined["release_date"].apply(_assign_split)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = FEATURES_DIR / "all_repos_release_level.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined release-level table written to {combined_path} ({len(combined)} releases total)")
    print("\nSplit sizes:")
    print(combined.groupby(["split", "repository_name"]).size().to_string())


if __name__ == "__main__":
    main()
