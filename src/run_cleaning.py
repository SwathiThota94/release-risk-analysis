"""
run_cleaning.py  (lives in src/, alongside cleaning.py)
----------------
Week 3 driver: reads each primary repository's raw tables from
data/tables/<owner>_<repo>/ (at the repo root, alongside main.py), runs the
cleaning.py functions, and writes cleaned intermediate tables plus a
combined data-quality report to data/clean/<owner>_<repo>/.

Prerequisite: both data/tables/kubernetes_kubernetes/ and
data/tables/apache_airflow/ must already exist locally -- pull them from
the shared drive first (they are gitignored, not committed to the repo).

Usage (run from either the repo root or from inside src/ -- paths resolve
relative to this file's location, not the current working directory):
    python src/run_cleaning.py
    -- or, from inside src/ --
    python run_cleaning.py
"""

from pathlib import Path
import pandas as pd

from cleaning import (
    clean_releases, build_release_cycles, clean_issues, clean_commits,
    clean_pull_requests, match_events_to_cycles, issues_open_at_release,
    clean_contributors, contributor_turnover_features, build_data_quality_report,
)

# repo folder name -> GitHub repo identifier (matches BUG_LABELS keys in cleaning.py)
REPOS = {
    "kubernetes_kubernetes": "kubernetes/kubernetes",
    "apache_airflow": "apache/airflow",
}

# This file lives in src/, but data/ sits at the repo root alongside main.py.
# Resolve paths from this file's location so the script works correctly
# regardless of which directory it's run from (repo root or src/ itself).
REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "data" / "tables"
CLEAN_DIR = REPO_ROOT / "data" / "clean"


def load_table(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return pd.DataFrame()
    return pd.read_csv(path)


def run_for_repo(folder_name: str, repo: str, quality_stats: list):
    print(f"\n=== Cleaning {repo} ===")
    raw_dir = TABLES_DIR / folder_name
    out_dir = CLEAN_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"  [skip] {raw_dir} not found -- pull this repo's data from the shared drive first")
        return

    # --- Releases + cycles ---
    releases_raw = load_table(raw_dir, "releases_table")
    if releases_raw.empty:
        print(f"  [skip] no releases_table.csv for {repo} -- cannot build cycles, skipping repo")
        return
    releases_clean = clean_releases(releases_raw, repo)
    releases_clean.to_csv(out_dir / "releases_clean.csv", index=False)
    quality_stats.append((f"{folder_name}: releases", releases_clean.attrs["cleaning_stats"]))
    print("Releases cleaned:", releases_clean.attrs["cleaning_stats"])

    cycles = build_release_cycles(releases_clean)
    cycles.to_csv(out_dir / "release_cycles.csv", index=False)
    quality_stats.append((f"{folder_name}: cycles", cycles.attrs["cleaning_stats"]))
    print("Release cycles built:", cycles.attrs["cleaning_stats"])

    # --- Issues ---
    issues_raw = load_table(raw_dir, "issues_table")
    if not issues_raw.empty:
        issues_clean = clean_issues(issues_raw, repo)
        issues_clean.to_csv(out_dir / "issues_clean.csv", index=False)
        quality_stats.append((f"{folder_name}: issues", issues_clean.attrs["cleaning_stats"]))
        print("Issues cleaned:", issues_clean.attrs["cleaning_stats"])

        backlog = issues_open_at_release(issues_clean, cycles, repo)
        backlog.to_csv(out_dir / "issues_open_at_release.csv", index=False)
        print(f"Backlog snapshot built for {len(backlog)} releases")

    # --- Commits ---
    commits_raw = load_table(raw_dir, "commits_table")
    commits_matched = pd.DataFrame()
    if not commits_raw.empty:
        commits_clean = clean_commits(commits_raw, repo)
        quality_stats.append((f"{folder_name}: commits", commits_clean.attrs["cleaning_stats"]))
        print("Commits cleaned:", commits_clean.attrs["cleaning_stats"])

        commits_matched = match_events_to_cycles(commits_clean, cycles, "committed_at", repo)
        commits_matched.to_csv(out_dir / "commits_matched_to_cycles.csv", index=False)
        print(f"Commits matched to cycles: {commits_matched['release_id'].notna().sum()} / {len(commits_matched)} matched")

    # --- Pull requests ---
    prs_raw = load_table(raw_dir, "pull_requests_table")
    if not prs_raw.empty:
        prs_clean = clean_pull_requests(prs_raw, repo)
        quality_stats.append((f"{folder_name}: pull_requests", prs_clean.attrs["cleaning_stats"]))
        print("PRs cleaned:", prs_clean.attrs["cleaning_stats"])

        prs_matched = match_events_to_cycles(prs_clean, cycles, "created_at", repo)
        prs_matched.to_csv(out_dir / "pull_requests_matched_to_cycles.csv", index=False)
        print(f"PRs matched to cycles: {prs_matched['release_id'].notna().sum()} / {len(prs_matched)} matched")

    # --- Contributors + RQ3 turnover features ---
    contributors_raw = load_table(raw_dir, "contributors_table")
    if not contributors_raw.empty:
        contributors_clean = clean_contributors(contributors_raw, repo)
        contributors_clean.to_csv(out_dir / "contributors_clean.csv", index=False)
        quality_stats.append((f"{folder_name}: contributors", contributors_clean.attrs["cleaning_stats"]))
        print("Contributors cleaned:", contributors_clean.attrs["cleaning_stats"])

        if not commits_matched.empty:
            turnover = contributor_turnover_features(commits_matched, contributors_clean, cycles, repo)
            turnover.to_csv(out_dir / "contributor_turnover_features.csv", index=False)
            print(f"Contributor turnover features built for {len(turnover)} releases")
        else:
            print("  [skip] contributor turnover features require commits_matched_to_cycles -- run commits collection first")


def main():
    quality_stats = []
    for folder_name, repo in REPOS.items():
        run_for_repo(folder_name, repo, quality_stats)

    report = build_data_quality_report(quality_stats)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(CLEAN_DIR / "data_quality_report.csv", index=False)
    print(f"\nData quality report written to {CLEAN_DIR / 'data_quality_report.csv'}")


if __name__ == "__main__":
    main()
