"""
main.py
-------
Entry point for Week 1-2 data collection (Section 27, Weeks 1-2).

Run:
    export GITHUB_TOKEN=ghp_your_token_here
    python main.py --repos kubernetes/kubernetes home-assistant/core --since 2021-01-01

This will produce, under data/raw/<repo>/ :
    repository.json, releases.json, commits.json, pull_requests.json,
    issues.json, contributors.json   (untouched API responses -- cache / audit trail)

and under data/tables/<repo1>__<repo2>.../ :
    repository_table.csv
    releases_table.csv
    commits_table.csv
    pull_requests_table.csv
    issues_table.csv
    contributors_table.csv           (tidy raw tables, one per Section 6.1)

    This folder is named after the --repos passed in, e.g. running
        python main.py --repos kubernetes/kubernetes ...
    writes to data/tables/kubernetes_kubernetes/, so each teammate's run
    for a different repo lands in its own folder instead of overwriting
    a previous run's CSVs.

Re-running the script reuses cached JSON automatically (Section 8: "resume
collection after interruption"). Pass --refresh to force re-fetching.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from github_client import GitHubClient
import collectors as coll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

TABLES_DIR = Path("data/tables")


def _run_tag(repos: list[str]) -> str:
    """Turn ['kubernetes/kubernetes'] into 'kubernetes_kubernetes', or join
    multiple repos with '__' so each teammate's run writes to its own folder
    instead of overwriting data/tables/*.csv from a previous run."""
    return "__".join(r.replace("/", "_") for r in repos)


def run_collection(repos: list[str], since: str | None, until: str | None,
                    refresh: bool, enrich: bool, enrich_limit: int | None, token: str | None):
    out_dir = TABLES_DIR / _run_tag(repos)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GitHubClient(token=token)

    # Sanity check on rate limit before starting a long run
    status = client.get_rate_limit_status()
    if status:
        core = status.get("resources", {}).get("core", {})
        logger.info("Rate limit at start: %s/%s remaining", core.get("remaining"), core.get("limit"))
        if not token:
            logger.warning(
                "No GITHUB_TOKEN set -- limited to 60 requests/hour. "
                "Set GITHUB_TOKEN for the 5000/hour authenticated limit."
            )

    all_repo_tables = []
    all_release_tables = []
    all_commit_tables = []
    all_pr_tables = []
    all_issue_tables = []
    all_contrib_tables = []

    for repo in repos:
        logger.info("=== Collecting %s ===", repo)
        try:
            repo_df = coll.collect_repository(client, repo, refresh=refresh)
            all_repo_tables.append(repo_df)

            releases_df = coll.collect_releases(client, repo, refresh=refresh)
            all_release_tables.append(releases_df)
            logger.info("%s: %d releases", repo, len(releases_df))

            commits_df = coll.collect_commits(client, repo, since=since, until=until, refresh=refresh)
            if enrich:
                commits_df = coll.enrich_commit_stats(client, repo, commits_df, refresh=refresh, limit=enrich_limit)
            all_commit_tables.append(commits_df)
            logger.info("%s: %d commits", repo, len(commits_df))

            pr_df = coll.collect_pull_requests(client, repo, refresh=refresh)
            if enrich:
                pr_df = coll.enrich_pr_reviews(client, repo, pr_df, refresh=refresh, limit=enrich_limit)
            all_pr_tables.append(pr_df)
            logger.info("%s: %d pull requests", repo, len(pr_df))

            issues_df = coll.collect_issues(client, repo, refresh=refresh)
            all_issue_tables.append(issues_df)
            logger.info("%s: %d issues (PRs excluded)", repo, len(issues_df))

            contrib_df = coll.collect_contributors(client, repo, refresh=refresh)
            all_contrib_tables.append(contrib_df)
            logger.info("%s: %d contributors", repo, len(contrib_df))

        except Exception:
            logger.exception("Collection failed for %s -- skipping to next repo", repo)
            continue

    def _save(frames, name):
        if not frames:
            logger.warning("No data collected for %s -- skipping write", name)
            return
        df = pd.concat(frames, ignore_index=True)
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))

    _save(all_repo_tables, "repository_table")
    _save(all_release_tables, "releases_table")
    _save(all_commit_tables, "commits_table")
    _save(all_pr_tables, "pull_requests_table")
    _save(all_issue_tables, "issues_table")
    _save(all_contrib_tables, "contributors_table")


def main():
    parser = argparse.ArgumentParser(description="GitHub raw data collection for release-risk capstone")
    parser.add_argument("--repos", nargs="+", required=True,
                         help="Repositories as owner/name, e.g. kubernetes/kubernetes")
    parser.add_argument("--since", default=None, help="ISO date, e.g. 2021-01-01 (bounds commit collection)")
    parser.add_argument("--until", default=None, help="ISO date, e.g. 2026-01-01")
    parser.add_argument("--refresh", action="store_true", help="Ignore JSON cache and re-fetch from API")
    parser.add_argument("--enrich", action="store_true",
                         help="Also fetch per-commit stats and per-PR review counts (many extra API calls)")
    parser.add_argument("--enrich-limit", type=int, default=None,
                         help="Cap the number of commits/PRs enriched per repo (useful for a pilot run)")
    parser.add_argument("--token", default=None, help="GitHub PAT (or set GITHUB_TOKEN env var)")
    args = parser.parse_args()

    run_collection(
        repos=args.repos,
        since=args.since,
        until=args.until,
        refresh=args.refresh,
        enrich=args.enrich,
        enrich_limit=args.enrich_limit,
        token=args.token,
    )


if __name__ == "__main__":
    main()
