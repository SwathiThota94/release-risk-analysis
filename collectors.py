"""
collectors.py
-------------
Implements the raw-table collection required in Section 6.1 of the
capstone instructions:

    - Repository table
    - Releases table
    - Commits table
    - Pull requests table
    - Issues table
    - Contributors table

Each collect_* function:
    1. Pulls data from the GitHub API via GitHubClient
    2. Caches the untouched API response as JSON under raw/<repo>/<endpoint>.json
       (this enables "resume after interruption" -- if the JSON cache exists
       and refresh=False, it is reused instead of re-hitting the API)
    3. Returns a pandas DataFrame shaped to the raw-table column list in the
       instructions. No feature engineering happens here -- that belongs to
       a later "processed" stage per Section 6.2, not this raw layer.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from github_client import GitHubClient

logger = logging.getLogger("collectors")

RAW_DIR = Path("data/raw")


def _cache_path(repo_full_name: str, endpoint_name: str) -> Path:
    safe_repo = repo_full_name.replace("/", "__")
    d = RAW_DIR / safe_repo
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{endpoint_name}.json"


def _load_or_fetch(client: GitHubClient, repo_full_name: str, endpoint_name: str,
                    endpoint_path: str, params: dict | None = None, refresh: bool = False):
    path = _cache_path(repo_full_name, endpoint_name)
    if path.exists() and not refresh:
        logger.info("Using cached %s for %s", endpoint_name, repo_full_name)
        with open(path) as f:
            return json.load(f)

    logger.info("Fetching %s for %s from GitHub API", endpoint_name, repo_full_name)
    items = client.paginated_get(endpoint_path, params=params)
    with open(path, "w") as f:
        json.dump(items, f)
    return items


# ----------------------------------------------------------------------
# Repository table
# ----------------------------------------------------------------------
def collect_repository(client: GitHubClient, repo_full_name: str, refresh: bool = False) -> pd.DataFrame:
    owner, name = repo_full_name.split("/")
    path = _cache_path(repo_full_name, "repository")
    if path.exists() and not refresh:
        with open(path) as f:
            repo = json.load(f)
    else:
        response = client._get(f"https://api.github.com/repos/{owner}/{name}")
        repo = response.json()
        with open(path, "w") as f:
            json.dump(repo, f)

    row = {
        "repository_id": repo.get("id"),
        "repository_name": repo.get("full_name"),
        "owner": owner,
        "primary_language": repo.get("language"),
        "repository_created_at": repo.get("created_at"),
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "open_issue_count": repo.get("open_issues_count"),
    }
    return pd.DataFrame([row])


# ----------------------------------------------------------------------
# Releases table
# ----------------------------------------------------------------------
def collect_releases(client: GitHubClient, repo_full_name: str, refresh: bool = False) -> pd.DataFrame:
    items = _load_or_fetch(
        client, repo_full_name, "releases",
        f"repos/{repo_full_name}/releases", refresh=refresh,
    )

    rows = []
    # Sort ascending by created_at so previous_release_tag can be derived
    items_sorted = sorted(items, key=lambda r: r.get("created_at") or "")
    previous_tag = None
    for rel in items_sorted:
        rows.append({
            "repository_name": repo_full_name,
            "release_id": rel.get("id"),
            "tag_name": rel.get("tag_name"),
            "release_name": rel.get("name"),
            "release_date": rel.get("published_at") or rel.get("created_at"),
            "prerelease_flag": rel.get("prerelease"),
            "draft_flag": rel.get("draft"),
            "target_commitish": rel.get("target_commitish"),
            "release_notes": rel.get("body"),
            "previous_release_tag": previous_tag,
        })
        previous_tag = rel.get("tag_name")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Commits table
# ----------------------------------------------------------------------
def collect_commits(client: GitHubClient, repo_full_name: str, since: str | None = None,
                     until: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """
    since / until: ISO-8601 timestamps to bound collection (recommended --
    a multi-year commit history for a large repo like Kubernetes is huge;
    bound it to your chosen historical scope, Section 4.2).

    NOTE: additions/deletions/files-changed are only present on the
    *detailed* commit endpoint (one call per SHA), which is expensive at
    scale. This function fetches the commit list cheaply, then optionally
    enriches with per-commit stats -- see `enrich_commit_stats`.
    """
    params = {}
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    items = _load_or_fetch(
        client, repo_full_name, "commits",
        f"repos/{repo_full_name}/commits", params=params, refresh=refresh,
    )

    rows = []
    for c in items:
        commit = c.get("commit", {})
        rows.append({
            "repository_name": repo_full_name,
            "commit_sha": c.get("sha"),
            "commit_date": commit.get("author", {}).get("date"),
            "author identifier": (c.get("author") or {}).get("login") or commit.get("author", {}).get("email"),
            "committer identifier": (c.get("committer") or {}).get("login") or commit.get("committer", {}).get("email"),
            "additions": None,       # filled in by enrich_commit_stats, if run
            "deletions": None,
            "total changes": None,
            "files changed": None,
            "commit message": commit.get("message"),
        })
    return pd.DataFrame(rows)


def enrich_commit_stats(client: GitHubClient, repo_full_name: str, commits_df: pd.DataFrame,
                         refresh: bool = False, limit: int | None = None) -> pd.DataFrame:
    """
    Optional enrichment: fetches additions/deletions/files-changed per commit.
    This is 1 API call per commit SHA -- expensive for large repos, so it is
    kept as a separate opt-in step rather than baked into collect_commits.
    Results are cached per-repository so re-running is cheap.
    """
    cache_path = _cache_path(repo_full_name, "commit_stats")
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    shas = commits_df["commit_sha"].tolist()
    if limit:
        shas = shas[:limit]

    for sha in shas:
        if sha in cache and not refresh:
            continue
        response = client._get(f"https://api.github.com/repos/{repo_full_name}/commits/{sha}")
        if response.status_code != 200:
            continue
        detail = response.json()
        stats = detail.get("stats", {})
        files = detail.get("files", [])
        cache[sha] = {
            "additions": stats.get("additions"),
            "deletions": stats.get("deletions"),
            "total": stats.get("total"),
            "files_changed": len(files),
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    enriched = commits_df.copy()
    enriched["additions"] = enriched["commit_sha"].map(lambda s: cache.get(s, {}).get("additions"))
    enriched["deletions"] = enriched["commit_sha"].map(lambda s: cache.get(s, {}).get("deletions"))
    enriched["total changes"] = enriched["commit_sha"].map(lambda s: cache.get(s, {}).get("total"))
    enriched["files changed"] = enriched["commit_sha"].map(lambda s: cache.get(s, {}).get("files_changed"))
    return enriched


# ----------------------------------------------------------------------
# Pull requests table
# ----------------------------------------------------------------------
def collect_pull_requests(client: GitHubClient, repo_full_name: str, state: str = "all",
                           refresh: bool = False) -> pd.DataFrame:
    """
    Uses the Search API-free `pulls` list endpoint (cheap: no per-PR call for
    the base fields). Review count / comment count require one extra call
    per PR (see enrich_pr_reviews) -- kept optional for cost control.
    """
    items = _load_or_fetch(
        client, repo_full_name, "pull_requests",
        f"repos/{repo_full_name}/pulls",
        params={"state": state, "sort": "created", "direction": "asc"},
        refresh=refresh,
    )

    rows = []
    for pr in items:
        rows.append({
            "repository_name": repo_full_name,
            "pull_request_id": pr.get("number"),
            "created_at": pr.get("created_at"),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "merge status": "merged" if pr.get("merged_at") else pr.get("state"),
            "review count": None,     # filled by enrich_pr_reviews, if run
            "comment count": pr.get("comments"),
            "changed files": pr.get("changed_files"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "author identifier": (pr.get("user") or {}).get("login"),
            "labels": ",".join(l.get("name", "") for l in pr.get("labels", [])),
            "milestone": (pr.get("milestone") or {}).get("title"),
        })
    return pd.DataFrame(rows)


def enrich_pr_reviews(client: GitHubClient, repo_full_name: str, pr_df: pd.DataFrame,
                       refresh: bool = False, limit: int | None = None) -> pd.DataFrame:
    """Optional: fetches review count per PR (1 API call per PR)."""
    cache_path = _cache_path(repo_full_name, "pr_reviews")
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    pr_numbers = pr_df["pull_request_id"].tolist()
    if limit:
        pr_numbers = pr_numbers[:limit]

    for pr_number in pr_numbers:
        key = str(pr_number)
        if key in cache and not refresh:
            continue
        reviews = client.paginated_get(f"repos/{repo_full_name}/pulls/{pr_number}/reviews")
        cache[key] = len(reviews)

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    enriched = pr_df.copy()
    enriched["review count"] = enriched["pull_request_id"].map(lambda n: cache.get(str(n)))
    return enriched


# ----------------------------------------------------------------------
# Issues table
# ----------------------------------------------------------------------
def collect_issues(client: GitHubClient, repo_full_name: str, state: str = "all",
                    refresh: bool = False) -> pd.DataFrame:
    """
    IMPORTANT (per Section 6.1 note): the `issues` endpoint also returns
    pull requests. Each raw item that has a `pull_request` key is a PR, not
    a genuine issue, and is filtered out here.
    """
    items = _load_or_fetch(
        client, repo_full_name, "issues",
        f"repos/{repo_full_name}/issues",
        params={"state": state, "sort": "created", "direction": "asc"},
        refresh=refresh,
    )

    rows = []
    for issue in items:
        if "pull_request" in issue:
            continue  # exclude PRs returned by the issues endpoint
        rows.append({
            "repository_name": repo_full_name,
            "issue_id": issue.get("number"),
            "created_at": issue.get("created_at"),
            "closed_at": issue.get("closed_at"),
            "labels": ",".join(l.get("name", "") for l in issue.get("labels", [])),
            "title": issue.get("title"),
            "body": issue.get("body"),
            "milestone": (issue.get("milestone") or {}).get("title"),
            "state": issue.get("state"),
            "author identifier": (issue.get("user") or {}).get("login"),
            "comment count": issue.get("comments"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Contributors table
# ----------------------------------------------------------------------
def collect_contributors(client: GitHubClient, repo_full_name: str, refresh: bool = False) -> pd.DataFrame:
    items = _load_or_fetch(
        client, repo_full_name, "contributors",
        f"repos/{repo_full_name}/contributors",
        params={"anon": "false"}, refresh=refresh,
    )

    rows = []
    for c in items:
        rows.append({
            "repository_name": repo_full_name,
            "contributor identifier": c.get("login"),
            "first contribution date": None,   # requires commit-history scan; derive downstream
            "number of contributions": c.get("contributions"),
            "release-cycle participation": None,  # derived later once release cycles are built
        })
    return pd.DataFrame(rows)
