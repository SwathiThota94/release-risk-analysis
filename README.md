# GitHub Release Risk Intelligence

## Project Overview

This project investigates whether **release risk** — the likelihood that a software release introduces bugs, regressions, or instability — can be predicted from publicly available GitHub activity data (commits, pull requests, issues, and release history).

Rather than relying on a single repository, this project deliberately collects and compares data across **four open-source projects with different scale, release cadence, and governance models**:

| Repository | Type | Why included |
|---|---|---|
| `kubernetes/kubernetes` | Large, CNCF/corporate-backed | High release frequency, rich commit/PR history |
| `microsoft/vscode` | Large, corporate-backed | Mature, high-velocity, comparable scale to Kubernetes |
| `apache/airflow` | Mid-size, foundation-backed | Different domain and governance model |
| `home-assistant/core` | Community-driven | Smaller scale, slower release cadence — tests generalization |

## Research Questions

- Can commit/PR/issue activity patterns predict release risk?
- Do release-risk signals generalize across repositories of very different scale and governance, or are they repository-specific?
- Which raw signals (commit volume, PR review counts, issue reopen rates, contributor churn, etc.) carry the most predictive signal?

## Project Structure

```
├── main.py                  # CLI entry point — orchestrates data collection
├── collectors.py            # Per-endpoint collection logic (repo, releases, commits, PRs, issues, contributors)
├── github_client.py         # GitHub API client with authentication and rate-limit handling
├── requirements.txt         # Python dependencies (requests, pandas)
├── data/
│   ├── raw/<repo>/          # Untouched API JSON responses (audit trail, resumable)
│   └── tables/<repo>/       # Tidy CSV tables, one per data type
└── logs/                    # Per-run API call logs
```

## Pipeline Stages

1. **Raw collection (this stage)** — pull repository metadata, releases, commits, PRs, issues, and contributors for each target repo via the GitHub API. Output is untouched raw tables only — no feature engineering yet.
2. **Processing & feature engineering** *(upcoming)* — derive release-risk features (e.g. commit churn before a release, PR review depth, issue reopen rate) from the raw tables.
3. **Modeling** *(upcoming)* — build both a **pooled model** across all four repos (tests generalization) and at least one **repository-specific model** (Kubernetes, our richest dataset) for comparison.

## Setup

```bash
pip install -r requirements.txt
```

Generate a GitHub Personal Access Token (Settings → Developer settings → Personal access tokens → Tokens (classic); no scopes needed for public repo data), then set it as an environment variable:

```bash
export GITHUB_TOKEN=ghp_your_token_here      # Mac/Linux
set GITHUB_TOKEN=ghp_your_token_here         # Windows cmd
$env:GITHUB_TOKEN="ghp_your_token_here"      # Windows PowerShell
```

## Running the Collection

```bash
python main.py --repos kubernetes/kubernetes --since 2021-01-01
```

Output lands in `data/tables/kubernetes_kubernetes/` — each repo gets its own subfolder so parallel team runs don't overwrite each other's data.

## Team

Data collection is split across four team members, one repository each — see `TEAM_SETUP.md` for exact branch names and commands.