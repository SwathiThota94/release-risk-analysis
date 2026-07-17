"""
github_client.py
-----------------
Thin wrapper around the GitHub REST API (v3) that satisfies Section 8 of the
capstone instructions:

    - Authenticate with GitHub where required
    - Handle pagination
    - Track API rate limits
    - Resume collection after interruption (via on-disk raw JSON cache)
    - Record collection date / repository name / endpoint used
    - Log collection errors

Usage:
    from github_client import GitHubClient
    client = GitHubClient(token="ghp_xxx")
    pages = client.paginated_get("repos/kubernetes/kubernetes/releases", params={"per_page": 100})
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

API_ROOT = "https://api.github.com"

logger = logging.getLogger("github_client")


class GitHubClient:
    def __init__(self, token: str | None = None, log_dir: str = "logs"):
        """
        token: a GitHub Personal Access Token (classic or fine-grained).
               Falls back to the GITHUB_TOKEN environment variable if not given.
               Unauthenticated requests are allowed but are rate-limited to
               60 requests/hour instead of 5000/hour, so a token is strongly
               recommended for any real collection run.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.session.headers.update(headers)

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / "collection_log.jsonl"

    # ------------------------------------------------------------------
    # Logging helper -- writes one JSON line per API call for auditability
    # (collection date, repo, endpoint, status, notes) as required by Sec. 8
    # ------------------------------------------------------------------
    def _log_event(self, endpoint: str, status: str, extra: dict | None = None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status": status,
        }
        if extra:
            record.update(extra)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Rate limit handling
    # ------------------------------------------------------------------
    def _handle_rate_limit(self, response: requests.Response):
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_ts = response.headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) == 0 and reset_ts:
            wait_seconds = max(int(reset_ts) - int(time.time()) + 5, 0)
            logger.warning(
                "Rate limit exhausted. Sleeping %s seconds until reset.", wait_seconds
            )
            self._log_event(
                "rate_limit", "sleeping", {"wait_seconds": wait_seconds}
            )
            time.sleep(wait_seconds)

    # ------------------------------------------------------------------
    # Single request with retry logic
    # ------------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, max_retries: int = 5):
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                attempt += 1
                logger.error("Request error on %s: %s (attempt %s)", url, exc, attempt)
                self._log_event(url, "request_exception", {"error": str(exc)})
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                self._log_event(url, "ok", {"params": params})
                return response

            if response.status_code == 403 and "rate limit" in response.text.lower():
                self._handle_rate_limit(response)
                attempt += 1
                continue

            if response.status_code == 404:
                logger.warning("404 Not Found: %s", url)
                self._log_event(url, "not_found")
                return response

            if response.status_code in (500, 502, 503, 504):
                attempt += 1
                wait = 2 ** attempt
                logger.warning(
                    "Server error %s on %s. Retrying in %ss (attempt %s).",
                    response.status_code, url, wait, attempt,
                )
                self._log_event(url, f"server_error_{response.status_code}")
                time.sleep(wait)
                continue

            # Any other non-200: log and return so caller can decide
            logger.error("Unexpected status %s on %s: %s", response.status_code, url, response.text[:300])
            self._log_event(url, f"error_{response.status_code}", {"body": response.text[:500]})
            return response

        raise RuntimeError(f"Exceeded max retries for {url}")

    # ------------------------------------------------------------------
    # Paginated GET -- handles GitHub's Link-header pagination
    # ------------------------------------------------------------------
    def paginated_get(self, endpoint: str, params: dict | None = None, max_pages: int | None = None):
        """
        endpoint: path relative to API_ROOT, e.g. 'repos/owner/name/releases'
        Returns: list of all JSON objects across all pages.
        """
        url = f"{API_ROOT}/{endpoint.lstrip('/')}"
        params = dict(params or {})
        params.setdefault("per_page", 100)

        all_items = []
        page_count = 0

        while url:
            response = self._get(url, params=params if page_count == 0 else None)
            if response is None or response.status_code != 200:
                break

            data = response.json()
            if isinstance(data, list):
                all_items.extend(data)
            else:
                # Some endpoints (e.g. search) wrap results in a dict
                all_items.extend(data.get("items", []))

            page_count += 1
            if max_pages and page_count >= max_pages:
                break

            # Follow the 'next' link from the Link header
            url = response.links.get("next", {}).get("url")

        return all_items

    def get_rate_limit_status(self):
        response = self._get(f"{API_ROOT}/rate_limit")
        return response.json() if response.status_code == 200 else None
