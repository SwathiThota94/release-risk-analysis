"""
audit_missing_data.py  (lives in src/)
----------------------------------------
Scans every column in every raw table, for both primary repositories, and
reports the missing-value rate for each. Flags anything above
FLAG_THRESHOLD as [SUSPICIOUS] -- these are candidates for the same silent
collection-gap pattern already confirmed three times in this project
(contributor first-contribution dates, PR review/comment/size fields,
commit additions/deletions/size fields).

This exists because each of those three gaps was found one at a time, by
accident, while building a specific feature -- rather than through a
single deliberate check. Run this BEFORE building any new feature on a
column you haven't explicitly verified.

Usage:
    python src/audit_missing_data.py
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "data" / "tables"

REPOS = ["kubernetes_kubernetes", "apache_airflow", "microsoft_vscode"]
RAW_TABLES = [
    "repository_table.csv",
    "releases_table.csv",
    "commits_table.csv",
    "pull_requests_table.csv",
    "issues_table.csv",
    "contributors_table.csv",
]

FLAG_THRESHOLD = 0.50  # columns missing more than this fraction get flagged

# Columns already confirmed broken in this project -- shown for reference,
# not re-flagged as "new" news, but still included so the report is complete.
KNOWN_ISSUES = {
    "first contribution date": "Confirmed 100% missing (contributors_table). Worked around via derived commit dates in cleaning.py.",
    "review count": "Confirmed 100% missing (pull_requests_table). RQ2 reduced to merge timing only.",
    "comment count": "Confirmed 100% missing (pull_requests_table, and separately in issues_table -- check both).",
    "changed files": "Confirmed 100% missing (pull_requests_table).",
    "additions": "Confirmed 100% missing in BOTH commits_table and pull_requests_table.",
    "deletions": "Confirmed 100% missing in BOTH commits_table and pull_requests_table.",
    "total changes": "Confirmed 100% missing (commits_table).",
    "files changed": "Confirmed 100% missing (commits_table).",
}


def audit_table(path: Path) -> list:
    if not path.exists():
        return [{"column": "(file not found)", "missing_pct": None, "note": str(path)}]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [{"column": "(read error)", "missing_pct": None, "note": str(e)}]

    results = []
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        # Also treat empty-string-only columns as effectively missing --
        # some raw fields may be blank strings rather than true NaN.
        if df[col].dtype == object:
            blank_pct = (df[col].astype(str).str.strip() == "").mean()
            missing_pct = max(missing_pct, blank_pct)
        results.append({"column": col, "missing_pct": missing_pct})
    return results


def main():
    print(f"{'REPO':<25}{'TABLE':<28}{'COLUMN':<28}{'MISSING %':<12}{'FLAG'}")
    print("-" * 110)

    any_new_flags = False

    for repo_folder in REPOS:
        for table_name in RAW_TABLES:
            path = TABLES_DIR / repo_folder / table_name
            rows = audit_table(path)
            for r in rows:
                col = r["column"]
                pct = r["missing_pct"]
                if pct is None:
                    print(f"{repo_folder:<25}{table_name:<28}{col:<28}{'--':<12}[skip: {r.get('note','')}]")
                    continue

                flag = ""
                if pct >= FLAG_THRESHOLD:
                    if col in KNOWN_ISSUES:
                        flag = "[KNOWN ISSUE]"
                    else:
                        flag = "[SUSPICIOUS -- NEW]"
                        any_new_flags = True

                print(f"{repo_folder:<25}{table_name:<28}{col:<28}{pct*100:>6.1f}%     {flag}")

    print("\n" + "=" * 110)
    if any_new_flags:
        print("[ACTION NEEDED] One or more columns above 50% missing were found that are NOT already")
        print("documented as a known issue. Investigate these before building features on them.")
    else:
        print("No new suspicious columns found beyond the three already-documented known issues.")
    print("\nKnown issues already documented in this project:")
    for col, note in KNOWN_ISSUES.items():
        print(f"  - {col}: {note}")


if __name__ == "__main__":
    main()
