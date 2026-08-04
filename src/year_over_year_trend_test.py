"""
year_over_year_trend_test.py  (lives in src/)
------------------------------------------------
Follow-up diagnostic to correlation_analysis.py: tests whether the apparent
decline in bugs_in_window over the TRAINING window (2021 - mid-2024) is a
statistically real trend, or noise -- this directly informs how much of
release_sequence / prior_releases_count's strong correlation with
elevated_risk reflects a genuine year-over-year decline versus an artifact
of how the Q75 threshold was calibrated across that same window.

Uses ONLY the 'train' split.

Two tests, run both pooled (all repos together) and per-repository:
    1. Kruskal-Wallis H-test across year groups -- tests whether
       bugs_in_window differs significantly across years AT ALL (an
       omnibus test; doesn't assume a monotonic direction).
    2. Spearman rank correlation between release_date (as ordinal rank)
       and bugs_in_window -- tests whether there's a significant MONOTONIC
       trend specifically (this is what "declining/rising over time" means).

Usage:
    python src/year_over_year_trend_test.py
"""

from pathlib import Path
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"


def run_tests(df: pd.DataFrame, label: str) -> dict:
    df = df.dropna(subset=["bugs_in_window"]).copy()
    if len(df) < 10:
        print(f"  [skip] {label}: too few rows ({len(df)}) for a meaningful test")
        return {}

    df["year"] = pd.to_datetime(df["release_date"]).dt.year
    year_groups = [g["bugs_in_window"].values for _, g in df.groupby("year") if len(g) >= 3]
    n_years = len(year_groups)

    result = {"group": label, "n": len(df), "n_years": n_years}

    if n_years >= 2:
        h_stat, kw_p = stats.kruskal(*year_groups)
        result["kruskal_wallis_H"] = h_stat
        result["kruskal_wallis_p"] = kw_p
    else:
        result["kruskal_wallis_H"] = None
        result["kruskal_wallis_p"] = None

    # Spearman trend: rank of release_date vs bugs_in_window
    date_rank = df["release_date"].rank()
    rho, spearman_p = stats.spearmanr(date_rank, df["bugs_in_window"])
    result["spearman_rho"] = rho
    result["spearman_p"] = spearman_p

    return result


def main():
    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found -- run run_features.py and preprocess_for_modeling.py first")
        return

    df = pd.read_csv(INPUT_PATH)
    if "split" not in df.columns:
        print("[error] No 'split' column found -- re-run run_features.py with the updated script.")
        return

    train = df[df["split"] == "train"].copy()
    print(f"Loaded {len(df)} total releases; using {len(train)} 'train' split releases.\n")

    results = []

    print("=== Pooled (all repositories) ===")
    pooled_year_stats = train.assign(year=pd.to_datetime(train["release_date"]).dt.year) \
        .groupby("year")["bugs_in_window"].agg(["count", "mean", "median"])
    print(pooled_year_stats.to_string())
    r = run_tests(train, "pooled")
    results.append(r)
    _print_result(r)

    for repo_name, g in train.groupby("repository_name"):
        print(f"\n=== {repo_name} ===")
        year_stats = g.assign(year=pd.to_datetime(g["release_date"]).dt.year) \
            .groupby("year")["bugs_in_window"].agg(["count", "mean", "median"])
        print(year_stats.to_string())
        r = run_tests(g, repo_name)
        results.append(r)
        _print_result(r)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_path = ANALYSIS_DIR / "year_over_year_trend_test.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote full results to {out_path}")


def _print_result(r: dict):
    if not r:
        return
    print(f"\n  Kruskal-Wallis across years: H={r.get('kruskal_wallis_H')}, "
          f"p={r.get('kruskal_wallis_p')}")
    print(f"  Spearman trend (date rank vs bugs_in_window): rho={r.get('spearman_rho'):.4f}, "
          f"p={r.get('spearman_p'):.4g}")
    if r.get("spearman_p") is not None and r["spearman_p"] < 0.05:
        direction = "DECLINING" if r["spearman_rho"] < 0 else "RISING"
        print(f"  -> Statistically significant {direction} trend over time within the training window.")
    else:
        print("  -> No statistically significant monotonic trend detected within the training window.")


if __name__ == "__main__":
    main()
