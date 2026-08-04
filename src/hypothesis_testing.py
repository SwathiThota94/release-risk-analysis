"""
hypothesis_testing.py  (lives in src/)
------------------------------------------
Week 5 (Descriptive and Diagnostic Analytics): formal hypothesis tests
comparing elevated-risk releases against non-risky releases on each feature.

Uses the Mann-Whitney U test (non-parametric) rather than a t-test, since
several features are right-skewed counts (commit_count, pr_count,
open_bugs_at_release) rather than roughly-normal measurements, and several
per-repository group sizes are modest -- Mann-Whitney does not assume
normality the way a t-test does.

Uses ONLY the 'train' split, consistent with correlation_analysis.py and
visualizations.py -- validation/test stay untouched until Week 6 evaluation.

For each feature, reports:
    - median for the not-risky group and the elevated-risk group
    - Mann-Whitney U statistic and p-value
    - rank-biserial correlation (effect size: -1 to +1, analogous to how
      far group ranks are shifted from a 50/50 tie; NOT the same scale as
      Pearson r, but interpretable the same directional way)
    - a Bonferroni-corrected significance flag, since testing ~20 features
      at once inflates the false-positive rate if judged at the raw 0.05
      threshold alone

Run pooled (all repositories) and separately per-repository, mirroring
correlation_analysis.py, since a difference that holds pooled may not hold
within each repository individually (relevant to RQ5).

Produces:
    data/analysis/hypothesis_test_results.csv          -- pooled results
    data/analysis/hypothesis_test_results_by_repo.csv  -- per-repository results

Usage:
    python src/hypothesis_testing.py
"""

from pathlib import Path
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Same exclusion logic as correlation_analysis.py: identifiers, dates, the
# split label, the raw ingredients the target is built from (testing those
# against the target would be circular), and the confirmed-constant
# is_prerelease column.
EXCLUDE_COLS = {
    "release_id", "repository_name", "release_date", "split",
    "bugs_in_window", "risk_threshold_used", "window_complete",
    "is_prerelease", TARGET_COL,
    # Same multicollinearity fix as correlation_analysis.py and every
    # modelling script: open_issues_at_release/prior_releases_count are
    # redundant with features already included; open_bugs_at_release and
    # prior_releases_avg_bugs (raw) are superseded by their repo-normalized
    # _repo_z counterparts -- see preprocess_for_modeling.py.
    "open_issues_at_release", "prior_releases_count",
    "open_bugs_at_release", "prior_releases_avg_bugs",
}


def mann_whitney_test(group0, group1):
    """
    Returns (median0, median1, u_stat, p_value, rank_biserial_effect_size).
    rank_biserial = 1 - (2*U) / (n0*n1); ranges -1 to 1, sign indicates
    which group tends to rank higher (positive means group1 > group0).
    """
    n0, n1 = len(group0), len(group1)
    u_stat, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    rank_biserial = 1 - (2 * u_stat) / (n0 * n1)
    return group0.median(), group1.median(), u_stat, p_value, rank_biserial


def run_hypothesis_tests(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        sub = df[[col, TARGET_COL]].dropna()
        g0 = sub.loc[sub[TARGET_COL] == 0, col]
        g1 = sub.loc[sub[TARGET_COL] == 1, col]
        if len(g0) < 3 or len(g1) < 3 or g0.nunique() <= 1 and g1.nunique() <= 1:
            continue  # not enough data / no variance to test meaningfully
        median0, median1, u, p, effect = mann_whitney_test(g0, g1)
        rows.append({
            "feature": col,
            "median_not_risky": median0, "median_elevated_risk": median1,
            "u_statistic": u, "p_value": p, "rank_biserial_effect_size": effect,
            "n_not_risky": len(g0), "n_elevated_risk": len(g1),
        })
    result = pd.DataFrame(rows).sort_values("p_value")
    if not result.empty:
        n_tests = len(result)
        bonferroni_alpha = 0.05 / n_tests
        result["bonferroni_alpha"] = bonferroni_alpha
        result["significant_bonferroni"] = result["p_value"] < bonferroni_alpha
        result["significant_uncorrected_0.05"] = result["p_value"] < 0.05
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

    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(train[c])]
    feature_cols = [c for c in feature_cols if train[c].nunique() > 1]  # drop zero-variance defensively

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pooled ---
    print("=== Pooled (all repositories) ===")
    pooled = run_hypothesis_tests(train, feature_cols)
    pooled.to_csv(ANALYSIS_DIR / "hypothesis_test_results.csv", index=False)
    print(pooled[["feature", "median_not_risky", "median_elevated_risk", "p_value",
                   "rank_biserial_effect_size", "significant_bonferroni"]].to_string(index=False))
    n_sig_bonf = int(pooled["significant_bonferroni"].sum()) if not pooled.empty else 0
    n_sig_raw = int(pooled["significant_uncorrected_0.05"].sum()) if not pooled.empty else 0
    print(f"\n{n_sig_raw} of {len(pooled)} features significant at uncorrected p<0.05; "
          f"{n_sig_bonf} survive Bonferroni correction (alpha={pooled['bonferroni_alpha'].iloc[0]:.5f} per test)"
          if not pooled.empty else "")

    # --- Per repository ---
    print("\n=== Per repository ===")
    by_repo_frames = []
    for repo_name, g in train.groupby("repository_name"):
        r = run_hypothesis_tests(g, feature_cols)
        if r.empty:
            continue
        r.insert(0, "repository_name", repo_name)
        by_repo_frames.append(r)
        print(f"\n--- {repo_name} ---")
        print(r[["feature", "median_not_risky", "median_elevated_risk", "p_value",
                  "rank_biserial_effect_size", "significant_uncorrected_0.05"]]
              .head(8).to_string(index=False))

    if by_repo_frames:
        by_repo = pd.concat(by_repo_frames, ignore_index=True)
        by_repo.to_csv(ANALYSIS_DIR / "hypothesis_test_results_by_repo.csv", index=False)
        print(f"\nWrote per-repository results to {ANALYSIS_DIR / 'hypothesis_test_results_by_repo.csv'}")

    print(f"\nWrote pooled results to {ANALYSIS_DIR / 'hypothesis_test_results.csv'}")
    print("\n[NOTE] With ~20 features tested simultaneously, some 'significant' results at the raw "
          "p<0.05 threshold are expected by chance alone. The Bonferroni-corrected flag is the more "
          "conservative, defensible standard to lead with in your report; treat uncorrected-only "
          "significant features as suggestive, not confirmed.")


if __name__ == "__main__":
    main()
