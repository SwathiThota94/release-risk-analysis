"""
explanatory_regression.py  (lives in src/)
------------------------------------------
Week 5 (Descriptive and Diagnostic Analytics): fits an EXPLANATORY logistic
regression model -- the goal here is understanding which features carry
independent effects on elevated_risk once other features are controlled
for, and whether repository identity matters, NOT optimizing predictive
accuracy (that's Week 6). Uses ONLY the 'train' split.

statsmodels is not available in this environment, so standard errors,
z-scores, p-values, and the likelihood-ratio test are computed manually
using the standard Wald-test formulas for logistic regression:
    - Point estimates: sklearn LogisticRegression, unregularized (mirrors
      classical MLE when penalty=None).
    - Covariance matrix of coefficients: inv(X^T W X), where W is a
      diagonal matrix of p_hat*(1-p_hat) at the fitted probabilities --
      this is the standard Fisher-information-based estimator statsmodels
      itself uses for a Logit model.
    - Wald z = coef / se; two-sided p-value from the standard normal.
    - Likelihood-ratio test for repository effects: fit the model with and
      without the repository dummies, compare 2*(LL_full - LL_reduced)
      against a chi-square distribution with df = difference in parameter
      counts.

Feature set deliberately excludes several redundant columns identified
during correlation analysis (see comments below) to keep the design
matrix numerically stable and coefficients interpretable -- an explanatory
model with severe multicollinearity produces unstable, misleading
coefficients even if predictive accuracy elsewhere is unaffected.

Produces, under data/analysis/:
    regression_coefficients.csv   -- coefficient, SE, z, p, odds ratio, 95% CI per feature
    regression_vif.csv            -- variance inflation factor per feature (multicollinearity check)
    regression_repo_effect_lr_test.csv  -- likelihood-ratio test result for repository effects

Usage:
    python src/explanatory_regression.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Base continuous/count features. Excluded, with rationale:
#   - median_time_to_merge_hours kept, avg_time_to_merge_hours dropped (r=0.93 with median -- near redundant)
#   - release_sequence kept, prior_releases_count dropped (r=0.999 with release_sequence -- near-perfect duplicate)
#   - had_commit_activity / had_pr_activity / had_contributor_activity / has_prior_release /
#     has_prior_release_history all dropped: had_commit_activity and had_contributor_activity
#     are an exact duplicate (r=1.0); the others are largely redundant with cycle_length_days
#     and release_sequence already being in the model, and add collinearity without adding
#     distinct explanatory value.
BASE_FEATURES = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "median_time_to_merge_hours", "distinct_contributors",
    "first_time_contributor_share", "top_contributor_share",
]
# open_bugs_at_release and prior_releases_avg_bugs are handled SEPARATELY,
# not via the pooled standardize() below -- see repo_standardize() and the
# note in main(). Both were found correlated at r=0.85-0.98 with
# repo_microsoft/vscode in earlier correlation analysis (VS Code's absolute
# backlog/bug-history scale is thousands of issues larger than the other two
# repos' scale). Standardizing them POOLED, then interacting with repo
# dummies, produced a design matrix with VIF = inf (a near-perfectly
# singular matrix -- repo identity was being encoded 5 different redundant
# ways at once). Standardizing WITHIN each repository instead measures "is
# this backlog high or low relative to THIS repo's own typical level" --
# genuinely repo-independent information -- and removes the confound.
REPO_CONFOUNDED_FEATURES = ["open_bugs_at_release", "prior_releases_avg_bugs"]
REPO_DUMMY_COLS = ["repo_kubernetes/kubernetes", "repo_microsoft/vscode"]


def repo_standardize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Z-score standardization computed WITHIN each repository_name group
    (using that group's own mean/std), not pooled across repos. This is
    what removes the repo-scale confound for backlog-type features -- see
    REPO_CONFOUNDED_FEATURES above."""
    out = df[cols].copy()
    for col in cols:
        out[col] = df.groupby("repository_name")[col].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1)
        )
        # Defensive backstop: a group with very few rows can produce a NaN
        # standard deviation (e.g. pandas' default ddof=1 is undefined for
        # a single-row group); treat "can't standardize meaningfully" as 0
        # (the group's own mean) rather than let NaN propagate silently.
        n_nan = out[col].isna().sum()
        if n_nan > 0:
            print(f"  [note] repo_standardize: {n_nan} NaN value(s) in '{col}' "
                  f"(likely a repository group too small to compute a standard "
                  f"deviation) -- filled with 0.")
            out[col] = out[col].fillna(0)
    return out


def standardize(df: pd.DataFrame, cols: list) -> tuple:
    """Z-score standardization fit on the given data only (train). Returns
    the standardized frame plus the means/stds used, so the same
    transform could be reapplied to other splits later if needed."""
    means = df[cols].mean()
    stds = df[cols].std().replace(0, 1)
    standardized = (df[cols] - means) / stds
    return standardized, means, stds


def fit_logistic(X: np.ndarray, y: np.ndarray):
    """Fits an effectively-unregularized logistic regression (C set very
    large to minimize shrinkage, mirroring classical MLE), returns
    (coefs_with_intercept, model)."""
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    model.fit(X, y)
    coefs = np.concatenate([model.intercept_, model.coef_.ravel()])
    return coefs, model


def wald_inference(X_with_const: np.ndarray, y: np.ndarray, coefs: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Manual Wald standard errors / z / p-values, matching what
    statsmodels' Logit.fit() would report, via the Fisher information
    matrix inv(X^T W X)."""
    z = X_with_const @ coefs
    p_hat = 1 / (1 + np.exp(-z))
    W = np.diag(p_hat * (1 - p_hat))

    # Small ridge term for numerical stability given remaining, non-exact
    # multicollinearity in the design matrix -- documented in the VIF output.
    xtwx = X_with_const.T @ W @ X_with_const
    xtwx_stable = xtwx + np.eye(xtwx.shape[0]) * 1e-8
    cov = np.linalg.inv(xtwx_stable)
    se = np.sqrt(np.diag(cov))

    z_scores = coefs / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    ci_low = coefs - 1.96 * se
    ci_high = coefs + 1.96 * se

    result = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "std_err": se,
        "z": z_scores,
        "p_value": p_values,
        "odds_ratio": np.exp(coefs),
        "ci_low_odds_ratio": np.exp(ci_low),
        "ci_high_odds_ratio": np.exp(ci_high),
    })
    result["significant_0.05"] = result["p_value"] < 0.05
    return result


def log_likelihood(X_with_const: np.ndarray, y: np.ndarray, coefs: np.ndarray) -> float:
    z = X_with_const @ coefs
    p_hat = 1 / (1 + np.exp(-z))
    eps = 1e-12
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return float(np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))


def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per feature: regress each feature on all
    others (OLS via lstsq), VIF = 1 / (1 - R^2)."""
    rows = []
    cols = X_df.columns.tolist()
    for col in cols:
        y_ = X_df[col].values
        X_ = X_df.drop(columns=[col]).values
        X_ = np.column_stack([np.ones(len(X_)), X_])
        coefs, *_ = np.linalg.lstsq(X_, y_, rcond=None)
        y_pred = X_ @ coefs
        ss_res = np.sum((y_ - y_pred) ** 2)
        ss_tot = np.sum((y_ - y_.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 0.999 else np.inf
        rows.append({"feature": col, "r_squared_vs_other_features": r2, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def main():
    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found -- run run_features.py and preprocess_for_modeling.py first")
        return

    df = pd.read_csv(INPUT_PATH)
    if "split" not in df.columns:
        print("[error] No 'split' column found -- re-run run_features.py with the updated script.")
        return

    train = df[df["split"] == "train"].copy()
    train = train.dropna(subset=BASE_FEATURES + REPO_CONFOUNDED_FEATURES + REPO_DUMMY_COLS + [TARGET_COL])
    print(f"Loaded {len(df)} total releases; using {len(train)} 'train' split releases "
          f"(after dropping rows with missing values in the model features).\n")

    y = train[TARGET_COL].values.astype(float)

    # Standardize base features pooled (these were not found repo-confounded
    # in the correlation analysis), and the two backlog features WITHIN each
    # repository (see REPO_CONFOUNDED_FEATURES note above).
    X_std, means, stds = standardize(train, BASE_FEATURES)
    X_backlog = repo_standardize(train, REPO_CONFOUNDED_FEATURES)
    X_full = pd.concat([
        X_std.reset_index(drop=True),
        X_backlog.reset_index(drop=True),
        train[REPO_DUMMY_COLS].reset_index(drop=True),
    ], axis=1)

    # --- Interaction term ---
    # commit_count x first_time_contributor_share: tests whether contributor
    # turnover matters more (or less) when release size/activity is also
    # high. (The open_bugs x repository interaction terms used in an earlier
    # version of this script are REMOVED: now that open_bugs_at_release is
    # repo-standardized rather than repo-confounded, a separate interaction
    # term is no longer the right way to test "does the backlog effect
    # differ by repo" -- that question is already answered more reliably by
    # the per-repository Mann-Whitney tests in hypothesis_testing.py, which
    # directly found opposite-signed effects for VS Code vs. the other two
    # repos without needing this pooled model to reproduce it.)
    X_full["commit_count_x_first_time_contrib"] = (
        X_full["commit_count"] * X_full["first_time_contributor_share"]
    )

    # Defensive check: diagnose and handle any NaN that made it into the
    # design matrix, rather than let sklearn crash uninformatively. This can
    # happen if a repository has very few rows for a REPO_CONFOUNDED_FEATURE
    # (a single-row group's standard deviation is undefined) or if an
    # upstream pipeline change introduced a gap not caught by earlier
    # NaN checks (re-run the full pipeline's own NaN checks if this fires
    # to track down the root cause).
    nan_counts = X_full.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f"\n[WARNING] NaN found in the design matrix after feature construction:")
        print(nan_cols.to_string())
        rows_before = len(X_full)
        valid_mask = ~X_full.isna().any(axis=1)
        X_full = X_full[valid_mask].reset_index(drop=True)
        y = y[valid_mask.values]
        print(f"Dropped {rows_before - len(X_full)} row(s) containing NaN; "
              f"{len(X_full)} rows remain for fitting. Investigate the flagged "
              f"column(s) above if this count is more than a handful of rows.")

    feature_names = ["intercept"] + X_full.columns.tolist()
    X_matrix = np.column_stack([np.ones(len(X_full)), X_full.values])

    print("Fitting full explanatory logistic regression "
          f"({len(feature_names) - 1} predictors + intercept)...")
    coefs, _ = fit_logistic(X_full.values, y)
    results = wald_inference(X_matrix, y, coefs, feature_names)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ANALYSIS_DIR / "regression_coefficients.csv", index=False)
    print("\n=== Coefficients (standardized features; repo dummies and interactions unstandardized 0/1 or product terms) ===")
    print(results.sort_values("p_value")[
        ["feature", "coef", "std_err", "p_value", "odds_ratio", "significant_0.05"]
    ].to_string(index=False))

    # --- VIF check on the (non-intercept) design matrix ---
    vif_df = compute_vif(X_full)
    vif_df.to_csv(ANALYSIS_DIR / "regression_vif.csv", index=False)
    print("\n=== Variance Inflation Factors (VIF > 10 signals problematic multicollinearity) ===")
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df["VIF"] > 10]
    if not high_vif.empty:
        print(f"\n[WARNING] {len(high_vif)} feature(s) show VIF > 10: {high_vif['feature'].tolist()}. "
              "Coefficients for these specific features should be interpreted cautiously.")

    # --- Likelihood-ratio test: do repository effects matter, jointly? ---
    # Reduced model: same features, MINUS the two repo dummies and the two
    # repo-interaction terms (since those are also repository-specific).
    reduced_cols = [c for c in X_full.columns if c not in REPO_DUMMY_COLS]
    X_reduced = X_full[reduced_cols].values
    X_reduced_const = np.column_stack([np.ones(len(X_reduced)), X_reduced])
    coefs_reduced, _ = fit_logistic(X_reduced, y)

    ll_full = log_likelihood(X_matrix, y, coefs)
    ll_reduced = log_likelihood(X_reduced_const, y, coefs_reduced)
    df_diff = X_matrix.shape[1] - X_reduced_const.shape[1]
    lr_stat = 2 * (ll_full - ll_reduced)
    lr_p = stats.chi2.sf(lr_stat, df_diff)

    lr_result = pd.DataFrame([{
        "log_likelihood_full_model": ll_full,
        "log_likelihood_reduced_model_no_repo_terms": ll_reduced,
        "lr_statistic": lr_stat,
        "degrees_of_freedom": df_diff,
        "p_value": lr_p,
        "repository_effects_significant_0.05": lr_p < 0.05,
    }])
    lr_result.to_csv(ANALYSIS_DIR / "regression_repo_effect_lr_test.csv", index=False)
    print("\n=== Likelihood-Ratio Test: Does repository identity (dummies) matter, jointly? ===")
    print(lr_result.to_string(index=False))
    if lr_p < 0.05:
        print("-> Repository effects ARE statistically significant: repository identity carries "
              "real explanatory power beyond the other features in the model.")
    else:
        print("-> Repository effects are NOT statistically significant once other features are controlled for.")


if __name__ == "__main__":
    main()
