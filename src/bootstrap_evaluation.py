"""
bootstrap_evaluation.py  (lives in src/)
------------------------------------------
Two things missing from the model comparison so far, both important given
how small the evaluation sets are (test set n=111):

1. BOOTSTRAP CONFIDENCE INTERVALS. Every metric reported so far (ROC-AUC,
   precision, recall, etc.) is a single point estimate from 111 test rows.
   This resamples the test set with replacement many times, recomputing
   metrics each time, to report a 95% confidence interval rather than a
   bare point estimate -- e.g. "ROC-AUC 0.91 (95% CI: 0.85-0.97)" is a much
   more honest claim than "0.91" alone on a sample this size.

2. PAIRED BOOTSTRAP MODEL COMPARISON. The project has compared several
   models' ROC-AUC point estimates (plain logistic regression, recency-
   weighted logistic regression, Elastic Net) without ever testing whether
   the differences between them are real or just noise on 111 rows. This
   uses the SAME bootstrap resample (paired) across all three models each
   iteration, computes the ROC-AUC difference each time, and reports what
   share of iterations favor each model plus a 95% CI for the difference --
   if that CI excludes 0, the difference is likely real; if it straddles 0,
   the models are statistically indistinguishable on this test set.

Also bootstraps precision/recall/F1 at the project's selected operating
threshold (0.32, chosen via threshold_analysis.py) for the final model.

Produces, under data/analysis/:
    bootstrap_roc_auc_cis.csv           -- CI per model
    bootstrap_pairwise_comparison.csv   -- paired differences between model pairs
    bootstrap_threshold_metrics_ci.csv  -- CI for precision/recall/F1 at the chosen threshold

Usage:
    python src/bootstrap_evaluation.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"
# Updated after the multicollinearity fix (see preprocess_for_modeling.py):
# open_issues_at_release removed (r=0.965 with open_bugs_at_release --
# redundant); prior_releases_count removed (r=0.999 with release_sequence
# -- redundant); open_bugs_at_release and prior_releases_avg_bugs replaced
# with their repo-normalized (_repo_z) versions, which resolved a severe
# confound with repository identity (correlation with repo_microsoft/vscode
# dropped from 0.969 to -0.087 after normalization).
FEATURE_COLS = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs_repo_z",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CONFIG_PATH = MODELS_DIR / "final_model_info.json"

MODELS_TO_COMPARE = {
    "logistic_regression": MODELS_DIR / "logistic_regression_model.joblib",
    "recency_weighted_logistic": MODELS_DIR / "recency_weighted_logistic_model.joblib",
    "elastic_net": MODELS_DIR / "elastic_net_model.joblib",
}


def load_final_model_config():
    """
    Reads model_name and threshold from the shared config file written by
    finalize_model.py and threshold_analysis.py, rather than hardcoding
    them here -- avoids this script silently going stale when the final
    model selection changes (as happened when the multicollinearity fix
    changed the final model from recency-weighted to plain logistic
    regression, but this constant was not updated to match).
    """
    if not CONFIG_PATH.exists():
        print(f"[warning] {CONFIG_PATH} not found -- run finalize_model.py and threshold_analysis.py first. "
              "Falling back to defaults (logistic_regression, threshold=0.5), which may not reflect the "
              "project's actual current final model choice.")
        return "logistic_regression", 0.5
    config = json.loads(CONFIG_PATH.read_text())
    model_name = config.get("model_name", "logistic_regression")
    threshold = config.get("threshold", 0.5)
    return model_name, threshold


def load_test_set():
    df = pd.read_csv(INPUT_PATH)
    test = df[df["split"] == "test"].copy()
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_test = scaler.transform(test[FEATURE_COLS])
    y_test = test[TARGET_COL].values.astype(int)
    return X_test, y_test


def get_probabilities(models, X_test):
    proba = {}
    for name, path in models.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        model = joblib.load(path)
        proba[name] = model.predict_proba(X_test)[:, 1]
    return proba


def bootstrap_indices(n, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_bootstrap)]


def main():
    final_model_name, chosen_threshold = load_final_model_config()
    print(f"Final model (from {CONFIG_PATH.name}): {final_model_name}, threshold: {chosen_threshold}\n")

    X_test, y_test = load_test_set()
    n = len(y_test)
    print(f"Test set: n={n}, positive_rate={y_test.mean():.3f}\n")

    proba = get_probabilities(MODELS_TO_COMPARE, X_test)
    if final_model_name not in proba:
        print(f"[error] final model '{final_model_name}' not available -- cannot proceed")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    idx_sets = bootstrap_indices(n, N_BOOTSTRAP, RANDOM_SEED)

    print(f"=== Bootstrap ROC-AUC 95% CIs ({N_BOOTSTRAP} resamples) ===")
    auc_samples = {name: [] for name in proba}
    n_skipped = 0
    for idx in idx_sets:
        y_resampled = y_test[idx]
        if len(np.unique(y_resampled)) < 2:
            n_skipped += 1
            continue
        for name, p in proba.items():
            auc_samples[name].append(roc_auc_score(y_resampled, p[idx]))

    ci_rows = []
    for name, samples in auc_samples.items():
        samples = np.array(samples)
        point_estimate = roc_auc_score(y_test, proba[name])
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        ci_rows.append({
            "model": name, "point_estimate_roc_auc": point_estimate,
            "bootstrap_mean": samples.mean(), "ci_low_95": ci_low, "ci_high_95": ci_high,
            "n_valid_bootstrap_iters": len(samples),
        })
        print(f"  {name:<28} ROC-AUC = {point_estimate:.4f}  (95% CI: {ci_low:.4f} - {ci_high:.4f}, "
              f"bootstrap mean={samples.mean():.4f})")
    if n_skipped > 0:
        print(f"  ({n_skipped} of {N_BOOTSTRAP} bootstrap resamples skipped -- only one class present)")

    pd.DataFrame(ci_rows).to_csv(ANALYSIS_DIR / "bootstrap_roc_auc_cis.csv", index=False)

    print(f"\n=== Paired Bootstrap Comparison (same resamples across models) ===")
    model_names = list(proba.keys())
    pairwise_rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            diffs = []
            wins_a = 0
            for idx in idx_sets:
                y_resampled = y_test[idx]
                if len(np.unique(y_resampled)) < 2:
                    continue
                auc_a = roc_auc_score(y_resampled, proba[name_a][idx])
                auc_b = roc_auc_score(y_resampled, proba[name_b][idx])
                diffs.append(auc_a - auc_b)
                if auc_a > auc_b:
                    wins_a += 1
            diffs = np.array(diffs)
            ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
            # Inclusive comparison: a CI that touches 0 at either edge (or,
            # degenerately, equals exactly [0, 0] when two models produce
            # identical predictions) must NOT be reported as a real
            # difference. A strict '<' here was found, during testing, to
            # incorrectly flag two literally identical models as
            # "significantly different" whenever their CI collapsed to
            # exactly [0, 0] -- fixed by requiring the CI to exclude 0
            # with a margin, not just fail to contain it at the boundary.
            significant = not (ci_low <= 0 <= ci_high)
            pct_a_wins = wins_a / len(diffs) * 100
            pairwise_rows.append({
                "model_a": name_a, "model_b": name_b,
                "mean_auc_diff_a_minus_b": diffs.mean(),
                "ci_low_95": ci_low, "ci_high_95": ci_high,
                "pct_iterations_a_better": pct_a_wins,
                "difference_likely_real": significant,
            })
            verdict = "LIKELY REAL DIFFERENCE" if significant else "NOT statistically distinguishable"
            print(f"  {name_a} vs {name_b}: mean diff={diffs.mean():+.4f} "
                  f"(95% CI: {ci_low:+.4f} to {ci_high:+.4f}) -- {name_a} better in {pct_a_wins:.1f}% "
                  f"of resamples -- {verdict}")

    pd.DataFrame(pairwise_rows).to_csv(ANALYSIS_DIR / "bootstrap_pairwise_comparison.csv", index=False)

    print(f"\n=== Bootstrap CI for final model at chosen threshold ({chosen_threshold}) ===")
    final_proba = proba[final_model_name]
    metric_samples = {"precision": [], "recall": [], "f1": []}
    for idx in idx_sets:
        y_resampled = y_test[idx]
        preds = (final_proba[idx] >= chosen_threshold).astype(int)
        if len(np.unique(y_resampled)) < 2:
            continue
        metric_samples["precision"].append(precision_score(y_resampled, preds, zero_division=0))
        metric_samples["recall"].append(recall_score(y_resampled, preds, zero_division=0))
        metric_samples["f1"].append(f1_score(y_resampled, preds, zero_division=0))

    threshold_ci_rows = []
    point_preds = (final_proba >= chosen_threshold).astype(int)
    point_metrics = {
        "precision": precision_score(y_test, point_preds, zero_division=0),
        "recall": recall_score(y_test, point_preds, zero_division=0),
        "f1": f1_score(y_test, point_preds, zero_division=0),
    }
    for metric_name, samples in metric_samples.items():
        samples = np.array(samples)
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        threshold_ci_rows.append({
            "metric": metric_name, "point_estimate": point_metrics[metric_name],
            "ci_low_95": ci_low, "ci_high_95": ci_high,
        })
        print(f"  {metric_name:<10} = {point_metrics[metric_name]:.3f}  (95% CI: {ci_low:.3f} - {ci_high:.3f})")

    pd.DataFrame(threshold_ci_rows).to_csv(ANALYSIS_DIR / "bootstrap_threshold_metrics_ci.csv", index=False)

    print(f"\nWrote bootstrap results to {ANALYSIS_DIR}")
    print("\n[NOTE] Given the test set's small size (n=111), wide confidence intervals are expected")
    print("and should be reported alongside any point estimate in the final write-up, rather than")
    print("presenting a single number as if it were precisely known.")


if __name__ == "__main__":
    main()
