"""
compute_all_model_metrics.py
-----------------------------
Standalone script: loads every saved model in data/models/, re-runs it on
the test split, and computes precision, recall, F1, and accuracy at the
0.5 threshold for each -- to fill out the full model-comparison table in
Section 5 (Table 12), which currently only has ROC-AUC/PR-AUC/Brier/accuracy.

Does not modify any existing pipeline script. Run this from the repo root
or from src/ -- paths resolve relative to this file's location.

Usage:
    python compute_all_model_metrics.py
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "data" / "models"
DATA_PATH = REPO_ROOT / "data" / "features" / "model_ready_release_level.csv"

# Map each saved model file to a friendly name matching Table 12.
# Adjust filenames here if yours differ -- check `ls data/models/` to confirm.
MODEL_FILES = {
    "logistic_regression_C=0.001": "logistic_regression_model.joblib",
    "elastic_net": "elastic_net_model.joblib",
    "recency_weighted_logistic": "recency_weighted_logistic_model.joblib",
    "naive_bayes": "naive_bayes_model.joblib",
    "decision_tree": "decision_tree_model.joblib",
    "random_forest": "random_forest_model.joblib",
    "gradient_boosting": "gradient_boosting_model.joblib",
}

FEATURE_COLUMNS = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs_repo_z",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]
# ^ confirmed exact 19-feature list from predictive_modeling.py's FEATURE_COLS.
# is_prerelease (constant) and had_contributor_activity (r=1.0 duplicate of
# had_commit_activity) are excluded, matching that script exactly.


def main():
    df = pd.read_csv(DATA_PATH)
    test = df[df["split"] == "test"]
    y_test = test["elevated_risk"]

    scaler_path = MODELS_DIR / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    results = []
    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"[skip] {model_path} not found")
            continue

        model = joblib.load(model_path)
        X_test = test[FEATURE_COLUMNS].copy()

        # Tree-based models typically don't need scaling; linear models do.
        # Adjust this condition if your pipeline scales everything uniformly.
        needs_scaling = model_name in {
            "logistic_regression_C=0.001", "elastic_net",
            "recency_weighted_logistic", "naive_bayes",
        }
        # predictive_modeling.py scales all features together in one call
        # (scaler.fit_transform(X) on the full FEATURE_COLS list) -- not a
        # partial continuous-only split, so match that exactly here.
        if needs_scaling and scaler is not None:
            X_test = pd.DataFrame(
                scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index,
            )

        y_pred = model.predict(X_test)

        results.append({
            "model": model_name,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
        })
        print(f"{model_name}: precision={results[-1]['precision']:.3f} "
              f"recall={results[-1]['recall']:.3f} f1={results[-1]['f1']:.3f} "
              f"accuracy={results[-1]['accuracy']:.3f}")

    out_df = pd.DataFrame(results)
    out_path = REPO_ROOT / "data" / "analysis" / "all_model_precision_recall_f1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
