"""
kfold_cross_validation.py  (lives in src/)
------------------------------------------
Supplementary analysis requested during report review: standard stratified
k-fold cross-validation on the final logistic regression specification.

IMPORTANT METHODOLOGICAL NOTE: this script deliberately does something the
rest of this project's pipeline avoids. Every other script (predictive_
modeling.py, tree_based_models.py, advanced_models.py, xgboost_model.py)
uses a strict chronological train/validation/test split specifically to
prevent future releases from leaking into predictions evaluated on earlier
ones. Standard k-fold cross-validation shuffles observations randomly
across folds, which reintroduces exactly that leakage on time-ordered data
-- a fold's "test" portion can easily contain releases that occurred
chronologically BEFORE releases in that fold's "training" portion.

This script exists as a supplementary comparison only, run on request, not
as a replacement for the project's primary time-based evaluation
methodology (Section 5.1). Results here should be read as "how would this
model perform under a leakage-permissive evaluation scheme conventionally
used for i.i.d. data" -- not as a more rigorous or more trustworthy
estimate than the time-based split results reported elsewhere.

Runs 5-fold and 10-fold stratified k-fold CV (StratifiedKFold preserves
the overall class ratio in each fold, appropriate given the ~24% pooled
positive rate) on the pooled full dataset (train+validation+test combined,
n=555), using the same 19-feature specification as the final model.

Produces, under data/analysis/:
    kfold_cv_results.csv

Usage:
    python src/kfold_cross_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Identical 19-feature, final-model specification used throughout this
# project (predictive_modeling.py, tree_based_models.py, advanced_models.py).
FEATURE_COLS = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs_repo_z",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]

FINAL_C = 0.001  # matches the final selected logistic regression (predictive_modeling.py)


def run_kfold(X, y, n_splits, label):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(C=FINAL_C, class_weight="balanced", max_iter=2000, solver="lbfgs")
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)

        fold_results.append({
            "cv_scheme": label, "fold": fold_idx, "n_test": len(y_test),
            "positive_rate": y_test.mean(),
            "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
            "brier_score": brier_score_loss(y_test, proba),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })
        print(f"  [{label} fold {fold_idx}] ROC-AUC={fold_results[-1]['roc_auc']:.4f}  "
              f"Precision={fold_results[-1]['precision']:.4f}  Recall={fold_results[-1]['recall']:.4f}  "
              f"F1={fold_results[-1]['f1']:.4f}  Accuracy={fold_results[-1]['accuracy']:.4f}  "
              f"(n={fold_results[-1]['n_test']}, pos_rate={fold_results[-1]['positive_rate']:.3f})")

    return fold_results


def main():
    print("=" * 70)
    print("SUPPLEMENTARY ANALYSIS: standard k-fold cross-validation")
    print("This evaluation scheme shuffles releases randomly across folds and")
    print("does NOT preserve chronological ordering. It permits information")
    print("from later releases to appear in a fold's training portion while")
    print("earlier releases appear in that same fold's test portion -- the")
    print("exact leakage risk this project's primary methodology (strict")
    print("time-based train/validation/test split, see Section 5.1) was")
    print("specifically designed to avoid. Results below should be read as")
    print("a supplementary comparison only, not a replacement for the")
    print("project's primary evaluation results.")
    print("=" * 70 + "\n")

    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)
    print(f"Pooled dataset (train+validation+test combined): n={len(df)}, "
          f"positive_rate={y.mean():.3f}\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=== 5-Fold Stratified Cross-Validation ===")
    all_results.extend(run_kfold(X, y, n_splits=5, label="5-fold"))

    print("\n=== 10-Fold Stratified Cross-Validation ===")
    all_results.extend(run_kfold(X, y, n_splits=10, label="10-fold"))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(ANALYSIS_DIR / "kfold_cv_results.csv", index=False)
    print(f"\nWrote {ANALYSIS_DIR / 'kfold_cv_results.csv'}")

    print("\n=== Summary (mean +/- std across folds) ===")
    for scheme in ["5-fold", "10-fold"]:
        sub = results_df[results_df["cv_scheme"] == scheme]
        print(f"\n{scheme}:")
        for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
            print(f"  {metric:<10} mean={sub[metric].mean():.4f}  std={sub[metric].std():.4f}")

    print("\n[REMINDER] Compare these figures against the time-based split results")
    print("in Table 12 (test ROC-AUC=0.929 for this same model specification).")
    print("A meaningful difference between the two is expected and informative --")
    print("it reflects the leakage this project's primary methodology avoids,")
    print("not an error in either evaluation.")


if __name__ == "__main__":
    main()
