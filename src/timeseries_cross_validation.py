"""
timeseries_cross_validation.py  (lives in src/)
------------------------------------------
Supplementary analysis: time-series cross-validation on the final logistic
regression specification, using sklearn's TimeSeriesSplit (expanding-
window / rolling-origin CV).

Unlike standard k-fold cross-validation (see kfold_cross_validation.py),
this approach respects chronological order: each fold trains only on
releases that occurred BEFORE the releases in that fold's validation
portion, and the training window expands with each successive fold. This
is consistent with -- not a departure from -- this project's primary
leakage-prevention methodology (see predictive_modeling.py, Section 5.1),
and can be read as a robustness check on the single time-based
train/validation/test split used elsewhere: does model performance hold
up across MULTIPLE chronological splits, not just the one split reported
as the project's primary result.

Runs on the full dataset sorted by release_date (n=555), using the same
19-feature specification as the final model. n_splits=5 by default,
producing 5 expanding-window folds; the earliest fold has the smallest
training set and the latest fold's training set is the largest.

Produces, under data/analysis/:
    timeseries_cv_results.csv

Usage:
    python src/timeseries_cross_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
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
N_SPLITS = 5


def main():
    print("=" * 70)
    print("SUPPLEMENTARY ANALYSIS: time-series cross-validation (TimeSeriesSplit)")
    print("Each fold trains only on releases occurring BEFORE that fold's")
    print("validation releases -- chronological order is preserved throughout,")
    print("consistent with this project's primary leakage-prevention approach")
    print("(Section 5.1). This is a robustness check across multiple")
    print("chronological splits, not a departure from the project's methodology.")
    print("=" * 70 + "\n")

    df = pd.read_csv(INPUT_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df.sort_values("release_date").reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)
    dates = df["release_date"]
    print(f"Full dataset, sorted chronologically: n={len(df)}, "
          f"positive_rate={y.mean():.3f}, "
          f"date range {dates.min().date()} to {dates.max().date()}\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_dates = dates.iloc[train_idx]
        val_dates = dates.iloc[val_idx]

        if len(np.unique(y_train)) < 2:
            print(f"  [fold {fold_idx}] skipped -- training portion has only one class present")
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(C=FINAL_C, class_weight="balanced", max_iter=2000, solver="lbfgs")
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_val_scaled)[:, 1]
        preds = model.predict(X_val_scaled)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(y_train), "n_val": len(y_val),
            "train_end_date": train_dates.max().date(),
            "val_start_date": val_dates.min().date(),
            "val_end_date": val_dates.max().date(),
            "train_positive_rate": y_train.mean(), "val_positive_rate": y_val.mean(),
            "roc_auc": roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_val, proba) if len(np.unique(y_val)) > 1 else np.nan,
            "brier_score": brier_score_loss(y_val, proba),
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
        })
        r = fold_results[-1]
        print(f"  [fold {fold_idx}] train n={r['n_train']} (through {r['train_end_date']}, "
              f"pos_rate={r['train_positive_rate']:.3f})  ->  "
              f"val n={r['n_val']} ({r['val_start_date']} to {r['val_end_date']}, "
              f"pos_rate={r['val_positive_rate']:.3f})")
        print(f"           ROC-AUC={r['roc_auc']:.4f}  Precision={r['precision']:.4f}  "
              f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}  Accuracy={r['accuracy']:.4f}\n")

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(ANALYSIS_DIR / "timeseries_cv_results.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'timeseries_cv_results.csv'}")

    print("\n=== Summary (mean +/- std across valid folds) ===")
    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
        print(f"  {metric:<10} mean={results_df[metric].mean():.4f}  std={results_df[metric].std():.4f}")

    print("\n[NOTE] Unlike standard k-fold, each fold here has a DIFFERENT, expanding")
    print("training window and a chronologically later validation window -- fold-to-")
    print("fold variation is expected and reflects genuine differences in the data")
    print("available and the base rate present at each point in the project's timeline,")
    print("not random noise from arbitrary partitioning.")


if __name__ == "__main__":
    main()
