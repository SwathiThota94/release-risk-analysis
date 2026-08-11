"""
xgboost_model.py  (lives in src/)
------------------------------------------
Follow-up model requested during report review: Extreme Gradient Boosting
(XGBoost), evaluated using the same validation-tuned methodology as
predictive_modeling.py and tree_based_models.py.

Uses the identical 19-feature specification (post multicollinearity fix --
see preprocess_for_modeling.py) as every other model in this project, for
direct comparability in model_comparison.csv.

Like tree_based_models.py's GradientBoostingClassifier, XGBoost does not
use class_weight; class-imbalance handling uses XGBoost's native
scale_pos_weight parameter instead, set from the TRAINING set's class
ratio, consistent with the class-imbalance handling approach used
throughout this project (e.g. class_weight="balanced" for logistic
regression, compute_sample_weight for sklearn's GradientBoostingClassifier).

All hyperparameter tuning is done against the VALIDATION set specifically,
not k-fold cross-validation, for the same time-ordering/leakage-prevention
reason established throughout this project (see predictive_modeling.py).

Produces, under data/models/:
    xgboost_hyperparameter_grid.csv   -- full grid, all combinations tried
    xgboost_model.joblib
Appends to data/models/model_comparison.csv.

Usage:
    python src/xgboost_model.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

TARGET_COL = "elevated_risk"

# Identical 19-feature specification used across predictive_modeling.py,
# tree_based_models.py, and advanced_models.py -- see preprocess_for_modeling.py
# for the multicollinearity-fix rationale (repo-z-scored backlog features,
# redundant prior_releases_count/open_issues_at_release removed).
FEATURE_COLS = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs_repo_z",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]


def load_splits():
    df = pd.read_csv(INPUT_PATH)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def xy(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[TARGET_COL].values.astype(int)


def evaluate(model, X, y, label: str, split: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "model": label, "split": split, "n": len(y), "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, proba),
        "accuracy": accuracy_score(y, preds),
    }


def report_and_collect(model, X_train, y_train, X_val, y_val, X_test, y_test, label, rows):
    for split_name, X_, y_ in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
        r = evaluate(model, X_, y_, label, split_name)
        rows.append(r)
        print(f"  [{split_name}] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"Brier={r['brier_score']:.4f}  Accuracy={r['accuracy']:.4f}  (n={r['n']}, pos_rate={r['positive_rate']:.3f})")


def print_top_importances(model, label, n=8):
    imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top feature importances ({label}):")
    print(imp.head(n).to_string(index=False))


def main():
    train, val, test = load_splits()
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)
    print(f"train={len(train)}, validation={len(val)}, test={len(test)}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # scale_pos_weight: XGBoost's native class-imbalance handling, set from
    # the TRAINING set's class ratio -- analogous to class_weight="balanced"
    # used for logistic regression elsewhere in this project.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight (train neg/pos ratio) = {scale_pos_weight:.4f}\n")

    print("=== XGBoost (tuning max_depth, n_estimators, learning_rate on validation set) ===")
    best_model, best_auc, best_params = None, -np.inf, None
    grid_rows = []
    for max_depth in [2, 3, 4]:
        for n_estimators in [100, 200]:
            for learning_rate in [0.05, 0.1]:
                m = XGBClassifier(
                    max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                    scale_pos_weight=scale_pos_weight, random_state=42,
                    eval_metric="logloss", use_label_encoder=False,
                )
                m.fit(X_train, y_train)
                proba_val = m.predict_proba(X_val)[:, 1]
                preds_val = m.predict(X_val)
                auc = roc_auc_score(y_val, proba_val)
                grid_rows.append({
                    "model": "xgboost", "max_depth": max_depth, "n_estimators": n_estimators,
                    "learning_rate": learning_rate, "validation_roc_auc": auc,
                    "validation_precision": precision_score(y_val, preds_val, zero_division=0),
                    "validation_recall": recall_score(y_val, preds_val, zero_division=0),
                    "validation_f1": f1_score(y_val, preds_val, zero_division=0),
                    "validation_accuracy": accuracy_score(y_val, preds_val),
                })
                if auc > best_auc:
                    best_auc, best_model, best_params = auc, m, (max_depth, n_estimators, learning_rate)

    print(f"Best params: max_depth={best_params[0]}, n_estimators={best_params[1]}, "
          f"learning_rate={best_params[2]} (validation ROC-AUC={best_auc:.4f})")
    report_and_collect(best_model, X_train, y_train, X_val, y_val, X_test, y_test, "xgboost", rows)
    print_top_importances(best_model, "xgboost")
    joblib.dump(best_model, MODELS_DIR / "xgboost_model.joblib")

    # --- Write full hyperparameter grid ---
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(MODELS_DIR / "xgboost_hyperparameter_grid.csv", index=False)
    print(f"\nWrote full hyperparameter grid ({len(grid_df)} combinations) to "
          f"{MODELS_DIR / 'xgboost_hyperparameter_grid.csv'}")

    # --- Append to the shared model comparison table ---
    new_rows_df = pd.DataFrame(rows)
    if COMPARISON_PATH.exists():
        existing = pd.read_csv(COMPARISON_PATH)
        combined = pd.concat([existing, new_rows_df], ignore_index=True)
    else:
        combined = new_rows_df
    combined.to_csv(COMPARISON_PATH, index=False)
    print(f"\nAppended results to {COMPARISON_PATH} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
