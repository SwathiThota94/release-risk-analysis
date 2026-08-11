"""
tree_based_models.py  (lives in src/)
------------------------------------------
Week 6 (Predictive Modelling), Part 2: decision tree, random forest, and
gradient boosting, following the same validation-tuned methodology as
predictive_modeling.py (Part 1: baseline + logistic regression).

Uses the FULL feature set (including repo dummies and release_sequence /
prior_releases_count) -- justified by sensitivity_analysis.py, which found
repo dummies contribute negligibly on their own (test ROC-AUC changes by
< 0.01 when removed) while release_sequence/prior_releases_count carry
substantial genuine out-of-sample signal (removing them alone drops test
ROC-AUC by ~0.13, and removing them together with repo dummies collapses
test ROC-AUC below 0.5) -- so there is no basis for excluding either from
these models.

Tree-based models do not require feature scaling (splits are scale-
invariant), so raw feature values are used directly here, unlike the
StandardScaler step used for logistic regression.

xgboost is not available in this environment (no network access to
install it); sklearn's GradientBoostingClassifier is used for the
"gradient boosting" task instead. GradientBoostingClassifier does not
support class_weight directly, so class-imbalance handling for it uses
per-sample weights (compute_sample_weight) instead of class_weight.

All hyperparameter tuning is done against the VALIDATION set specifically,
not k-fold cross-validation, for the same time-ordering reason established
throughout this project.

Appends to data/models/model_comparison.csv (created by
predictive_modeling.py if it exists; created fresh otherwise).

Usage:
    python src/tree_based_models.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

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
    if not hasattr(model, "feature_importances_"):
        return
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

    # --- Decision Tree ---
    print("=== Decision Tree (tuning max_depth, min_samples_leaf on validation set) ===")
    best_dt, best_dt_auc, best_dt_params = None, -np.inf, None
    grid_rows = []
    for max_depth in [2, 3, 4, 5, 7, None]:
        for min_samples_leaf in [5, 10, 20]:
            m = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        class_weight="balanced", random_state=42)
            m.fit(X_train, y_train)
            proba_val = m.predict_proba(X_val)[:, 1]
            preds_val = m.predict(X_val)
            auc = roc_auc_score(y_val, proba_val)
            grid_rows.append({"model": "decision_tree", "max_depth": max_depth,
                               "min_samples_leaf": min_samples_leaf, "n_estimators": None,
                               "learning_rate": None, "validation_roc_auc": auc,
                               "validation_precision": precision_score(y_val, preds_val, zero_division=0),
                               "validation_recall": recall_score(y_val, preds_val, zero_division=0),
                               "validation_f1": f1_score(y_val, preds_val, zero_division=0),
                               "validation_accuracy": accuracy_score(y_val, preds_val)})
            if auc > best_dt_auc:
                best_dt_auc, best_dt, best_dt_params = auc, m, (max_depth, min_samples_leaf)
    print(f"Best params: max_depth={best_dt_params[0]}, min_samples_leaf={best_dt_params[1]} "
          f"(validation ROC-AUC={best_dt_auc:.4f})")
    report_and_collect(best_dt, X_train, y_train, X_val, y_val, X_test, y_test, "decision_tree", rows)
    print_top_importances(best_dt, "decision_tree")
    joblib.dump(best_dt, MODELS_DIR / "decision_tree_model.joblib")

    # --- Random Forest ---
    print("\n=== Random Forest (tuning n_estimators, max_depth on validation set) ===")
    best_rf, best_rf_auc, best_rf_params = None, -np.inf, None
    for n_estimators in [100, 300]:
        for max_depth in [3, 5, 7, None]:
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
            m.fit(X_train, y_train)
            proba_val = m.predict_proba(X_val)[:, 1]
            preds_val = m.predict(X_val)
            auc = roc_auc_score(y_val, proba_val)
            grid_rows.append({"model": "random_forest", "max_depth": max_depth,
                               "min_samples_leaf": None, "n_estimators": n_estimators,
                               "learning_rate": None, "validation_roc_auc": auc,
                               "validation_precision": precision_score(y_val, preds_val, zero_division=0),
                               "validation_recall": recall_score(y_val, preds_val, zero_division=0),
                               "validation_f1": f1_score(y_val, preds_val, zero_division=0),
                               "validation_accuracy": accuracy_score(y_val, preds_val)})
            if auc > best_rf_auc:
                best_rf_auc, best_rf, best_rf_params = auc, m, (n_estimators, max_depth)
    print(f"Best params: n_estimators={best_rf_params[0]}, max_depth={best_rf_params[1]} "
          f"(validation ROC-AUC={best_rf_auc:.4f})")
    report_and_collect(best_rf, X_train, y_train, X_val, y_val, X_test, y_test, "random_forest", rows)
    print_top_importances(best_rf, "random_forest")
    joblib.dump(best_rf, MODELS_DIR / "random_forest_model.joblib")

    # --- Gradient Boosting (sklearn substitute for XGBoost -- see module docstring) ---
    print("\n=== Gradient Boosting (tuning n_estimators, max_depth, learning_rate on validation set) ===")
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    best_gb, best_gb_auc, best_gb_params = None, -np.inf, None
    for n_estimators in [100, 200]:
        for max_depth in [2, 3]:
            for learning_rate in [0.05, 0.1]:
                m = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                learning_rate=learning_rate, random_state=42)
                m.fit(X_train, y_train, sample_weight=sample_weight_train)
                proba_val = m.predict_proba(X_val)[:, 1]
                preds_val = m.predict(X_val)
                auc = roc_auc_score(y_val, proba_val)
                grid_rows.append({"model": "gradient_boosting", "max_depth": max_depth,
                                   "min_samples_leaf": None, "n_estimators": n_estimators,
                                   "learning_rate": learning_rate, "validation_roc_auc": auc,
                                   "validation_precision": precision_score(y_val, preds_val, zero_division=0),
                                   "validation_recall": recall_score(y_val, preds_val, zero_division=0),
                                   "validation_f1": f1_score(y_val, preds_val, zero_division=0),
                                   "validation_accuracy": accuracy_score(y_val, preds_val)})
                if auc > best_gb_auc:
                    best_gb_auc, best_gb, best_gb_params = auc, m, (n_estimators, max_depth, learning_rate)
    print(f"Best params: n_estimators={best_gb_params[0]}, max_depth={best_gb_params[1]}, "
          f"learning_rate={best_gb_params[2]} (validation ROC-AUC={best_gb_auc:.4f})")
    report_and_collect(best_gb, X_train, y_train, X_val, y_val, X_test, y_test, "gradient_boosting", rows)
    print_top_importances(best_gb, "gradient_boosting")
    joblib.dump(best_gb, MODELS_DIR / "gradient_boosting_model.joblib")

    # --- Write full hyperparameter grid (all combinations tried, not just winners) ---
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(MODELS_DIR / "tree_hyperparameter_grid.csv", index=False)
    print(f"\nWrote full hyperparameter grid ({len(grid_df)} combinations) to {MODELS_DIR / 'tree_hyperparameter_grid.csv'}")

    # --- Append to the shared model comparison table ---
    new_rows_df = pd.DataFrame(rows)
    if COMPARISON_PATH.exists():
        existing = pd.read_csv(COMPARISON_PATH)
        combined = pd.concat([existing, new_rows_df], ignore_index=True)
    else:
        combined = new_rows_df
    combined.to_csv(COMPARISON_PATH, index=False)
    print(f"\nAppended results to {COMPARISON_PATH} ({len(combined)} total rows)")

    print("\n=== Validation ROC-AUC summary (this script's models) ===")
    print(f"Decision Tree:      {best_dt_auc:.4f}")
    print(f"Random Forest:      {best_rf_auc:.4f}")
    print(f"Gradient Boosting:  {best_gb_auc:.4f}")



if __name__ == "__main__":
    main()
