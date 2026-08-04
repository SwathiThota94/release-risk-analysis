"""
predictive_modeling.py  (lives in src/)
------------------------------------------
Week 6 (Predictive Modelling), Part 1: data preparation, baseline model,
and logistic regression.

Uses the existing 'split' column (train/validation/test) from
model_ready_release_level.csv -- the same 60/20/20 time-based split
established and leakage-audited in Week 4/5. Hyperparameters are tuned
against the VALIDATION set specifically (not k-fold cross-validation),
since this is time-ordered data -- shuffling folds would leak future
information into training, the same leakage risk already addressed
throughout this project.

Given the documented class-distribution drift between train and
validation/test (see the Target-Variable Definition document, Section 6),
model comparison leads with ROC-AUC and PR-AUC (threshold-independent,
robust to a shifting base rate) rather than raw accuracy.

Produces, under data/models/:
    model_comparison.csv          -- appended to by each script in this series
    logistic_regression_model.joblib
    scaler.joblib

Usage:
    python src/predictive_modeling.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

# Feature set for predictive modelling. Unlike explanatory_regression.py,
# multicollinearity (VIF) is not a correctness concern for prediction itself
# -- it affects coefficient interpretability, not a regularized model's
# predictive accuracy -- so the repo-confounded backlog features are used
# in their plain (pooled-standardized) form here rather than needing
# per-repository standardization. is_prerelease (confirmed constant) and
# had_contributor_activity (exact duplicate of had_commit_activity,
# r=1.0) are still excluded, since a constant or duplicate column adds
# nothing for any model type and only wastes a feature slot.
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found -- run the feature engineering pipeline first")
    df = pd.read_csv(INPUT_PATH)
    if "split" not in df.columns:
        raise ValueError("No 'split' column found -- re-run run_features.py with the updated script.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns not found: {missing}")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    print(f"Loaded {len(df)} releases -> train={len(train)}, validation={len(val)}, test={len(test)}")
    return train, val, test


def prepare_xy(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values.astype(int)
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y


def evaluate(model, X, y, label: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    result = {
        "model": label,
        "n": len(y),
        "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, proba),
        "accuracy": accuracy_score(y, preds),
    }
    return result


def main():
    train, val, test = load_splits()

    scaler = StandardScaler()
    X_train, y_train = prepare_xy(train, scaler, fit_scaler=True)
    X_val, y_val = prepare_xy(val, scaler)
    X_test, y_test = prepare_xy(test, scaler)

    print(f"\nTraining set positive rate: {y_train.mean():.3f}")
    print(f"Validation set positive rate: {y_val.mean():.3f} (expected to differ -- see known drift limitation)")
    print(f"Test set positive rate: {y_test.mean():.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = []

    # --- Baseline model: always predicts the majority class ---
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    # DummyClassifier's predict_proba is degenerate (all 0 or all 1), so
    # ROC-AUC/PR-AUC are undefined for it -- report accuracy only, which is
    # the point of a baseline: "how good is guessing the majority class."
    baseline_val_acc = accuracy_score(y_val, baseline.predict(X_val))
    print(f"\n=== Baseline (always predict majority class) ===")
    print(f"Validation accuracy: {baseline_val_acc:.3f} (ROC-AUC/PR-AUC undefined -- baseline never varies its prediction)")
    comparison_rows.append({
        "model": "baseline_majority_class", "split": "validation", "n": len(y_val),
        "positive_rate": y_val.mean(), "roc_auc": np.nan, "pr_auc": np.nan,
        "brier_score": np.nan, "accuracy": baseline_val_acc,
    })

    # --- Logistic regression, with class-imbalance handling ---
    # class_weight='balanced' addresses the class-imbalance requirement:
    # it reweights the loss function inversely proportional to class
    # frequency in the TRAINING set, so the model isn't just biased toward
    # whichever class happens to be more common there.
    print(f"\n=== Logistic Regression (tuning C on validation set) ===")
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_C, best_val_auc, best_model = None, -np.inf, None
    for C in C_grid:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")
        m.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        print(f"  C={C:<8} validation ROC-AUC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_C, best_model = val_auc, C, m

    print(f"Best C: {best_C} (validation ROC-AUC={best_val_auc:.4f})")

    for split_name, X_, y_ in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
        r = evaluate(best_model, X_, y_, f"logistic_regression_C={best_C}")
        r["split"] = split_name
        comparison_rows.append(r)
        print(f"  [{split_name}] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"Brier={r['brier_score']:.4f}  Accuracy={r['accuracy']:.4f}  (n={r['n']}, pos_rate={r['positive_rate']:.3f})")

    # --- Save artifacts ---
    joblib.dump(best_model, MODELS_DIR / "logistic_regression_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"\nSaved model to {MODELS_DIR / 'logistic_regression_model.joblib'}")
    print(f"Saved scaler to {MODELS_DIR / 'scaler.joblib'}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Wrote model comparison table to {comparison_path}")

    # --- Coefficient inspection (quick sanity check against Week 5 findings) ---
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": best_model.coef_[0]})
    coef_df = coef_df.sort_values("coef", key=lambda s: s.abs(), ascending=False)
    print("\nTop logistic regression coefficients (standardized features):")
    print(coef_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
