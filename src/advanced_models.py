"""
advanced_models.py  (lives in src/)
------------------------------------------
Additional models beyond the required Week 6 list, each chosen because it
directly targets a property of this dataset already CONFIRMED by earlier
analysis (not a generic "try more models" exercise):

1. ELASTIC NET logistic regression. Correlation analysis (Week 5) found
   several tight multicollinearity clusters (release-size features;
   backlog/repo-identity features). Elastic Net's L1 component can zero
   out redundant features automatically, rather than manually picking one
   feature from each correlated pair as explanatory_regression.py did.

2. RECENCY-WEIGHTED logistic regression. The sensitivity/year-over-year
   analysis confirmed real, directional drift in bug-reporting rates over
   the training window. Weighting training samples by recency nudges the
   model toward the CURRENT relationship rather than treating all training
   years as equally representative of "now."

3. NAIVE BAYES. Tree-based models catastrophically failed under confirmed
   covariate shift (test ROC-AUC as low as 0.237) while logistic
   regression generalized well (0.912). This tests whether an even
   simpler model with strong independence assumptions does comparably or
   better -- informative either way about how much "smoothness/simplicity"
   is buying under this specific kind of drift.

4. REPOSITORY-AWARE MODELS. Correlation analysis, hypothesis testing, and
   the explanatory regression's likelihood-ratio test all confirmed
   repository identity matters, and that at least one relationship
   (open_bugs_at_release) reverses sign entirely across repositories.
   statsmodels/lme4-style mixed-effects modeling is not available in this
   environment, so two practical alternatives are implemented instead:
     (a) Fully separate per-repository models (each repository gets its
         own model, trained and tuned only on its own data) -- the
         "no pooling" extreme.
     (b) A shrinkage/partial-pooling approximation: each repository's
         final coefficients are a weighted blend of its own per-repository
         coefficients and the pooled (all-repository) coefficients, with
         the weight determined by that repository's sample size relative
         to a shrinkage constant -- conceptually similar to an empirical-
         Bayes / random-effects shrinkage estimator, without requiring a
         true mixed-effects modelling library.

Appends to data/models/model_comparison.csv, consistent with
predictive_modeling.py and tree_based_models.py.

Usage:
    python src/advanced_models.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
REPOS = ["kubernetes/kubernetes", "apache/airflow", "microsoft/vscode"]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_splits():
    df = pd.read_csv(INPUT_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"])
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def xy(df, scaler=None, fit=False):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)
    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    return X, y


def evaluate(model, X, y, label, split):
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "model": label, "split": split, "n": len(y), "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, proba),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }


def report(rows, model, X_train, y_train, X_val, y_val, X_test, y_test, label):
    for split_name, X_, y_ in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
        r = evaluate(model, X_, y_, label, split_name)
        rows.append(r)
        print(f"  [{split_name}] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"Brier={r['brier_score']:.4f}  Accuracy={r['accuracy']:.4f}  (n={r['n']}, pos_rate={r['positive_rate']:.3f})")


def main():
    train, val, test = load_splits()
    scaler = StandardScaler()
    X_train, y_train = xy(train, scaler, fit=True)
    X_val, y_val = xy(val, scaler)
    X_test, y_test = xy(test, scaler)
    print(f"train={len(train)}, validation={len(val)}, test={len(test)}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # =========================================================
    # 1. ELASTIC NET
    # =========================================================
    print("=== Elastic Net Logistic Regression (tuning C, l1_ratio on validation) ===")
    best_model, best_auc, best_params = None, -np.inf, None
    for C in C_GRID:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # NOTE: penalty='elasticnet' is REQUIRED here for correctness on
            # most sklearn versions -- omitting it (as an earlier version of
            # this script did, to silence a deprecation warning on a newer
            # sklearn) causes many sklearn versions to silently fall back to
            # plain L2 and ignore l1_ratio entirely, which is a correctness
            # bug, not a cosmetic issue: it means Elastic Net never actually
            # ran. A harmless FutureWarning on newer sklearn versions is a
            # far smaller cost than silently getting the wrong model.
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=FutureWarning)
                m = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                                        class_weight="balanced", max_iter=5000)
                m.fit(X_train, y_train)
            auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            if auc > best_auc:
                best_auc, best_model, best_params = auc, m, (C, l1_ratio)
    print(f"Best params: C={best_params[0]}, l1_ratio={best_params[1]} (validation ROC-AUC={best_auc:.4f})")
    n_zeroed = int((np.abs(best_model.coef_[0]) < 1e-6).sum())
    print(f"Features zeroed out by L1 penalty: {n_zeroed} / {len(FEATURE_COLS)}")
    report(rows, best_model, X_train, y_train, X_val, y_val, X_test, y_test, "elastic_net")
    joblib.dump(best_model, MODELS_DIR / "elastic_net_model.joblib")

    # =========================================================
    # 2. RECENCY-WEIGHTED LOGISTIC REGRESSION
    # =========================================================
    print("\n=== Recency-Weighted Logistic Regression (tuning C on validation) ===")
    order = train["release_date"].rank(method="first").values
    recency_weight = 0.2 + 0.8 * (order - 1) / (len(order) - 1)
    class_counts = train[TARGET_COL].value_counts()
    class_weight_map = {0: len(train) / (2 * class_counts.get(0, 1)), 1: len(train) / (2 * class_counts.get(1, 1))}
    class_weight_arr = train[TARGET_COL].map(class_weight_map).values
    sample_weight = recency_weight * class_weight_arr

    best_model, best_auc, best_C = None, -np.inf, None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        m.fit(X_train, y_train, sample_weight=sample_weight)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_model, best_C = auc, m, C
    print(f"Best C: {best_C} (validation ROC-AUC={best_auc:.4f})")
    report(rows, best_model, X_train, y_train, X_val, y_val, X_test, y_test, "recency_weighted_logistic")
    joblib.dump(best_model, MODELS_DIR / "recency_weighted_logistic_model.joblib")

    # =========================================================
    # 3. NAIVE BAYES
    # =========================================================
    print("\n=== Gaussian Naive Bayes ===")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, nb.predict_proba(X_val)[:, 1])
    print(f"Validation ROC-AUC={val_auc:.4f}")
    report(rows, nb, X_train, y_train, X_val, y_val, X_test, y_test, "naive_bayes")
    joblib.dump(nb, MODELS_DIR / "naive_bayes_model.joblib")

    # =========================================================
    # 4a. FULLY SEPARATE PER-REPOSITORY MODELS (no pooling)
    # =========================================================
    print("\n=== Repository-Specific Models (no pooling -- separate model per repository) ===")
    repo_col_map = {"kubernetes/kubernetes": "repo_kubernetes/kubernetes", "microsoft/vscode": "repo_microsoft/vscode"}
    repo_models = {}
    for repo_name in REPOS:
        if repo_name == "apache/airflow":
            repo_train = train[(train["repo_kubernetes/kubernetes"] == 0) & (train["repo_microsoft/vscode"] == 0)]
            repo_val = val[(val["repo_kubernetes/kubernetes"] == 0) & (val["repo_microsoft/vscode"] == 0)]
            repo_test = test[(test["repo_kubernetes/kubernetes"] == 0) & (test["repo_microsoft/vscode"] == 0)]
        else:
            col = repo_col_map[repo_name]
            repo_train = train[train[col] == 1]
            repo_val = val[val[col] == 1]
            repo_test = test[test[col] == 1]

        print(f"\n  --- {repo_name} (train={len(repo_train)}, val={len(repo_val)}, test={len(repo_test)}) ---")
        if len(repo_train) < 20 or repo_train[TARGET_COL].nunique() < 2:
            print("    [skip] too few training rows or no class variation for a repo-specific model")
            continue

        repo_features = [c for c in FEATURE_COLS if not c.startswith("repo_")]
        repo_scaler = StandardScaler()
        Xr_train = repo_scaler.fit_transform(repo_train[repo_features])
        yr_train = repo_train[TARGET_COL].values.astype(int)
        Xr_val = repo_scaler.transform(repo_val[repo_features]) if len(repo_val) > 0 else None
        yr_val = repo_val[TARGET_COL].values.astype(int) if len(repo_val) > 0 else None

        best_r, best_r_auc = None, -np.inf
        for C in C_GRID:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")
            m.fit(Xr_train, yr_train)
            if Xr_val is not None and len(np.unique(yr_val)) > 1:
                auc = roc_auc_score(yr_val, m.predict_proba(Xr_val)[:, 1])
            else:
                auc = roc_auc_score(yr_train, m.predict_proba(Xr_train)[:, 1])
            if auc > best_r_auc:
                best_r_auc, best_r = auc, m
        repo_models[repo_name] = (best_r, repo_scaler, repo_features)

        if len(repo_test) > 0 and repo_test[TARGET_COL].nunique() > 1:
            Xr_test = repo_scaler.transform(repo_test[repo_features])
            yr_test = repo_test[TARGET_COL].values.astype(int)
            r = evaluate(best_r, Xr_test, yr_test, f"repo_specific_{repo_name}", "test")
            rows.append(r)
            print(f"    [test] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  Accuracy={r['accuracy']:.4f}  "
                  f"(n={r['n']}, pos_rate={r['positive_rate']:.3f})")
        else:
            print("    [note] test set for this repository has < 2 classes present or 0 rows -- "
                  "ROC-AUC not computable; this itself reflects the documented class-distribution drift.")

    # =========================================================
    # 4b. SHRINKAGE / PARTIAL-POOLING APPROXIMATION
    # =========================================================
    print("\n=== Partial-Pooling (Shrinkage) Approximation ===")
    print("Blends each repository's own coefficients with the pooled model's coefficients,")
    print("weighted by that repository's training sample size (more data -> trust the repo-specific")
    print("estimate more; less data -> shrink toward the pooled estimate). A lightweight stand-in for")
    print("a true random-intercept/mixed-effects model, which requires a library not available here.\n")

    pooled_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    pooled_lr.fit(X_train, y_train)
    pooled_coef = pooled_lr.coef_[0]
    pooled_intercept = pooled_lr.intercept_[0]

    SHRINKAGE_K = 50  # shrinkage constant: repo sample size at which repo-specific and pooled estimates are weighted equally
    for repo_name in REPOS:
        if repo_name not in repo_models:
            continue
        repo_model, repo_scaler, repo_features = repo_models[repo_name]
        if repo_name == "apache/airflow":
            repo_train_n = len(train[(train["repo_kubernetes/kubernetes"] == 0) & (train["repo_microsoft/vscode"] == 0)])
            repo_test_df = test[(test["repo_kubernetes/kubernetes"] == 0) & (test["repo_microsoft/vscode"] == 0)]
        else:
            col = repo_col_map[repo_name]
            repo_train_n = len(train[train[col] == 1])
            repo_test_df = test[test[col] == 1]

        alpha = repo_train_n / (repo_train_n + SHRINKAGE_K)

        pooled_coef_map = dict(zip(FEATURE_COLS, pooled_coef))
        repo_coef_map = dict(zip(repo_features, repo_model.coef_[0]))
        blended_coef = np.array([
            alpha * repo_coef_map.get(f, 0.0) + (1 - alpha) * pooled_coef_map[f]
            for f in FEATURE_COLS
        ])
        blended_intercept = alpha * repo_model.intercept_[0] + (1 - alpha) * pooled_intercept

        print(f"  {repo_name}: n_train={repo_train_n}, shrinkage weight (alpha, toward repo-specific)={alpha:.3f}")

        if len(repo_test_df) > 0 and repo_test_df[TARGET_COL].nunique() > 1:
            X_repo_test_pooled_space = scaler.transform(repo_test_df[FEATURE_COLS].values)
            z = X_repo_test_pooled_space @ blended_coef + blended_intercept
            proba = 1 / (1 + np.exp(-z))
            preds = (proba >= 0.5).astype(int)
            y_repo_test = repo_test_df[TARGET_COL].values.astype(int)
            r = {
                "model": f"partial_pooling_{repo_name}", "split": "test", "n": len(y_repo_test),
                "positive_rate": y_repo_test.mean(),
                "roc_auc": roc_auc_score(y_repo_test, proba),
                "pr_auc": average_precision_score(y_repo_test, proba),
                "brier_score": brier_score_loss(y_repo_test, proba),
                "accuracy": accuracy_score(y_repo_test, preds),
                "precision": precision_score(y_repo_test, preds, zero_division=0),
                "recall": recall_score(y_repo_test, preds, zero_division=0),
                "f1": f1_score(y_repo_test, preds, zero_division=0),
            }
            rows.append(r)
            print(f"    [test] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  Accuracy={r['accuracy']:.4f}")
        else:
            print("    [note] test set for this repository has < 2 classes present -- ROC-AUC not computable.")

    # =========================================================
    # Append to shared comparison table
    # =========================================================
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
