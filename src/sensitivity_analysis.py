"""
sensitivity_analysis.py  (lives in src/)
------------------------------------------
Week 6 follow-up: two things that need to happen before trusting the
logistic regression result from predictive_modeling.py, or before adding
more model types on top of it.

1. FEATURE-ABLATION SENSITIVITY TEST. The full feature set includes
   repo_kubernetes/kubernetes and repo_microsoft/vscode, and separately
   release_sequence / prior_releases_count. Given the documented
   class-distribution drift (Kubernetes near-0% positive in the test
   period; Airflow/VS Code 87-90% positive), a model could achieve a
   deceptively high test ROC-AUC simply by learning "if this isn't
   Kubernetes, predict risky" -- without having learned anything genuinely
   useful about release-level behavior. This script refits logistic
   regression under several feature configurations (full, without repo
   dummies, without release_sequence/prior_releases_count, without both)
   and compares validation/test performance across them, tuning C
   against the validation set the same way for every configuration.

2. CALIBRATION CURVE. Required by the Week 6 task list, and directly
   explains a puzzling pattern already observed: high test ROC-AUC/PR-AUC
   alongside very low test accuracy. This happens when a model's predicted
   probabilities are calibrated to the TRAINING set's base rate (~24%
   positive) but the test set's real base rate is very different (~61%
   positive in one real run) -- the model still RANKS releases correctly
   (good AUC) but its probabilities sit on the wrong side of the default
   0.5 threshold far too often (poor accuracy). The calibration curve
   makes this visible directly.

Produces, under data/analysis/:
    sensitivity_ablation_results.csv
    calibration_curve.png

Usage:
    python src/sensitivity_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.calibration import calibration_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"

TARGET_COL = "elevated_risk"

FULL_FEATURES = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_issues_at_release", "open_bugs_at_release",
    "prior_releases_avg_bugs", "prior_releases_count",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]

REPO_COLS = ["repo_kubernetes/kubernetes", "repo_microsoft/vscode"]
SEQUENCE_COLS = ["release_sequence", "prior_releases_count"]

CONFIGURATIONS = {
    "full_feature_set": FULL_FEATURES,
    "without_repo_dummies": [c for c in FULL_FEATURES if c not in REPO_COLS],
    "without_release_sequence_features": [c for c in FULL_FEATURES if c not in SEQUENCE_COLS],
    "without_repo_and_sequence": [c for c in FULL_FEATURES if c not in REPO_COLS + SEQUENCE_COLS],
}

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_splits():
    df = pd.read_csv(INPUT_PATH)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def fit_and_tune(train, val, feature_cols):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train[TARGET_COL].values.astype(int)
    X_val = scaler.transform(val[feature_cols])
    y_val = val[TARGET_COL].values.astype(int)

    best_model, best_auc = None, -np.inf
    for C in C_GRID:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_model = auc, m
    return best_model, scaler


def evaluate_config(model, scaler, df, feature_cols, split_name, config_name):
    X = scaler.transform(df[feature_cols])
    y = df[TARGET_COL].values.astype(int)
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "config": config_name, "split": split_name, "n": len(y), "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": accuracy_score(y, preds),
    }


def main():
    train, val, test = load_splits()
    print(f"train={len(train)}, validation={len(val)}, test={len(test)}\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    fitted_models = {}

    for config_name, feature_cols in CONFIGURATIONS.items():
        print(f"=== {config_name} ({len(feature_cols)} features) ===")
        model, scaler = fit_and_tune(train, val, feature_cols)
        fitted_models[config_name] = (model, scaler, feature_cols)
        for split_name, split_df in [("train", train), ("validation", val), ("test", test)]:
            r = evaluate_config(model, scaler, split_df, feature_cols, split_name, config_name)
            results.append(r)
            print(f"  [{split_name}] ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
                  f"Accuracy={r['accuracy']:.4f}  (n={r['n']}, pos_rate={r['positive_rate']:.3f})")
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv(ANALYSIS_DIR / "sensitivity_ablation_results.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'sensitivity_ablation_results.csv'}")

    # --- Interpretation: how much does test ROC-AUC drop when repo dummies
    # or sequence features are removed? ---
    test_results = results_df[results_df["split"] == "test"].set_index("config")
    full_auc = test_results.loc["full_feature_set", "roc_auc"]
    print(f"\n=== Sensitivity summary (test set ROC-AUC) ===")
    print(f"Full feature set:              {full_auc:.4f}")
    for cfg in ["without_repo_dummies", "without_release_sequence_features", "without_repo_and_sequence"]:
        auc = test_results.loc[cfg, "roc_auc"]
        drop = full_auc - auc
        print(f"{cfg:<35} {auc:.4f}  (change from full: {-drop:+.4f})")

    repo_change = test_results.loc["without_repo_dummies", "roc_auc"] - full_auc
    seq_change = test_results.loc["without_release_sequence_features", "roc_auc"] - full_auc
    if abs(repo_change) > 0.05:
        if repo_change < 0:
            print(f"\n[NOTE] Removing repo dummies DROPS test ROC-AUC by {-repo_change:.3f}. "
                  "Repository identity appears to carry genuine, stable predictive value that "
                  "transfers from training to the test period.")
        else:
            print(f"\n[NOTE] Removing repo dummies IMPROVES test ROC-AUC by {repo_change:.3f}. "
                  "This suggests the full model's repo-dummy coefficients -- fit on training-period "
                  "data -- do NOT transfer well to the test period's different repo-level behavior "
                  "(consistent with the documented class-distribution drift). Rather than adding real "
                  "signal, including these dummies may cause the model to rely on a training-period "
                  "repo pattern that no longer holds by the test period. The without-repo-dummies "
                  "result may be the more honest estimate of real generalization here.")
    if abs(seq_change) > 0.05:
        direction = "drops" if seq_change < 0 else "improves"
        print(f"[NOTE] Removing release_sequence/prior_releases_count {direction} test ROC-AUC by "
              f"{abs(seq_change):.3f}. This quantifies how much of the model's test-set performance "
              "depends on the training-window-specific drift pattern documented in the Target-Variable "
              "Definition, versus other features.")

    # --- Calibration curve, full-feature-set model, validation AND test ---
    model, scaler, feature_cols = fitted_models["full_feature_set"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (split_name, split_df) in zip(axes, [("validation", val), ("test", test)]):
        X = scaler.transform(split_df[feature_cols])
        y = split_df[TARGET_COL].values.astype(int)
        proba = model.predict_proba(X)[:, 1]
        if len(np.unique(y)) > 1:
            frac_pos, mean_pred = calibration_curve(y, proba, n_bins=5, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.set_title(f"Calibration: {split_name} (base rate={y.mean():.2f})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction actually positive")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Calibration Curve: full-feature-set logistic regression")
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "calibration_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")
    print("[NOTE] If the model's curve sits well below the diagonal on a split, the model is "
          "systematically UNDER-predicting risk there (its probabilities were calibrated to a "
          "different base rate) -- this is expected here given the documented train/test base-rate drift.")


if __name__ == "__main__":
    main()
