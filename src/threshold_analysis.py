"""
threshold_analysis.py  (lives in src/)
------------------------------------------
Follow-up to model comparison: ROC-AUC/PR-AUC (used to SELECT the best
model) don't require picking a decision threshold. Precision, recall, F1,
and accuracy all DO -- and the default 0.5 cutoff is not necessarily the
right one, especially given the documented base-rate drift between train
(~24% positive) and validation/test (much higher in the observed data).

This script sweeps a full range of thresholds against the VALIDATION set
(never test, to avoid tuning against your final holdout), plots precision/
recall/F1 vs. threshold, and recommends an operating threshold based on
maximum F1. The chosen threshold is then applied to the test set exactly
once, as a final, honest check -- test is never used to pick the threshold
itself.

Uses the recency-weighted logistic regression model (the project's
selected final model as of the last model-comparison round) and the
scaler saved by predictive_modeling.py -- scaling parameters depend only
on the training feature values, which are identical across all logistic-
regression variants in this project, so the same saved scaler is valid to
reuse here without refitting.

Produces, under data/analysis/:
    threshold_sweep.csv            -- precision/recall/F1/accuracy at each threshold, validation set
    threshold_sweep_chart.png      -- visualization
    threshold_final_evaluation.txt -- chosen threshold + one-time test-set confirmation

Usage:
    python src/threshold_analysis.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

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

# Reads the GENERIC final_model.joblib/final_scaler.joblib -- whichever
# model finalize_model.py most recently selected -- rather than hardcoding
# a specific model's filename. This was previously hardcoded to
# recency_weighted_logistic_model.joblib, which silently went stale when
# the final model selection changed to plain logistic regression after the
# multicollinearity fix; reading the generic artifact avoids that class of
# bug recurring.
MODEL_PATH = MODELS_DIR / "final_model.joblib"
SCALER_PATH = MODELS_DIR / "final_scaler.joblib"
CONFIG_PATH = MODELS_DIR / "final_model_info.json"


def load_splits():
    df = pd.read_csv(INPUT_PATH)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def metrics_at_threshold(y_true, proba, threshold):
    preds = (proba >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    # F2 weights recall twice as heavily as precision (beta=2): appropriate
    # here because this is a PRE-DEPLOYMENT risk check -- missing a
    # genuinely risky release (a false negative) is a more costly error
    # than an extra false alarm that just triggers additional QA review.
    # F1 treats both errors as equally costly, which does not match that
    # framing.
    beta = 2
    if precision + recall > 0:
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        f2 = 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy_score(y_true, preds),
        "n_predicted_positive": int(preds.sum()),
    }


def main():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print(f"[error] {MODEL_PATH} or {SCALER_PATH} not found -- run predictive_modeling.py and "
              "advanced_models.py first")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_name = json.loads(CONFIG_PATH.read_text()).get("model_name", "unknown") if CONFIG_PATH.exists() else "unknown"
    train, val, test = load_splits()

    X_val = scaler.transform(val[FEATURE_COLS])
    y_val = val[TARGET_COL].values.astype(int)
    X_test = scaler.transform(test[FEATURE_COLS])
    y_test = test[TARGET_COL].values.astype(int)

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    print(f"Validation set: n={len(y_val)}, positive_rate={y_val.mean():.3f}")
    print(f"Test set:       n={len(y_test)}, positive_rate={y_test.mean():.3f}\n")

    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = [metrics_at_threshold(y_val, proba_val, t) for t in thresholds]
    sweep_df = pd.DataFrame(rows)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(ANALYSIS_DIR / "threshold_sweep.csv", index=False)

    best_f2_row = sweep_df.loc[sweep_df["f2"].idxmax()]
    best_threshold = best_f2_row["threshold"]
    print(f"=== Threshold recommendation (validation set, max F2 -- recall weighted 2x precision) ===")
    print(f"Rationale: this is a pre-deployment risk check; missing a genuinely risky release (false")
    print(f"negative) is treated as more costly than an extra false alarm (false positive), so F2 is")
    print(f"used instead of F1 to select the operating threshold.")
    print(f"\nBest threshold: {best_threshold:.2f}  "
          f"(precision={best_f2_row['precision']:.3f}, recall={best_f2_row['recall']:.3f}, "
          f"F1={best_f2_row['f1']:.3f}, F2={best_f2_row['f2']:.3f}, accuracy={best_f2_row['accuracy']:.3f})")

    # Write the discovered threshold to the shared config file (alongside
    # the model_name finalize_model.py already wrote), so bootstrap_evaluation.py
    # and anything else downstream reads this dynamically rather than needing
    # a hardcoded, manually-synced constant.
    config = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
    config["threshold"] = round(float(best_threshold), 2)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Wrote chosen threshold to {CONFIG_PATH}")

    default_row = metrics_at_threshold(y_val, proba_val, 0.5)
    print(f"\nFor comparison, default threshold 0.50: "
          f"precision={default_row['precision']:.3f}, recall={default_row['recall']:.3f}, "
          f"F1={default_row['f1']:.3f}, F2={default_row['f2']:.3f}")

    # --- Transparent sensitivity table across several candidate thresholds ---
    # Shown alongside the F2-selected threshold so the tradeoff is visible
    # and inspectable, not just asserted by a single automatically-chosen number.
    candidates = sorted(set([0.15, 0.20, 0.25, 0.30, round(float(best_threshold), 2), 0.35, 0.40, 0.50]))
    print(f"\n=== Sensitivity table: precision/recall/F1/F2 at candidate thresholds (validation set) ===")
    sens_rows = [metrics_at_threshold(y_val, proba_val, t) for t in candidates]
    sens_df = pd.DataFrame(sens_rows)
    print(sens_df.to_string(index=False))
    sens_df.to_csv(ANALYSIS_DIR / "threshold_candidates_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#4C72B0")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#DD8452")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", color="#55A868", linewidth=1, linestyle="--")
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2 (recall-weighted)", color="#C44E52", linewidth=2)
    ax.axvline(best_threshold, color="gray", linestyle="--", label=f"Best F2 threshold ({best_threshold:.2f})")
    ax.axvline(0.5, color="lightgray", linestyle=":", label="Default threshold (0.50)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 / F2 vs. Decision Threshold\n(validation set, {model_name})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    chart_path = ANALYSIS_DIR / "threshold_sweep_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {chart_path}")

    test_at_best = metrics_at_threshold(y_test, proba_test, best_threshold)
    test_at_default = metrics_at_threshold(y_test, proba_test, 0.5)
    cm_best = confusion_matrix(y_test, (proba_test >= best_threshold).astype(int))

    summary_lines = [
        "THRESHOLD SELECTION AND FINAL TEST-SET CONFIRMATION",
        "=" * 55,
        f"Model: {model_name}",
        f"Threshold selected on VALIDATION set (max F2, recall weighted 2x precision): {best_threshold:.2f}",
        "Rationale: this is a pre-deployment risk check where missing a genuinely risky release",
        "is treated as more costly than an extra false alarm, so F2 (not F1) was used to select",
        "the operating threshold.",
        "",
        f"Validation set at chosen threshold: precision={best_f2_row['precision']:.3f}, "
        f"recall={best_f2_row['recall']:.3f}, F1={best_f2_row['f1']:.3f}, F2={best_f2_row['f2']:.3f}",
        "",
        "TEST SET (evaluated once, using the validation-selected threshold -- not tuned on test):",
        f"  At threshold {best_threshold:.2f}: precision={test_at_best['precision']:.3f}, "
        f"recall={test_at_best['recall']:.3f}, F1={test_at_best['f1']:.3f}, F2={test_at_best['f2']:.3f}, "
        f"accuracy={test_at_best['accuracy']:.3f}",
        f"  At default 0.50:      precision={test_at_default['precision']:.3f}, "
        f"recall={test_at_default['recall']:.3f}, F1={test_at_default['f1']:.3f}, F2={test_at_default['f2']:.3f}, "
        f"accuracy={test_at_default['accuracy']:.3f}",
        "",
        f"Confusion matrix on test set at threshold {best_threshold:.2f} "
        f"([[TN, FP], [FN, TP]]):",
        f"  {cm_best.tolist()}",
        "",
        "Interpretation note: precision/recall/F1/F2 at any single threshold should be read alongside",
        "the documented test-set base-rate drift (test positive rate differs substantially from",
        "training) -- these figures describe performance at ONE specific operating point chosen from",
        "validation data, not the model's overall discrimination ability (see ROC-AUC/PR-AUC for that).",
        "See threshold_candidates_sensitivity.csv for the full tradeoff across several candidate",
        "thresholds, so this choice can be reviewed rather than taken on faith.",
    ]
    summary_text = "\n".join(summary_lines)
    (ANALYSIS_DIR / "threshold_final_evaluation.txt").write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
