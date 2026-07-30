"""
finalize_model.py  (lives in src/)
------------------------------------------
Final step: consolidates results from predictive_modeling.py,
tree_based_models.py, advanced_models.py, threshold_analysis.py, and
bootstrap_evaluation.py into one clean comparison table, chart, and
selection summary. Selects and saves the final model + its chosen
operating threshold.

SELECTION RATIONALE (full evidence trail):

1. Tree-based models (decision tree, random forest, gradient boosting)
   achieved strong validation ROC-AUC but collapsed on the test set (as
   low as 0.237, worse than random), traced to covariate shift: several
   key features take test-period values entirely outside their
   training-period range, which hard-threshold tree splits cannot
   extrapolate across. This ruled out the entire tree-based family.

2. Among the remaining linear-family models, Elastic Net achieved the
   highest point-estimate test ROC-AUC (0.919), but a paired bootstrap
   comparison (2000 resamples) found it is NOT statistically
   distinguishable from recency-weighted logistic regression (95% CI of
   the difference: -0.012 to 0.000) -- both are a statistical tie at the
   top, and both are a confirmed, statistically real improvement over
   plain logistic regression alone (95% CI: -0.017 to -0.001, excluding
   zero).

3. Given that discrimination (ROC-AUC) tie, CALIBRATION was used as the
   tiebreaker: recency-weighted logistic regression's test Brier score
   (0.306) is substantially better than Elastic Net's (0.569) -- meaning
   its predicted probabilities are considerably more trustworthy, not
   just its rankings.

4. Recency-weighted logistic regression is also the most directly
   motivated choice methodologically: it was built specifically to
   counter the project's own documented, statistically confirmed
   training-window drift finding, rather than being a regularization
   choice that happened to score well.

Recency-weighted logistic regression is therefore the final selected
model, evaluated at an operating threshold of 0.32 (chosen via
threshold_analysis.py using max F2 on the validation set, since missing a
genuinely risky release was judged more costly than an extra false alarm
for this pre-deployment use case).

Produces, under data/models/:
    model_comparison_summary.csv   -- pivoted, one row per model, columns per split
    model_comparison_chart.png     -- ROC-AUC by model and split
    final_model.joblib             -- copy of the selected final model
    final_scaler.joblib            -- copy of its associated scaler
    final_model_summary.txt        -- full plain-text selection rationale and evaluation

Usage:
    python src/finalize_model.py
"""

from pathlib import Path
import shutil
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "data" / "models"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
CONFIG_PATH = MODELS_DIR / "final_model_info.json"

# Selected after the multicollinearity fix (see preprocess_for_modeling.py):
# prior to that fix, recency-weighted logistic regression was the final
# choice, selected via a calibration tiebreaker against Elastic Net. After
# the fix, plain logistic regression's test ROC-AUC and calibration both
# improved substantially on their own (test ROC-AUC 0.912 -> 0.929, Brier
# 0.257 -> 0.213), and a fresh bootstrap comparison found all three linear
# models statistically indistinguishable. Given that tie, plain logistic
# regression is preferred as the simplest model -- recency-weighting's
# drift-specific motivation remains valid (the year-over-year drift finding
# was re-confirmed, unchanged, after this fix), but it no longer
# demonstrably improves on the plain model now that the separate
# multicollinearity problem is resolved, so its added complexity is not
# earning its place.
FINAL_MODEL_NAME = "logistic_regression"
FINAL_MODEL_SOURCE = MODELS_DIR / "logistic_regression_model.joblib"
FINAL_SCALER_SOURCE = MODELS_DIR / "scaler.joblib"

CHART_MODELS = [
    "baseline_majority_class", "decision_tree", "random_forest", "gradient_boosting",
    "logistic_regression_C=0.001", "logistic_regression_C=1.0", "logistic_regression_C=0.01",
    "recency_weighted_logistic", "elastic_net", "naive_bayes",
]


def load_optional_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main():
    if not COMPARISON_PATH.exists():
        print(f"[error] {COMPARISON_PATH} not found -- run predictive_modeling.py, tree_based_models.py, "
              "and advanced_models.py first")
        return

    df = pd.read_csv(COMPARISON_PATH)

    # Deduplicate by (model, split), keeping the LAST occurrence. This
    # matters because scripts like advanced_models.py can be re-run after a
    # bug fix (as happened here with Elastic Net), and each run APPENDS
    # rather than replaces -- without deduplication, pivot_table() silently
    # AVERAGES the old (buggy) and new (fixed) rows together, corrupting
    # the summary table and chart with a blended, meaningless number, even
    # though the actual saved .joblib model file is correctly the fixed one.
    n_before = len(df)
    df = df.drop_duplicates(subset=["model", "split"], keep="last").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[note] Dropped {n_dropped} duplicate (model, split) row(s) from model_comparison.csv, "
              f"keeping the most recent result for each -- likely from re-running a script after a fix.")

    print("=== Full comparison table (deduplicated) ===")
    print(df.to_string(index=False))

    pivot = df.pivot_table(index="model", columns="split", values="roc_auc")
    pivot = pivot[[c for c in ["train", "validation", "test"] if c in pivot.columns]]
    pivot.to_csv(MODELS_DIR / "model_comparison_summary.csv")
    print("\n=== ROC-AUC by model and split ===")
    print(pivot.to_string())

    chart_models = [m for m in CHART_MODELS if m in pivot.index]
    splits = [c for c in ["train", "validation", "test"] if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(chart_models))
    width = 0.25
    colors = {"train": "#4C72B0", "validation": "#DD8452", "test": "#C44E52"}
    for i, split_name in enumerate(splits):
        values = [pivot.loc[m, split_name] if m in pivot.index else float("nan") for m in chart_models]
        ax.bar([xi + i * width for xi in x], values, width=width, label=split_name, color=colors.get(split_name))
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(chart_models, rotation=25, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random guessing (0.5)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model Comparison: ROC-AUC by Split\n(tree models collapse on test due to covariate shift; linear models generalize)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = MODELS_DIR / "model_comparison_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {chart_path}")

    if not FINAL_MODEL_SOURCE.exists():
        print(f"[warning] {FINAL_MODEL_SOURCE} not found -- cannot copy final model")
    else:
        shutil.copy(FINAL_MODEL_SOURCE, MODELS_DIR / "final_model.joblib")
        shutil.copy(FINAL_SCALER_SOURCE, MODELS_DIR / "final_scaler.joblib")
        print(f"Saved final model ({FINAL_MODEL_NAME}) to {MODELS_DIR / 'final_model.joblib'}")

    # Write/update a shared config file so downstream scripts (threshold_analysis.py,
    # bootstrap_evaluation.py) can read the current final model name dynamically,
    # rather than each hardcoding it separately -- exactly the kind of manual-sync
    # gap that caused today's model-selection change to require touching 3 files.
    existing_config = {}
    if CONFIG_PATH.exists():
        existing_config = json.loads(CONFIG_PATH.read_text())
    existing_config["model_name"] = FINAL_MODEL_NAME
    CONFIG_PATH.write_text(json.dumps(existing_config, indent=2))
    print(f"Wrote {CONFIG_PATH} (model_name={FINAL_MODEL_NAME}) -- threshold_analysis.py will add "
          f"the chosen operating threshold to this same file once it determines it.")

    bootstrap_cis = load_optional_csv(ANALYSIS_DIR / "bootstrap_roc_auc_cis.csv")
    pairwise = load_optional_csv(ANALYSIS_DIR / "bootstrap_pairwise_comparison.csv")
    threshold_cis = load_optional_csv(ANALYSIS_DIR / "bootstrap_threshold_metrics_ci.csv")

    # Model names in model_comparison.csv include the tuned hyperparameter
    # (e.g. "logistic_regression_C=0.001"), not the bare FINAL_MODEL_NAME --
    # match by prefix rather than exact equality.
    test_rows = df[df["split"] == "test"]
    final_row = test_rows[test_rows["model"].astype(str).str.startswith(FINAL_MODEL_NAME)]

    threshold_display = existing_config.get("threshold", "not yet determined -- run threshold_analysis.py")

    summary_lines = [
        "FINAL MODEL SELECTION SUMMARY",
        "=" * 60,
        f"Selected model: {FINAL_MODEL_NAME}",
        f"Chosen operating threshold: {threshold_display}",
        "",
        "SELECTION RATIONALE:",
        "1. Tree-based models (decision tree, random forest, gradient boosting) achieved strong",
        "   validation ROC-AUC but collapsed on the test set (as low as 0.237-0.30, worse than or",
        "   near random), due to covariate shift in several key features. This ruled out the entire",
        "   tree-based family.",
        "2. A severe multicollinearity issue was found and fixed (see preprocess_for_modeling.py):",
        "   open_bugs_at_release and prior_releases_avg_bugs were confounded with repository identity",
        "   (VS Code's absolute backlog scale is thousands of issues larger than the other two repos).",
        "   Fixing this via repository-wise normalization substantially improved every linear model's",
        "   test ROC-AUC and calibration -- plain logistic regression's test ROC-AUC rose from 0.912 to",
        "   0.929 and its Brier score improved from 0.257 to 0.213, purely from this fix.",
        "3. Before the multicollinearity fix, recency-weighted logistic regression was selected over",
        "   Elastic Net via a calibration tiebreaker (both were statistically tied on ROC-AUC). AFTER",
        "   the fix, a fresh bootstrap comparison found all three linear models (plain logistic",
        "   regression, recency-weighted, Elastic Net) statistically indistinguishable on ROC-AUC, and",
        "   plain logistic regression now has the BEST calibration of the three (Brier 0.213 vs. 0.258",
        "   for Elastic Net and 0.293 for recency-weighted).",
        "4. Given that three-way tie, plain logistic regression is selected as the simplest model.",
        "   Recency-weighting's original motivation remains valid -- a year-over-year trend test,",
        "   re-run and unchanged after the multicollinearity fix, still confirms a real, significant",
        "   drift in bug-reporting rates over the training window (e.g. kubernetes/kubernetes rho=-0.79,",
        "   p=2.3e-37) -- but that fix no longer demonstrably improves on the plain model now that the",
        "   separate multicollinearity problem is resolved, so its added complexity is not earning its",
        "   place. The drift remains a documented limitation of the target variable (see the",
        "   Target-Variable Definition document), independent of which model is selected.",
        "",
        "TEST-SET EVALUATION (point estimates):",
    ]
    if not final_row.empty:
        r = final_row.iloc[0]
        summary_lines += [
            f"  n = {int(r['n'])}, positive rate = {r['positive_rate']:.3f}",
            f"  ROC-AUC  = {r['roc_auc']:.4f}",
            f"  PR-AUC   = {r['pr_auc']:.4f}",
            f"  Brier    = {r['brier_score']:.4f}",
        ]
    else:
        summary_lines.append(f"  [warning] No '{FINAL_MODEL_NAME}' test row found in model_comparison.csv")

    if bootstrap_cis is not None:
        summary_lines.append("")
        summary_lines.append("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples, test set):")
        for _, row in bootstrap_cis.iterrows():
            summary_lines.append(
                f"  {row['model']:<28} ROC-AUC = {row['point_estimate_roc_auc']:.4f} "
                f"(95% CI: {row['ci_low_95']:.4f} - {row['ci_high_95']:.4f})"
            )

    if pairwise is not None:
        summary_lines.append("")
        summary_lines.append("PAIRWISE STATISTICAL COMPARISON:")
        for _, row in pairwise.iterrows():
            verdict = "LIKELY REAL DIFFERENCE" if row["difference_likely_real"] else "not statistically distinguishable"
            summary_lines.append(
                f"  {row['model_a']} vs {row['model_b']}: diff={row['mean_auc_diff_a_minus_b']:+.4f} "
                f"(95% CI: {row['ci_low_95']:+.4f} to {row['ci_high_95']:+.4f}) -- {verdict}"
            )

    if threshold_cis is not None:
        summary_lines.append("")
        summary_lines.append(f"OPERATING POINT (threshold={threshold_display}) WITH BOOTSTRAP 95% CIs:")
        for _, row in threshold_cis.iterrows():
            summary_lines.append(
                f"  {row['metric']:<10} = {row['point_estimate']:.3f}  "
                f"(95% CI: {row['ci_low_95']:.3f} - {row['ci_high_95']:.3f})"
            )

    summary_lines += [
        "",
        "NOTE ON ACCURACY: accuracy is deliberately NOT used as a primary metric in this project,",
        "since the target variable's positive rate differs substantially between training (~24%) and",
        "the test period (documented, investigated base-rate drift -- see the Target-Variable",
        "Definition document). ROC-AUC, PR-AUC, and threshold-specific precision/recall/F1 are the",
        "primary evaluation criteria.",
    ]

    summary_text = "\n".join(summary_lines)
    (MODELS_DIR / "final_model_summary.txt").write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
