"""
explainability.py  (lives in src/)
------------------------------------------
Week 7: SHAP explanations, global and local risk drivers.

The `shap` library is not installable in this environment (no network
access). However, the final model (recency-weighted logistic regression)
is LINEAR in its standardized feature space, and exact Shapley values for
a linear model have a known closed-form solution (this is what the shap
library itself calls "Linear SHAP" internally -- no sampling or
approximation needed):

    phi_i(x) = coef_i * (x_i_scaled - baseline_i)

where baseline_i is the reference/background value for feature i. Since
StandardScaler was fit on the TRAINING set, the training set's mean in
scaled space is exactly 0 for every feature -- so using the training
distribution as the SHAP background (the conventional choice) simplifies
this to:

    phi_i(x) = coef_i * x_i_scaled

This satisfies the SHAP "efficiency" property exactly (not approximately):
    intercept + sum_i(phi_i(x)) == model's raw log-odds output for x
This is verified directly in this script before trusting the output.

Produces, under data/analysis/:
    shap_values_all_releases.csv    -- per-release, per-feature contribution (phi_i), all splits
    global_risk_drivers.png         -- mean |phi_i| across the test set (global importance)
    global_risk_drivers.csv
    local_explanation_examples.png  -- waterfall-style breakdown for 3 example releases
    local_explanation_examples.csv

Usage:
    python src/explainability.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CHOSEN_THRESHOLD = 0.32


def load_data_and_model():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    scaler = joblib.load(MODELS_DIR / "final_scaler.joblib")
    df = pd.read_csv(INPUT_PATH)
    return model, scaler, df


def compute_shap_values(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact linear-model Shapley values: phi_i = coef_i * x_i_scaled, with the
    training set's scaled mean (0, by construction of StandardScaler) as
    the implicit background/baseline.
    """
    X_scaled = scaler.transform(df[FEATURE_COLS].values)
    coef = model.coef_[0]
    phi = X_scaled * coef
    phi_df = pd.DataFrame(phi, columns=[f"shap_{c}" for c in FEATURE_COLS], index=df.index)
    phi_df.insert(0, "release_id", df["release_id"].values)
    phi_df.insert(1, "repository_name", df["repository_name"].values)
    phi_df.insert(2, "split", df["split"].values)
    phi_df["shap_base_value"] = model.intercept_[0]
    phi_df["shap_sum_plus_base"] = phi_df["shap_base_value"] + phi.sum(axis=1)
    return phi_df


def verify_efficiency_property(model, scaler, df: pd.DataFrame, phi_df: pd.DataFrame):
    """
    Sanity check: intercept + sum(phi_i) must EXACTLY equal the model's raw
    decision function output (pre-sigmoid log-odds) for every row. This is
    not an approximation -- if this check fails, something is wrong with
    the computation, not with the method.
    """
    X_scaled = scaler.transform(df[FEATURE_COLS].values)
    raw_decision = model.decision_function(X_scaled)
    reconstructed = phi_df["shap_sum_plus_base"].values
    max_abs_diff = np.max(np.abs(raw_decision - reconstructed))
    print(f"Efficiency property check: max |reconstructed - actual raw model output| = {max_abs_diff:.2e}")
    if max_abs_diff > 1e-8:
        print("[WARNING] Efficiency property does NOT hold within numerical precision -- investigate before trusting these values.")
    else:
        print("Efficiency property holds exactly (within floating-point precision) -- these are exact Shapley values, not an approximation.")


def plot_global_drivers(phi_df: pd.DataFrame, split_name: str = "test"):
    shap_cols = [c for c in phi_df.columns if c.startswith("shap_") and c not in ("shap_base_value", "shap_sum_plus_base")]
    subset = phi_df[phi_df["split"] == split_name]
    mean_abs = subset[shap_cols].abs().mean().sort_values(ascending=True)
    mean_abs.index = [c.replace("shap_", "") for c in mean_abs.index]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(mean_abs.index, mean_abs.values, color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (contribution to log-odds)")
    ax.set_title(f"Global Risk Drivers -- Mean Absolute Feature Contribution ({split_name} set)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "global_risk_drivers.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    mean_abs_df = mean_abs.sort_values(ascending=False).reset_index()
    mean_abs_df.columns = ["feature", "mean_abs_shap_value"]
    mean_abs_df.to_csv(ANALYSIS_DIR / "global_risk_drivers.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'global_risk_drivers.csv'}")
    return mean_abs_df


def plot_local_examples(phi_df: pd.DataFrame, df: pd.DataFrame, model, scaler):
    """
    Waterfall-style breakdown for 3 example releases from the test set:
    the most confidently-predicted risky release, the most confidently
    predicted not-risky release, and the release closest to the decision
    threshold (0.32) -- the "most uncertain" case, often the most
    informative one to explain to a release manager in practice.
    """
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS].values)
    proba_test = model.predict_proba(X_test_scaled)[:, 1]
    test_phi = phi_df[phi_df["split"] == "test"].reset_index(drop=True)

    idx_most_risky = int(np.argmax(proba_test))
    idx_most_safe = int(np.argmin(proba_test))
    idx_closest_to_threshold = int(np.argmin(np.abs(proba_test - CHOSEN_THRESHOLD)))

    examples = {
        "Highest predicted risk": idx_most_risky,
        "Lowest predicted risk": idx_most_safe,
        f"Closest to decision threshold ({CHOSEN_THRESHOLD})": idx_closest_to_threshold,
    }

    shap_cols = [c for c in phi_df.columns if c.startswith("shap_") and c not in ("shap_base_value", "shap_sum_plus_base")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    local_rows = []

    for ax, (label, idx) in zip(axes, examples.items()):
        row = test_phi.iloc[idx]
        release_id = row["release_id"]
        repo = row["repository_name"]
        proba = proba_test[idx]

        contributions = row[shap_cols].sort_values(key=lambda s: s.abs(), ascending=True)
        feature_names = [c.replace("shap_", "") for c in contributions.index]
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in contributions.values]

        ax.barh(feature_names, contributions.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}\nrelease_id={release_id}, {repo}\npredicted P(risk)={proba:.3f}", fontsize=10)
        ax.set_xlabel("Contribution to log-odds")
        ax.tick_params(axis="y", labelsize=7)

        for feat, val in contributions.items():
            local_rows.append({
                "example": label, "release_id": release_id, "repository_name": repo,
                "predicted_probability": proba, "feature": feat.replace("shap_", ""), "shap_value": val,
            })

    fig.suptitle("Local Explanations: Individual Release Risk Breakdowns\n"
                 "(red = pushes toward elevated risk, blue = pushes toward not-risky)", fontsize=12)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "local_explanation_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(ANALYSIS_DIR / "local_explanation_examples.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'local_explanation_examples.csv'}")


def main():
    model, scaler, df = load_data_and_model()
    print(f"Loaded {len(df)} releases; model = {type(model).__name__}\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    phi_df = compute_shap_values(model, scaler, df)
    verify_efficiency_property(model, scaler, df, phi_df)

    phi_df.to_csv(ANALYSIS_DIR / "shap_values_all_releases.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'shap_values_all_releases.csv'} ({len(phi_df)} releases)\n")

    print("=== Global risk drivers (test set) ===")
    global_drivers = plot_global_drivers(phi_df, split_name="test")
    print(global_drivers.head(10).to_string(index=False))

    print("\n=== Local explanation examples ===")
    plot_local_examples(phi_df, df, model, scaler)


if __name__ == "__main__":
    main()
