"""
recommendation_framework.py  (lives in src/)
------------------------------------------
Week 7: recommendation rules, release-readiness framework, and scenario
analysis -- built directly on top of explainability.py's exact SHAP
values (run that script first).

RELEASE-READINESS FRAMEWORK: maps a release's predicted probability into
a three-tier readiness level (Low / Medium / High risk), using the
project's chosen decision threshold (0.32, from threshold_analysis.py) as
the Medium/High boundary, and the midpoint between 0 and that threshold
as the Low/Medium boundary -- so "Low" means comfortably below the
operating point, not just "under 0.5."

RECOMMENDATION RULES: a rule-based mapping from each feature to a
plain-language explanation and a suggested action, applied to a release's
TOP contributing SHAP drivers (not all 21 features -- only the ones that
actually mattered for that specific release). This directly uses the
project's confirmed findings (e.g. the release_sequence/prior_releases_count
drift finding, the open_bugs_at_release repository sign-flip) rather than
generic, made-up advice.

SCENARIO ANALYSIS: since the final model is linear, the effect of
changing one feature's value can be computed EXACTLY via simple
arithmetic (no re-fitting needed) -- this answers "what if this release
had had fewer open bugs at release time" type questions precisely.

Produces, under data/analysis/:
    release_readiness_report.csv   -- risk tier + top drivers + recommendations, all test-set releases
    scenario_analysis_examples.csv -- what-if examples for the 3 releases used in explainability.py

Usage:
    python src/recommendation_framework.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
INPUT_PATH = FEATURES_DIR / "model_ready_release_level.csv"
SHAP_PATH = ANALYSIS_DIR / "shap_values_all_releases.csv"

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
TOP_N_DRIVERS = 3

# --- Recommendation rules: feature -> (plain-language meaning, action when
# this feature is pushing TOWARD elevated risk for a given release) ---
# Grounded directly in this project's confirmed findings, not generic advice.
FEATURE_RULES = {
    "release_sequence": (
        "This release's position in its repository's history is associated with elevated risk "
        "(a project-wide finding, strongest and most consistent signal across every analysis method used).",
        "Cross-check against recent actual bug trends for this repository before relying heavily on this "
        "signal alone -- it partly reflects a documented training-period drift pattern that may not hold "
        "indefinitely (see the Target-Variable Definition document)."
    ),
    "prior_releases_count": (
        "Same underlying signal as release_sequence (the two are nearly identical, r=0.999) -- how far "
        "into this repository's release history this release falls.",
        "See release_sequence recommendation; treat these two as one signal, not two independent ones."
    ),
    "open_bugs_at_release_repo_z": (
        "The pre-release open-bug backlog is unusually high or low relative to THIS REPOSITORY'S OWN "
        "typical level (a repo-normalized value, not a raw count -- see the raw open_bugs_at_release "
        "column for the actual count). NOTE: this relationship runs in OPPOSITE directions for "
        "microsoft/vscode versus kubernetes/kubernetes and apache/airflow -- confirm which repository "
        "this release belongs to before interpreting the direction.",
        "For kubernetes/kubernetes or apache/airflow: a high backlog suggests unresolved technical debt "
        "risk -- consider prioritizing backlog triage before this release ships. For microsoft/vscode: "
        "a LOW backlog was associated with higher risk in this dataset -- investigate whether this reflects "
        "under-reporting rather than genuine stability before treating it as reassuring."
    ),
    "prior_releases_avg_bugs_repo_z": (
        "This repository's historical average post-release bug count (from releases whose own outcome "
        "was already known at this release's time), expressed relative to this repository's own typical "
        "level (repo-normalized, not a raw count).",
        "Review whether recent releases in this repository have been trending toward more or fewer "
        "post-release bugs; a persistently high historical average may warrant a broader process review, "
        "not just extra scrutiny on this one release."
    ),
    "commit_count": (
        "The volume of commits before this release is unusually high or low.",
        "A very high commit count (a large release) may warrant proportionally more testing time; a very "
        "low count paired with elevated risk may indicate the release was small but touched a "
        "disproportionately risky area of the codebase -- review change scope, not just change volume."
    ),
    "pr_count": (
        "The number of pull requests merged before this release is unusually high or low.",
        "Consider whether review capacity kept pace with PR volume for this release cycle."
    ),
    "avg_time_to_merge_hours": (
        "Pull requests before this release merged unusually quickly or slowly on average.",
        "Very fast merging paired with elevated risk may indicate reduced review scrutiny; consider a "
        "second reviewer pass on the highest-impact changes before shipping."
    ),
    "median_time_to_merge_hours": (
        "Same signal as avg_time_to_merge_hours, less sensitive to a few outlier PRs.",
        "See avg_time_to_merge_hours recommendation."
    ),
    "distinct_contributors": (
        "The number of distinct people committing before this release is unusually high or low.",
        "A low distinct-contributor count concentrates release risk on fewer people's work; consider "
        "an additional independent review pass."
    ),
    "first_time_contributor_count": (
        "This release involved an unusually high number of first-time contributors.",
        "Consider additional review scrutiny specifically on first-time contributors' changes, given "
        "this project's confirmed (though not fully independent of other factors) association between "
        "contributor turnover and elevated risk."
    ),
    "first_time_contributor_share": (
        "Same signal as first_time_contributor_count, expressed as a share of all contributors.",
        "See first_time_contributor_count recommendation."
    ),
    "top_contributor_share": (
        "This release's commits were unusually concentrated in (or spread across) a small number of "
        "contributors.",
        "High concentration in one contributor increases bus-factor risk; consider pairing or review "
        "redundancy for that contributor's changes."
    ),
    "cycle_length_days": (
        "The time since the previous release is unusually long or short.",
        "A very short cycle may mean limited integration/soak time before shipping; consider whether "
        "this release had adequate testing time relative to its scope."
    ),
    "open_issues_at_release": (
        "The total open-issue backlog (not just bugs) at release time is unusually high or low relative "
        "to this repository's own typical level.",
        "Review whether backlog growth reflects genuine user-facing risk or simply issue-triage lag."
    ),
    "had_commit_activity": ("Whether any human commit activity was observed in this release's cycle at all.", "N/A -- structural flag, not independently actionable."),
    "had_pr_activity": ("Whether any human PR activity was observed in this release's cycle at all.", "N/A -- structural flag, not independently actionable."),
    "has_prior_release_history": ("Whether this repository had any prior release to compare against.", "N/A -- structural flag, not independently actionable."),
    "has_prior_release": ("Whether a prior release exists to measure a pre-release cycle from.", "N/A -- structural flag, not independently actionable."),
    "pct_merged": (
        "The share of this cycle's PRs that were merged (vs. closed unmerged) is unusually high or low.",
        "A low merge rate paired with elevated risk may indicate churn or disagreement during review; "
        "investigate closed-unmerged PRs for recurring concerns."
    ),
    "repo_kubernetes/kubernetes": (
        "Repository identity itself (kubernetes/kubernetes vs. the baseline, apache/airflow) contributes "
        "to this prediction.",
        "Interpret cautiously -- confirmed statistically significant as a joint effect, but this project "
        "also found this repository's true risk direction can reverse once other features are controlled "
        "for (a Simpson's-paradox-style finding); do not treat repository identity alone as a strong signal."
    ),
    "repo_microsoft/vscode": (
        "Repository identity itself (microsoft/vscode vs. the baseline, apache/airflow) contributes to "
        "this prediction.",
        "Interpret cautiously -- see repo_kubernetes/kubernetes recommendation; the same caveat about "
        "repository-specific reversals applies."
    ),
}


def get_risk_tier(proba: float) -> str:
    low_medium_boundary = CHOSEN_THRESHOLD / 2
    if proba < low_medium_boundary:
        return "Low"
    elif proba < CHOSEN_THRESHOLD:
        return "Medium"
    else:
        return "High"


def build_readiness_report(df: pd.DataFrame, shap_df: pd.DataFrame, model, scaler, split_name: str = "test") -> pd.DataFrame:
    subset_df = df[df["split"] == split_name].reset_index(drop=True)
    subset_shap = shap_df[shap_df["split"] == split_name].reset_index(drop=True)
    X_scaled = scaler.transform(subset_df[FEATURE_COLS].values)
    proba = model.predict_proba(X_scaled)[:, 1]

    shap_cols = [c for c in shap_df.columns if c.startswith("shap_") and c not in ("shap_base_value", "shap_sum_plus_base")]

    rows = []
    for i in range(len(subset_df)):
        release_id = subset_df.loc[i, "release_id"]
        repo = subset_df.loc[i, "repository_name"]
        p = proba[i]
        tier = get_risk_tier(p)

        contributions = subset_shap.loc[i, shap_cols]
        # Only drivers pushing TOWARD elevated risk (positive contribution)
        # are actionable recommendations for a risky release; for a safe
        # release, the top POSITIVE contributors are still worth surfacing
        # as "watch items," since they're the closest thing to a risk signal
        # even if outweighed by other factors.
        top_positive = contributions.sort_values(ascending=False).head(TOP_N_DRIVERS)

        driver_descriptions = []
        recommendations = []
        for feat_col, shap_val in top_positive.items():
            feat = feat_col.replace("shap_", "")
            if feat in FEATURE_RULES and shap_val > 0:
                desc, action = FEATURE_RULES[feat]
                driver_descriptions.append(f"{feat} (contribution={shap_val:.3f}): {desc}")
                recommendations.append(action)

        rows.append({
            "release_id": release_id, "repository_name": repo,
            "predicted_probability": p, "risk_tier": tier,
            "top_driver_1": driver_descriptions[0] if len(driver_descriptions) > 0 else "",
            "top_driver_2": driver_descriptions[1] if len(driver_descriptions) > 1 else "",
            "top_driver_3": driver_descriptions[2] if len(driver_descriptions) > 2 else "",
            "recommendation_1": recommendations[0] if len(recommendations) > 0 else "",
            "recommendation_2": recommendations[1] if len(recommendations) > 1 else "",
            "recommendation_3": recommendations[2] if len(recommendations) > 2 else "",
        })

    return pd.DataFrame(rows)


# Maps a human-readable feature name to its actual model-input column, for
# the two features that were repo-normalized to fix multicollinearity.
# scenario_analysis() accepts the RAW, human-readable name/value (e.g. "what
# if open_bugs_at_release had been 350") and converts internally -- asking
# someone to reason in z-score units would defeat the purpose of a
# scenario tool meant for release managers, not statisticians.
REPO_NORMALIZED_FEATURE_MAP = {
    "open_bugs_at_release": "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs": "prior_releases_avg_bugs_repo_z",
}


def scenario_analysis(df: pd.DataFrame, model, scaler, release_id, feature: str, new_raw_value: float) -> dict:
    """
    Exact what-if analysis: since the model is linear in scaled-feature
    space, changing one feature's raw value shifts the log-odds by exactly
    coef_i * (new_scaled_value - old_scaled_value) -- no re-fitting needed,
    and no approximation involved.

    `feature` may be given as either a direct FEATURE_COLS name, or as the
    human-readable raw name for a repo-normalized feature (e.g.
    "open_bugs_at_release" instead of "open_bugs_at_release_repo_z") -- in
    the latter case, new_raw_value should be a real, human-readable value
    (e.g. an actual bug count), and this function converts it to the
    correct repo-specific z-score internally, using that repository's own
    TRAINING-split mean/std (matching exactly how preprocess_for_modeling.py
    built the column in the first place).
    """
    row = df[df["release_id"] == release_id].iloc[0]
    repo = row["repository_name"]

    if feature in REPO_NORMALIZED_FEATURE_MAP:
        model_feature = REPO_NORMALIZED_FEATURE_MAP[feature]
        train_rows = df[(df["repository_name"] == repo) & (df["split"] == "train")]
        mean, std = train_rows[feature].mean(), train_rows[feature].std()
        if pd.isna(std) or std == 0:
            std = 1
        new_value_for_model = (new_raw_value - mean) / std
        display_old_value = row[feature]  # show the RAW old value, not its z-score
    else:
        model_feature = feature
        new_value_for_model = new_raw_value
        display_old_value = row[feature]

    old_raw = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
    new_raw = old_raw.copy()
    feat_idx = FEATURE_COLS.index(model_feature)
    new_raw[0, feat_idx] = new_value_for_model

    old_scaled = scaler.transform(old_raw)
    new_scaled = scaler.transform(new_raw)

    old_proba = model.predict_proba(old_scaled)[0, 1]
    new_proba = model.predict_proba(new_scaled)[0, 1]

    return {
        "release_id": release_id, "repository_name": repo, "feature": feature,
        "old_value": display_old_value, "new_value": new_raw_value,
        "old_probability": old_proba, "new_probability": new_proba,
        "old_risk_tier": get_risk_tier(old_proba), "new_risk_tier": get_risk_tier(new_proba),
        "probability_change": new_proba - old_proba,
    }


def main():
    if not SHAP_PATH.exists():
        print(f"[error] {SHAP_PATH} not found -- run explainability.py first")
        return

    model = joblib.load(MODELS_DIR / "final_model.joblib")
    scaler = joblib.load(MODELS_DIR / "final_scaler.joblib")
    df = pd.read_csv(INPUT_PATH)
    shap_df = pd.read_csv(SHAP_PATH)
    print(f"Loaded {len(df)} releases and {len(shap_df)} SHAP rows\n")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building release-readiness report (test set) ===")
    report = build_readiness_report(df, shap_df, model, scaler, split_name="test")
    report.to_csv(ANALYSIS_DIR / "release_readiness_report.csv", index=False)
    print(f"Wrote {ANALYSIS_DIR / 'release_readiness_report.csv'} ({len(report)} releases)")
    print(f"\nRisk tier distribution:\n{report['risk_tier'].value_counts().to_string()}")
    print("\n[NOTE] The final model uses strong L2 regularization (C=0.001), which compresses individual")
    print("feature coefficients toward small values. This means per-release driver contributions are often")
    print("close together in magnitude -- the 'top driver' ranking for a given release should be read as")
    print("directional guidance, not a sharply decisive ordering.")

    print("\n=== Example readiness report entries ===")
    for tier in ["High", "Medium", "Low"]:
        example = report[report["risk_tier"] == tier].head(1)
        if not example.empty:
            r = example.iloc[0]
            print(f"\n--- {tier} risk example: release_id={r['release_id']}, {r['repository_name']} "
                  f"(P={r['predicted_probability']:.3f}) ---")
            for i in range(1, 4):
                driver = r[f"top_driver_{i}"]
                rec = r[f"recommendation_{i}"]
                if driver:
                    print(f"  Driver {i}: {driver}")
                    print(f"    -> Recommendation: {rec}")

    # --- Scenario analysis on the same 3 examples used in explainability.py ---
    print("\n=== Scenario analysis examples ===")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS].values)
    proba_test = model.predict_proba(X_test_scaled)[:, 1]

    idx_most_risky = int(np.argmax(proba_test))
    example_release_id = test_df.loc[idx_most_risky, "release_id"]
    example_row = test_df.loc[idx_most_risky]

    scenario_rows = []
    # Scenario: what if the open bug backlog had been at this repository's median instead?
    repo_median_backlog = df[df["repository_name"] == example_row["repository_name"]]["open_bugs_at_release"].median()
    s1 = scenario_analysis(df, model, scaler, example_release_id, "open_bugs_at_release", repo_median_backlog)
    scenario_rows.append(s1)
    print(f"\nScenario 1 -- release_id={example_release_id} ({example_row['repository_name']}): if "
          f"open_bugs_at_release had been {repo_median_backlog:.0f} (this repo's median) instead of "
          f"{s1['old_value']:.0f}:")
    print(f"  Predicted probability: {s1['old_probability']:.3f} -> {s1['new_probability']:.3f} "
          f"(risk tier: {s1['old_risk_tier']} -> {s1['new_risk_tier']})")
    print(f"  NOTE: interpret the direction of this change alongside which repository this release belongs to --")
    print(f"  open_bugs_at_release's relationship with risk runs in OPPOSITE directions for microsoft/vscode")
    print(f"  versus kubernetes/kubernetes and apache/airflow (see the Week 5 sign-flip finding).")

    # Scenario: what if commit_count had been half its actual value?
    half_commits = example_row["commit_count"] / 2
    s2 = scenario_analysis(df, model, scaler, example_release_id, "commit_count", half_commits)
    scenario_rows.append(s2)
    print(f"\nScenario 2 -- release_id={example_release_id} ({example_row['repository_name']}): if "
          f"commit_count had been {half_commits:.0f} (half actual) instead of {s2['old_value']:.0f}:")
    print(f"  Predicted probability: {s2['old_probability']:.3f} -> {s2['new_probability']:.3f} "
          f"(risk tier: {s2['old_risk_tier']} -> {s2['new_risk_tier']})")

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(ANALYSIS_DIR / "scenario_analysis_examples.csv", index=False)
    print(f"\nWrote {ANALYSIS_DIR / 'scenario_analysis_examples.csv'}")


if __name__ == "__main__":
    main()
