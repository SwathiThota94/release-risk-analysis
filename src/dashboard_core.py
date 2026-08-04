"""
dashboard_core.py  (lives in src/)
------------------------------------------
Week 7: core data-loading, prediction, and scenario logic for the
dashboard. Deliberately separated from dashboard_app.py (the Streamlit UI
layer) so this logic can be tested with plain Python, independent of
Streamlit -- this file has no Streamlit dependency at all.

Usage (as a library, imported by dashboard_app.py):
    from dashboard_core import DashboardData
    data = DashboardData.load()
    proba, tier = data.predict_release(release_id)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"

TARGET_COL = "elevated_risk"
FEATURE_COLS = [
    "cycle_length_days", "release_sequence", "commit_count", "pr_count",
    "pct_merged", "avg_time_to_merge_hours", "median_time_to_merge_hours",
    "distinct_contributors", "first_time_contributor_count", "first_time_contributor_share",
    "top_contributor_share", "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs_repo_z",
    "had_commit_activity", "had_pr_activity", "has_prior_release_history", "has_prior_release",
    "repo_kubernetes/kubernetes", "repo_microsoft/vscode",
]

REPO_NORMALIZED_FEATURE_MAP = {
    "open_bugs_at_release": "open_bugs_at_release_repo_z",
    "prior_releases_avg_bugs": "prior_releases_avg_bugs_repo_z",
}


def _read_optional_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _read_optional_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


class DashboardData:
    """
    Bundles everything the dashboard needs: the release-level dataset, the
    final model + scaler, and whichever pre-computed analysis artifacts are
    available (gracefully handles any that are missing, rather than
    crashing -- lets the dashboard run even if e.g. bootstrap_evaluation.py
    hasn't been run yet).
    """

    def __init__(self, df, model, scaler, model_name, threshold,
                 shap_df, global_drivers, comparison_summary,
                 threshold_sweep, readiness_report):
        self.df = df
        self.model = model
        self.scaler = scaler
        self.model_name = model_name
        self.threshold = threshold
        self.shap_df = shap_df
        self.global_drivers = global_drivers
        self.comparison_summary = comparison_summary
        self.threshold_sweep = threshold_sweep
        self.readiness_report = readiness_report

    @classmethod
    def load(cls):
        df_path = FEATURES_DIR / "model_ready_release_level.csv"
        if not df_path.exists():
            raise FileNotFoundError(
                f"{df_path} not found -- run the full pipeline (preprocess_for_modeling.py "
                "onward) before launching the dashboard."
            )
        df = pd.read_csv(df_path)

        model_path = MODELS_DIR / "final_model.joblib"
        scaler_path = MODELS_DIR / "final_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"{model_path} or {scaler_path} not found -- run finalize_model.py first."
            )
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        config = _read_optional_json(MODELS_DIR / "final_model_info.json")
        model_name = config.get("model_name", "unknown")
        threshold = config.get("threshold", 0.5)

        shap_df = _read_optional_csv(ANALYSIS_DIR / "shap_values_all_releases.csv")
        global_drivers = _read_optional_csv(ANALYSIS_DIR / "global_risk_drivers.csv")
        comparison_summary = _read_optional_csv(MODELS_DIR / "model_comparison_summary.csv")
        threshold_sweep = _read_optional_csv(ANALYSIS_DIR / "threshold_sweep.csv")
        readiness_report = _read_optional_csv(ANALYSIS_DIR / "release_readiness_report.csv")

        return cls(df, model, scaler, model_name, threshold, shap_df,
                    global_drivers, comparison_summary, threshold_sweep, readiness_report)

    def get_risk_tier(self, proba: float) -> str:
        low_medium_boundary = self.threshold / 2
        if proba < low_medium_boundary:
            return "Low"
        elif proba < self.threshold:
            return "Medium"
        else:
            return "High"

    def predict_release(self, release_id) -> dict:
        """Returns the model's predicted probability and risk tier for a
        release already present in the dataset (looked up by release_id)."""
        row = self.df[self.df["release_id"] == release_id]
        if row.empty:
            raise ValueError(f"release_id {release_id} not found in dataset")
        row = row.iloc[0]
        X = self.scaler.transform(row[FEATURE_COLS].values.astype(float).reshape(1, -1))
        proba = self.model.predict_proba(X)[0, 1]
        return {
            "release_id": release_id,
            "repository_name": row["repository_name"],
            "release_date": row["release_date"],
            "probability": float(proba),
            "risk_tier": self.get_risk_tier(proba),
            "actual_elevated_risk": row[TARGET_COL] if TARGET_COL in row else None,
            "split": row.get("split", None),
        }

    def get_shap_breakdown(self, release_id, top_n: int = 8) -> pd.DataFrame:
        """Returns this release's top-N SHAP contributions, sorted by
        absolute magnitude, if shap_values_all_releases.csv is available."""
        if self.shap_df is None:
            return pd.DataFrame()
        row = self.shap_df[self.shap_df["release_id"] == release_id]
        if row.empty:
            return pd.DataFrame()
        row = row.iloc[0]
        shap_cols = [c for c in self.shap_df.columns
                     if c.startswith("shap_") and c not in ("shap_base_value", "shap_sum_plus_base")]
        contributions = row[shap_cols].sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
        out = pd.DataFrame({
            "feature": [c.replace("shap_", "") for c in contributions.index],
            "contribution": contributions.values,
        })
        return out

    def scenario_what_if(self, release_id, feature: str, new_raw_value: float) -> dict:
        """
        Exact what-if recalculation (linear model, no re-fitting needed).
        `feature` may be a raw human-readable name for a repo-normalized
        feature (e.g. 'open_bugs_at_release') -- converted internally to
        the correct repo-specific z-score, using that repository's own
        TRAINING-split mean/std (matching preprocess_for_modeling.py).
        """
        row = self.df[self.df["release_id"] == release_id].iloc[0]
        repo = row["repository_name"]

        if feature in REPO_NORMALIZED_FEATURE_MAP:
            model_feature = REPO_NORMALIZED_FEATURE_MAP[feature]
            train_rows = self.df[(self.df["repository_name"] == repo) & (self.df["split"] == "train")]
            mean, std = train_rows[feature].mean(), train_rows[feature].std()
            if pd.isna(std) or std == 0:
                std = 1
            new_value_for_model = (new_raw_value - mean) / std
            display_old_value = row[feature]
        else:
            model_feature = feature
            new_value_for_model = new_raw_value
            display_old_value = row[feature]

        old_raw = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        new_raw = old_raw.copy()
        feat_idx = FEATURE_COLS.index(model_feature)
        new_raw[0, feat_idx] = new_value_for_model

        old_proba = self.model.predict_proba(self.scaler.transform(old_raw))[0, 1]
        new_proba = self.model.predict_proba(self.scaler.transform(new_raw))[0, 1]

        return {
            "release_id": release_id, "repository_name": repo, "feature": feature,
            "old_value": display_old_value, "new_value": new_raw_value,
            "old_probability": old_proba, "new_probability": new_proba,
            "old_risk_tier": self.get_risk_tier(old_proba), "new_risk_tier": self.get_risk_tier(new_proba),
        }

    def filter_releases(self, repository=None, split=None, risk_tier=None) -> pd.DataFrame:
        """Returns releases matching the given filters (any combination),
        with predicted probability and risk tier attached -- used by the
        Release Explorer page."""
        subset = self.df.copy()
        if repository and repository != "All":
            subset = subset[subset["repository_name"] == repository]
        if split and split != "All":
            subset = subset[subset["split"] == split]

        X = self.scaler.transform(subset[FEATURE_COLS].values)
        subset = subset.copy()
        subset["predicted_probability"] = self.model.predict_proba(X)[:, 1]
        subset["risk_tier"] = subset["predicted_probability"].apply(self.get_risk_tier)

        if risk_tier and risk_tier != "All":
            subset = subset[subset["risk_tier"] == risk_tier]

        return subset[["release_id", "repository_name", "release_date", "split",
                        "predicted_probability", "risk_tier"] +
                       ([TARGET_COL] if TARGET_COL in subset.columns else [])]
