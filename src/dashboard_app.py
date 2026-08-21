"""
dashboard_app.py  (lives in src/)
------------------------------------------
Week 7: functional dashboard, built on Streamlit. All data-loading,
prediction, and scenario logic lives in dashboard_core.py (tested
independently with plain Python, since streamlit itself is not
installable/testable in the environment this was developed in) -- this
file is purely the UI layer on top of that already-verified logic.

Pages:
    1. Overview          -- project summary, key stats, model performance chart
    2. Release Explorer  -- browse/filter releases with predictions, risk tiers
    3. Release Detail     -- SHAP breakdown, recommendations, scenario analysis for one release
    4. Model Performance  -- comparison table, calibration curve, threshold sweep
    5. About / Methodology -- documentation, known limitations

Usage:
    streamlit run src/dashboard_app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from dashboard_core import DashboardData, FEATURE_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
MODELS_DIR = REPO_ROOT / "data" / "models"

st.set_page_config(page_title="Release Risk Intelligence Platform", layout="wide")


@st.cache_resource
def load_data():
    return DashboardData.load()


def render_overview(data: DashboardData):
    st.header("Release Risk Intelligence Platform")
    st.caption("Predicting elevated post-release software quality risk from GitHub repository analytics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total releases", len(data.df))
    col2.metric("Repositories", data.df["repository_name"].nunique())
    col3.metric("Final model", data.model_name)
    col4.metric("Operating threshold", f"{data.threshold:.2f}" if isinstance(data.threshold, float) else data.threshold)
        # Final model performance on the test set
    test_df = data.df[data.df["split"] == "test"].copy()

    X_test = data.scaler.transform(test_df[FEATURE_COLS].values)
    y_test = test_df["elevated_risk"].astype(int).values

    test_prob = data.model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= float(data.threshold)).astype(int)

    roc_auc = roc_auc_score(y_test, test_prob)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    accuracy = accuracy_score(y_test, test_pred)

    st.subheader("Final model performance — test set")

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("ROC-AUC", f"{roc_auc:.3f}")
    p2.metric("Precision", f"{precision:.1%}")
    p3.metric("Recall", f"{recall:.1%}")
    p4.metric("F1 Score", f"{f1:.1%}")
    p5.metric("Accuracy", f"{accuracy:.1%}")

    st.subheader("Repository breakdown")
    repo_counts = data.df["repository_name"].value_counts().rename("releases")
    st.bar_chart(repo_counts)

    if data.comparison_summary is not None:
        st.subheader("Model comparison (ROC-AUC by split)")
        main_models = data.comparison_summary[
            data.comparison_summary["model"].isin([
                "baseline_majority_class", "decision_tree", "random_forest", "gradient_boosting",
                "elastic_net", "naive_bayes",
            ]) | data.comparison_summary["model"].str.startswith("logistic_regression_C=")
            | data.comparison_summary["model"].str.startswith("recency_weighted_logistic")
        ]
        st.dataframe(main_models, use_container_width=True)
        st.caption(
            "Repository-specific and partial-pooling models (evaluated only on their own test subsets, "
            "not train/validation) are omitted from this summary for clarity -- see "
            "data/models/model_comparison_summary.csv for the complete table."
        )
        chart_path = MODELS_DIR / "model_comparison_chart.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="Model comparison across train/validation/test")

    st.info(
        "This dashboard reflects a model trained on data through the project's training window. "
        "See the 'About / Methodology' page for known limitations, including a documented "
        "class-distribution drift between the training and evaluation periods."
    )


def render_release_explorer(data: DashboardData):
    st.header("Release Explorer")
    st.caption("Browse releases with model-predicted risk, filterable by repository, data split, and risk tier.")

    col1, col2, col3 = st.columns(3)
    repo_options = ["All"] + sorted(data.df["repository_name"].unique().tolist())
    repository = col1.selectbox("Repository", repo_options)
    split_options = ["All"] + sorted(data.df["split"].dropna().unique().tolist()) if "split" in data.df.columns else ["All"]
    split = col2.selectbox("Data split", split_options)
    risk_tier = col3.selectbox("Risk tier", ["All", "Low", "Medium", "High"])

    filtered = data.filter_releases(repository=repository, split=split, risk_tier=risk_tier)
    st.write(f"**{len(filtered)} releases match these filters.**")
    st.dataframe(
        filtered.sort_values("predicted_probability", ascending=False),
        use_container_width=True,
        column_config={
            "predicted_probability": st.column_config.ProgressColumn(
                "Predicted risk", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )

    st.caption(
        "Select a release_id above and use the 'Release Detail' page to see its full explanation, "
        "recommendations, and scenario analysis."
    )


def render_release_detail(data: DashboardData):
    st.header("Release Detail")
    st.caption("Full risk breakdown, plain-language drivers, and what-if scenario analysis for one release.")

    release_ids = data.df["release_id"].tolist()
    release_id = st.selectbox("Select a release_id", release_ids)

    if release_id is None:
        return

    result = data.predict_release(release_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("Repository", result["repository_name"])
    col2.metric("Predicted risk", f"{result['probability']:.3f}")
    col3.metric("Risk tier", result["risk_tier"])

    if result["actual_elevated_risk"] is not None and result["split"] in ("train", "validation"):
        st.caption(f"Actual label (non-test split, informational only): elevated_risk = {int(result['actual_elevated_risk'])}")

    st.subheader("Top contributing factors (exact Shapley values)")
    shap_breakdown = data.get_shap_breakdown(release_id)
    if not shap_breakdown.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in shap_breakdown["contribution"]]
        ax.barh(shap_breakdown["feature"][::-1], shap_breakdown["contribution"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution to log-odds (red = toward risk, blue = toward safe)")
        st.pyplot(fig)
    else:
        st.warning("SHAP values not available -- run explainability.py first.")

    if data.readiness_report is not None:
        rec_row = data.readiness_report[data.readiness_report["release_id"] == release_id]
        if not rec_row.empty:
            st.subheader("Recommendations")
            r = rec_row.iloc[0]
            for i in range(1, 4):
                driver = r.get(f"top_driver_{i}", "")
                rec = r.get(f"recommendation_{i}", "")
                if isinstance(driver, str) and driver:
                    st.markdown(f"**{driver}**")
                    st.markdown(f"> {rec}")

    st.subheader("Scenario analysis (what-if)")
    st.caption("Adjust one feature and see the model's prediction update, using the model's exact linear math.")
    scenario_feature = st.selectbox(
        "Feature to adjust",
        ["open_bugs_at_release", "prior_releases_avg_bugs", "commit_count", "pr_count", "cycle_length_days"],
    )
    row = data.df[data.df["release_id"] == release_id].iloc[0]
    current_value = float(row[scenario_feature]) if scenario_feature in row else 0.0
    new_value = st.slider(
        f"New value for {scenario_feature}",
        min_value=0.0, max_value=max(current_value * 3, 10.0), value=current_value,
    )
    if st.button("Run scenario"):
        scenario = data.scenario_what_if(release_id, scenario_feature, new_value)
        st.write(f"Predicted probability: **{scenario['old_probability']:.3f}** -> **{scenario['new_probability']:.3f}**")
        st.write(f"Risk tier: **{scenario['old_risk_tier']}** -> **{scenario['new_risk_tier']}**")
        if scenario_feature in ("open_bugs_at_release", "prior_releases_avg_bugs"):
            st.caption(
                "Note: this feature's relationship with risk runs in opposite directions for "
                "microsoft/vscode versus the other two repositories -- interpret the direction "
                "alongside which repository this release belongs to."
            )


def render_model_performance(data: DashboardData):
    st.header("Model Performance")
        # Confusion matrix for final model on the test set
    test_df = data.df[data.df["split"] == "test"].copy()

    X_test = data.scaler.transform(test_df[FEATURE_COLS].values)
    y_test = test_df["elevated_risk"].astype(int).values

    test_prob = data.model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= float(data.threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    st.subheader(f"Final results at threshold {data.threshold:.2f}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Risky releases caught", tp)
    c2.metric("Risky releases missed", fn)
    c3.metric("Safe releases identified", tn)
    c4.metric("False alarms", fp)

    st.subheader("Confusion matrix")

    cm_table = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=["Actually safe", "Actually elevated risk"],
        columns=["Predicted safe", "Predicted elevated risk"],
    )

    st.dataframe(cm_table, use_container_width=True)

    if data.comparison_summary is not None:
        st.subheader("ROC-AUC by model and split")
        main_models = data.comparison_summary[
            data.comparison_summary["model"].isin([
                "baseline_majority_class", "decision_tree", "random_forest", "gradient_boosting",
                "elastic_net", "naive_bayes",
            ]) | data.comparison_summary["model"].str.startswith("logistic_regression_C=")
            | data.comparison_summary["model"].str.startswith("recency_weighted_logistic")
        ]
        st.dataframe(main_models, use_container_width=True)

    calibration_path = ANALYSIS_DIR / "calibration_curve.png"
    if calibration_path.exists():
        st.subheader("Calibration")
        st.image(str(calibration_path))

    if data.threshold_sweep is not None:
        st.subheader("Threshold sensitivity (validation set)")
        st.line_chart(data.threshold_sweep.set_index("threshold")[["precision", "recall", "f1", "f2"]])

    global_drivers_path = ANALYSIS_DIR / "global_risk_drivers.png"
    if global_drivers_path.exists():
        st.subheader("Global risk drivers")
        st.image(str(global_drivers_path))

def render_data_insights(data: DashboardData):
    st.header("Data & Repository Insights")
    st.caption(
        "How release risk changes over time and how patterns differ across repositories."
    )

    split_rates = (
        data.df.groupby("split")["elevated_risk"]
        .mean()
        .reindex(["train", "validation", "test"])
    )

    st.subheader("Risk distribution across time periods")

    c1, c2, c3 = st.columns(3)

    c1.metric("Training risk rate", f"{split_rates['train']:.1%}")
    c2.metric("Validation risk rate", f"{split_rates['validation']:.1%}")
    c3.metric("Test risk rate", f"{split_rates['test']:.1%}")
    st.warning(
        "The share of elevated-risk releases changes quite a lot across the three time periods. "
        "This shows that the data distribution has shifted over time, so the model should be "
        "monitored and re-evaluated as new release data becomes available."
    )

    trend_path = ANALYSIS_DIR / "trend_over_time.png"

    if trend_path.exists():
        st.subheader("Risk and bug trends over time")
        st.image(
            str(trend_path),
            caption="Elevated-risk rate and average post-release bug count over time",
        )

        sign_flip_path = ANALYSIS_DIR / "open_bugs_sign_flip.png"

    if sign_flip_path.exists():
        st.subheader("How risk patterns differ by repository")

        st.write(
            "For Apache Airflow and Kubernetes, elevated-risk releases tend to have "
            "higher open-bug backlogs. For VS Code, the relationship goes in the opposite direction."
        )

        st.image(
            str(sign_flip_path),
            caption="Open bug backlog relationship by repository",
        )

        st.info(
            "This shows that the same risk signal does not behave the same way across all repositories, "
            "so repository context matters when interpreting the model."
        )

def render_about():
    st.header("About / Methodology")
    st.markdown("""
This dashboard supports **release risk assessment**, not automated release blocking. Predictions
should inform, not replace, human release-readiness judgment.

**Known limitations:**
- Code-churn and pull-request review-depth features could not be collected (confirmed unavailable
  at the GitHub API collection source for all three repositories); the model relies on commit/PR
  volume and timing, contributor turnover, issue backlog, and release-timing/history features only.
- A documented, statistically confirmed drift exists in bug-reporting rates over time, differing by
  repository. Test-period evaluation results should be interpreted with this in mind.
- `open_bugs_at_release`'s relationship with risk reverses direction between microsoft/vscode and
  the other two repositories -- a confirmed, real finding, not a modelling artifact.
- This model has been validated on three repositories only and should not be assumed to generalize
  to a repository with substantially different scale or governance without re-validation.

See the project's Progress Report, Target-Variable Definition, and Model Card documents for full detail.
""")


def main():
    try:
        data = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Release Explorer", "Release Detail", "Model Performance", "Data & Repository Insights", "About / Methodology"],
    )

    if page == "Overview":
        render_overview(data)
    elif page == "Release Explorer":
        render_release_explorer(data)
    elif page == "Release Detail":
        render_release_detail(data)
    elif page == "Model Performance":
        render_model_performance(data)
    elif page == "Data & Repository Insights":
        render_data_insights(data)
    elif page == "About / Methodology":
        render_about()


if __name__ == "__main__":
    main()
