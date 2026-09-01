# Release Risk Intelligence Platform

**Predicting elevated post-release software quality risk from GitHub repository analytics**

DAMO 699 Capstone Project — Master of Data Analytics
Group 3: Bogapriya Muralikannan, Kanishka Skandaraj, Swathi Thota, Nihal Vellaramkallingal Sulaiman

## Live Dashboard

🔗 **[View the live Release Risk Intelligence Platform](https://release-risk-intelligence.streamlit.app/)**

## Project Overview

This project investigates whether **release risk** — the likelihood that a software release will experience elevated post-release bug activity — can be predicted from pre-release GitHub activity data (commit volume, pull-request activity, contributor turnover, issue backlog, and release timing), using only signals available *before* a release is published.

The analysis spans three major open-source repositories, deliberately selected for their divergence in scale, governance model, and release cadence:

| Repository | Scale / Governance | Role in the study |
|---|---|---|
| `kubernetes/kubernetes` | Very large, CNCF-governed, community-driven | High release frequency, largest contributor base |
| `apache/airflow` | Mid-sized, Apache Software Foundation-governed | Longer release cycles, smaller contributor community |
| `microsoft/vscode` | Large, corporate-sponsored (Microsoft) | High commit volume, centralized development model |

555 eligible releases were collected across the three repositories via the GitHub REST API (since January 1, 2021) and used to build, validate, and explain a deployed release-risk classifier.

## Research Questions

- **RQ1:** Does pre-release commit volume predict elevated post-release bug activity?
- **RQ2:** Does pull-request activity (volume, merge outcome, timing) predict elevated risk?
- **RQ3:** Does contributor turnover during the pre-release window predict elevated risk?
- **RQ4:** Does release timing (cycle length, release history) predict elevated risk?
- **RQ5:** Do these relationships generalize across repositories of different scale, governance, and cadence?

Full research questions, hypotheses, and evidence-based findings for each are documented in the project report (Sections 1 and 4).

## Key Results

- **Final model:** L2-regularized Logistic Regression (C=0.001), 19 features, operating threshold 0.43
- **Test-set performance:** ROC-AUC 0.929, Precision 94.4%, Recall 100.0%, F1 97.1%, Accuracy 96.4%
- **Confusion matrix:** zero false negatives, 4 false positives (out of 111 test releases)
- Tree-based models (decision tree, random forest, gradient boosting, XGBoost) were tested and rejected due to a confirmed covariate-shift collapse on the chronological test set — see the full report for details.
- Model selection, hyperparameter tuning, threshold selection, bootstrap validation, and both standard k-fold and time-series cross-validation are all documented and reproducible.

## Project Structure

```
├── src/
│ ├── main.py # CLI entry point — orchestrates raw data collection
│ ├── collectors.py # Per-endpoint GitHub API collection logic
│ ├── github_client.py # GitHub API client (auth, pagination, rate limits)
│ ├── cleaning.py / run_cleaning.py # Data cleaning and release-cycle construction
│ ├── run_features.py / preprocess_for_modeling.py # Feature engineering and preprocessing
│ ├── correlation_analysis.py, hypothesis_testing.py, explanatory_regression.py # Descriptive/diagnostic analytics
│ ├── predictive_modeling.py # Baseline + logistic regression
│ ├── tree_based_models.py # Decision tree, random forest, gradient boosting
│ ├── advanced_models.py # Elastic net, recency-weighted logistic, naive Bayes, repository-aware models
│ ├── xgboost_model.py # XGBoost candidate model
│ ├── sensitivity_analysis.py # Feature-ablation sensitivity + calibration curve
│ ├── kfold_cross_validation.py / timeseries_cross_validation.py # Supplementary CV analyses
│ ├── threshold_analysis.py # Threshold selection and sensitivity
│ ├── explainability.py # SHAP-based global/local explanations
│ ├── dashboard_core.py # Dashboard data-loading and prediction logic (no Streamlit dependency)
│ └── dashboard_app.py # Streamlit dashboard UI
├── data/
│ ├── raw/<repo>/ # Untouched API JSON responses (gitignored)
│ ├── tables/<repo>/ # Tidy raw CSV tables (gitignored)
│ ├── clean/<repo>/ # Cleaned, standardized tables
│ ├── features/<repo>/ # Engineered, model-ready feature tables
│ ├── models/ # Trained models, scalers, comparison results
│ └── analysis/ # Diagnostic outputs, SHAP values, cross-validation results
├── logs/ # API collection logs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For data collection, generate a GitHub Personal Access Token (Settings → Developer settings → Personal access tokens → Tokens (classic); no scopes needed for public repo data), then set it as an environment variable:

```bash
export GITHUB_TOKEN=ghp_your_token_here      # Mac/Linux
set GITHUB_TOKEN=ghp_your_token_here         # Windows cmd
$env:GITHUB_TOKEN="ghp_your_token_here"      # Windows PowerShell
```

## Running the Pipeline

```bash
# 1. Data collection
python src/main.py --repos kubernetes/kubernetes apache/airflow microsoft/vscode --since 2021-01-01

# 2. Cleaning
python src/run_cleaning.py

# 3. Feature engineering and preprocessing
python src/run_features.py
python src/preprocess_for_modeling.py

# 4. Predictive modeling
python src/predictive_modeling.py
python src/tree_based_models.py
python src/advanced_models.py
python src/xgboost_model.py

# 5. Explainability
python src/explainability.py
```

## Running the Dashboard Locally

```bash
streamlit run src/dashboard_app.py
```

Or view the live deployed version: **[release-risk-intelligence.streamlit.app](https://release-risk-intelligence.streamlit.app/)**

## Documentation

Full methodology, findings, and limitations are documented in the project's Comprehensive Report (Sections 1–7 plus Appendix A: Feature-Level Data Dictionary), included in this repository.

## Team

Bogapriya Muralikannan · Kanishka Skandaraj · Swathi Thota · Nihal Vellaramkallingal Sulaiman