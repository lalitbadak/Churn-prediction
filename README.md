# Customer Churn Prediction & Analysis

Predict which telecom customers are likely to leave, understand *why*, and score individual customers through an interactive web app.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)

**[Live demo](https://customer-churn-prediction-and-analysis-66prjho2qqhprceqcqnnbo.streamlit.app/)**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key Business Insights](#key-business-insights)
- [Streamlit App](#streamlit-app)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations & Next Steps](#limitations--next-steps)
- [Tech Stack](#tech-stack)

---

## Overview

Customer churn is expensive: keeping an existing customer is generally cheaper than acquiring a new one. This project works through the full analytics workflow on the Telco Customer Churn dataset:

1. Clean and encode the raw data
2. Explore what separates churners from loyal customers
3. Train and compare three classifiers (Logistic Regression, Random Forest, XGBoost)
4. Turn model findings into retention recommendations
5. Deploy the trained XGBoost model in a Streamlit app that returns a churn probability and a risk level

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (IBM sample data, via Kaggle).

| Property | Value |
|---|---|
| Raw rows | 7,043 customers |
| Rows after cleaning | 7,032 |
| Churn rate | ~26.6% (imbalanced) |
| Feature groups | Demographics, account info, subscribed services, billing |
| Target | `Churn` (Yes / No) |

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in `data/`.

---

## Approach

**Cleaning**
- Converted `TotalCharges` to numeric; 11 rows were blank (new customers with `tenure = 0`) and were dropped
- Dropped the `customerID` identifier

**Encoding**
- Binary columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) mapped to 0/1
- Multi-class columns (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`) one-hot encoded
- Result: 40 model features (see [`app/model/feature_names.json`](app/model/feature_names.json))

**Modeling**
- Stratified 70/30 train/test split (`random_state=42`)
- Class imbalance handled with `class_weight='balanced'` for Logistic Regression and Random Forest
- Evaluated with precision, recall, F1, and ROC AUC on the held-out test set (2,110 customers)

---

## Results

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.74 | 0.50 | **0.80** | 0.62 | **0.838** |
| Random Forest (balanced, 100 trees) | 0.78 | 0.61 | 0.47 | 0.53 | 0.811 |
| XGBoost (default parameters) | 0.78 | 0.59 | 0.52 | 0.55 | 0.812 |

**Reading the table:** Logistic Regression has the best ROC AUC and by far the best recall on churners, so it catches the most at-risk customers at the cost of more false alarms. Random Forest and XGBoost are more precise but miss roughly half of actual churners at the default 0.5 threshold. Which model is "best" depends on the relative cost of a missed churner versus a wasted retention offer.

The XGBoost model is the one saved and served in the app (`app/model/churn_model.json`).

### Top drivers by model

| Logistic Regression (coefficients) | Random Forest (importance) | XGBoost (importance) |
|---|---|---|
| Fiber optic internet (+) | TotalCharges | Fiber optic internet |
| Month-to-month contract (+) | tenure | Month-to-month contract |
| Two-year contract (-) | MonthlyCharges | DSL internet |
| DSL internet (-) | Month-to-month contract | One-year contract |
| Streaming movies / TV (+) | No tech support | Streaming movies |

---

## Key Business Insights

- **Contract type is the strongest lever.** Month-to-month customers churn far more than those on one- or two-year contracts. Offer loyalty discounts or perks to move them onto longer terms.
- **Fiber optic customers are at higher risk.** Consider targeted retention offers, bundles, or a review of price-to-value for this segment.
- **Missing support services signal risk.** Customers without online security or tech support churn more. Bundle or promote these services.
- **Higher monthly charges and streaming add-ons correlate with churn.** Personalized discounts or value packages may help.
- **Electronic check users churn more.** Encourage a switch to automatic payment methods.
- **Newer customers are the most vulnerable.** Low tenure and low total charges are strong churn signals, so invest in early-lifecycle engagement.

> These are correlations found in the data, not proven causes. Validate with A/B tests before committing budget.

---

## Streamlit App

The app loads the saved XGBoost booster and returns a churn probability with a risk band.

**Inputs (sidebar):** tenure, monthly charges, total charges, contract type, internet service, online security, tech support, payment method.

**Output:**

| Churn probability | Risk level |
|---|---|
| > 60% | High: consider immediate retention action |
| 30% - 60% | Medium: consider engagement strategies |
| < 30% | Low |

Fields not exposed in the UI (e.g. gender, partner, dependents, phone service, paperless billing) are fixed to default values, and the remaining one-hot service columns default to 0. See [Limitations](#limitations--next-steps).

---

## Project Structure

```
.
├── customer_churn.ipynb        # EDA, modeling, and model export
├── app/
│   ├── streamlit_app.py        # Streamlit web app
│   └── model/
│       ├── churn_model.json    # Saved XGBoost booster
│       └── feature_names.json  # Ordered list of model features
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction-and-Analysis.git
cd Customer-Churn-Prediction-and-Analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app

Run from the **repository root**, since the app loads the model using relative paths:

```bash
streamlit run app/streamlit_app.py
```

### 3. Reproduce the analysis (optional)

1. Place the dataset in `data/`
2. Install notebook dependencies: `pip install jupyter matplotlib seaborn`
3. Open `customer_churn.ipynb`, update the dataset path in the first code cells to match your environment, and run all cells

Running the notebook regenerates `app/model/churn_model.json` and `app/model/feature_names.json`.

---

## Limitations & Next Steps

**Current limitations**
- Several inputs in the app are hard-coded (e.g. gender, partner status), so predictions reflect a partly assumed customer profile.
- The app sets `OnlineSecurity` and `TechSupport` to just Yes/No. For customers with no internet service, the dataset uses a separate "No internet service" category that the app does not populate.
- XGBoost was trained with default hyperparameters and no imbalance handling, and the 0.5 decision threshold was not tuned.
- Evaluation uses a single train/test split rather than cross-validation.
- The dataset is a static sample; real churn behavior shifts over time.

**Ideas for improvement**
- Hyperparameter tuning (`GridSearchCV` / Optuna) and cross-validation
- Handle imbalance in XGBoost (`scale_pos_weight`) or with SMOTE, and tune the decision threshold to a business cost
- Add SHAP explanations to the app for per-customer drivers
- Expose all features in the UI, or add a CSV batch-scoring mode
- Add a cost-benefit view: expected revenue saved vs. retention offer cost

---

## Tech Stack

- **Python:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **App:** Streamlit
- **Development:** Jupyter Notebook, Git, GitHub Codespaces

---

## Acknowledgements

Dataset: IBM Telco Customer Churn sample data, shared on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).
