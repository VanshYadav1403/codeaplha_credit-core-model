# Credit Score Prediction Model

A machine learning project that predicts an individual's creditworthiness from financial history data using classification algorithms. Built and tested in Python, designed to run end-to-end in Google Colab with zero external dataset downloads.

## Objective

Predict whether a loan applicant is likely to be creditworthy (low default risk) or a high default risk, based on their financial profile — income, debt, credit utilization, payment history, and loan details.

## Approach

The pipeline generates a realistic synthetic credit dataset (2,000 applicants, 20 features) with engineered ratios commonly used in real credit scoring — debt-to-income, credit utilization, payment-to-income, and loan-to-income ratios. Three classification models are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

Each model is evaluated using 5-fold stratified cross-validation and standard classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 74.5% | 77.8% | 89.8% | 83.4% | **77.5%** | 70.2% |
| Decision Tree | 70.3% | 74.7% | 88.1% | 80.8% | 59.9% | 61.0% |
| Random Forest | 72.3% | 73.8% | 94.7% | 82.9% | 71.7% | 68.9% |

Logistic Regression performs best on ROC-AUC, consistent with the underlying risk relationship being largely linear — a common finding in real credit scoring models.

## Features Used

**Demographics & employment:** age, annual income, employment status, employment years, home ownership

**Credit history:** existing loans, number of credit accounts, credit limit, credit used, credit utilization ratio, credit age, average account age, late payments (last 12 months)

**Loan details:** loan amount, loan term, loan purpose

**Engineered features:** debt-to-income ratio, payment-to-income ratio, loan-to-income ratio

## Visualizations

The pipeline generates and saves:

- Exploratory data analysis (target distribution, correlation heatmap, feature distributions)
- Confusion matrices for all three models
- ROC curves comparing all models
- Model performance comparison bar chart
- Feature importance — Random Forest importances and Logistic Regression coefficients (signed, showing direction of effect)

## Getting Started

### Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `credit_score.py` or paste its contents into a notebook cell
3. Run all cells — no dataset download required

### Run locally

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python credit_score.py
```

## Output Structure

```
outputs/
├── credit_data.csv
├── model_comparison.csv
├── models/
│   └── credit_score_best.pkl
└── plots/
    ├── eda_overview.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── metrics_comparison.png
    └── feature_importance.png
```

## Tech Stack

Python · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn

## Future Improvements

- Train on a real-world dataset (e.g. Kaggle "Give Me Some Credit" or UCI Statlog German Credit)
- Add XGBoost / LightGBM for comparison
- Hyperparameter tuning with GridSearchCV / Optuna
- SHAP values for model interpretability
- Deploy as a web app with Streamlit or Flask
