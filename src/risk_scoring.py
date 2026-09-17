"""
risk_scoring.py
------------------
Converts fraud probabilities into a business-friendly Risk Score (0-100) and
Risk Level (Low / Medium / High), and generates a risk report for the test set.

Thresholds here are PROJECT-DESIGN CHOICES — tune them based on your
precision/recall trade-off analysis (see reports/model_comparison.csv and
reports/pr_curves_comparison.png).

Run:
    python src/risk_scoring.py
Output:
    reports/risk_scored_transactions.csv
"""
import json
import joblib
import pandas as pd

from features import build_features, encode_features
from train import time_based_split

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


def probability_to_risk(prob: float) -> int:
    """Simple linear mapping: Risk Score = Fraud Probability * 100."""
    return round(prob * 100)


def risk_level(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "LOW"
    elif prob < HIGH_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def recommended_action(level: str) -> str:
    return {
        "LOW": "Approve",
        "MEDIUM": "Additional verification (OTP / step-up auth)",
        "HIGH": "Flag for manual review / hold transaction",
    }[level]


def main():
    model = joblib.load("models/xgboost_model.pkl")
    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)

    raw = pd.read_csv("data/transactions.csv")
    featured = build_features(raw)
    train_df, val_df, test_df = time_based_split(featured, feature_cols=None)

    X_all, y_all, _ = encode_features(featured)
    X_test = X_all.loc[test_df.index][feature_cols]

    y_prob = model.predict_proba(X_test)[:, 1]

    result = test_df[["transaction_id", "customer_id", "amount", "merchant_category",
                       "payment_type", "is_fraud"]].copy()
    result["fraud_probability"] = y_prob
    result["risk_score"] = [probability_to_risk(p) for p in y_prob]
    result["risk_level"] = [risk_level(p) for p in y_prob]
    result["recommended_action"] = result["risk_level"].apply(recommended_action)

    result = result.sort_values("risk_score", ascending=False)
    result.to_csv("reports/risk_scored_transactions.csv", index=False)

    print("Risk level distribution:")
    print(result["risk_level"].value_counts())
    print("\nTop 10 highest-risk transactions:")
    print(result.head(10).to_string(index=False))
    print("\nSaved: reports/risk_scored_transactions.csv")


if __name__ == "__main__":
    main()
