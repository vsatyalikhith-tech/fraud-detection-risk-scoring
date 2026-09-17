"""
predict.py
------------
Loads the trained XGBoost model and scores a single new transaction
(as a dict of raw fields) end-to-end: raw fields -> features -> probability
-> risk score/level. Used by the FastAPI service in api/app.py.
"""
import json
import joblib
import numpy as np
import pandas as pd

from risk_scoring import probability_to_risk, risk_level, recommended_action

MODEL_PATH = "models/xgboost_model.pkl"
FEATURES_PATH = "models/feature_columns.json"

_model = None
_feature_cols = None


def _load():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


def score_transaction(payload: dict) -> dict:
    """
    payload example (simplified, single-transaction, no customer history needed):
    {
        "amount": 8500,
        "hour": 2,
        "day_of_week": 5,
        "merchant_category": "electronics",
        "payment_type": "credit_card",
        "device": "web_chrome",
        "new_device": 1,
        "transactions_last_1h": 8,
        "transactions_last_24h": 12,
        "cust_running_avg_amount": 1500,
        "location_change": 0.0
    }
    Missing fields are filled with sensible defaults.
    """
    model, feature_cols = _load()

    amount = float(payload.get("amount", 0))
    hour = int(payload.get("hour", 12))
    day_of_week = int(payload.get("day_of_week", 0))
    cust_avg = float(payload.get("cust_running_avg_amount", amount if amount > 0 else 1))

    row = {
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week in (5, 6) else 0,
        "is_night": 1 if hour <= 5 else 0,
        "cust_running_avg_amount": cust_avg,
        "amount_vs_customer_avg": amount / (cust_avg + 1e-6),
        "transactions_last_24h": float(payload.get("transactions_last_24h", 1)),
        "transactions_last_1h": float(payload.get("transactions_last_1h", 0)),
        "new_device": int(payload.get("new_device", 0)),
        "location_change": float(payload.get("location_change", 0.0)),
    }

    # One-hot encode categoricals to match training columns; unseen categories -> all zeros
    for col in feature_cols:
        for prefix in ["merchant_category_", "payment_type_", "device_"]:
            if col.startswith(prefix):
                field = prefix[:-1]  # e.g. "merchant_category"
                category_value = payload.get(field)
                row[col] = 1 if (category_value is not None and col == f"{prefix}{category_value}") else 0

    X = pd.DataFrame([row])
    # Ensure all expected columns exist, in the right order
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    prob = float(model.predict_proba(X)[:, 1][0])
    level = risk_level(prob)

    return {
        "fraud_probability": round(prob, 4),
        "risk_score": probability_to_risk(prob),
        "risk_level": level,
        "recommended_action": recommended_action(level),
    }


if __name__ == "__main__":
    sample = {
        "amount": 8500,
        "hour": 2,
        "day_of_week": 5,
        "merchant_category": "electronics",
        "payment_type": "credit_card",
        "device": "web_chrome",
        "new_device": 1,
        "transactions_last_1h": 8,
        "transactions_last_24h": 12,
        "cust_running_avg_amount": 1500,
    }
    print(score_transaction(sample))
