"""
features.py
-------------
Feature engineering for the fraud detection model.

IMPORTANT (leakage prevention):
All rolling/customer-behavior features are computed using only PAST
transactions relative to each row (expanding/rolling window ordered by time),
never using information from the future.

Run standalone:
    python src/features.py
Output:
    data/transactions_features.csv
"""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df = df.sort_values(["customer_id", "transaction_time"]).reset_index(drop=True)

    # --- Time features ---
    df["hour"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 0 and h <= 5) else 0)

    # --- Amount features ---
    df["log_amount"] = np.log1p(df["amount"])

    # Expanding (past-only) mean amount per customer, shifted by 1 to exclude current txn
    df["cust_running_avg_amount"] = (
        df.groupby("customer_id")["amount"]
          .apply(lambda s: s.shift(1).expanding().mean())
          .reset_index(level=0, drop=True)
    )
    df["cust_running_avg_amount"] = df["cust_running_avg_amount"].fillna(df["amount"].median())
    df["amount_vs_customer_avg"] = df["amount"] / (df["cust_running_avg_amount"] + 1e-6)

    # --- Customer velocity features (past-only, time-based rolling windows) ---
    df = df.set_index("transaction_time")
    df["transactions_last_24h"] = (
        df.groupby("customer_id")["amount"]
          .rolling("24h", closed="left").count()
          .reset_index(level=0, drop=True)
    )
    df["transactions_last_1h"] = (
        df.groupby("customer_id")["amount"]
          .rolling("1h", closed="left").count()
          .reset_index(level=0, drop=True)
    )
    df = df.reset_index()
    df["transactions_last_24h"] = df["transactions_last_24h"].fillna(0)
    df["transactions_last_1h"] = df["transactions_last_1h"].fillna(0)

    # --- Device / novelty features (past-only) ---
    df["seen_device_before"] = (
        df.groupby(["customer_id", "device"]).cumcount()
    )
    df["new_device"] = (df["seen_device_before"] == 0).astype(int)
    df = df.drop(columns=["seen_device_before"])

    # --- Location change (simple distance-like proxy, past-only) ---
    df["prev_lat"] = df.groupby("customer_id")["location_lat"].shift(1)
    df["prev_lon"] = df.groupby("customer_id")["location_lon"].shift(1)
    df["location_change"] = np.sqrt(
        (df["location_lat"] - df["prev_lat"]) ** 2 + (df["location_lon"] - df["prev_lon"]) ** 2
    ).fillna(0)
    df = df.drop(columns=["prev_lat", "prev_lon"])

    return df


CATEGORICAL_COLS = ["merchant_category", "payment_type", "device"]

NUMERIC_FEATURE_COLS = [
    "amount", "log_amount", "hour", "day_of_week", "is_weekend", "is_night",
    "cust_running_avg_amount", "amount_vs_customer_avg",
    "transactions_last_24h", "transactions_last_1h",
    "new_device", "location_change",
]


def encode_features(df: pd.DataFrame):
    """One-hot encode categoricals, return X, y, and feature name list."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    dummy_cols = [c for c in df_encoded.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)]
    feature_cols = NUMERIC_FEATURE_COLS + dummy_cols
    X = df_encoded[feature_cols]
    y = df_encoded["is_fraud"]
    return X, y, feature_cols


if __name__ == "__main__":
    raw = pd.read_csv("data/transactions.csv")
    featured = build_features(raw)
    featured.to_csv("data/transactions_features.csv", index=False)
    print(f"Feature engineering complete. Shape: {featured.shape}")
    print("New columns:", [c for c in featured.columns if c not in raw.columns])
