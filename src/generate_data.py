"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC transaction dataset for the Fraud Detection &
Risk Scoring project. Replace this with your real dataset (e.g. a Kaggle
credit-card / e-commerce transactions dataset) by loading it into the same
column schema used below.

Run:
    python src/generate_data.py
Output:
    data/transactions.csv
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 3000
N_TRANSACTIONS = 60000
FRAUD_RATE = 0.012  # ~1.2% fraud, realistic imbalance

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "fashion",
                        "food_delivery", "utilities", "entertainment", "jewelry"]
PAYMENT_TYPES = ["credit_card", "debit_card", "upi", "netbanking", "wallet"]
DEVICES = ["mobile_ios", "mobile_android", "web_chrome", "web_safari", "pos_terminal"]

def random_timestamp(start, end):
    delta = end - start
    return start + timedelta(seconds=np.random.randint(0, int(delta.total_seconds())))

def generate():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    customer_ids = np.random.randint(10000, 10000 + N_CUSTOMERS, size=N_TRANSACTIONS)
    is_fraud = np.random.binomial(1, FRAUD_RATE, size=N_TRANSACTIONS)

    rows = []
    for i in range(N_TRANSACTIONS):
        fraud = is_fraud[i]
        ts = random_timestamp(start_date, end_date)

        # Fraud tends to happen at odd hours, higher amounts, new devices
        if fraud:
            hour = np.random.choice(range(0, 24), p=_night_heavy_probs())
            amount = np.round(np.random.lognormal(mean=6.5, sigma=1.1), 2)  # higher amounts
            device = np.random.choice(DEVICES, p=[0.30, 0.30, 0.15, 0.10, 0.15])
            merchant = np.random.choice(MERCHANT_CATEGORIES,
                                         p=[0.05, 0.20, 0.20, 0.10, 0.05, 0.05, 0.10, 0.25])
        else:
            hour = ts.hour
            amount = np.round(np.random.lognormal(mean=4.2, sigma=0.9), 2)  # normal amounts
            device = np.random.choice(DEVICES, p=[0.30, 0.35, 0.20, 0.10, 0.05])
            merchant = np.random.choice(MERCHANT_CATEGORIES,
                                         p=[0.25, 0.15, 0.08, 0.15, 0.15, 0.12, 0.08, 0.02])

        ts = ts.replace(hour=hour)

        rows.append({
            "transaction_id": f"TXN{100000 + i}",
            "customer_id": customer_ids[i],
            "amount": amount,
            "transaction_time": ts,
            "merchant_category": merchant,
            "payment_type": np.random.choice(PAYMENT_TYPES),
            "device": device,
            "location_lat": np.round(np.random.uniform(8.0, 35.0), 4),
            "location_lon": np.round(np.random.uniform(68.0, 90.0), 4),
            "is_fraud": fraud,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("transaction_time").reset_index(drop=True)
    return df

def _night_heavy_probs():
    # weight late-night/early-morning hours higher for fraud
    probs = np.ones(24)
    for h in [0, 1, 2, 3, 4, 23]:
        probs[h] = 4
    return probs / probs.sum()

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df):,} transactions -> data/transactions.csv")
    print(df["is_fraud"].value_counts(normalize=True))
