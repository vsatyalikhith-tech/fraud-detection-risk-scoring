# Fraud Detection & Risk Scoring System

An end-to-end fraud detection and risk-scoring pipeline: SQL-based transaction
analysis → EDA & statistics → class-imbalance handling → feature engineering
→ Logistic Regression → Random Forest → XGBoost → evaluation → SHAP
explainability → risk scoring → FastAPI service → Streamlit dashboard.

## Project structure

```
fraud-detection-risk-scoring/
├── data/                       # raw + processed data, SQLite DB (generated)
├── sql/
│   └── fraud_analysis.sql      # standalone SQL queries
├── src/
│   ├── generate_data.py        # synthetic dataset generator (swap for real data)
│   ├── sql_analysis.py         # loads CSV -> SQLite, runs business queries
│   ├── eda_statistics.py       # EDA plots + hypothesis tests
│   ├── features.py             # feature engineering (leakage-safe)
│   ├── train.py                # trains LogReg, RF, XGBoost + evaluation
│   ├── shap_analysis.py        # SHAP explainability
│   ├── risk_scoring.py         # probability -> risk score/level
│   └── predict.py              # single-transaction scoring used by the API
├── api/
│   └── app.py                  # FastAPI risk-scoring service
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── models/                     # saved model artifacts (generated)
├── reports/                    # plots, metrics, scored transactions (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline, in order

```bash
# 1. Generate data (replace with a real dataset by matching the same schema)
python src/generate_data.py

# 2. SQL extraction & analysis
python src/sql_analysis.py

# 3. EDA + statistical tests
python src/eda_statistics.py

# 4. Train models (Logistic Regression -> Random Forest -> XGBoost)
python src/train.py

# 5. SHAP explainability
python src/shap_analysis.py

# 6. Convert probabilities into risk scores
python src/risk_scoring.py
```

## Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 8500,
    "hour": 2,
    "day_of_week": 5,
    "merchant_category": "electronics",
    "payment_type": "credit_card",
    "device": "web_chrome",
    "new_device": 1,
    "transactions_last_1h": 8,
    "transactions_last_24h": 12,
    "cust_running_avg_amount": 1500
  }'
```

Response:

```json
{
  "fraud_probability": 0.87,
  "risk_score": 87,
  "risk_level": "HIGH",
  "recommended_action": "Flag for manual review / hold transaction"
}
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Using a real dataset instead of the synthetic one

Replace `data/transactions.csv` with your real dataset, keeping (or mapping to)
these columns: `transaction_id, customer_id, amount, transaction_time,
merchant_category, payment_type, device, location_lat, location_lon, is_fraud`.
A public credit-card fraud dataset (e.g. from Kaggle) works well — you may
need to rename/derive columns to match this schema, or adjust `features.py`
to match the columns you actually have.

## Notes on methodology

- **Leakage prevention**: all customer-behavior features (running averages,
  transaction velocity, device novelty) are computed using only *past*
  transactions relative to each row.
- **Time-based split**: train/validation/test are split chronologically
  (70/15/15) rather than randomly, to simulate predicting future transactions
  from past data.
- **Class imbalance**: handled via `class_weight="balanced"` (Logistic
  Regression, Random Forest) and `scale_pos_weight` (XGBoost) rather than
  naive oversampling; accuracy is intentionally not used as the primary
  metric — precision, recall, F1, ROC-AUC and PR-AUC are.
- **Risk score** = `fraud_probability * 100`, bucketed into LOW (<30),
  MEDIUM (30–70), HIGH (>70). These thresholds are project-design choices —
  tune them against your precision/recall trade-off, not a universal standard.

## Resume line

> Built an end-to-end fraud detection system using SQL-based transaction
> analysis, statistical analysis, feature engineering, Logistic Regression,
> Random Forest and XGBoost; evaluated models using precision, recall, F1,
> ROC-AUC and PR-AUC, and converted fraud probabilities into explainable
> transaction risk scores using SHAP, served via a FastAPI risk-scoring
> endpoint.
