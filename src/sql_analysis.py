"""
sql_analysis.py
-----------------
Loads transactions.csv into a SQLite database and runs SQL analysis queries.
This is the "SQL Extraction" stage of the pipeline — demonstrates SQL skills
on top of the raw data before EDA.

Run:
    python src/sql_analysis.py
Output:
    data/fraud.db (SQLite database)
    Printed query results
"""
import sqlite3
import pandas as pd

DB_PATH = "data/fraud.db"
CSV_PATH = "data/transactions.csv"


def load_to_sql():
    df = pd.read_csv(CSV_PATH, parse_dates=["transaction_time"])
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df):,} rows into {DB_PATH} (table: transactions)")


QUERIES = {
    "total_transactions": "SELECT COUNT(*) AS total FROM transactions;",

    "fraud_vs_legit": """
        SELECT is_fraud, COUNT(*) AS count,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 3) AS pct
        FROM transactions
        GROUP BY is_fraud;
    """,

    "fraud_by_payment_type": """
        SELECT payment_type,
               COUNT(*) AS transactions,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY payment_type
        ORDER BY fraud_rate_pct DESC;
    """,

    "fraud_by_merchant_category": """
        SELECT merchant_category,
               COUNT(*) AS transactions,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
               ROUND(AVG(amount), 2) AS avg_amount
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_rate_pct DESC;
    """,

    "fraud_by_hour": """
        SELECT CAST(strftime('%H', transaction_time) AS INTEGER) AS hour,
               COUNT(*) AS transactions,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY hour
        ORDER BY hour;
    """,

    "top_risky_devices": """
        SELECT device,
               COUNT(*) AS transactions,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY device
        ORDER BY fraud_rate_pct DESC;
    """,

    "high_value_fraud": """
        SELECT transaction_id, customer_id, amount, merchant_category, payment_type
        FROM transactions
        WHERE is_fraud = 1
        ORDER BY amount DESC
        LIMIT 10;
    """,
}


def run_queries():
    conn = sqlite3.connect(DB_PATH)
    for name, sql in QUERIES.items():
        print(f"\n=== {name} ===")
        result = pd.read_sql_query(sql, conn)
        print(result.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    load_to_sql()
    run_queries()
