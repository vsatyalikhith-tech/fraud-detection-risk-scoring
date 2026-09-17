-- fraud_analysis.sql
-- Standalone SQL queries for the Fraud Detection & Risk Scoring project.
-- Run these directly against data/fraud.db (created by src/sql_analysis.py)
-- using: sqlite3 data/fraud.db < sql/fraud_analysis.sql

-- 1. Total transactions
SELECT COUNT(*) AS total FROM transactions;

-- 2. Fraud vs legitimate class balance
SELECT is_fraud, COUNT(*) AS count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 3) AS pct
FROM transactions
GROUP BY is_fraud;

-- 3. Fraud rate by payment type
SELECT payment_type,
       COUNT(*) AS transactions,
       SUM(is_fraud) AS fraud_count,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY payment_type
ORDER BY fraud_rate_pct DESC;

-- 4. Fraud rate by merchant category (with average amount)
SELECT merchant_category,
       COUNT(*) AS transactions,
       SUM(is_fraud) AS fraud_count,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
       ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;

-- 5. Fraud rate by hour of day
SELECT CAST(strftime('%H', transaction_time) AS INTEGER) AS hour,
       COUNT(*) AS transactions,
       SUM(is_fraud) AS fraud_count,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;

-- 6. Fraud rate by device
SELECT device,
       COUNT(*) AS transactions,
       SUM(is_fraud) AS fraud_count,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY device
ORDER BY fraud_rate_pct DESC;

-- 7. Top 10 highest-value fraud transactions
SELECT transaction_id, customer_id, amount, merchant_category, payment_type
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 10;

-- 8. Customers with unusually high transaction counts (possible velocity abuse)
SELECT customer_id, COUNT(*) AS txn_count, SUM(is_fraud) AS fraud_count
FROM transactions
GROUP BY customer_id
HAVING txn_count > 40
ORDER BY txn_count DESC;
