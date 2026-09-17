"""
eda_statistics.py
-------------------
Exploratory Data Analysis + statistical hypothesis testing for the fraud
dataset. Produces plots (saved as PNGs) and prints statistical test results.

Run:
    python src/eda_statistics.py
Output:
    reports/*.png
    Printed statistical summary
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("reports", exist_ok=True)


def load_data():
    df = pd.read_csv("data/transactions.csv", parse_dates=["transaction_time"])
    df["hour"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_time"].dt.dayofweek
    return df


def basic_overview(df):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass balance:\n", df["is_fraud"].value_counts(normalize=True))
    print("\nDescribe (amount):\n", df["amount"].describe())


def plot_fraud_distribution(df):
    counts = df["is_fraud"].value_counts()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar", color=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["Legitimate", "Fraud"], rotation=0)
    plt.title("Fraud vs Legitimate Transaction Count")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/fraud_distribution.png", dpi=120)
    plt.close()


def plot_amount_by_class(df):
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [df[df.is_fraud == 0]["amount"], df[df.is_fraud == 1]["amount"]],
        tick_labels=["Legitimate", "Fraud"],
        showfliers=False,
    )
    plt.title("Transaction Amount by Class (outliers hidden)")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig("reports/amount_by_class.png", dpi=120)
    plt.close()


def plot_fraud_by_hour(df):
    hourly = df.groupby("hour")["is_fraud"].mean() * 100
    plt.figure(figsize=(7, 4))
    hourly.plot(kind="line", marker="o", color="#C44E52")
    plt.title("Fraud Rate (%) by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Rate (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/fraud_by_hour.png", dpi=120)
    plt.close()


def plot_fraud_by_merchant(df):
    rate = (df.groupby("merchant_category")["is_fraud"].mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    rate.plot(kind="bar", color="#55A868")
    plt.title("Fraud Rate (%) by Merchant Category")
    plt.ylabel("Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig("reports/fraud_by_merchant.png", dpi=120)
    plt.close()


def statistical_tests(df):
    fraud_amt = df[df.is_fraud == 1]["amount"]
    legit_amt = df[df.is_fraud == 0]["amount"]

    print("\n--- Descriptive stats: amount ---")
    for label, series in [("Fraud", fraud_amt), ("Legit", legit_amt)]:
        print(f"{label}: mean={series.mean():.2f}, median={series.median():.2f}, "
              f"std={series.std():.2f}, q1={series.quantile(.25):.2f}, q3={series.quantile(.75):.2f}")

    # Mann-Whitney U test: amount distributions differ between fraud/legit?
    u_stat, p_value = stats.mannwhitneyu(fraud_amt, legit_amt, alternative="two-sided")
    print(f"\nMann-Whitney U test (amount, fraud vs legit): U={u_stat:.1f}, p-value={p_value:.6f}")
    if p_value < 0.05:
        print("=> Reject H0: fraud and legitimate transaction amounts come from different distributions.")
    else:
        print("=> Fail to reject H0: no significant difference detected.")

    # Chi-square test: is fraud rate independent of payment_type?
    contingency = pd.crosstab(df["payment_type"], df["is_fraud"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    print(f"\nChi-square test (payment_type vs is_fraud): chi2={chi2:.2f}, p-value={p_chi2:.6f}")
    if p_chi2 < 0.05:
        print("=> Fraud rate is significantly associated with payment type.")
    else:
        print("=> No significant association detected.")

    # Correlation of hour with fraud (point-biserial-like via simple corr)
    corr = df[["hour", "is_fraud"]].corr().iloc[0, 1]
    print(f"\nCorrelation (hour, is_fraud): {corr:.4f}")


if __name__ == "__main__":
    df = load_data()
    basic_overview(df)
    plot_fraud_distribution(df)
    plot_amount_by_class(df)
    plot_fraud_by_hour(df)
    plot_fraud_by_merchant(df)
    statistical_tests(df)
    print("\nPlots saved to reports/")
