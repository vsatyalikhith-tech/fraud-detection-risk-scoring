"""
shap_analysis.py
------------------
SHAP explainability for the trained XGBoost model.

Run:
    python src/shap_analysis.py
Output:
    reports/shap_summary.png
    reports/shap_feature_importance.png
    Printed per-transaction explanation for a sample high-risk transaction
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import build_features, encode_features
from train import time_based_split


def main():
    model = joblib.load("models/xgboost_model.pkl")
    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)

    raw = pd.read_csv("data/transactions.csv")
    featured = build_features(raw)
    train_df, val_df, test_df = time_based_split(featured, feature_cols=None)

    X_all, y_all, _ = encode_features(featured)
    X_test = X_all.loc[test_df.index][feature_cols]
    y_test = y_all.loc[test_df.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance (summary plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=120)
    plt.close()

    # Bar chart of mean |SHAP value| per feature
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_feature_importance.png", dpi=120)
    plt.close()

    # Explain a single high-risk transaction
    y_prob = model.predict_proba(X_test)[:, 1]
    top_idx_local = int(np.argmax(y_prob))  # position within X_test
    top_row = X_test.iloc[[top_idx_local]]
    top_shap = shap_values[top_idx_local]

    print(f"\nHighest predicted fraud probability in test set: {y_prob[top_idx_local]:.4f}")
    print(f"Actual label: {y_test.iloc[top_idx_local]}")
    contrib = pd.Series(top_shap, index=feature_cols).sort_values(key=abs, ascending=False)
    print("\nTop contributing features (SHAP values):")
    print(contrib.head(8).to_string())

    print("\nSHAP plots saved to reports/shap_summary.png and reports/shap_feature_importance.png")


if __name__ == "__main__":
    main()
