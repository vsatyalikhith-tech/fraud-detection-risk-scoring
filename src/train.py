"""
train.py
---------
Trains and compares three models on the fraud dataset:
  1. Logistic Regression (baseline, with class_weight='balanced')
  2. Random Forest
  3. XGBoost (main model)

Uses a TIME-BASED split (train on earlier transactions, test on later ones)
to avoid leakage and to simulate real deployment (predicting future fraud
from past patterns).

Run:
    python src/train.py
Output:
    models/logreg_model.pkl
    models/random_forest_model.pkl
    models/xgboost_model.pkl
    models/feature_columns.json
    reports/model_comparison.csv
    reports/*.png (ROC curves, PR curves, confusion matrices)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

from features import build_features, encode_features

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def time_based_split(df, feature_cols, target_col="is_fraud", train_frac=0.70, val_frac=0.15):
    df = df.sort_values("transaction_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def evaluate_model(name, model, X_test, y_test, scaler=None):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Pred Legit", "Pred Fraud"])
    plt.yticks([0, 1], ["Actual Legit", "Actual Fraud"])
    plt.tight_layout()
    plt.savefig(f"reports/confusion_matrix_{name}.png", dpi=120)
    plt.close()

    return metrics, y_prob


def plot_roc_pr_curves(results):
    plt.figure(figsize=(6, 5))
    for name, (y_test, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curves_comparison.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    for name, (y_test, y_prob) in results.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/pr_curves_comparison.png", dpi=120)
    plt.close()


def main():
    raw = pd.read_csv("data/transactions.csv")
    featured = build_features(raw)

    train_df, val_df, test_df = time_based_split(featured, feature_cols=None)

    # Encode using columns fit on the FULL featured set so train/val/test share dummy columns
    X_all, y_all, feature_cols = encode_features(featured)
    X_train = X_all.loc[train_df.index]
    y_train = y_all.loc[train_df.index]
    X_val = X_all.loc[val_df.index]
    y_val = y_all.loc[val_df.index]
    X_test = X_all.loc[test_df.index]
    y_test = y_all.loc[test_df.index]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

    all_results = []
    curve_data = {}

    # ---------- 1. Logistic Regression ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    metrics, y_prob = evaluate_model("LogisticRegression", logreg, X_test, y_test, scaler=scaler)
    all_results.append(metrics)
    curve_data["LogisticRegression"] = (y_test, y_prob)
    joblib.dump(logreg, "models/logreg_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # ---------- 2. Random Forest ----------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    metrics, y_prob = evaluate_model("RandomForest", rf, X_test, y_test)
    all_results.append(metrics)
    curve_data["RandomForest"] = (y_test, y_prob)
    joblib.dump(rf, "models/random_forest_model.pkl")

    # ---------- 3. XGBoost ----------
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metrics, y_prob = evaluate_model("XGBoost", xgb, X_test, y_test)
    all_results.append(metrics)
    curve_data["XGBoost"] = (y_test, y_prob)
    joblib.dump(xgb, "models/xgboost_model.pkl")

    # ---------- Save comparison ----------
    results_df = pd.DataFrame(all_results)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("reports/model_comparison.csv", index=False)

    plot_roc_pr_curves(curve_data)

    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("\nModels saved to models/. Reports saved to reports/.")


if __name__ == "__main__":
    main()
