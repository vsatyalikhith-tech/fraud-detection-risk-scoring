"""
Streamlit dashboard for the Fraud Detection & Risk Scoring project.

Run:
    streamlit run dashboard/app.py

Requires reports/risk_scored_transactions.csv and reports/model_comparison.csv
to already exist (run the pipeline first: generate_data -> sql_analysis ->
eda_statistics -> train -> risk_scoring).
"""
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Fraud Detection & Risk Scoring", layout="wide")

st.title("Fraud Detection & Risk Scoring Dashboard")

try:
    scored = pd.read_csv("reports/risk_scored_transactions.csv")
    comparison = pd.read_csv("reports/model_comparison.csv")
except FileNotFoundError:
    st.error("Run the pipeline first: generate_data.py -> train.py -> risk_scoring.py")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Fraud Overview", "Risk Monitoring", "Model Performance"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(scored):,}")
    col2.metric("Fraud Transactions", f"{int(scored['is_fraud'].sum()):,}")
    col3.metric("Fraud Rate", f"{scored['is_fraud'].mean() * 100:.2f}%")
    col4.metric("High-Risk Flagged", f"{(scored['risk_level'] == 'HIGH').sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(scored, x="merchant_category", color="risk_level",
                            title="Risk Level by Merchant Category", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.pie(scored, names="risk_level", title="Risk Level Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Transaction Risk Monitoring")
    level_filter = st.multiselect("Filter by risk level", ["LOW", "MEDIUM", "HIGH"],
                                   default=["HIGH", "MEDIUM"])
    filtered = scored[scored["risk_level"].isin(level_filter)] if level_filter else scored
    st.dataframe(
        filtered[["transaction_id", "customer_id", "amount", "merchant_category",
                  "fraud_probability", "risk_score", "risk_level", "recommended_action"]]
        .sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

with tab3:
    st.subheader("Model Comparison")
    st.dataframe(comparison.set_index("model"), use_container_width=True)
    fig3 = px.bar(comparison, x="model", y=["precision", "recall", "f1", "roc_auc", "pr_auc"],
                  barmode="group", title="Model Metrics Comparison")
    st.plotly_chart(fig3, use_container_width=True)
