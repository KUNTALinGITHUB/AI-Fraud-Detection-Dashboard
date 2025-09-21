# step4_dashboard_simple.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# -------------------------------
# Load models
# -------------------------------
rf = joblib.load("models/random_forest.joblib")
xgb = joblib.load("models/xgboost.joblib")
lr = joblib.load("models/logistic_regression.joblib")
autoencoder = load_model("models/autoencoder.h5", custom_objects={"mse": MeanSquaredError()})

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection Dashboard")

st.write("Upload a CSV file with transactions to check for **fraud risk**.")

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### 📋 Uploaded Data Preview", data.head())

    if "Class" in data.columns:
        X = data.drop(columns=["Class"])
        y = data["Class"]
    else:
        X = data
        y = None

    # -------------------------------
    # Model Predictions
    # -------------------------------
    st.subheader("🔮 Fraud Risk Predictions")

    rf_probs = rf.predict_proba(X)[:, 1]
    xgb_probs = xgb.predict_proba(X)[:, 1]
    lr_probs = lr.predict_proba(X)[:, 1]

    # Autoencoder anomaly score
    X_np = np.array(X)
    X_recon = autoencoder.predict(X_np)
    mse = np.mean(np.square(X_np - X_recon), axis=1)
    ae_scores = (mse - mse.min()) / (mse.max() - mse.min())

    results = pd.DataFrame({
        "RandomForest_Prob": rf_probs,
        "XGBoost_Prob": xgb_probs,
        "LogReg_Prob": lr_probs,
        "AutoEnc_Score": ae_scores,
    })

    # Average fraud risk across models
    results["Avg_Fraud_Risk"] = results.mean(axis=1)

    # Show summary
    st.write("### 📊 Fraud Risk Scores (Top 10)")
    st.dataframe(results.head(10))

    # -------------------------------
    # Fraud Risk Distribution
    # -------------------------------
    st.subheader("📈 Fraud Risk Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(results["Avg_Fraud_Risk"], bins=20, kde=True, color="red", ax=ax)
    ax.set_title("Fraud Risk Distribution")
    ax.set_xlabel("Fraud Risk Score (0 = Safe, 1 = High Risk)")
    st.pyplot(fig)

    # -------------------------------
    # Bar Chart: Fraud vs Normal
    # -------------------------------
    st.subheader("🚨 Fraud vs Normal (Based on Risk > 0.5)")
    results["Fraud_Flag"] = (results["Avg_Fraud_Risk"] > 0.5).astype(int)

    fraud_count = results["Fraud_Flag"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=fraud_count.index.map({0: "Normal", 1: "Fraud"}), y=fraud_count.values, palette=["green", "red"], ax=ax)
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # -------------------------------
    # Fraud Risk Gauge (Single Selection)
    # -------------------------------
    st.subheader("🎯 Check Risk of a Single Transaction")
    idx = st.number_input("Select transaction index", min_value=0, max_value=len(results)-1, value=0)
    risk_score = results.loc[idx, "Avg_Fraud_Risk"]

    st.metric(label="Fraud Risk Score", value=f"{risk_score:.2f}")

    if risk_score < 0.3:
        st.success("✅ Low Risk Transaction")
    elif risk_score < 0.7:
        st.warning("⚠️ Medium Risk Transaction")
    else:
        st.error("🚨 High Risk Transaction")

    # -------------------------------
    # SHAP Global Feature Importance
    # -------------------------------
    st.subheader("📊 Key Features Driving Predictions (SHAP)")
    sample_data = X.sample(min(200, len(X)), random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample_data)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_to_plot, sample_data, show=False)
    st.pyplot(fig)
    st.write("The plot above shows the most important features influencing the fraud risk predictions.")
    st.write("Red features increase risk, blue features decrease risk.")