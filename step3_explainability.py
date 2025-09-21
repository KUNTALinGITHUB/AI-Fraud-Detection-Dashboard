# step3_explainability.py
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os

os.makedirs("explainability", exist_ok=True)

# -------------------------------
# 1. Load processed data
# -------------------------------
test = pd.read_csv("processed/test.csv")
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

# -------------------------------
# 2. Load saved models
# -------------------------------
lr = joblib.load("models/logistic_regression.joblib")
rf = joblib.load("models/random_forest.joblib")
xgb = joblib.load("models/xgboost.joblib")
autoencoder = load_model("models/autoencoder.h5", custom_objects={"mse": MeanSquaredError()})

print("✅ Models loaded successfully!")

# -------------------------------
# 3. SHAP Explainability (Global)
# -------------------------------
print("\n🔹 Running SHAP for Random Forest...")

sample_data = X_test.sample(1000, random_state=42)
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(sample_data)

# Handle binary vs multiclass output
if isinstance(shap_values_rf, list):
    # For binary classification → pick class 1 (fraud)
    shap_values_to_plot = shap_values_rf[1]
else:
    # If it's already a numpy array
    shap_values_to_plot = shap_values_rf

shap.summary_plot(shap_values_to_plot, sample_data, show=False)
plt.savefig("explainability/shap_rf_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved: explainability/shap_rf_summary.png")

# -------------------------------
# 4. LIME Explainability (Local)
# -------------------------------
print("\n🔹 Running LIME on a fraud sample...")

# Pick one fraud sample
fraud_sample = X_test[y_test == 1].iloc[0]

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test),
    feature_names=X_test.columns,
    class_names=["Normal", "Fraud"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    data_row=fraud_sample.values,
    predict_fn=rf.predict_proba
)

exp.save_to_file("explainability/lime_fraud_example.html")
print("✅ LIME explanation saved: explainability/lime_fraud_example.html")

print("\n🎯 Step 3 complete: SHAP + LIME outputs generated.")
