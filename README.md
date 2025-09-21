 AI-Fraud-Detection-Dashboard
 💳 Fraud Detection Dashboard  An interactive Streamlit dashboard that combines multiple machine learning models and explainable AI to assess fraud risk in transactional data.


# 💳 Fraud Detection Dashboard

An interactive Streamlit dashboard that combines multiple machine learning models and explainable AI to assess fraud risk in transactional data.

## 🚀 Features

- **Multi-model predictions** using:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Autoencoder (for anomaly detection)
- **Risk scoring**: Aggregates model outputs into a unified fraud risk score.
- **Visual analytics**:
  - Histogram of fraud risk distribution
  - Bar chart of flagged fraud vs normal transactions
  - Transaction-level risk gauge
- **Explainability**:
  - SHAP summary plot to show global feature importance
  - Highlights which features increase or decrease fraud risk

## 📂 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/KUNTALinGITHUB/AI-Fraud-Detection-Dashboard.git
   cd AI-Fraud-Detection-Dashboard
Install dependencies:

bash
pip install -r requirements.txt
Run the dashboard:

bash
streamlit run step4_dashboard_simple.py
Upload a CSV file with transaction data. If it contains a Class column, it will be used as ground truth.

🧠 Models Used
Random Forest / XGBoost / Logistic Regression: Trained on labeled transaction data to predict fraud probability.

Autoencoder: Learns normal transaction patterns and flags anomalies based on reconstruction error.

📊 Explainability with SHAP
SHAP (SHapley Additive exPlanations) is used to interpret the Random Forest model:

Red bars = features that increase fraud risk

Blue bars = features that reduce fraud risk

This helps users understand why a transaction is flagged and builds trust in the model.

📸 Screenshots
Fraud Risk Distribution

Fraud vs Normal Transactions

SHAP Feature Importance Plot

Transaction-level Risk Gauge


https://github.com/user-attachments/assets/8586a344-3d76-480d-a0d2-5833c0933502



🛠 Tech Stack
Tool/Library	Purpose
Streamlit	Interactive dashboard UI
scikit-learn	ML models
XGBoost	Gradient boosting model
TensorFlow/Keras	Autoencoder for anomaly scoring
SHAP	Model interpretability
Seaborn/Matplotlib	Data visualization
Pandas/Numpy	Data manipulation
📈 Use Cases
Banking & Fintech: Flag suspicious transactions

E-commerce: Detect fraudulent purchases

Insurance: Identify claim anomalies

📬 Contact
Feel free to connect on LinkedIn or raise an issue in the repo.
