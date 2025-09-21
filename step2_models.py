# --- Imports ---
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# --- Load Data ---
train = pd.read_csv("processed/train.csv")
test  = pd.read_csv("processed/test.csv")

X_train = train.drop("Class", axis=1).values
y_train = train["Class"].values

X_test = test.drop("Class", axis=1).values
y_test = test["Class"].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Helper for Evaluation ---
def evaluate_model(y_true, y_pred, name):
    print(f"\n🔹 Results for {name}:")
    print(classification_report(y_true, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))


# ======================================================
# 1️⃣ UNSUPERVISED MODELS
# ======================================================

## Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=0.001, random_state=42)
iso.fit(X_train)
y_pred_iso = iso.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # convert -1 (outlier) -> fraud=1
evaluate_model(y_test, y_pred_iso, "Isolation Forest")

## Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
y_pred_lof = lof.fit_predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
evaluate_model(y_test, y_pred_lof, "Local Outlier Factor")


## Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 14  # half of features

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-4))(input_layer)
encoder = Dense(int(encoding_dim/2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim/2), activation="relu")(encoder)
decoder = Dense(input_dim, activation="tanh")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train, X_train,
                epochs=5,  # keep low to test quickly, can increase later
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=1)

# Reconstruction error threshold
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 99.5)  # top 0.5% suspicious
y_pred_ae = (mse > threshold).astype(int)
evaluate_model(y_test, y_pred_ae, "Autoencoder")


# ======================================================
# 2️⃣ SUPERVISED BASELINES
# ======================================================

## Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
evaluate_model(y_test, y_pred_lr, "Logistic Regression")

## Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, "Random Forest")

## XGBoost
xgb = XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=10, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
evaluate_model(y_test, y_pred_xgb, "XGBoost")





# Create folder if not exists
os.makedirs("models", exist_ok=True)

# Save supervised models
joblib.dump(log_reg, "models/logistic_regression.joblib")
joblib.dump(rf, "models/random_forest.joblib")
joblib.dump(xgb, "models/xgboost.joblib")

# Save unsupervised models
joblib.dump(iso, "models/isolation_forest.joblib")
joblib.dump(lof, "models/local_outlier_factor.joblib")

# Save autoencoder
autoencoder.save("models/autoencoder.h5")

print("✅ All models saved in 'models/' folder")
