#!/usr/bin/env python3
"""
Step 1: Load, inspect, preprocess, and split the credit card dataset.
Saves:
 - processed/train.csv
 - processed/test.csv
 - processed/scalers.joblib
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded DataFrame with shape: {df.shape}")
    return df

def quick_eda(df: pd.DataFrame):
    print("\n=== Quick EDA ===")
    print("Columns:", list(df.columns))
    print("\nDtypes:\n", df.dtypes)
    print("\nTop 5 rows:\n", df.head().to_string(index=False))
    missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing}")
    class_counts = df['Class'].value_counts().sort_index()
    total = len(df)
    fraud = int(class_counts.get(1, 0))
    nonfraud = int(class_counts.get(0, 0))
    print(f"\nClass distribution:\n  Non-fraud (0): {nonfraud}\n  Fraud     (1): {fraud}")
    print(f"  Fraud percentage: {fraud/total*100:.5f}%")
    return

def preprocess(df: pd.DataFrame):
    """
    - Scale 'Time' and 'Amount' using StandardScaler and add columns Time_scaled, Amount_scaled.
    - Keep V1..V28 as-is (they are already PCA/standardized in original dataset).
    - Drop original Time/Amount columns (optional).
    """
    df_proc = df.copy()

    # Scale Amount
    amount_scaler = StandardScaler()
    df_proc['Amount_scaled'] = amount_scaler.fit_transform(df_proc[['Amount']])

    # Scale Time
    time_scaler = StandardScaler()
    df_proc['Time_scaled'] = time_scaler.fit_transform(df_proc[['Time']])

    # Option: drop original columns to avoid duplicated info
    df_proc = df_proc.drop(columns=['Time', 'Amount'])

    scalers = {
        'amount_scaler': amount_scaler,
        'time_scaler': time_scaler
    }

    return df_proc, scalers

def split_and_save(df_proc: pd.DataFrame, out_dir: Path):
    # features / label
    X = df_proc.drop(columns=['Class'])
    y = df_proc['Class']

    print(f"\nFeatures shape: {X.shape}, Labels shape: {y.shape}")

    # stratified split to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_fp = out_dir / "train.csv"
    test_fp = out_dir / "test.csv"

    train_df.to_csv(train_fp, index=False)
    test_df.to_csv(test_fp, index=False)

    print(f"\nSaved train -> {train_fp} ({train_df.shape})")
    print(f"Saved test  -> {test_fp} ({test_df.shape})")

    return train_fp, test_fp

def save_scalers(scalers: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    scalers_fp = out_dir / "scalers.joblib"
    joblib.dump(scalers, scalers_fp)
    print(f"Saved scalers to: {scalers_fp}")
    return scalers_fp

def main(csv_path: str, out_dir: str):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    df = load_data(csv_path)
    quick_eda(df)
    df_proc, scalers = preprocess(df)
    print("\nAfter preprocessing, example head:\n", df_proc.head().to_string(index=False))
    train_fp, test_fp = split_and_save(df_proc, out_dir)
    save_scalers(scalers, out_dir)
    print("\nStep 1 complete. Next: model-building (unsupervised detectors + supervised baseline).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step1: data prep for credit card fraud project")
    parser.add_argument("--csv", type=str, default="creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--out_dir", type=str, default="processed", help="Directory to save processed files")
    args = parser.parse_args()
    main(args.csv, args.out_dir)
