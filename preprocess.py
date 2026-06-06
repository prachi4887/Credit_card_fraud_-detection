"""
preprocess.py
-------------
Data preprocessing utilities for Credit Card Fraud Detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def check_missing(df: pd.DataFrame) -> None:
    """Print missing value summary."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[INFO] No missing values found.")
    else:
        print("[WARNING] Missing values detected:")
        print(missing[missing > 0])


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale 'Amount' and 'Time' using StandardScaler.
    Drops originals and adds scaled versions.
    """
    scaler = StandardScaler()
    df = df.copy()
    df['scaled_Amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_Time']   = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    print("[INFO] 'Amount' and 'Time' scaled and replaced.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"[INFO] Train fraud rate: {y_train.mean()*100:.3f}%")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE oversampling to handle class imbalance.

    Returns:
        X_resampled, y_resampled
    """
    print(f"[INFO] Before SMOTE -> Genuine: {(y_train==0).sum():,} | Fraud: {(y_train==1).sum():,}")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After  SMOTE -> Genuine: {(y_res==0).sum():,} | Fraud: {(y_res==1).sum():,}")
    return X_res, y_res


def apply_undersampling(X_train, y_train, random_state: int = 42):
    """
    Apply random undersampling to handle class imbalance.

    Returns:
        X_resampled, y_resampled
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"[INFO] After Undersampling -> Genuine: {(y_res==0).sum():,} | Fraud: {(y_res==1).sum():,}")
    return X_res, y_res


def full_pipeline(filepath: str, method: str = 'smote'):
    """
    Run the full preprocessing pipeline.

    Args:
        filepath: Path to the raw CSV file.
        method: Imbalance handling method - 'smote' or 'undersample'.

    Returns:
        X_train, X_test, y_train, y_test (after resampling)
    """
    df = load_data(filepath)
    check_missing(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    if method == 'smote':
        X_train, y_train = apply_smote(X_train, y_train)
    elif method == 'undersample':
        X_train, y_train = apply_undersampling(X_train, y_train)
    else:
        print("[INFO] No resampling applied.")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = full_pipeline('../data/creditcard.csv', method='smote')
    print("\n[SUCCESS] Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
