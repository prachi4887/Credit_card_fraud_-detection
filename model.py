"""
model.py
--------
Model training utilities for Credit Card Fraud Detection.
"""

import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train, y_train, max_iter: int = 1000, random_state: int = 42):
    """
    Train a Logistic Regression classifier.

    Returns:
        Trained LogisticRegression model
    """
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    print("[INFO] Logistic Regression training complete.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest classifier.

    Returns:
        Trained RandomForestClassifier model
    """
    print("[INFO] Training Random Forest (this may take a minute)...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest training complete.")
    return model


def save_model(model, name: str, save_dir: str = '../models/'):
    """Save a trained model to disk using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.pkl')
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")


def load_model(name: str, load_dir: str = '../models/'):
    """Load a saved model from disk."""
    path = os.path.join(load_dir, f'{name}.pkl')
    model = joblib.load(path)
    print(f"[INFO] Model loaded from: {path}")
    return model


if __name__ == '__main__':
    from preprocess import full_pipeline

    X_train, X_test, y_train, y_test = full_pipeline('../data/creditcard.csv')

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    save_model(lr, 'logistic_regression')
    save_model(rf, 'random_forest')

    print("\n[SUCCESS] All models trained and saved.")
