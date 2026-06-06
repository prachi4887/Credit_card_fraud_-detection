"""
main.py
-------
Entry point for Credit Card Fraud Detection project.
Run this file to train all models and generate evaluation outputs.

Usage:
    python main.py
    python main.py --method undersample
    python main.py --data path/to/creditcard.csv
"""

import argparse
import os
import pandas as pd

from src.preprocess import full_pipeline
from src.model import train_logistic_regression, train_random_forest, save_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_metrics_comparison
)


def main(data_path: str, method: str):
    print("\n" + "="*60)
    print("   CREDIT CARD FRAUD DETECTION - ML Pipeline")
    print("="*60)

    # Step 1: Preprocessing
    print("\n[STEP 1] Preprocessing data...")
    X_train, X_test, y_train, y_test = full_pipeline(data_path, method=method)

    # Step 2: Train Models
    print("\n[STEP 2] Training models...")
    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # Step 3: Save Models
    print("\n[STEP 3] Saving models...")
    os.makedirs('models', exist_ok=True)
    save_model(lr, 'logistic_regression', save_dir='models/')
    save_model(rf, 'random_forest',       save_dir='models/')

    # Step 4: Evaluate
    print("\n[STEP 4] Evaluating models...")
    os.makedirs('outputs', exist_ok=True)

    models = {
        'Logistic Regression': lr,
        'Random Forest':       rf,
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_test, y_test))

    # Step 5: Save plots
    print("\n[STEP 5] Generating plots...")
    plot_confusion_matrices(models, X_test, y_test, save_dir='outputs/')
    plot_roc_curves(models, X_test, y_test,          save_dir='outputs/')
    plot_precision_recall_curves(models, X_test, y_test, save_dir='outputs/')
    plot_metrics_comparison(results,                 save_dir='outputs/')

    # Final Summary
    print("\n" + "="*60)
    print("   FINAL RESULTS SUMMARY")
    print("="*60)
    print(pd.DataFrame(results).to_string(index=False))
    print("\n[DONE] Pipeline complete. Check 'outputs/' folder for plots.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    parser.add_argument('--data',   type=str, default='data/creditcard.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--method', type=str, default='smote',
                        choices=['smote', 'undersample', 'none'],
                        help='Imbalance handling method')
    args = parser.parse_args()

    main(data_path=args.data, method=args.method)
