"""
evaluate.py
-----------
Model evaluation utilities for Credit Card Fraud Detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve
)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Evaluate a model and print classification metrics.

    Returns:
        dict with Precision, Recall, F1, ROC-AUC scores
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"  Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))
    print(f"  ROC-AUC Score:             {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Average Precision Score:   {average_precision_score(y_test, y_proba):.4f}")

    return {
        'Model':     name,
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1-Score':  round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4),
        'Avg-Prec':  round(average_precision_score(y_test, y_proba), 4),
    }


def plot_confusion_matrices(models_dict: dict, X_test, y_test, save_dir: str = '../outputs/'):
    """
    Plot confusion matrices for all models side by side.

    Args:
        models_dict: {'Model Name': model_object}
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Genuine', 'Fraud'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=12)

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved: {path}")


def plot_roc_curves(models_dict: dict, X_test, y_test, save_dir: str = '../outputs/'):
    """
    Plot ROC curves for all models on one chart.

    Args:
        models_dict: {'Model Name': model_object}
    """
    os.makedirs(save_dir, exist_ok=True)
    colors = ['steelblue', 'crimson', 'darkorange', 'green']

    plt.figure(figsize=(8, 6))
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=color, lw=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved: {path}")


def plot_precision_recall_curves(models_dict: dict, X_test, y_test, save_dir: str = '../outputs/'):
    """Plot Precision-Recall curves for all models."""
    os.makedirs(save_dir, exist_ok=True)
    colors = ['steelblue', 'crimson', 'darkorange', 'green']

    plt.figure(figsize=(8, 6))
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(rec, prec, label=f'{name} (AP={ap:.4f})', color=color, lw=2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'precision_recall_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved: {path}")


def plot_metrics_comparison(results: list, save_dir: str = '../outputs/'):
    """Bar chart comparing all model metrics."""
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results).set_index('Model')
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    ax = df[metrics].plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.set_xticklabels(df.index, rotation=0, fontsize=11)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

    plt.tight_layout()
    path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved: {path}")


if __name__ == '__main__':
    from preprocess import full_pipeline
    from model import train_logistic_regression, train_random_forest

    X_train, X_test, y_train, y_test = full_pipeline('../data/creditcard.csv')

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    models = {
        'Logistic Regression': lr,
        'Random Forest':       rf,
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_test, y_test))

    print("\n===== Final Summary =====")
    print(pd.DataFrame(results).to_string(index=False))

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_metrics_comparison(results)
