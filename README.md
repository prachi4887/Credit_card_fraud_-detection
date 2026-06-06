# 💳 Credit Card Fraud Detection

> **Task 5 | Machine Learning Internship Project**  
> Detecting fraudulent credit card transactions using classification algorithms.

---

## 📋 Project Overview

Credit card fraud is a major problem in the financial industry. This project builds and evaluates machine learning models to automatically identify fraudulent transactions from a highly imbalanced real-world dataset.

### Objectives
- Preprocess and normalize transaction data
- Handle severe class imbalance using SMOTE
- Train Logistic Regression and Random Forest classifiers
- Evaluate models using Precision, Recall, F1-Score, and ROC-AUC

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/                        # Dataset folder (not in repo - see below)
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb    # Full analysis notebook (EDA + Modeling)
│
├── src/
│   ├── preprocess.py            # Data loading, scaling, splitting, SMOTE
│   ├── model.py                 # Model training and saving
│   └── evaluate.py              # Metrics, confusion matrix, ROC curves
│
├── outputs/                     # Generated plots and visualizations
│
├── models/                      # Saved trained models (.pkl)
│
├── main.py                      # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions
- **Features:** 30 (V1–V28 are PCA-transformed, plus Time and Amount)
- **Target:** `Class` (0 = Genuine, 1 = Fraud)
- **Imbalance:** Only ~0.17% of transactions are fraudulent

> ⚠️ The dataset is not included in this repository due to its size. Download it from Kaggle and place it in the `data/` folder.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in:
```
data/creditcard.csv
```

---

## 🚀 How to Run

### Option A: Run the full pipeline (recommended)
```bash
python main.py
```

With custom options:
```bash
python main.py --data data/creditcard.csv --method smote
# method options: smote | undersample | none
```

### Option B: Jupyter Notebook (step-by-step)
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier, fast and interpretable |
| **Random Forest** | Ensemble of decision trees, handles non-linearity well |

---

## 📈 Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | ~0.XX | ~0.XX | ~0.XX | ~0.XX |
| Random Forest | ~0.XX | ~0.XX | ~0.XX | ~0.XX |

> ℹ️ Run the pipeline to see your actual results — they will vary slightly based on random seed.

**Key Metric for Fraud Detection → Recall**  
We want to catch as many frauds as possible (minimize false negatives).

---

## 🔬 Techniques Used

- **StandardScaler** — Normalize `Amount` and `Time` features
- **SMOTE** — Synthetic Minority Oversampling to fix class imbalance
- **Train-Test Split** — 80% training / 20% testing (stratified)
- **Classification Report** — Precision, Recall, F1 per class
- **ROC-AUC Curve** — Model discrimination ability
- **Confusion Matrix** — Visual breakdown of predictions

---

## 📷 Output Visualizations

All plots are saved to `outputs/`:
- `class_distribution.png` — Fraud vs Genuine ratio
- `amount_distribution.png` — Transaction amount by class
- `correlation_heatmap.png` — Feature correlations
- `smote_comparison.png` — Before/after SMOTE
- `confusion_matrices.png` — True/False Positive breakdown
- `roc_curves.png` — ROC-AUC comparison
- `precision_recall_curves.png` — PR curves
- `feature_importance.png` — Top features (Random Forest)
- `metrics_comparison.png` — Side-by-side model comparison

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 👤 Author

**Your Name**  
Internship Project — Task 5  
[GitHub](https://github.com/psirohi4887) | [LinkedIn](https://linkedin.com/in/prachi-sirohi-24395335b)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
