# A Gradient Boosting Tree Framework for Intelligent Phishing URL Classification

## 1. Project Overview

This repository contains a complete, scientifically rigorous, end-to-end machine learning research pipeline for phishing website detection. The study investigates the predictive capacity and learning dynamics of **Gradient Tree Boosting** (`sklearn.ensemble.GradientBoostingClassifier`), comparing its sequential pseudo-residual fitting mechanism against parallel bagging ensembles (**Random Forest**), single-tree baselines (**Decision Tree**), and linear baselines (**Logistic Regression**).

---

## 2. Critical Academic Scope Statement

> **IMPORTANT SCOPE DEFINITION**
> 
> This project operates on **engineered phishing-related website, URL, and host characteristics**, **NOT** raw unparsed URL text strings.
> 
> The input is a 30-dimensional discrete vector containing lexical, domain-level, and HTML/JavaScript properties (e.g., SSL certificate validity, anchor URL domain consistency, prefix/suffix hyphens, traffic rank, popup scripts). 
> 
> The resulting model does **not** accept raw text strings like `https://example.com/login` directly unless coupled with an upstream feature extraction parser that computes these 30 attributes.

---

## 3. Repository Architecture

```text
CNS_Project/
│
├── README.md                           # Comprehensive documentation and instructions
├── requirements.txt                    # Exact reproducible dependency pins
├── .gitignore                          # Standard git exclusion rules
├── pytest.ini                          # Pytest configuration
├── config.py                           # Central experiment configuration & parameters
├── run_pipeline.py                     # Master execution pipeline
│
├── data/
│   ├── raw/                            # Original authentic Kaggle dataset files
│   └── processed/                      # Cleaned dataset & split metadata
│
├── preprocessing/
│   ├── __init__.py
│   ├── audit.py                        # Dataset structural, missing, and duplicate audit
│   ├── clean.py                        # Identifier excision & binary target normalization
│   ├── leakage.py                      # Univariate predictor, identifier, and split contamination audits
│   └── split.py                        # Stratified 80/20 train/test splitting
│
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py          # StandardScaler + LogisticRegression Pipeline
│   ├── decision_tree.py                # Single CART baseline
│   ├── gradient_boosting.py            # Gradient Boosting classifier builder
│   └── random_forest.py                # Random Forest comparison builder
│
├── experiments/
│   ├── __init__.py
│   ├── baseline_experiment.py          # 5-Fold CV comparison on development set
│   ├── hyperparameter_search.py        # RandomizedSearchCV on development set
│   ├── staged_analysis.py              # Staged loss, error, F1, and pseudo-residual tracking
│   └── comparison_experiment.py        # Final holdout test evaluation and model comparison
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                      # Multi-metric classification evaluation engine
│   ├── curves.py                       # Overlaid ROC and PR curve visualization
│   ├── confusion.py                    # Annotated confusion matrix generator
│   └── error_analysis.py               # False negative, false positive, and borderline auditor
│
├── interpretability/
│   ├── __init__.py
│   ├── feature_importance.py           # Gini impurity (MDI) feature ranking
│   └── permutation_importance.py       # Test set permutation importance (20 repeats)
│
├── plots/                              # Publication-grade figures (300 DPI)
├── results/
│   ├── metrics/                        # Quantitative CSV/JSON performance tables
│   ├── audits/                         # Missing value, duplicate, and leakage audits
│   ├── tuning/                         # Hyperparameter search results and best parameters
│   ├── predictions/                    # Sample-level predictions and error files
│   └── learning_dynamics/              # Staged iteration metrics table
│
├── artifacts/
│   └── models/                         # Serialized joblib model & environment metadata
│
├── reports/
│   ├── dataset_audit.md                # Observed structural data audit report
│   ├── leakage_analysis.md             # Split contamination & leakage investigation
│   ├── methodology.md                  # Comprehensive mathematical methodology
│   ├── experimental_results.md         # Empirical findings compiled from execution
│   └── final_summary.md                # Final research synthesis and conclusions
│
└── tests/
    ├── test_data.py                    # Data integrity and splitting unit tests
    ├── test_metrics.py                 # Handcrafted metric array validation tests
    └── test_leakage.py                 # Index disjointness and feature isolation tests
```

---

## 4. Environment Setup

### 4.1 Clone Repository & Create Virtual Environment
```bash
# Navigate to project directory
cd CNS_Project

# Create clean virtual environment
python -m venv .venv
```

### 4.2 Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Dataset Acquisition

The project utilizes the authentic **Kaggle: Phishing Website Detector** benchmark:
- Identifier: `eswarchandt/phishing-website-detector`
- URL: [https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector)

### Automated Acquisition
The pipeline automatically attempts to acquire the dataset via `kagglehub` upon first execution.

### Manual Acquisition (Fallback)
If internet connectivity or authentication is unavailable:
1. Download the archive from the Kaggle dataset page.
2. Unzip and place `phishing.csv` into `data/raw/phishing.csv`.
3. The pipeline will automatically detect the local file.

---

## 6. Running the Complete Research Pipeline

To execute all 10 phases end-to-end, run:
```bash
python run_pipeline.py
```

The pipeline executes sequentially:
1. Dataset acquisition & schema verification.
2. Structural audit (missing values, class balance, duplicates).
3. Data cleaning (dropping artificial `Index`, mapping target: $0 = \text{Legitimate}$, $1 = \text{Phishing}$).
4. Stratified train/test partitioning (80% development, 20% untouched holdout test).
5. Leakage & cross-split contamination audit.
6. 5-Fold cross-validation across all baseline classifiers.
7. 50-iteration randomized hyperparameter optimization for Gradient Boosting.
8. Staged learning dynamics analysis (training deviance, validation log loss, pseudo-residuals).
9. Final model retraining on full development set & evaluation on the untouched test partition.
10. Feature importance (MDI vs Permutation), sample-level error analysis, report generation, and programmatic artifact verification.

---

## 7. Running Tests

To run the automated test suite:
```bash
pytest
```
Expected output: 8 unit tests passed across data integrity, evaluation metrics, and leakage checks.

---

## 8. Reproducibility & Research Discipline

- **Deterministic Random Seed**: `RANDOM_STATE = 42` is strictly maintained across all splits, cross-validation folds, classifiers, randomized searches, and permutation shuffles.
- **Untouched Test Partition**: The 20% holdout test partition ($n = 2,211$) is strictly isolated. It never participates in preprocessing fitting, cross-validation, hyperparameter tuning, or model selection.
- **Runtime Environment Tracking**: Runtime system details (Python version, OS platform, package versions) are recorded in `results/environment.json` and `artifacts/models/model_metadata.json`.

---

## 9. Limitations & Cybersecurity Context

1. **Engineered Feature Representation**: The model requires 30 pre-extracted behavioral and structural attributes. It does not perform NLP-based tokenization on raw URL strings.
2. **Historical Concept Drift**: Real-world phishing strategies evolve dynamically. Attackers increasingly leverage valid SSL certificates (e.g., Let's Encrypt), reducing the predictive dominance of features like `HTTPS` over time.
3. **Discrete Feature Cardinality**: In a coarse 30-feature discrete space ($\{-1, 0, 1\}$), distinct URLs can map to identical feature vectors. The 64 observed conflicting feature vectors define the theoretical Bayes error floor for this representation.
4. **Asymmetric Operational Costs**: False Negatives (admitting a phishing site) represent high-severity security breaches, whereas False Positives (blocking a legitimate site) cause operational friction. Operational deployments require calibrating decision thresholds according to institutional risk tolerance.
