"""
Central Configuration Module for Phishing URL Classification Research Pipeline.
Defines parameters, directory paths, and reproducibility settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data Files
RAW_CSV_PATH = RAW_DATA_DIR / "phishing.csv"
RAW_TXT_PATH = RAW_DATA_DIR / "phishing.txt"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"
SPLIT_METADATA_PATH = PROCESSED_DATA_DIR / "split_metadata.json"

# Results Directories
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
AUDITS_DIR = RESULTS_DIR / "audits"
TUNING_DIR = RESULTS_DIR / "tuning"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LEARNING_DYNAMICS_DIR = RESULTS_DIR / "learning_dynamics"

# Artifacts & Reports
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "gradient_boosting_final.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

# Core Experiment Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Target Settings
# The raw dataset uses 'class' with values {-1, 1}.
# The cleaned/processed dataset standardizes the column name to 'Result'.
# Label semantics in raw UCI/Kaggle dataset:
#  1 = legitimate
# -1 = phishing
# Internal mapping to standard binary classification:
#  0 = legitimate (negative class)
#  1 = phishing (positive class)
RAW_TARGET_CANDIDATES = ["class", "Result"]
TARGET_COLUMN = "Result"
RAW_LABEL_MAPPING = {
    1: 0,   # Legitimate -> 0
    -1: 1   # Phishing -> 1
}

# Identifier columns to drop if present (e.g., 'Index', 'Unnamed: 0')
IDENTIFIER_COLUMNS = ["Index", "Unnamed: 0", "id"]

# Kaggle Dataset Identifier
KAGGLE_DATASET_ID = "eswarchandt/phishing-website-detector"

# Hyperparameter Search Space for Gradient Boosting
PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 150, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "max_features": [None, "sqrt", "log2"]
}
RANDOM_SEARCH_N_ITER = 50

# Seed list for robustness analysis
ROBUSTNESS_SEEDS = [21, 42, 84, 123, 2026]

# Plot formatting
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)

def ensure_directories():
    """Ensure all required output directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        AUDITS_DIR,
        TUNING_DIR,
        PREDICTIONS_DIR,
        LEARNING_DYNAMICS_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
