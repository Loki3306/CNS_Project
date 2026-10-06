"""
Dataset Cleaning and Standardization Module.
Standardizes column identifiers, maps target labels, validates types, and saves cleaned data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    IDENTIFIER_COLUMNS,
    PROCESSED_CSV_PATH,
    PROCESSED_DATA_DIR,
    RAW_CSV_PATH,
    RAW_LABEL_MAPPING,
    RAW_TARGET_CANDIDATES,
    TARGET_COLUMN,
)


def clean_dataset(
    raw_path: Path = RAW_CSV_PATH,
    output_path: Path = PROCESSED_CSV_PATH,
    drop_identifiers: bool = True
) -> pd.DataFrame:
    """
    Clean, validate, and standardize the raw phishing dataset.
    
    1. Removes artificial index/identifier columns.
    2. Validates numerical datatypes.
    3. Normalizes target column name to 'Result'.
    4. Maps raw labels {-1, 1} to binary classification standards:
       - 0 = Legitimate (negative class)
       - 1 = Phishing (positive class)
    5. Asserts valid binary domain {0, 1}.
    6. Saves output to data/processed/cleaned_data.csv without mutating raw data.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # 1. Detect and standardize target column
    target_candidate = None
    for cand in RAW_TARGET_CANDIDATES:
        if cand in df.columns:
            target_candidate = cand
            break
    if target_candidate is None:
        raise ValueError(f"Could not identify target column among candidates: {RAW_TARGET_CANDIDATES}")

    # 2. Drop artificial identifiers
    removed_cols = []
    if drop_identifiers:
        for id_col in IDENTIFIER_COLUMNS:
            if id_col in df.columns:
                df = df.drop(columns=[id_col])
                removed_cols.append(id_col)

    # 3. Rename target column to standard TARGET_COLUMN
    if target_candidate != TARGET_COLUMN:
        df = df.rename(columns={target_candidate: TARGET_COLUMN})

    # 4. Map target values
    raw_unique_targets = set(df[TARGET_COLUMN].unique())
    expected_raw_targets = set(RAW_LABEL_MAPPING.keys())
    if not raw_unique_targets.issubset(expected_raw_targets):
        raise ValueError(
            f"Unexpected target values {raw_unique_targets}; expected subset of {expected_raw_targets}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(RAW_LABEL_MAPPING)

    # 5. Type and Domain Assertions
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}, (
        f"Target domain must be strictly {{0, 1}}, got {set(df[TARGET_COLUMN].unique())}"
    )

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Non-numeric column detected in cleaned dataset: {col}")

    # Ensure integer types for all discrete feature columns
    df = df.astype(int)

    # 6. Save cleaned dataset
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    from config import ensure_directories
    ensure_directories()
    cleaned = clean_dataset()
    print("Dataset cleaned successfully:")
    print("Shape:", cleaned.shape)
    print("Target distribution:\n", cleaned[TARGET_COLUMN].value_counts())
