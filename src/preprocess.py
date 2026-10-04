"""Data cleaning and preprocessing design module for PHDA Diabetes dataset."""

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    load_raw_data,
    validate_schema,
)

# Columns where a value of 0 is clinically/medically invalid and represents missing data
ZERO_INVALID_COLUMNS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "diabetes_cleaned.csv"


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces medically invalid zero values with NaN for specified clinical features.

    Note: Pregnancies (can be 0) and Outcome (binary 0/1) are explicitly left intact.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with invalid zero entries replaced with np.nan.
    """
    validate_schema(df)
    df_cleaned = df.copy()

    for col in ZERO_INVALID_COLUMNS:
        df_cleaned[col] = df_cleaned[col].replace(0, np.nan)

    return df_cleaned


def check_duplicates(df: pd.DataFrame) -> int:
    """Checks for duplicate rows in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Number of duplicate rows found.
    """
    return int(df.duplicated().sum())


def get_missing_values_report(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Generates a missing values summary report for the cleaned dataset.

    Args:
        df_cleaned: DataFrame where invalid zeros have been replaced by NaN.

    Returns:
        DataFrame summarizing total missing count and missing percentage per feature.
    """
    total_rows = len(df_cleaned)
    missing_count = df_cleaned.isnull().sum()
    missing_pct = (missing_count / total_rows) * 100

    report = pd.DataFrame(
        {
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_pct.round(2),
        }
    ).sort_values(by="Missing_Count", ascending=False)

    return report


def build_preprocessor(
    feature_cols: List[str] = None,
    impute_strategy: str = "median",
    scale_features: bool = True,
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for feature imputation and scaling.

    Designed to be fit ONLY on training data during ML model development to prevent
    data leakage across train/validation/test splits.

    Args:
        feature_cols: List of feature column names (excluding target).
        impute_strategy: Imputation strategy ('median', 'mean', etc.).
        scale_features: Whether to include StandardScaler in pipeline.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    if feature_cols is None:
        feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]

    pipeline_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if scale_features:
        pipeline_steps.append(("scaler", StandardScaler()))

    feature_pipeline = Pipeline(pipeline_steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", feature_pipeline, feature_cols)],
        remainder="passthrough",
    )

    return preprocessor


def save_cleaned_data(
    df_cleaned: pd.DataFrame, destination_path: Path = DEFAULT_CLEANED_CSV_PATH
) -> Path:
    """Saves the cleaned DataFrame (with NaNs for invalid zeros) to processed data directory.

    Args:
        df_cleaned: Cleaned DataFrame.
        destination_path: Path to save the output CSV.

    Returns:
        Path where file was saved.
    """
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(destination_path, index=False)
    print(f"[INFO] Saved cleaned dataset to: {destination_path}")
    return destination_path


def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs data loading, zero-invalidation, duplicate check, and saves cleaned dataset.

    Returns:
        Tuple of (Cleaned DataFrame, Missing values report DataFrame).
    """
    df_raw = load_raw_data()
    num_dups = check_duplicates(df_raw)
    print(f"[INFO] Duplicate rows count: {num_dups}")

    df_cleaned = replace_invalid_zeros(df_raw)
    missing_report = get_missing_values_report(df_cleaned)

    save_cleaned_data(df_cleaned)

    return df_cleaned, missing_report


if __name__ == "__main__":
    print("=== PHDA Data Cleaning & Preprocessing Design ===")
    df_cleaned, missing_report = run_preprocessing_pipeline()

    print("\nMissing Values Report (after replacing medically invalid 0s with NaN):")
    print(missing_report)

    print("\nPre-processing ColumnTransformer initialized cleanly (Ready for ML pipeline).")
