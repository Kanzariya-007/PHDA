"""Data acquisition, loading, and schema validation module for PHDA."""

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

# Default dataset metadata & configuration
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/npradaschnor/"
    "Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
)

EXPECTED_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

TARGET_COLUMN: str = "Outcome"

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RAW_CSV_PATH = RAW_DATA_DIR / "diabetes.csv"


def fetch_raw_data(
    url: str = DEFAULT_DATA_URL,
    destination_path: Path = DEFAULT_RAW_CSV_PATH,
    force_download: bool = False,
) -> Path:
    """Acquires raw dataset from a remote URL and saves to local destination.

    Args:
        url: Remote CSV dataset URL.
        destination_path: Path where raw dataset should be stored.
        force_download: If True, re-downloads even if file already exists.

    Returns:
        Path to local raw dataset file.
    """
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() and not force_download:
        print(f"[INFO] Raw dataset already present at: {destination_path}")
        return destination_path

    print(f"[INFO] Downloading dataset from: {url}")
    try:
        urllib.request.urlretrieve(url, destination_path)
        print(f"[INFO] Dataset successfully saved to: {destination_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset from {url}: {exc}") from exc

    return destination_path


def load_raw_data(file_path: Path = DEFAULT_RAW_CSV_PATH) -> pd.DataFrame:
    """Loads raw dataset CSV into a pandas DataFrame.

    Args:
        file_path: Path to dataset CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        fetch_raw_data(destination_path=file_path)

    df = pd.read_csv(file_path)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validates schema requirements for the PIMA Diabetes dataset.

    Checks:
        1. DataFrame is non-empty.
        2. Expected column names match exact required schema.
        3. All features and target are numeric.
        4. Target column 'Outcome' contains binary values (0/1).

    Args:
        df: Pandas DataFrame to validate.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if df.empty:
        raise ValueError("Schema Validation Failed: DataFrame is empty.")

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema Validation Failed: Missing columns {missing_cols}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        raise ValueError(f"Schema Validation Failed: Unexpected columns {extra_cols}")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Schema Validation Failed: Column order mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\nGot: {list(df.columns)}"
        )

    # Validate feature data types (must be numeric)
    non_numeric_cols = [
        col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"Schema Validation Failed: Non-numeric columns found: {non_numeric_cols}"
        )

    # Validate target column values
    unique_target_values = set(df[TARGET_COLUMN].unique())
    valid_target_values = {0, 1}
    if not unique_target_values.issubset(valid_target_values):
        raise ValueError(
            f"Schema Validation Failed: Target column '{TARGET_COLUMN}' contains invalid values "
            f"{unique_target_values - valid_target_values}. Expected subset of {valid_target_values}."
        )

    return True


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates summary statistics and data profile for dataset.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary containing dataset summary information.
    """
    validate_schema(df)

    rows, cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    
    # In PIMA dataset, zeros in medical metrics (Glucose, BP, SkinThickness, Insulin, BMI) represent missing values
    zeros_counts = {
        col: int((df[col] == 0).sum())
        for col in df.columns
        if col not in ["Pregnancies", "Outcome"]
    }

    summary = {
        "num_rows": rows,
        "num_columns": cols,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": null_counts,
        "zeros_counts": zeros_counts,
        "target_distribution": target_counts,
    }

    return summary


def run_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs data acquisition, loading, schema validation, and profiling pipeline.

    Returns:
        Tuple of (DataFrame, summary dictionary).
    """
    raw_path = fetch_raw_data()
    df = load_raw_data(raw_path)
    validate_schema(df)
    summary = get_dataset_summary(df)
    return df, summary


if __name__ == "__main__":
    print("=== PHDA Data Ingestion & Schema Validation ===")
    df, summary = run_pipeline()
    print(f"\nDataset successfully loaded and validated.")
    print(f"Shape: {summary['num_rows']} rows x {summary['num_columns']} columns")
    print("\nColumns and Data Types:")
    for col, dtype in summary["dtypes"].items():
        print(f"  - {col}: {dtype}")
    print("\nTarget ('Outcome') Distribution:")
    for val, count in summary["target_distribution"].items():
        label = "Non-Diabetic (0)" if val == 0 else "Diabetic (1)"
        pct = (count / summary['num_rows']) * 100
        print(f"  - {label}: {count} ({pct:.2f}%)")
    print("\nClinically Invalid Zero Values (Potential Missing Data):")
    for col, z_count in summary["zeros_counts"].items():
        pct = (z_count / summary['num_rows']) * 100
        print(f"  - {col}: {z_count} zero entries ({pct:.2f}%)")
