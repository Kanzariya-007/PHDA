"""Unit tests for dataset loading, acquisition, and schema validation."""

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    fetch_raw_data,
    load_raw_data,
    validate_schema,
    get_dataset_summary,
    run_pipeline,
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture returning a valid mock DataFrame matching PIMA schema."""
    data = {
        "Pregnancies": [1, 2, 5],
        "Glucose": [85, 120, 140],
        "BloodPressure": [66, 70, 80],
        "SkinThickness": [23, 30, 0],
        "Insulin": [94, 0, 180],
        "BMI": [26.6, 32.0, 35.5],
        "DiabetesPedigreeFunction": [0.351, 0.5, 0.627],
        "Age": [31, 45, 50],
        "Outcome": [0, 1, 1],
    }
    return pd.DataFrame(data)


def test_validate_schema_success(sample_valid_df):
    """Test validate_schema passes on valid DataFrame."""
    assert validate_schema(sample_valid_df) is True


def test_validate_schema_empty_df():
    """Test validate_schema raises ValueError on empty DataFrame."""
    empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_schema(empty_df)


def test_validate_schema_missing_column(sample_valid_df):
    """Test validate_schema fails when a column is missing."""
    invalid_df = sample_valid_df.drop(columns=["Outcome"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(invalid_df)


def test_validate_schema_unexpected_column(sample_valid_df):
    """Test validate_schema fails when an unexpected column is added."""
    invalid_df = sample_valid_df.copy()
    invalid_df["ExtraCol"] = 1
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_schema(invalid_df)


def test_validate_schema_column_order_mismatch(sample_valid_df):
    """Test validate_schema fails if columns are reordered."""
    cols = list(sample_valid_df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    invalid_df = sample_valid_df[cols]
    with pytest.raises(ValueError, match="Column order mismatch"):
        validate_schema(invalid_df)


def test_validate_schema_non_numeric():
    """Test validate_schema fails if non-numeric column exists."""
    data = {col: [1.0, 2.0] for col in EXPECTED_COLUMNS}
    data["Glucose"] = ["high", "low"]
    invalid_df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Non-numeric columns found"):
        validate_schema(invalid_df)


def test_validate_schema_invalid_target_values(sample_valid_df):
    """Test validate_schema fails if Outcome target contains values other than 0 or 1."""
    invalid_df = sample_valid_df.copy()
    invalid_df.loc[0, "Outcome"] = 2
    with pytest.raises(ValueError, match="Target column 'Outcome' contains invalid values"):
        validate_schema(invalid_df)


def test_get_dataset_summary(sample_valid_df):
    """Test get_dataset_summary output metrics."""
    summary = get_dataset_summary(sample_valid_df)
    assert summary["num_rows"] == 3
    assert summary["num_columns"] == 9
    assert summary["columns"] == EXPECTED_COLUMNS
    assert summary["target_distribution"] == {0: 1, 1: 2}


def test_live_data_pipeline(tmp_path):
    """Integration test verifying full data acquisition, schema, and statistics on real dataset."""
    dest_path = tmp_path / "test_raw_diabetes.csv"
    
    # 1. Fetch raw data
    fetched_file = fetch_raw_data(destination_path=dest_path)
    assert fetched_file.exists()
    assert fetched_file.stat().st_size > 0

    # 2. Load data
    df = load_raw_data(dest_path)
    assert isinstance(df, pd.DataFrame)

    # 3. Validate schema
    assert validate_schema(df) is True
    assert df.shape == (768, 9)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}

    # 4. Profile summary
    summary = get_dataset_summary(df)
    assert summary["num_rows"] == 768
    assert summary["num_columns"] == 9
    assert summary["target_distribution"][0] == 500
    assert summary["target_distribution"][1] == 268
