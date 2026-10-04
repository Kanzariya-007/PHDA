"""Unit tests for data cleaning and preprocessing design module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data_loader import EXPECTED_COLUMNS
from src.preprocess import (
    ZERO_INVALID_COLUMNS,
    build_preprocessor,
    check_duplicates,
    get_missing_values_report,
    replace_invalid_zeros,
    save_cleaned_data,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture returning sample raw DataFrame with valid and invalid zeros."""
    data = {
        "Pregnancies": [0, 2, 5],
        "Glucose": [0, 120, 140],
        "BloodPressure": [66, 0, 80],
        "SkinThickness": [23, 30, 0],
        "Insulin": [0, 0, 180],
        "BMI": [26.6, 0.0, 35.5],
        "DiabetesPedigreeFunction": [0.351, 0.5, 0.627],
        "Age": [31, 45, 50],
        "Outcome": [0, 1, 0],
    }
    return pd.DataFrame(data)


def test_replace_invalid_zeros(sample_raw_df):
    """Test that invalid zeros are replaced by NaN only in specified clinical columns."""
    df_cleaned = replace_invalid_zeros(sample_raw_df)

    # Pregnancies zero should NOT be converted to NaN
    assert df_cleaned.loc[0, "Pregnancies"] == 0
    assert not pd.isna(df_cleaned.loc[0, "Pregnancies"])

    # Outcome zero should NOT be converted to NaN
    assert df_cleaned.loc[0, "Outcome"] == 0
    assert not pd.isna(df_cleaned.loc[0, "Outcome"])

    # Glucose zero SHOULD be converted to NaN
    assert pd.isna(df_cleaned.loc[0, "Glucose"])

    # BloodPressure zero SHOULD be converted to NaN
    assert pd.isna(df_cleaned.loc[1, "BloodPressure"])

    # SkinThickness zero SHOULD be converted to NaN
    assert pd.isna(df_cleaned.loc[2, "SkinThickness"])

    # Insulin zero SHOULD be converted to NaN
    assert pd.isna(df_cleaned.loc[0, "Insulin"])
    assert pd.isna(df_cleaned.loc[1, "Insulin"])
    assert df_cleaned.loc[2, "Insulin"] == 180

    # BMI zero SHOULD be converted to NaN
    assert pd.isna(df_cleaned.loc[1, "BMI"])


def test_check_duplicates(sample_raw_df):
    """Test duplicate row detection."""
    assert check_duplicates(sample_raw_df) == 0

    # Add duplicate row
    duplicated_df = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]], ignore_index=True)
    assert check_duplicates(duplicated_df) == 1


def test_get_missing_values_report(sample_raw_df):
    """Test missing values profiling report."""
    df_cleaned = replace_invalid_zeros(sample_raw_df)
    report = get_missing_values_report(df_cleaned)

    assert "Missing_Count" in report.columns
    assert "Missing_Percentage" in report.columns
    assert report.loc["Insulin", "Missing_Count"] == 2
    assert report.loc["Pregnancies", "Missing_Count"] == 0


def test_build_preprocessor_no_leakage(sample_raw_df):
    """Test preprocessor construction and train/test separation (no leakage)."""
    df_cleaned = replace_invalid_zeros(sample_raw_df)
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    # Split into mock train and test sets
    train_df = df_cleaned.iloc[:2]
    test_df = df_cleaned.iloc[2:]

    feature_cols = [c for c in EXPECTED_COLUMNS if c != "Outcome"]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Fit preprocessor strictly on train data
    X_train_trans = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_train_trans).any()

    # Transform test data using fitted train parameters
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_test_trans).any()


def test_save_cleaned_data(sample_raw_df, tmp_path):
    """Test saving cleaned CSV file."""
    df_cleaned = replace_invalid_zeros(sample_raw_df)
    out_file = tmp_path / "cleaned.csv"
    saved_path = save_cleaned_data(df_cleaned, destination_path=out_file)
    assert saved_path.exists()

    loaded = pd.read_csv(saved_path)
    assert len(loaded) == 3
    assert pd.isna(loaded.loc[0, "Glucose"])
