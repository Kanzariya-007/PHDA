"""Unit and integration tests for model inference, validation, and parameter indicators."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.predict import (
    DEFAULT_MODEL_PKL_PATH,
    FEATURE_CONTRACT,
    classify_parameter_indicators,
    format_and_validate_input,
    load_model,
    make_prediction,
)
from src.train import train_and_evaluate_models


@pytest.fixture(scope="module")
def ensure_model_pkl():
    """Fixture ensuring models/diabetes_model.pkl exists before running inference tests."""
    if not DEFAULT_MODEL_PKL_PATH.exists():
        train_and_evaluate_models()
    return DEFAULT_MODEL_PKL_PATH


@pytest.fixture
def sample_patient_dict() -> dict:
    """Fixture returning sample patient feature dictionary."""
    return {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 90,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }


def test_load_model_success(ensure_model_pkl):
    """Test that load_model successfully loads sklearn Pipeline from .pkl artifact."""
    model = load_model(DEFAULT_MODEL_PKL_PATH, use_cache=False)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_load_model_missing_file_raises_error(tmp_path):
    """Test load_model with auto_train_if_missing=False raises FileNotFoundError when file is missing."""
    missing_path = tmp_path / "non_existent_model.pkl"
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(missing_path, use_cache=False, auto_train_if_missing=False)


def test_load_model_corrupt_file_raises_error(tmp_path):
    """Test load_model raises RuntimeError when loading a corrupted file."""
    corrupt_path = tmp_path / "corrupt_model.pkl"
    corrupt_path.write_text("invalid binary garbage content")
    with pytest.raises(RuntimeError, match="Failed to load model artifact"):
        load_model(corrupt_path, use_cache=False, auto_train_if_missing=False)


def test_format_and_validate_input_valid(sample_patient_dict):
    """Test format_and_validate_input with valid feature dictionary."""
    df_val = format_and_validate_input(sample_patient_dict)
    assert isinstance(df_val, pd.DataFrame)
    assert len(df_val) == 1
    assert list(df_val.columns) == FEATURE_CONTRACT


def test_format_and_validate_input_shuffled_keys(sample_patient_dict):
    """Test that input feature columns are strictly ordered according to FEATURE_CONTRACT."""
    shuffled_dict = {k: sample_patient_dict[k] for k in reversed(list(sample_patient_dict.keys()))}
    df_val = format_and_validate_input(shuffled_dict)
    assert list(df_val.columns) == FEATURE_CONTRACT


def test_format_and_validate_input_negative_values(sample_patient_dict):
    """Test that negative input values raise ValueError."""
    # Negative BMI
    invalid_bmi = sample_patient_dict.copy()
    invalid_bmi["BMI"] = -5.0
    with pytest.raises(ValueError, match="cannot be negative"):
        format_and_validate_input(invalid_bmi)

    # Negative Age
    invalid_age = sample_patient_dict.copy()
    invalid_age["Age"] = -10
    with pytest.raises(ValueError, match="cannot be negative"):
        format_and_validate_input(invalid_age)

    # Negative Pregnancies
    invalid_preg = sample_patient_dict.copy()
    invalid_preg["Pregnancies"] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        format_and_validate_input(invalid_preg)


def test_format_and_validate_input_out_of_bounds(sample_patient_dict):
    """Test that out-of-bounds input values raise ValueError."""
    # Glucose above validation limit
    invalid_glucose = sample_patient_dict.copy()
    invalid_glucose["Glucose"] = 500.0
    with pytest.raises(ValueError, match="outside the application validation range"):
        format_and_validate_input(invalid_glucose)

    # Age above validation limit
    invalid_age = sample_patient_dict.copy()
    invalid_age["Age"] = 150
    with pytest.raises(ValueError, match="outside the application validation range"):
        format_and_validate_input(invalid_age)


def test_format_and_validate_input_invalid_zeros(sample_patient_dict):
    """Test that 0 in clinical metrics becomes NaN, while Pregnancies=0 remains 0."""
    zero_dict = sample_patient_dict.copy()
    zero_dict["Pregnancies"] = 0
    zero_dict["Glucose"] = 0
    zero_dict["Insulin"] = 0

    df_val = format_and_validate_input(zero_dict)

    # Pregnancies zero remains 0
    assert df_val.loc[0, "Pregnancies"] == 0

    # Glucose and Insulin zeros become NaN for median imputer
    assert pd.isna(df_val.loc[0, "Glucose"])
    assert pd.isna(df_val.loc[0, "Insulin"])


def test_classify_parameter_indicators(sample_patient_dict):
    """Test parameter reference band indicator engine."""
    indicators = classify_parameter_indicators(sample_patient_dict)
    assert len(indicators) == 8

    # Verify keys present in indicator dictionaries
    for ind in indicators:
        assert "feature" in ind
        assert "severity" in ind
        assert ind["severity"] in ["Very High", "High", "Normal", "Low", "Very Low"]
        assert "tag" in ind
        assert ind["tag"] in ["normal", "warning", "danger"]
        assert "badge_color" in ind
        assert "description" in ind

    # High Glucose case (Glucose = 150 -> Very High)
    high_patient = sample_patient_dict.copy()
    high_patient["Glucose"] = 150.0
    high_inds = classify_parameter_indicators(high_patient)
    glucose_ind = next(i for i in high_inds if i["feature"] == "Glucose")
    assert glucose_ind["severity"] == "Very High"
    assert glucose_ind["tag"] == "danger"


def test_make_prediction_structure(ensure_model_pkl, sample_patient_dict):
    """Test prediction execution and output schema."""
    res = make_prediction(sample_patient_dict, model_path=DEFAULT_MODEL_PKL_PATH)

    assert "prediction" in res
    assert res["prediction"] in [0, 1]
    assert "label" in res
    assert res["label"] in ["Non-Diabetic", "Diabetic"]
    assert "probability_non_diabetic" in res
    assert "probability_diabetic" in res
    assert "confidence" in res

    total_prob = res["probability_non_diabetic"] + res["probability_diabetic"]
    assert pytest.approx(total_prob, abs=1e-3) == 1.0


def test_model_round_trip_serialization(sample_patient_dict, tmp_path):
    """Test saving model to a new .pkl path and making predictions from the loaded artifact."""
    model = load_model(DEFAULT_MODEL_PKL_PATH)

    tmp_pkl_path = tmp_path / "diabetes_model_test.pkl"
    joblib.dump(model, tmp_pkl_path)
    assert tmp_pkl_path.exists()

    res_orig = make_prediction(sample_patient_dict, model=model)
    res_loaded = make_prediction(sample_patient_dict, model_path=tmp_pkl_path)

    assert res_orig["prediction"] == res_loaded["prediction"]
    assert res_orig["probability_diabetic"] == res_loaded["probability_diabetic"]
