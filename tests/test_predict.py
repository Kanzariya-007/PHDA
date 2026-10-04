"""Unit and integration tests for model inference and serialization."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.predict import (
    DEFAULT_MODEL_PKL_PATH,
    FEATURE_CONTRACT,
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


def test_load_model(ensure_model_pkl):
    """Test that load_model successfully loads sklearn Pipeline from .pkl artifact."""
    model = load_model(DEFAULT_MODEL_PKL_PATH, use_cache=False)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_format_and_validate_input_valid(sample_patient_dict):
    """Test format_and_validate_input with valid feature dictionary."""
    df_val = format_and_validate_input(sample_patient_dict)
    assert isinstance(df_val, pd.DataFrame)
    assert len(df_val) == 1
    assert list(df_val.columns) == FEATURE_CONTRACT


def test_format_and_validate_input_shuffled_keys(sample_patient_dict):
    """Test that input feature columns are strictly ordered according to FEATURE_CONTRACT."""
    # Reverse dictionary order
    shuffled_dict = {k: sample_patient_dict[k] for k in reversed(list(sample_patient_dict.keys()))}
    df_val = format_and_validate_input(shuffled_dict)
    assert list(df_val.columns) == FEATURE_CONTRACT


def test_format_and_validate_input_missing_feature(sample_patient_dict):
    """Test that missing required features raise ValueError."""
    incomplete_dict = sample_patient_dict.copy()
    del incomplete_dict["Glucose"]
    with pytest.raises(ValueError, match="Missing required feature"):
        format_and_validate_input(incomplete_dict)


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

    # Probabilities sum to 1.0
    total_prob = res["probability_non_diabetic"] + res["probability_diabetic"]
    assert pytest.approx(total_prob, abs=1e-3) == 1.0


def test_model_round_trip_serialization(sample_patient_dict, tmp_path):
    """Test saving model to a new .pkl path and making predictions from the loaded artifact."""
    model = load_model(DEFAULT_MODEL_PKL_PATH)

    tmp_pkl_path = tmp_path / "diabetes_model_test.pkl"
    joblib.dump(model, tmp_pkl_path)
    assert tmp_pkl_path.exists()

    # Make prediction using temporary artifact
    res_orig = make_prediction(sample_patient_dict, model=model)
    res_loaded = make_prediction(sample_patient_dict, model_path=tmp_pkl_path)

    assert res_orig["prediction"] == res_loaded["prediction"]
    assert res_orig["probability_diabetic"] == res_loaded["probability_diabetic"]
