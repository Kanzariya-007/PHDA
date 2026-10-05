"""Prediction and inference module for PHDA Diabetes Prediction model.

Provides a clean, reusable inference interface for Streamlit UI or API integration.
"""

from pathlib import Path
from typing import Dict, Any, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocess import ZERO_INVALID_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PKL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"

# Exact Feature Contract for Inference
FEATURE_CONTRACT: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Sensible ranges, defaults, units, and help texts for UI controls based on PIMA dataset & clinical defaults
FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Pregnancies": {
        "min": 0,
        "max": 17,
        "default": 1,
        "step": 1,
        "unit": "count",
        "label": "Pregnancies",
        "help": "Number of times pregnant.",
    },
    "Glucose": {
        "min": 0.0,
        "max": 200.0,
        "default": 120.0,
        "step": 1.0,
        "unit": "mg/dL",
        "label": "Glucose Level",
        "help": "Plasma glucose concentration at 2 hours in an oral glucose tolerance test. (0 represents unknown/unrecorded).",
    },
    "BloodPressure": {
        "min": 0.0,
        "max": 122.0,
        "default": 70.0,
        "step": 1.0,
        "unit": "mm Hg",
        "label": "Diastolic Blood Pressure",
        "help": "Diastolic blood pressure in mm Hg. (0 represents unknown/unrecorded).",
    },
    "SkinThickness": {
        "min": 0.0,
        "max": 99.0,
        "default": 20.0,
        "step": 1.0,
        "unit": "mm",
        "label": "Triceps Skinfold Thickness",
        "help": "Triceps skin fold thickness in mm. (0 represents unknown/unrecorded).",
    },
    "Insulin": {
        "min": 0.0,
        "max": 846.0,
        "default": 80.0,
        "step": 1.0,
        "unit": "μU/ml",
        "label": "2-Hour Serum Insulin",
        "help": "2-Hour serum insulin in μU/ml. (0 represents unknown/unrecorded).",
    },
    "BMI": {
        "min": 0.0,
        "max": 67.1,
        "default": 28.5,
        "step": 0.1,
        "unit": "kg/m²",
        "label": "Body Mass Index (BMI)",
        "help": "Body Mass Index (weight in kg / (height in m)²). (0 represents unknown/unrecorded).",
    },
    "DiabetesPedigreeFunction": {
        "min": 0.078,
        "max": 2.420,
        "default": 0.470,
        "step": 0.001,
        "unit": "score",
        "label": "Diabetes Pedigree Function",
        "help": "Genetic risk score calculated from family history of diabetes.",
    },
    "Age": {
        "min": 21,
        "max": 81,
        "default": 33,
        "step": 1,
        "unit": "years",
        "label": "Age",
        "help": "Age of the patient in years.",
    },
}

_CACHED_MODEL: Union[Pipeline, None] = None


def load_model(model_path: Path = DEFAULT_MODEL_PKL_PATH, use_cache: bool = True) -> Pipeline:
    """Loads trained sklearn pipeline model artifact from disk.

    Args:
        model_path: Path to serialized model artifact (.pkl / .joblib).
        use_cache: If True, returns cached model in memory if already loaded.

    Returns:
        Loaded sklearn Pipeline model object.
    """
    global _CACHED_MODEL
    model_path = Path(model_path)

    if use_cache and _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not model_path.exists():
        print(f"[INFO] Model artifact not found at {model_path}. Training model...")
        from src.train import train_and_evaluate_models
        _, _, model = train_and_evaluate_models()
        _CACHED_MODEL = model
        return model

    try:
        model = joblib.load(model_path)
        _CACHED_MODEL = model
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact from {model_path}: {exc}") from exc


def format_and_validate_input(input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Validates feature names, types, ordering, and replaces invalid zeros with NaN.

    Args:
        input_data: Single prediction dictionary or pandas DataFrame.

    Returns:
        Validated pandas DataFrame formatted to exact feature contract.

    Raises:
        ValueError: If required features are missing or invalid.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input data must be a dictionary or a pandas DataFrame.")

    # 1. Validate missing features
    missing_features = set(FEATURE_CONTRACT) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Input validation error: Missing required feature(s): {sorted(list(missing_features))}"
        )

    # 2. Select and order columns according to exact FEATURE_CONTRACT
    df_ordered = df[FEATURE_CONTRACT].copy()

    # 3. Validate numeric data types
    non_numeric_cols = [
        col for col in FEATURE_CONTRACT if not pd.api.types.is_numeric_dtype(df_ordered[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Input validation error: Features must be numeric. Invalid: {non_numeric_cols}")

    # 4. Convert medically invalid zeros to NaN for clinical metric features
    for col in ZERO_INVALID_COLUMNS:
        if col in df_ordered.columns:
            df_ordered[col] = df_ordered[col].replace(0, np.nan)

    return df_ordered


def make_prediction(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model: Union[Pipeline, None] = None,
    model_path: Path = DEFAULT_MODEL_PKL_PATH,
) -> Dict[str, Any]:
    """Executes prediction on input features and returns classification results and probabilities.

    Args:
        input_data: Feature dictionary or DataFrame matching FEATURE_CONTRACT.
        model: Optional pre-loaded Pipeline instance. If None, loads from model_path.
        model_path: Path to serialized model artifact (.pkl).

    Returns:
        Dictionary containing prediction integer, human-readable label, and probability metrics.
    """
    if model is None:
        model = load_model(model_path)

    df_validated = format_and_validate_input(input_data)

    prediction = int(model.predict(df_validated)[0])
    label = "Diabetic" if prediction == 1 else "Non-Diabetic"

    result = {
        "prediction": prediction,
        "label": label,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_validated)[0]
        result["probability_non_diabetic"] = float(round(probabilities[0], 4))
        result["probability_diabetic"] = float(round(probabilities[1], 4))
        result["confidence"] = float(round(max(probabilities), 4))

    return result


if __name__ == "__main__":
    print("=== PHDA Inference Module Demo ===")
    sample_patient = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,  # Invalid zero (will be handled as NaN by imputer)
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    print(f"\nInput Patient Profile:\n{sample_patient}")
    prediction_result = make_prediction(sample_patient)
    print(f"\nPrediction Result:\n{prediction_result}")
