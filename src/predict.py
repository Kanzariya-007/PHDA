"""Prediction, input validation, and parameter indicator classification module for PHDA."""

from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

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

# Sensible ranges, dataset bounds, application validation limits, units, and descriptions
FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Pregnancies": {
        "dataset_min": 0,
        "dataset_max": 17,
        "val_min": 0,
        "val_max": 20,
        "default": 1,
        "step": 1,
        "unit": "count",
        "label": "Pregnancies",
        "help": "Number of times pregnant (PIMA Dataset range: 0–17).",
    },
    "Glucose": {
        "dataset_min": 0.0,
        "dataset_max": 199.0,
        "val_min": 40.0,
        "val_max": 300.0,
        "default": 120.0,
        "step": 1.0,
        "unit": "mg/dL",
        "label": "Glucose Level",
        "help": "2-Hour oral glucose tolerance test concentration (PIMA Dataset range: 0–199 mg/dL; 0 = unrecorded).",
    },
    "BloodPressure": {
        "dataset_min": 0.0,
        "dataset_max": 122.0,
        "val_min": 30.0,
        "val_max": 180.0,
        "default": 70.0,
        "step": 1.0,
        "unit": "mm Hg",
        "label": "Diastolic Blood Pressure",
        "help": "Diastolic blood pressure (PIMA Dataset range: 0–122 mm Hg; 0 = unrecorded).",
    },
    "SkinThickness": {
        "dataset_min": 0.0,
        "dataset_max": 99.0,
        "val_min": 5.0,
        "val_max": 100.0,
        "default": 20.0,
        "step": 1.0,
        "unit": "mm",
        "label": "Triceps Skinfold Thickness",
        "help": "Triceps skin fold thickness (PIMA Dataset range: 0–99 mm; 0 = unrecorded).",
    },
    "Insulin": {
        "dataset_min": 0.0,
        "dataset_max": 846.0,
        "val_min": 14.0,
        "val_max": 900.0,
        "default": 80.0,
        "step": 1.0,
        "unit": "μU/ml",
        "label": "2-Hour Serum Insulin",
        "help": "2-Hour serum insulin (PIMA Dataset range: 0–846 μU/ml; 0 = unrecorded).",
    },
    "BMI": {
        "dataset_min": 0.0,
        "dataset_max": 67.1,
        "val_min": 10.0,
        "val_max": 75.0,
        "default": 28.5,
        "step": 0.1,
        "unit": "kg/m²",
        "label": "Body Mass Index (BMI)",
        "help": "Body Mass Index (weight in kg / height in m²) (PIMA Dataset range: 0.0–67.1 kg/m²; 0 = unrecorded).",
    },
    "DiabetesPedigreeFunction": {
        "dataset_min": 0.078,
        "dataset_max": 2.420,
        "val_min": 0.050,
        "val_max": 3.000,
        "default": 0.470,
        "step": 0.001,
        "unit": "score",
        "label": "Diabetes Pedigree Function",
        "help": "Genetic risk score based on family diabetes history (PIMA Dataset range: 0.078–2.420).",
    },
    "Age": {
        "dataset_min": 21,
        "dataset_max": 81,
        "val_min": 18,
        "val_max": 120,
        "default": 33,
        "step": 1,
        "unit": "years",
        "label": "Age",
        "help": "Age of the patient in years (PIMA Dataset range: 21–81 years).",
    },
}

_CACHED_MODEL: Union[Pipeline, None] = None


def load_model(
    model_path: Path = DEFAULT_MODEL_PKL_PATH,
    use_cache: bool = True,
    auto_train_if_missing: bool = True,
) -> Pipeline:
    """Loads trained sklearn pipeline model artifact from disk.

    Args:
        model_path: Path to serialized model artifact (.pkl / .joblib).
        use_cache: If True, returns cached model in memory if already loaded.
        auto_train_if_missing: If True, trains model if missing. Otherwise raises FileNotFoundError.

    Returns:
        Loaded sklearn Pipeline model object.

    Raises:
        FileNotFoundError: If model file is missing and auto_train_if_missing is False.
        RuntimeError: If model file is corrupted or fails to load.
    """
    global _CACHED_MODEL
    model_path = Path(model_path)

    if use_cache and _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not model_path.exists():
        if auto_train_if_missing:
            print(f"[INFO] Model artifact not found at {model_path}. Training model...")
            from src.train import train_and_evaluate_models

            _, _, model = train_and_evaluate_models()
            _CACHED_MODEL = model
            return model
        else:
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Please run 'python -m src.train' to generate the model artifact."
            )

    try:
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid scikit-learn model/pipeline.")
        _CACHED_MODEL = model
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model artifact from {model_path}: {exc}. "
            f"The model file may be corrupt. Re-generate by running 'python -m src.train'."
        ) from exc


def format_and_validate_input(input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Validates feature names, types, ordering, range bounds, and replaces invalid zeros with NaN.

    Args:
        input_data: Single prediction dictionary or pandas DataFrame.

    Returns:
        Validated pandas DataFrame formatted to exact feature contract.

    Raises:
        ValueError: If required features are missing, non-numeric, negative, or out of bounds.
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

    # 4. Strict validation of impossible negative & out-of-bounds values
    for col in FEATURE_CONTRACT:
        val = df_ordered[col].iloc[0]
        cfg = FEATURE_CONFIG[col]

        # Check negative values
        if val < 0:
            raise ValueError(
                f"Input validation error: '{col}' cannot be negative ({val} {cfg['unit']})."
            )

        # Allow 0 for ZERO_INVALID_COLUMNS (as unrecorded/missing data), but validate non-zero bounds
        if col in ZERO_INVALID_COLUMNS and val == 0:
            continue

        if val < cfg["val_min"] or val > cfg["val_max"]:
            raise ValueError(
                f"Input validation error: '{col}' value {val} {cfg['unit']} is outside "
                f"the application validation range [{cfg['val_min']}, {cfg['val_max']}]."
            )

    # 5. Convert medically invalid zeros to NaN for clinical metric features
    for col in ZERO_INVALID_COLUMNS:
        if col in df_ordered.columns:
            df_ordered[col] = df_ordered[col].replace(0, np.nan)

    return df_ordered


def classify_parameter_indicators(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Classifies each input feature value against reference/normal bands into severity indicators.

    Severity Labels: 'Very High', 'High', 'Normal', 'Low', 'Very Low'
    Tags/Colors:
        - danger (Red '#DC2626') for strongly abnormal / high-risk levels
        - warning (Yellow '#D97706') for mild elevation or low deviation
        - normal (Green '#16A34A') for expected reference range

    Note: Indicators reflect relative value levels against reference bands and do NOT
    independently establish a medical diagnosis of diabetes.

    Args:
        input_data: Input feature dictionary.

    Returns:
        List of dictionaries with parameter indicator metrics and visual tags.
    """
    indicators = []

    for col in FEATURE_CONTRACT:
        val = float(input_data.get(col, FEATURE_CONFIG[col]["default"]))
        unit = FEATURE_CONFIG[col]["unit"]
        label = FEATURE_CONFIG[col]["label"]

        # Classification logic per parameter based on clinical reference thresholds & dataset stats
        if col == "Glucose":
            ref_band = "70–99 mg/dL"
            if val == 0 or val < 70:
                severity = "Low" if val > 0 else "Unrecorded"
                deviation = "Below" if val > 0 else "Unrecorded"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Below standard fasting baseline (< 70 mg/dL)."
            elif 70 <= val <= 99:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within normal reference band (70–99 mg/dL)."
            elif 100 <= val <= 125:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above reference band; prediabetes range (100–125 mg/dL)."
            else:  # val >= 126
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above reference band (≥ 126 mg/dL)."

        elif col == "BloodPressure":
            ref_band = "60–80 mm Hg"
            if val == 0 or val < 60:
                severity = "Low" if val > 0 else "Unrecorded"
                deviation = "Below" if val > 0 else "Unrecorded"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Below standard diastolic baseline (< 60 mm Hg)."
            elif 60 <= val <= 80:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within normal diastolic reference band (60–80 mm Hg)."
            elif 81 <= val <= 89:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above reference band; mild diastolic elevation (81–89 mm Hg)."
            else:  # val >= 90
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above reference band (≥ 90 mm Hg)."

        elif col == "BMI":
            ref_band = "18.5–24.9 kg/m²"
            if val == 0 or val < 18.5:
                severity = "Low" if val > 0 else "Unrecorded"
                deviation = "Below" if val > 0 else "Unrecorded"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Below standard normal BMI baseline (< 18.5 kg/m²)."
            elif 18.5 <= val <= 24.9:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within normal healthy BMI band (18.5–24.9 kg/m²)."
            elif 25.0 <= val <= 29.9:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above reference band; overweight BMI category (25.0–29.9 kg/m²)."
            else:  # val >= 30.0
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above reference band; obese category (≥ 30.0 kg/m²)."

        elif col == "SkinThickness":
            ref_band = "10–30 mm"
            if val == 0 or val < 10:
                severity = "Low" if val > 0 else "Unrecorded"
                deviation = "Below" if val > 0 else "Unrecorded"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Below standard skinfold baseline (< 10 mm)."
            elif 10 <= val <= 30:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within normal skinfold thickness band (10–30 mm)."
            elif 31 <= val <= 40:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above reference band (31–40 mm)."
            else:  # val > 40
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above reference band (> 40 mm)."

        elif col == "Insulin":
            ref_band = "15–166 μU/ml"
            if val == 0 or val < 15:
                severity = "Low" if val > 0 else "Unrecorded"
                deviation = "Below" if val > 0 else "Unrecorded"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Below standard insulin baseline (< 15 μU/ml)."
            elif 15 <= val <= 166:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within normal serum insulin reference band (15–166 μU/ml)."
            elif 167 <= val <= 250:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above reference band (167–250 μU/ml)."
            else:  # val > 250
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above reference band (> 250 μU/ml)."

        elif col == "Pregnancies":
            ref_band = "0–4 count"
            if val <= 4:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within typical population baseline (0–4)."
            elif 5 <= val <= 8:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above typical reference band (5–8)."
            else:  # val > 8
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above typical reference band (> 8)."

        elif col == "DiabetesPedigreeFunction":
            ref_band = "0.078–0.499 score"
            if val <= 0.499:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Within typical pedigree risk score band (≤ 0.50)."
            elif 0.500 <= val <= 0.799:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Above typical pedigree score band (0.50–0.80)."
            else:  # val >= 0.800
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Significantly above typical pedigree score band (≥ 0.80)."

        elif col == "Age":
            ref_band = "18–44 years"
            if val <= 44:
                severity = "Normal"
                deviation = "Within"
                tag = "normal"
                badge_color = "#16A34A"
                desc = "Younger / early middle-age cohort (18–44 yrs)."
            elif 45 <= val <= 64:
                severity = "High"
                deviation = "Above"
                tag = "warning"
                badge_color = "#D97706"
                desc = "Middle-age cohort with elevated demographic baseline (45–64 yrs)."
            else:  # val >= 65
                severity = "Very High"
                deviation = "Above"
                tag = "danger"
                badge_color = "#DC2626"
                desc = "Senior age cohort (≥ 65 yrs)."

        else:
            ref_band = "Standard"
            severity = "Normal"
            deviation = "Within"
            tag = "normal"
            badge_color = "#16A34A"
            desc = "Within expected parameters."

        indicators.append(
            {
                "feature": col,
                "label": label,
                "value": val,
                "unit": unit,
                "ref_band": ref_band,
                "deviation": deviation,
                "severity": severity,
                "tag": tag,
                "badge_color": badge_color,
                "description": desc,
            }
        )

    return indicators


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
        "Insulin": 0,  # Invalid zero (handled as NaN by imputer)
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    print(f"\nInput Patient Profile:\n{sample_patient}")
    prediction_result = make_prediction(sample_patient)
    print(f"\nPrediction Result:\n{prediction_result}")

    print("\nParameter Indicators:")
    indicators = classify_parameter_indicators(sample_patient)
    for ind in indicators:
        print(f"  - {ind['label']}: {ind['value']} {ind['unit']} [{ind['severity']}] -> {ind['description']}")
