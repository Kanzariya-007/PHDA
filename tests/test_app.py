"""Unit tests for Streamlit application UI helpers and configurations."""

import pytest
from sklearn.pipeline import Pipeline

from src.predict import FEATURE_CONFIG, FEATURE_CONTRACT, load_model, make_prediction


def test_feature_config_completeness():
    """Verify FEATURE_CONFIG contains all contract features with valid ranges and defaults."""
    assert set(FEATURE_CONFIG.keys()) == set(FEATURE_CONTRACT)

    for feature_name, cfg in FEATURE_CONFIG.items():
        assert "min" in cfg
        assert "max" in cfg
        assert "default" in cfg
        assert "step" in cfg
        assert "unit" in cfg
        assert "label" in cfg
        assert "help" in cfg

        # Range and default sanity check
        assert cfg["min"] <= cfg["default"] <= cfg["max"]
        assert cfg["step"] > 0


def test_default_inputs_prediction():
    """Verify that default values from FEATURE_CONFIG produce a valid prediction."""
    default_inputs = {feat: FEATURE_CONFIG[feat]["default"] for feat in FEATURE_CONTRACT}
    res = make_prediction(default_inputs)

    assert "prediction" in res
    assert res["prediction"] in [0, 1]
    assert "probability_diabetic" in res
    assert 0.0 <= res["probability_diabetic"] <= 1.0


def test_app_model_loader():
    """Verify app's model loader interface executes cleanly."""
    model = load_model()
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
