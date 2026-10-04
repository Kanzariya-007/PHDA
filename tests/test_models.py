"""Unit and integration tests for ML model training, evaluation, and selection."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_loader import EXPECTED_COLUMNS
from src.train import (
    BEST_MODEL_PATH,
    EVALUATION_JSON_PATH,
    build_model_pipelines,
    evaluate_model,
    get_train_test_data,
    train_and_evaluate_models,
)


def test_get_train_test_data():
    """Test train/test split shape, stratification, and reproducibility."""
    X_train1, X_test1, y_train1, y_test1 = get_train_test_data(test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = get_train_test_data(test_size=0.2, random_state=42)

    # 1. Total samples check (768 total in PIMA)
    assert len(X_train1) + len(X_test1) == 768
    assert len(X_test1) == 154  # 20% of 768 rounded

    # 2. Stratification check (Outcome positive ratio ~34.9%)
    train_pos_pct = (y_train1.sum() / len(y_train1)) * 100
    test_pos_pct = (y_test1.sum() / len(y_test1)) * 100
    assert abs(train_pos_pct - test_pos_pct) < 1.0

    # 3. Reproducibility check with fixed random seed
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)


def test_build_model_pipelines():
    """Test creation of model pipelines."""
    pipelines = build_model_pipelines()
    assert "Logistic Regression" in pipelines
    assert "Random Forest" in pipelines
    assert isinstance(pipelines["Logistic Regression"], Pipeline)
    assert isinstance(pipelines["Random Forest"], Pipeline)


def test_evaluate_model():
    """Test model evaluation metric structure and valid metric bounds."""
    X_train, X_test, y_train, y_test = get_train_test_data(random_state=42)
    pipelines = build_model_pipelines(random_state=42)
    lr_model = pipelines["Logistic Regression"]
    lr_model.fit(X_train, y_train)

    metrics = evaluate_model(lr_model, X_test, y_test)

    required_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]
    for key in required_keys:
        assert key in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_train_and_evaluate_integration(tmp_path):
    """Integration test verifying full model training, evaluation, selection, and joblib export."""
    results, best_name, best_model = train_and_evaluate_models(random_state=42)

    assert "Logistic Regression" in results
    assert "Random Forest" in results
    assert best_name in ["Logistic Regression", "Random Forest"]

    # Verify best model matches higher accuracy
    lr_acc = results["Logistic Regression"]["accuracy"]
    rf_acc = results["Random Forest"]["accuracy"]
    if best_name == "Logistic Regression":
        assert lr_acc >= rf_acc
    else:
        assert rf_acc >= lr_acc

    # Verify files created
    assert EVALUATION_JSON_PATH.exists()
    assert BEST_MODEL_PATH.exists()

    # Load saved model artifact and test prediction capabilities
    loaded_model = joblib.load(BEST_MODEL_PATH)
    assert isinstance(loaded_model, Pipeline)

    X_train, X_test, y_train, y_test = get_train_test_data(random_state=42)
    preds = loaded_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    probas = loaded_model.predict_proba(X_test)
    assert probas.shape == (len(X_test), 2)
