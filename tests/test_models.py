"""Unit and integration tests for ML model training, 5-fold cross-validation, and selection."""

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
    build_candidate_pipelines,
    evaluate_model_pipeline,
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


def test_build_candidate_pipelines():
    """Test creation of candidate model pipelines."""
    pipelines = build_candidate_pipelines()
    assert "Baseline Logistic Regression" in pipelines
    assert "Baseline Random Forest" in pipelines
    assert "Tuned Random Forest" in pipelines
    assert "Gradient Boosting" in pipelines
    assert isinstance(pipelines["Baseline Random Forest"], Pipeline)


def test_evaluate_model_pipeline():
    """Test 5-fold cross-validation and test evaluation metric structure."""
    X_train, X_test, y_train, y_test = get_train_test_data(random_state=42)
    pipelines = build_candidate_pipelines(random_state=42)
    rf_pipeline = pipelines["Baseline Random Forest"]

    metrics = evaluate_model_pipeline(
        "Baseline Random Forest", rf_pipeline, X_train, y_train, X_test, y_test, random_state=42
    )

    required_keys = [
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
        "test_roc_auc",
        "confusion_matrix",
    ]
    for key in required_keys:
        assert key in metrics

    assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_train_and_evaluate_integration():
    """Integration test verifying full model training, cross-validation, selection, and joblib export."""
    results, best_name, best_model = train_and_evaluate_models(random_state=42)

    assert "Baseline Logistic Regression" in results
    assert "Baseline Random Forest" in results
    assert best_name == "Baseline Random Forest"

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
