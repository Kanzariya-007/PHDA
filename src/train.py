"""Model training, evaluation, and selection module for PHDA."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import EXPECTED_COLUMNS, TARGET_COLUMN, load_raw_data
from src.preprocess import ZERO_INVALID_COLUMNS, replace_invalid_zeros

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_JSON_PATH = REPORTS_DIR / "model_evaluation.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def get_train_test_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads cleaned PIMA dataset and performs stratified train/test split.

    Args:
        test_size: Ratio of test dataset split (default 0.2).
        random_state: Random state seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df_raw = load_raw_data()
    df_cleaned = replace_invalid_zeros(df_raw)

    feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    X = df_cleaned[feature_cols]
    y = df_cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def build_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Constructs sklearn Pipelines for Logistic Regression and Random Forest.

    Preprocessing (median imputation + scaling) is integrated into each pipeline
    to guarantee zero data leakage during cross-validation and evaluation.

    Args:
        random_state: Random state seed.

    Returns:
        Dictionary mapping model names to unfitted sklearn Pipelines.
    """
    # Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=random_state, max_iter=1000),
            ),
        ]
    )

    # Random Forest Pipeline
    rf_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=random_state),
            ),
        ]
    )

    return {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
    }


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluates a trained model pipeline on test data across multiple metrics.

    Args:
        model: Trained sklearn Pipeline.
        X_test: Test feature DataFrame.
        y_test: Test target Series.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)) if roc_auc is not None else None,
        "confusion_matrix": cm,
    }

    return metrics


def train_and_evaluate_models(
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], str, Pipeline]:
    """Trains candidate models, evaluates metrics, selects the best model by accuracy, and saves artifacts.

    Args:
        random_state: Random state seed.

    Returns:
        Tuple of (Evaluation results dict, Best model name, Best trained model pipeline).
    """
    X_train, X_test, y_train, y_test = get_train_test_data(random_state=random_state)
    pipelines = build_model_pipelines(random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Pipeline] = {}

    for name, pipeline in pipelines.items():
        print(f"[INFO] Training {name}...")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        trained_models[name] = pipeline

    # Select best model based on Accuracy (primary metric), using F1 as secondary tie-breaker
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]["accuracy"], results[k]["f1_score"]),
    )
    best_model = trained_models[best_model_name]

    print(
        f"\n[INFO] Best Model Selected: '{best_model_name}' "
        f"with Accuracy = {results[best_model_name]['accuracy']:.4f}"
    )

    # Save evaluation results JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_summary = {
        "random_state": random_state,
        "test_split_ratio": 0.2,
        "best_model": best_model_name,
        "models": results,
    }
    with open(EVALUATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"[INFO] Saved model evaluation results to: {EVALUATION_JSON_PATH}")

    # Save best trained model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"[INFO] Saved best trained model artifact to: {BEST_MODEL_PATH}")

    return results, best_model_name, best_model


if __name__ == "__main__":
    print("=== PHDA Model Training & Evaluation Pipeline ===")
    results, best_name, _ = train_and_evaluate_models()

    print("\nModel Comparison Table:")
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 80)
    for model_name, m in results.items():
        print(
            f"{model_name:<22} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | "
            f"{m['recall']:<8.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f}"
        )
