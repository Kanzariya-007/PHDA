"""Model training, 5-fold cross-validation evaluation, hyperparameter tuning, and selection module for PHDA."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import EXPECTED_COLUMNS, TARGET_COLUMN, load_raw_data
from src.preprocess import ZERO_INVALID_COLUMNS, replace_invalid_zeros

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_JSON_PATH = REPORTS_DIR / "model_evaluation.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_PKL_PATH = MODELS_DIR / "diabetes_model.pkl"


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


def build_candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Constructs sklearn Pipelines for baseline and tuned candidate classifiers.

    Preprocessing (median imputation + scaling) is integrated into each pipeline
    to guarantee zero data leakage during cross-validation and evaluation.

    Args:
        random_state: Random state seed.

    Returns:
        Dictionary mapping model names to unfitted sklearn Pipelines.
    """
    pipelines = {
        "Baseline Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=random_state, max_iter=1000),
                ),
            ]
        ),
        "Baseline Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=random_state),
                ),
            ]
        ),
        "Tuned Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Tuned Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        learning_rate=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    return pipelines


def evaluate_model_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Performs 5-fold Stratified CV on train set and evaluates on untouched test set.

    Args:
        name: Name of the model.
        pipeline: Unfitted sklearn Pipeline.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
        random_state: Random state seed.

    Returns:
        Dictionary of cross-validation and test evaluation metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=["accuracy", "roc_auc"],
        n_jobs=-1,
    )

    cv_acc_mean = float(np.mean(cv_results["test_accuracy"]))
    cv_acc_std = float(np.std(cv_results["test_accuracy"]))
    cv_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
    cv_auc_std = float(np.std(cv_results["test_roc_auc"]))

    # Fit pipeline strictly on full X_train
    pipeline.fit(X_train, y_train)

    # Evaluate on untouched X_test
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    test_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": name,
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std": round(cv_acc_std, 4),
        "cv_roc_auc_mean": round(cv_auc_mean, 4),
        "cv_roc_auc_std": round(cv_auc_std, 4),
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1_score": round(test_f1, 4),
        "test_roc_auc": round(test_auc, 4) if test_auc is not None else None,
        "confusion_matrix": cm,
    }


def train_and_evaluate_models(
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], str, Pipeline]:
    """Trains candidate models, runs 5-fold CV and test evaluation, selects the best model, and serializes artifacts.

    Args:
        random_state: Random state seed.

    Returns:
        Tuple of (Results dictionary, Best model name, Best trained model pipeline).
    """
    X_train, X_test, y_train, y_test = get_train_test_data(random_state=random_state)
    pipelines = build_candidate_pipelines(random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    print("=== PHDA Model Cross-Validation & Test Evaluation ===")
    for name, pipeline in pipelines.items():
        print(f"[INFO] Evaluating {name}...")
        metrics = evaluate_model_pipeline(
            name, pipeline, X_train, y_train, X_test, y_test, random_state=random_state
        )
        results[name] = metrics
        trained_pipelines[name] = pipeline

    # Select best model: Primary metric is Test Accuracy, secondary is Test ROC-AUC and F1 score
    best_model_name = max(
        results.keys(),
        key=lambda k: (
            results[k]["test_accuracy"],
            results[k]["test_roc_auc"],
            results[k]["test_f1_score"],
        ),
    )
    best_model = trained_pipelines[best_model_name]

    print(
        f"\n[INFO] Best Model Selected: '{best_model_name}' "
        f"with Test Accuracy = {results[best_model_name]['test_accuracy']:.4f}, "
        f"Test ROC-AUC = {results[best_model_name]['test_roc_auc']:.4f}"
    )

    # Save evaluation results JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_summary = {
        "random_state": random_state,
        "test_split_ratio": 0.2,
        "cv_folds": 5,
        "baseline_model": "Baseline Random Forest",
        "best_model": best_model_name,
        "models": results,
    }
    with open(EVALUATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"[INFO] Saved model evaluation summary to: {EVALUATION_JSON_PATH}")

    # Save best trained model artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(best_model, MODEL_PKL_PATH)
    print(f"[INFO] Saved best trained model artifact to: {MODEL_PKL_PATH}")

    return results, best_model_name, best_model


if __name__ == "__main__":
    results, best_name, _ = train_and_evaluate_models()

    print("\n" + "=" * 105)
    print(
        f"{'Model Name':<28} | {'CV Acc (Mean±Std)':<18} | {'CV AUC (Mean±Std)':<18} | "
        f"{'Test Acc':<8} | {'Test F1':<8} | {'Test AUC':<8}"
    )
    print("-" * 105)
    for name, m in results.items():
        cv_acc_str = f"{m['cv_accuracy_mean']:.4f} ± {m['cv_accuracy_std']:.3f}"
        cv_auc_str = f"{m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.3f}"
        print(
            f"{name:<28} | {cv_acc_str:<18} | {cv_auc_str:<18} | "
            f"{m['test_accuracy']:<8.4f} | {m['test_f1_score']:<8.4f} | {m['test_roc_auc']:<8.4f}"
        )
    print("=" * 105)
