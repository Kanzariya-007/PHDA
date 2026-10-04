"""Unit tests for Exploratory Data Analysis (EDA) module."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.eda import (
    generate_descriptive_stats,
    plot_class_distribution,
    plot_missing_values,
    plot_feature_distributions,
    plot_bmi_glucose_by_outcome,
    plot_correlation_heatmap,
    run_eda_pipeline,
)
from src.preprocess import replace_invalid_zeros


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    """Fixture returning a sample cleaned DataFrame with NaNs."""
    data = {
        "Pregnancies": [0, 2, 5, 1, 3],
        "Glucose": [85, 120, 140, None, 100],
        "BloodPressure": [66, None, 80, 72, 64],
        "SkinThickness": [23, 30, None, 19, 25],
        "Insulin": [None, None, 180, 88, 105],
        "BMI": [26.6, None, 35.5, 30.1, 28.4],
        "DiabetesPedigreeFunction": [0.351, 0.5, 0.627, 0.234, 0.45],
        "Age": [31, 45, 50, 22, 36],
        "Outcome": [0, 1, 1, 0, 0],
    }
    return pd.DataFrame(data)


def test_generate_descriptive_stats(sample_cleaned_df):
    """Test generating descriptive statistics profile."""
    summary = generate_descriptive_stats(sample_cleaned_df)
    assert summary["dataset_dimensions"]["rows"] == 5
    assert summary["target_balance"]["negative_cases_0"] == 3
    assert summary["target_balance"]["positive_cases_1"] == 2
    assert "Glucose" in summary["feature_summary"]
    assert "missing_pct" in summary["feature_summary"]["Glucose"]


def test_plot_generation(sample_cleaned_df, tmp_path):
    """Test that all EDA plotting functions generate valid PNG files."""
    class_fig = tmp_path / "class.png"
    missing_fig = tmp_path / "missing.png"
    dist_fig = tmp_path / "dist.png"
    bmi_fig = tmp_path / "bmi.png"
    corr_fig = tmp_path / "corr.png"

    plot_class_distribution(sample_cleaned_df, class_fig)
    plot_missing_values(sample_cleaned_df, missing_fig)
    plot_feature_distributions(sample_cleaned_df, dist_fig)
    plot_bmi_glucose_by_outcome(sample_cleaned_df, bmi_fig)
    plot_correlation_heatmap(sample_cleaned_df, corr_fig)

    assert class_fig.exists() and class_fig.stat().st_size > 0
    assert missing_fig.exists() and missing_fig.stat().st_size > 0
    assert dist_fig.exists() and dist_fig.stat().st_size > 0
    assert bmi_fig.exists() and bmi_fig.stat().st_size > 0
    assert corr_fig.exists() and corr_fig.stat().st_size > 0


def test_run_eda_pipeline_integration():
    """Integration test running full EDA pipeline on actual raw data."""
    summary, fig_paths = run_eda_pipeline()
    assert summary["dataset_dimensions"]["rows"] == 768
    assert len(fig_paths) == 5
    for name, path in fig_paths.items():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
