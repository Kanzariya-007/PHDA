"""Exploratory Data Analysis (EDA) module for PHDA Diabetes dataset."""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import TARGET_COLUMN, load_raw_data
from src.preprocess import ZERO_INVALID_COLUMNS, replace_invalid_zeros, get_missing_values_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_SUMMARY_JSON_PATH = REPORTS_DIR / "eda_summary.json"


def generate_descriptive_stats(df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive descriptive statistics for raw and cleaned features.

    Args:
        df_cleaned: Cleaned DataFrame with invalid zeros replaced by NaN.

    Returns:
        Dictionary containing summary statistics per feature and target class metrics.
    """
    feature_cols = [c for c in df_cleaned.columns if c != TARGET_COLUMN]
    
    stats_df = df_cleaned[feature_cols].describe().T
    stats_df["median"] = df_cleaned[feature_cols].median()
    stats_df["skewness"] = df_cleaned[feature_cols].skew()
    stats_df["missing_pct"] = (df_cleaned[feature_cols].isnull().sum() / len(df_cleaned)) * 100

    target_counts = df_cleaned[TARGET_COLUMN].value_counts().to_dict()
    total_samples = len(df_cleaned)

    summary = {
        "dataset_dimensions": {"rows": total_samples, "columns": df_cleaned.shape[1]},
        "target_balance": {
            "negative_cases_0": int(target_counts.get(0, 0)),
            "positive_cases_1": int(target_counts.get(1, 0)),
            "negative_pct": round((target_counts.get(0, 0) / total_samples) * 100, 2),
            "positive_pct": round((target_counts.get(1, 0) / total_samples) * 100, 2),
            "imbalance_ratio": round(target_counts.get(0, 0) / max(1, target_counts.get(1, 0)), 2),
        },
        "feature_summary": stats_df.round(3).to_dict(orient="index"),
        "zero_invalid_columns": ZERO_INVALID_COLUMNS,
    }

    return summary


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Plots target variable Outcome class distribution."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x=TARGET_COLUMN,
        data=df,
        palette=["#2b5c8f", "#d95f02"],
        hue=TARGET_COLUMN,
        legend=False,
    )
    plt.title("Target Variable (Outcome) Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Diabetes Diagnosis (0 = Non-Diabetic, 1 = Diabetic)", fontsize=11)
    plt.ylabel("Patient Count", fontsize=11)

    total = len(df)
    for p in ax.patches:
        height = int(p.get_height())
        pct = (height / total) * 100
        ax.annotate(
            f"{height}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_missing_values(df_cleaned: pd.DataFrame, output_path: Path) -> Path:
    """Plots percentage of invalid zeros (missing values) per feature."""
    missing_report = get_missing_values_report(df_cleaned)
    missing_report_filtered = missing_report[missing_report["Missing_Count"] > 0]

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x=missing_report_filtered.index,
        y="Missing_Percentage",
        data=missing_report_filtered,
        palette="Reds_r",
        hue=missing_report_filtered.index,
        legend=False,
    )
    plt.title("Percentage of Medically Invalid Zero Values (Missing Data)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Clinical Features", fontsize=11)
    plt.ylabel("Missing Percentage (%)", fontsize=11)
    plt.xticks(rotation=15)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height + 1),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.ylim(0, max(missing_report["Missing_Percentage"]) + 8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_distributions(df_cleaned: pd.DataFrame, output_path: Path) -> Path:
    """Plots feature distribution histograms with KDE overlays colored by Outcome."""
    features = [c for c in df_cleaned.columns if c != TARGET_COLUMN]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes[idx]
        sns.histplot(
            data=df_cleaned,
            x=col,
            hue=TARGET_COLUMN,
            kde=True,
            palette=["#2b5c8f", "#d95f02"],
            bins=30,
            ax=ax,
            alpha=0.6,
        )
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density / Count", fontsize=10)

    plt.suptitle("Feature Distributions grouped by Outcome (NaN-handled)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_bmi_glucose_by_outcome(df_cleaned: pd.DataFrame, output_path: Path) -> Path:
    """Plots scatter plot & boxplots comparing Glucose and BMI by Outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot Glucose vs BMI
    sns.scatterplot(
        data=df_cleaned,
        x="Glucose",
        y="BMI",
        hue=TARGET_COLUMN,
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.7,
        s=60,
        ax=axes[0],
    )
    axes[0].set_title("BMI vs Glucose Concentration by Outcome", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Glucose (mg/dL)", fontsize=11)
    axes[0].set_ylabel("BMI (kg/m²)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot of Glucose & BMI side-by-side melted
    df_melted = df_cleaned.melt(
        id_vars=[TARGET_COLUMN],
        value_vars=["Glucose", "BMI"],
        var_name="Feature",
        value_name="Value",
    )
    sns.boxplot(
        data=df_melted,
        x="Feature",
        y="Value",
        hue=TARGET_COLUMN,
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[1],
    )
    axes[1].set_title("Glucose & BMI Distributions across Diagnoses", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Clinical Metric", fontsize=11)
    axes[1].set_ylabel("Value", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_correlation_heatmap(df_cleaned: pd.DataFrame, output_path: Path) -> Path:
    """Plots Pearson correlation heatmap of features and target."""
    plt.figure(figsize=(10, 8))
    corr = df_cleaned.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
    )
    plt.title("Feature Correlation Heatmap (PIMA Diabetes Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def run_eda_pipeline() -> tuple[Dict[str, Any], Dict[str, Path]]:
    """Runs complete EDA pipeline, computes descriptive stats, and saves visualizations.

    Returns:
        Tuple of (Summary dictionary, Dictionary of saved figure paths).
    """
    df_raw = load_raw_data()
    df_cleaned = replace_invalid_zeros(df_raw)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_stats = generate_descriptive_stats(df_cleaned)

    # Save summary statistics to JSON
    with open(EDA_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"[INFO] Saved EDA summary stats to: {EDA_SUMMARY_JSON_PATH}")

    # Generate charts
    fig_paths = {
        "class_distribution": plot_class_distribution(df_cleaned, FIGURES_DIR / "class_distribution.png"),
        "missing_values": plot_missing_values(df_cleaned, FIGURES_DIR / "missing_values_profile.png"),
        "feature_distributions": plot_feature_distributions(df_cleaned, FIGURES_DIR / "feature_distributions.png"),
        "bmi_glucose_by_outcome": plot_bmi_glucose_by_outcome(df_cleaned, FIGURES_DIR / "bmi_glucose_by_outcome.png"),
        "correlation_heatmap": plot_correlation_heatmap(df_cleaned, FIGURES_DIR / "correlation_heatmap.png"),
    }

    for name, path in fig_paths.items():
        print(f"[INFO] Saved figure '{name}' to: {path}")

    return summary_stats, fig_paths


if __name__ == "__main__":
    print("=== PHDA Exploratory Data Analysis (EDA) Pipeline ===")
    summary, fig_paths = run_eda_pipeline()
    print("\nEDA Pipeline completed successfully.")
    print(f"Total Rows: {summary['dataset_dimensions']['rows']}")
    print(f"Target Balance: {summary['target_balance']['negative_cases_0']} Non-Diabetic vs {summary['target_balance']['positive_cases_1']} Diabetic")
