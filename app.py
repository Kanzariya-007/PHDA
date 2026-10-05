"""Predictive Healthcare Diagnostic App (PHDA) - Streamlit Web Interface."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.predict import (
    FEATURE_CONFIG,
    FEATURE_CONTRACT,
    classify_parameter_indicators,
    load_model,
    make_prediction,
)

# Page Configuration
st.set_page_config(
    page_title="Predictive Healthcare Diagnostic App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Premium Aesthetics
st.markdown(
    """
    <style>
    /* Global Typography & Headers */
    .main-header {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Card Layouts */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    .result-card-positive {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 1.5rem;
        color: #991B1B;
    }
    .result-card-negative {
        background-color: #F0FDF4;
        border: 2px solid #16A34A;
        border-radius: 12px;
        padding: 1.5rem;
        color: #166534;
    }
    
    /* Indicator Badges */
    .badge-very-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    
    /* Input Help Range Tags */
    .range-info {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_model():
    """Loads and caches the trained model pipeline to prevent reloading/retraining on app re-runs."""
    return load_model(auto_train_if_missing=False)


def main():
    # Header Section (Clean Title, No Clutter)
    st.markdown('<div class="main-header">Predictive Healthcare Diagnostic App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Machine Learning Risk Assessment based on Clinical Health Metrics</div>',
        unsafe_allow_html=True,
    )

    # Load Model Pipeline with Graceful Error Handling
    try:
        model = get_cached_model()
    except Exception as exc:
        st.error(
            f"⚠️ **Model Artifact Error**: {exc}\n\n"
            "Please run `python -m src.train` in your terminal to generate the trained model artifact."
        )
        st.stop()

    # Sidebar Population & Model Specifications
    with st.sidebar:
        st.header("📋 Population Context")
        st.info(
            "**Training Population Note:**\n"
            "The underlying Random Forest model is trained on the PIMA Female Patient Population (Ages 21+).\n\n"
            "- **Pregnancies = 0** represents a patient with no pregnancy history.\n"
            "- Model predictions reflect statistical risk profile for this demographic."
        )
        st.divider()
        st.subheader("Model Specifications")
        st.markdown("- **Algorithm:** Random Forest Classifier")
        st.markdown("- **Preprocessing:** Median Imputation & Feature Scaling")
        st.markdown("- **Primary Metric (Accuracy):** ~77.3%")
        st.markdown("- **ROC-AUC Score:** ~0.818")
        st.divider()
        st.caption("PHDA MVP Version 1.1.0")

    # Main Form Inputs
    st.markdown('<div class="section-title">📊 Patient Clinical Parameters</div>', unsafe_allow_html=True)
    st.write("Enter clinical measurements below. Each input displays its dataset range and application limits.")

    col1, col2 = st.columns(2)
    user_inputs = {}

    with col1:
        st.markdown("##### Demographics & Reproductive History")

        user_inputs["Age"] = st.number_input(
            label=f"{FEATURE_CONFIG['Age']['label']} ({FEATURE_CONFIG['Age']['unit']})",
            min_value=int(FEATURE_CONFIG['Age']['val_min']),
            max_value=int(FEATURE_CONFIG['Age']['val_max']),
            value=int(FEATURE_CONFIG['Age']['default']),
            step=int(FEATURE_CONFIG['Age']['step']),
            help=FEATURE_CONFIG['Age']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["Age"]["dataset_min"]}–{FEATURE_CONFIG["Age"]["dataset_max"]} yrs | Validation Limit: {FEATURE_CONFIG["Age"]["val_min"]}–{FEATURE_CONFIG["Age"]["val_max"]} yrs</div>',
            unsafe_allow_html=True,
        )

        user_inputs["Pregnancies"] = st.number_input(
            label=f"{FEATURE_CONFIG['Pregnancies']['label']} ({FEATURE_CONFIG['Pregnancies']['unit']})",
            min_value=int(FEATURE_CONFIG['Pregnancies']['val_min']),
            max_value=int(FEATURE_CONFIG['Pregnancies']['val_max']),
            value=int(FEATURE_CONFIG['Pregnancies']['default']),
            step=int(FEATURE_CONFIG['Pregnancies']['step']),
            help=FEATURE_CONFIG['Pregnancies']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["Pregnancies"]["dataset_min"]}–{FEATURE_CONFIG["Pregnancies"]["dataset_max"]} | Validation Limit: {FEATURE_CONFIG["Pregnancies"]["val_min"]}–{FEATURE_CONFIG["Pregnancies"]["val_max"]}</div>',
            unsafe_allow_html=True,
        )

        user_inputs["DiabetesPedigreeFunction"] = st.number_input(
            label=f"{FEATURE_CONFIG['DiabetesPedigreeFunction']['label']} ({FEATURE_CONFIG['DiabetesPedigreeFunction']['unit']})",
            min_value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['val_min']),
            max_value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['val_max']),
            value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['default']),
            step=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['step']),
            format="%.3f",
            help=FEATURE_CONFIG['DiabetesPedigreeFunction']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["DiabetesPedigreeFunction"]["dataset_min"]}–{FEATURE_CONFIG["DiabetesPedigreeFunction"]["dataset_max"]} | Validation Limit: {FEATURE_CONFIG["DiabetesPedigreeFunction"]["val_min"]}–{FEATURE_CONFIG["DiabetesPedigreeFunction"]["val_max"]}</div>',
            unsafe_allow_html=True,
        )

        user_inputs["BMI"] = st.number_input(
            label=f"{FEATURE_CONFIG['BMI']['label']} ({FEATURE_CONFIG['BMI']['unit']})",
            min_value=float(FEATURE_CONFIG['BMI']['val_min']),
            max_value=float(FEATURE_CONFIG['BMI']['val_max']),
            value=float(FEATURE_CONFIG['BMI']['default']),
            step=float(FEATURE_CONFIG['BMI']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['BMI']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["BMI"]["dataset_min"]}–{FEATURE_CONFIG["BMI"]["dataset_max"]} kg/m² | Validation Limit: {FEATURE_CONFIG["BMI"]["val_min"]}–{FEATURE_CONFIG["BMI"]["val_max"]} kg/m²</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("##### Metabolic & Blood Parameters")

        user_inputs["Glucose"] = st.number_input(
            label=f"{FEATURE_CONFIG['Glucose']['label']} ({FEATURE_CONFIG['Glucose']['unit']})",
            min_value=float(FEATURE_CONFIG['Glucose']['val_min']),
            max_value=float(FEATURE_CONFIG['Glucose']['val_max']),
            value=float(FEATURE_CONFIG['Glucose']['default']),
            step=float(FEATURE_CONFIG['Glucose']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['Glucose']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["Glucose"]["dataset_min"]}–{FEATURE_CONFIG["Glucose"]["dataset_max"]} mg/dL | Validation Limit: {FEATURE_CONFIG["Glucose"]["val_min"]}–{FEATURE_CONFIG["Glucose"]["val_max"]} mg/dL</div>',
            unsafe_allow_html=True,
        )

        user_inputs["BloodPressure"] = st.number_input(
            label=f"{FEATURE_CONFIG['BloodPressure']['label']} ({FEATURE_CONFIG['BloodPressure']['unit']})",
            min_value=float(FEATURE_CONFIG['BloodPressure']['val_min']),
            max_value=float(FEATURE_CONFIG['BloodPressure']['val_max']),
            value=float(FEATURE_CONFIG['BloodPressure']['default']),
            step=float(FEATURE_CONFIG['BloodPressure']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['BloodPressure']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["BloodPressure"]["dataset_min"]}–{FEATURE_CONFIG["BloodPressure"]["dataset_max"]} mm Hg | Validation Limit: {FEATURE_CONFIG["BloodPressure"]["val_min"]}–{FEATURE_CONFIG["BloodPressure"]["val_max"]} mm Hg</div>',
            unsafe_allow_html=True,
        )

        user_inputs["Insulin"] = st.number_input(
            label=f"{FEATURE_CONFIG['Insulin']['label']} ({FEATURE_CONFIG['Insulin']['unit']})",
            min_value=float(FEATURE_CONFIG['Insulin']['val_min']),
            max_value=float(FEATURE_CONFIG['Insulin']['val_max']),
            value=float(FEATURE_CONFIG['Insulin']['default']),
            step=float(FEATURE_CONFIG['Insulin']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['Insulin']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["Insulin"]["dataset_min"]}–{FEATURE_CONFIG["Insulin"]["dataset_max"]} μU/ml | Validation Limit: {FEATURE_CONFIG["Insulin"]["val_min"]}–{FEATURE_CONFIG["Insulin"]["val_max"]} μU/ml</div>',
            unsafe_allow_html=True,
        )

        user_inputs["SkinThickness"] = st.number_input(
            label=f"{FEATURE_CONFIG['SkinThickness']['label']} ({FEATURE_CONFIG['SkinThickness']['unit']})",
            min_value=float(FEATURE_CONFIG['SkinThickness']['val_min']),
            max_value=float(FEATURE_CONFIG['SkinThickness']['val_max']),
            value=float(FEATURE_CONFIG['SkinThickness']['default']),
            step=float(FEATURE_CONFIG['SkinThickness']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['SkinThickness']['help'],
        )
        st.markdown(
            f'<div class="range-info">Dataset Range: {FEATURE_CONFIG["SkinThickness"]["dataset_min"]}–{FEATURE_CONFIG["SkinThickness"]["dataset_max"]} mm | Validation Limit: {FEATURE_CONFIG["SkinThickness"]["val_min"]}–{FEATURE_CONFIG["SkinThickness"]["val_max"]} mm</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Action Button
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        predict_btn = st.button("🔍 Assess Diabetes Risk Profile", use_container_width=True)

    # Execution of Prediction & Assessment Results
    if predict_btn:
        st.divider()

        try:
            # 1. Execute Inference via Reusable Module
            result = make_prediction(user_inputs, model=model)
            prediction = result["prediction"]
            label = result["label"]
            prob_diabetic = result.get("probability_diabetic", 0.0)
            prob_non_diabetic = result.get("probability_non_diabetic", 0.0)
            confidence = result.get("confidence", 0.0)

            # 2. Section I: ML Model Prediction Output
            st.markdown('<div class="section-title">🎯 Machine Learning Model Prediction</div>', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-positive">
                            <h3 style="margin:0; color:#991B1B;">⚠️ High Risk Profile Predicted</h3>
                            <h4 style="margin-top:0.3rem; font-weight:600; color:#B91C1C;">Classification Label: {label}</h4>
                            <p style="margin-bottom:0; font-size:0.95rem;">The model estimates a <strong>{prob_diabetic * 100:.1f}%</strong> statistical likelihood of diabetes based on the overall feature profile.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-negative">
                            <h3 style="margin:0; color:#166534;">✅ Low Risk Profile Predicted</h3>
                            <h4 style="margin-top:0.3rem; font-weight:600; color:#15803D;">Classification Label: {label}</h4>
                            <p style="margin-bottom:0; font-size:0.95rem;">The model estimates a <strong>{prob_non_diabetic * 100:.1f}%</strong> statistical likelihood of non-diabetic status based on the overall feature profile.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown("##### Estimated Probability Distribution")
                st.write(f"**Diabetic Risk Probability:** {prob_diabetic * 100:.1f}%")
                st.progress(float(prob_diabetic))

                st.write(f"**Non-Diabetic Probability:** {prob_non_diabetic * 100:.1f}%")
                st.progress(float(prob_non_diabetic))

                st.metric(label="Model Confidence Score", value=f"{confidence * 100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Section II: Parameter-Level Reference Band Indicators
            st.markdown('<div class="section-title">🩺 Parameter Reference Band Indicators</div>', unsafe_allow_html=True)
            st.caption(
                "Parameter indicators classify individual metric values relative to reference bands (Very Low, Low, Normal, High, Very High). "
                "These indicators describe parameter levels and do NOT independently establish a clinical diagnosis."
            )

            # Visual Legend
            st.markdown(
                """
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px; margin-bottom: 15px; font-size: 0.85rem;">
                    <strong>Visual Tag Legend:</strong> &nbsp;
                    <span class="badge-normal">🟢 Normal</span> Expected Reference Range &nbsp;|&nbsp;
                    <span class="badge-high">🟡 High / Low</span> Mild Deviation &nbsp;|&nbsp;
                    <span class="badge-very-high">🔴 Very High / Very Low</span> Significant Deviation
                </div>
                """,
                unsafe_allow_html=True,
            )

            indicators = classify_parameter_indicators(user_inputs)

            # Split indicators into Normal vs Elevated/Abnormal
            normal_inds = [i for i in indicators if i["severity"] == "Normal"]
            abnormal_inds = [i for i in indicators if i["severity"] != "Normal"]

            ind_col1, ind_col2 = st.columns(2)

            with ind_col1:
                st.markdown("##### ⚠️ Elevated / Deviating Parameters")
                if abnormal_inds:
                    for ind in abnormal_inds:
                        badge_class = (
                            "badge-very-high"
                            if ind["severity"] in ["Very High", "Very Low"]
                            else "badge-high"
                        )
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>{ind['label']}</strong>
                                    <span class="{badge_class}">{ind['severity']}</span>
                                </div>
                                <div style="font-size:1.1rem; font-weight:600; margin:4px 0;">{ind['value']} {ind['unit']}</div>
                                <div style="font-size:0.82rem; color:#64748B;">{ind['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("All clinical parameters are within normal reference bands.")

            with ind_col2:
                st.markdown("##### ✅ Normal Reference Parameters")
                if normal_inds:
                    for ind in normal_inds:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>{ind['label']}</strong>
                                    <span class="badge-normal">Normal</span>
                                </div>
                                <div style="font-size:1.1rem; font-weight:600; margin:4px 0;">{ind['value']} {ind['unit']}</div>
                                <div style="font-size:0.82rem; color:#64748B;">{ind['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No parameters are in the normal reference band.")

        except ValueError as val_err:
            st.warning(f"⚠️ **Validation Notice**: {val_err}")
        except Exception as exc:
            st.error(f"Error during risk assessment execution: {exc}")


if __name__ == "__main__":
    main()
