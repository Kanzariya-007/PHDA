"""Predictive Healthcare Diagnostic App (PHDA) - Streamlit Web Interface."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.predict import (
    FEATURE_CONFIG,
    FEATURE_CONTRACT,
    load_model,
    make_prediction,
)

# Page Configuration
st.set_page_config(
    page_title="PHDA - Diabetes Risk Diagnostic App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Premium Aesthetics
st.markdown(
    """
    <style>
    /* Global Styles & Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1A365D;
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4A5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #92400E;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .result-card-positive {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.5rem;
        color: #991B1B;
    }
    .result-card-negative {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 1.5rem;
        color: #166534;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.6rem 2rem;
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
    return load_model()


def main():
    # Header Section
    st.markdown('<div class="main-header">🩺 Predictive Healthcare Diagnostic App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">ML-powered Diabetes Risk Assessment based on Clinical Health Metrics</div>',
        unsafe_allow_html=True,
    )

    # Medical Disclaimer Callout
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational & Research Disclaimer:</strong> 
            This application is a machine-learning-based risk estimation tool developed for educational and research demonstration purposes only. 
            It does <strong>not</strong> provide medical diagnoses, professional healthcare opinions, or treatment plans. Always consult a licensed healthcare professional for medical concerns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Model Pipeline
    try:
        model = get_cached_model()
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        st.stop()

    # Sidebar Information
    with st.sidebar:
        st.header("📋 About PHDA")
        st.write(
            """
            This app utilizes a **Random Forest Classifier** trained on the **PIMA Indians Diabetes Dataset** (NIDDK / UCI Repository).
            """
        )
        st.divider()
        st.subheader("Model Specifications")
        st.markdown("- **Algorithm:** Random Forest Classifier")
        st.markdown("- **Pre-processing:** Median Imputation & Feature Scaling")
        st.markdown("- **Primary Metric (Accuracy):** ~77.3%")
        st.markdown("- **ROC-AUC Score:** ~0.818")
        st.divider()
        st.caption("PHDA MVP Version 1.0.0")

    # Main Form Inputs
    st.subheader("📊 Patient Clinical Measurements")
    st.write("Adjust the clinical parameters below to evaluate diabetes risk.")

    col1, col2 = st.columns(2)

    user_inputs = {}

    with col1:
        st.markdown("##### Basic Metrics & History")
        user_inputs["Age"] = st.number_input(
            label=f"{FEATURE_CONFIG['Age']['label']} ({FEATURE_CONFIG['Age']['unit']})",
            min_value=int(FEATURE_CONFIG['Age']['min']),
            max_value=int(FEATURE_CONFIG['Age']['max']),
            value=int(FEATURE_CONFIG['Age']['default']),
            step=int(FEATURE_CONFIG['Age']['step']),
            help=FEATURE_CONFIG['Age']['help'],
        )

        user_inputs["Pregnancies"] = st.number_input(
            label=f"{FEATURE_CONFIG['Pregnancies']['label']} ({FEATURE_CONFIG['Pregnancies']['unit']})",
            min_value=int(FEATURE_CONFIG['Pregnancies']['min']),
            max_value=int(FEATURE_CONFIG['Pregnancies']['max']),
            value=int(FEATURE_CONFIG['Pregnancies']['default']),
            step=int(FEATURE_CONFIG['Pregnancies']['step']),
            help=FEATURE_CONFIG['Pregnancies']['help'],
        )

        user_inputs["DiabetesPedigreeFunction"] = st.number_input(
            label=f"{FEATURE_CONFIG['DiabetesPedigreeFunction']['label']} ({FEATURE_CONFIG['DiabetesPedigreeFunction']['unit']})",
            min_value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['min']),
            max_value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['max']),
            value=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['default']),
            step=float(FEATURE_CONFIG['DiabetesPedigreeFunction']['step']),
            format="%.3f",
            help=FEATURE_CONFIG['DiabetesPedigreeFunction']['help'],
        )

        user_inputs["BMI"] = st.number_input(
            label=f"{FEATURE_CONFIG['BMI']['label']} ({FEATURE_CONFIG['BMI']['unit']})",
            min_value=float(FEATURE_CONFIG['BMI']['min']),
            max_value=float(FEATURE_CONFIG['BMI']['max']),
            value=float(FEATURE_CONFIG['BMI']['default']),
            step=float(FEATURE_CONFIG['BMI']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['BMI']['help'],
        )

    with col2:
        st.markdown("##### Blood & Metabolic Metrics")
        user_inputs["Glucose"] = st.number_input(
            label=f"{FEATURE_CONFIG['Glucose']['label']} ({FEATURE_CONFIG['Glucose']['unit']})",
            min_value=float(FEATURE_CONFIG['Glucose']['min']),
            max_value=float(FEATURE_CONFIG['Glucose']['max']),
            value=float(FEATURE_CONFIG['Glucose']['default']),
            step=float(FEATURE_CONFIG['Glucose']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['Glucose']['help'],
        )

        user_inputs["BloodPressure"] = st.number_input(
            label=f"{FEATURE_CONFIG['BloodPressure']['label']} ({FEATURE_CONFIG['BloodPressure']['unit']})",
            min_value=float(FEATURE_CONFIG['BloodPressure']['min']),
            max_value=float(FEATURE_CONFIG['BloodPressure']['max']),
            value=float(FEATURE_CONFIG['BloodPressure']['default']),
            step=float(FEATURE_CONFIG['BloodPressure']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['BloodPressure']['help'],
        )

        user_inputs["Insulin"] = st.number_input(
            label=f"{FEATURE_CONFIG['Insulin']['label']} ({FEATURE_CONFIG['Insulin']['unit']})",
            min_value=float(FEATURE_CONFIG['Insulin']['min']),
            max_value=float(FEATURE_CONFIG['Insulin']['max']),
            value=float(FEATURE_CONFIG['Insulin']['default']),
            step=float(FEATURE_CONFIG['Insulin']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['Insulin']['help'],
        )

        user_inputs["SkinThickness"] = st.number_input(
            label=f"{FEATURE_CONFIG['SkinThickness']['label']} ({FEATURE_CONFIG['SkinThickness']['unit']})",
            min_value=float(FEATURE_CONFIG['SkinThickness']['min']),
            max_value=float(FEATURE_CONFIG['SkinThickness']['max']),
            value=float(FEATURE_CONFIG['SkinThickness']['default']),
            step=float(FEATURE_CONFIG['SkinThickness']['step']),
            format="%.1f",
            help=FEATURE_CONFIG['SkinThickness']['help'],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Action Button
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        predict_btn = st.button("🔍 Assess Diabetes Risk", use_container_width=True)

    # Results Display Section
    if predict_btn:
        st.divider()
        st.subheader("🎯 Assessment Results")

        try:
            # Execute inference using reusable module
            result = make_prediction(user_inputs, model=model)
            prediction = result["prediction"]
            label = result["label"]
            prob_diabetic = result.get("probability_diabetic", 0.0)
            prob_non_diabetic = result.get("probability_non_diabetic", 0.0)
            confidence = result.get("confidence", 0.0)

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-positive">
                            <h3 style="margin:0; color:#991B1B;">⚠️ Positive Outcome Predicted</h3>
                            <h4 style="margin-top:0.3rem; font-weight:normal; color:#B91C1C;">Model Estimate: High Diabetes Risk ({label})</h4>
                            <p style="margin-bottom:0;">The model estimates a <strong>{prob_diabetic * 100:.1f}%</strong> probability of diabetes based on the provided clinical metrics.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-negative">
                            <h3 style="margin:0; color:#166534;">✅ Negative Outcome Predicted</h3>
                            <h4 style="margin-top:0.3rem; font-weight:normal; color:#15803D;">Model Estimate: Low Diabetes Risk ({label})</h4>
                            <p style="margin-bottom:0;">The model estimates a <strong>{prob_non_diabetic * 100:.1f}%</strong> probability of non-diabetic status based on the provided clinical metrics.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown("##### Probability Breakdown")
                st.write(f"**Diabetic Risk Probability:** {prob_diabetic * 100:.1f}%")
                st.progress(float(prob_diabetic))

                st.write(f"**Non-Diabetic Probability:** {prob_non_diabetic * 100:.1f}%")
                st.progress(float(prob_non_diabetic))

                st.metric(label="Model Confidence Score", value=f"{confidence * 100:.1f}%")

        except Exception as exc:
            st.error(f"Error during prediction assessment: {exc}")


if __name__ == "__main__":
    main()
