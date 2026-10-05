# Predictive Healthcare Diagnostic App (PHDA) - Diabetes Prediction App

A machine-learning-powered healthcare web application that estimates diabetes risk based on clinical health metrics using the PIMA Indians Diabetes Dataset. Built with Python, Pandas, Scikit-Learn, and Streamlit.

---

## 1. Project Overview

The **Predictive Healthcare Diagnostic App (PHDA)** is an end-to-end Machine Learning MVP designed to evaluate patient diabetes likelihood based on 8 clinical features. 

### Key Capabilities
- **Reproducible Data Acquisition & Schema Validation**: Automated data ingestion with strict schema enforcement.
- **Data Cleaning & Medical Preprocessing**: Handles medically invalid zero entries (in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI) as missing data using leakage-free imputation pipelines.
- **Exploratory Data Analysis (EDA)**: Statistical profiling and visualization plots.
- **Leakage-Free 5-Fold Cross-Validation & Model Training**: Evaluates Baseline & Tuned Logistic Regression, Random Forest, Gradient Boosting, and HistGradientBoosting classifiers.
- **Interactive Web UI**: A Streamlit application for entering clinical metrics, obtaining prediction results, probability distributions, and parameter-level reference band indicators.

---

## 2. Model Retraining Behavior Notice

> **Random Forest is retrained during normal app use: NO**

Model training ([src/train.py](file:///d:/Programming/Data%20Science/PHDA/src/train.py)) is completely decoupled from the Streamlit web application ([app.py](file:///d:/Programming/Data%20Science/PHDA/app.py)). The application loads the pre-trained pipeline artifact ([models/diabetes_model.pkl](file:///d:/Programming/Data%20Science/PHDA/models/diabetes_model.pkl)) using `@st.cache_resource`. Streamlit reruns or user interactions do **not** trigger model retraining or repeated disk loading.

---

## 3. Prerequisites

- **Python**: Recommended **Python 3.9+** (Tested on Python 3.14).
- **Package Manager**: `pip` (bundled with Python).
- **Virtual Environment**: Strongly recommended (`venv`) to isolate dependencies.

---

## 4. Installation on a Fresh PC

Follow these steps to set up the project on a new computer from a fresh clone.

### Step 1: Clone the Repository & Navigate to Directory
```bash
git clone https://github.com/Kanzariya-007/PHDA.git
cd PHDA
```

### Step 2: Create a Virtual Environment
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv venv
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Activate the Virtual Environment
- **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*
- **Windows Command Prompt:**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Upgrade pip & Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Project Structure

The project follows a clean, modular data science architecture:

```text
PHDA/
├── data/
│   ├── raw/                  # Raw dataset files (git-ignored)
│   └── processed/            # Cleaned dataset artifacts (git-ignored)
├── models/
│   ├── best_model.joblib     # Saved Random Forest model artifact (git-ignored)
│   └── diabetes_model.pkl    # Serialized pipeline for inference (git-ignored)
├── reports/
│   ├── eda_summary.json      # Profiling statistics summary
│   ├── model_evaluation.json # Reproducible 5-fold CV & test evaluation metrics
│   ├── PBL_REPORT.md         # Problem-Based Learning technical report
│   └── figures/              # Generated EDA visualization charts (.png)
├── src/
│   ├── __init__.py           # Package initializer
│   ├── data_loader.py        # Data ingestion & schema validation
│   ├── preprocess.py         # Data cleaning & ColumnTransformer pipeline
│   ├── eda.py                # EDA statistics & chart generation
│   ├── train.py              # Model training, 5-fold CV, & selection
│   └── predict.py            # Reusable inference module & parameter indicators
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py   # Ingestion & schema tests
│   ├── test_preprocess.py    # Preprocessing & leakage isolation tests
│   ├── test_eda.py           # EDA profiling & plot generation tests
│   ├── test_models.py        # Training & metrics tests
│   ├── test_predict.py       # Inference & serialization tests
│   └── test_app.py           # Streamlit UI configuration tests
├── app.py                    # Main Streamlit web application entrypoint
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules for data & model artifacts
└── README.md                 # Project documentation
```

---

## 6. Running the Application

After activating your virtual environment and installing dependencies:

```bash
streamlit run app.py
```
*(If `streamlit` command is not in PATH, use: `python -m streamlit run app.py`)*

### Accessing the Web App
1. Open your browser and navigate to `http://localhost:8501`.
2. Enter clinical measurements in the form controls.
3. Click **"🔍 Assess Diabetes Risk Profile"** to view prediction results, probability breakdown, and parameter reference band indicators.

---

## 7. Model Evaluation & Comparison Results

Models were evaluated using 5-fold Stratified Cross-Validation on the 80% training set ($X_{\text{train}}$) and evaluated once on the 20% holdout test set ($X_{\text{test}}$).

### Experimental Metrics Table

| Model Pipeline | 5-Fold CV Accuracy (Mean $\pm$ Std) | 5-Fold CV ROC-AUC (Mean $\pm$ Std) | Holdout Test Accuracy | Holdout Test Precision | Holdout Test Recall | Holdout Test F1-Score | Holdout Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.7882 $\pm$ 0.019 | 0.8434 $\pm$ 0.018 | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| **Baseline Random Forest** *(Selected)* | **0.7639 $\pm$ 0.016** | **0.8154 $\pm$ 0.023** | **0.7727** | **0.7021** | **0.6111** | **0.6535** | **0.8181** |
| **Tuned Random Forest** | 0.7606 $\pm$ 0.017 | 0.8349 $\pm$ 0.026 | 0.7403 | 0.6119 | 0.7593 | 0.6721 | 0.8215 |
| **Tuned Logistic Regression** | 0.7622 $\pm$ 0.011 | 0.8449 $\pm$ 0.015 | 0.7338 | 0.5970 | 0.7407 | 0.6496 | 0.8126 |
| **Gradient Boosting** | 0.7524 $\pm$ 0.018 | 0.8222 $\pm$ 0.018 | 0.7403 | 0.6591 | 0.5370 | 0.5918 | 0.8222 |
| **HistGradientBoosting** | 0.7671 $\pm$ 0.027 | 0.8161 $\pm$ 0.017 | 0.7403 | 0.6522 | 0.5556 | 0.6154 | 0.8254 |

- **Selected Model:** **Baseline Random Forest** was selected for achieving the highest Test Accuracy (**0.7727** / 77.27%) and Precision (**0.7021** / 70.21%) with stable cross-validation performance.
- **ROC-AUC Explanation:** Area Under ROC Curve (0.8181) measures the model's ability to rank higher-risk patients above lower-risk patients across decision thresholds. It is **not** a measure of clinical diagnostic accuracy.

---

## 8. Basic Usage & Input Features

### Feature Inputs
The model accepts 8 clinical measurements:

| Feature | Description | Unit | Dataset Range | Validation Limit |
|---|---|---|---|---|
| `Pregnancies` | Number of times pregnant | count | 0 – 17 | 0 – 20 |
| `Glucose` | 2-Hour oral glucose concentration | mg/dL | 0 – 199 | 40.0 – 300.0 |
| `BloodPressure` | Diastolic blood pressure | mm Hg | 0 – 122 | 30.0 – 180.0 |
| `SkinThickness` | Triceps skin fold thickness | mm | 0 – 99 | 5.0 – 100.0 |
| `Insulin` | 2-Hour serum insulin | μU/ml | 0 – 846 | 14.0 – 900.0 |
| `BMI` | Body Mass Index ($\text{kg/m}^2$) | $\text{kg/m}^2$ | 0.0 – 67.1 | 10.0 – 75.0 |
| `DiabetesPedigreeFunction` | Family history risk score | score | 0.078 – 2.420 | 0.050 – 3.000 |
| `Age` | Patient age | years | 21 – 81 | 18 – 120 |

---

## 9. Testing & Verification

Run the complete automated test suite using `pytest`:

```bash
pytest
```
*(Or run via python module: `python -m pytest`)*

The test suite consists of **35 unit and integration tests** covering:
- Schema validation & data ingestion ([tests/test_data_loader.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_data_loader.py))
- Invalid zero handling & leakage prevention ([tests/test_preprocess.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_preprocess.py))
- EDA summary statistics & plot rendering ([tests/test_eda.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_eda.py))
- 5-Fold CV model training & metrics selection ([tests/test_models.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_models.py))
- Inference interface, input bounds, & `.pkl` serialization ([tests/test_predict.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_predict.py))
- Streamlit UI configuration & bounds ([tests/test_app.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_app.py))

---

## 10. Troubleshooting

| Symptom / Issue | Likely Cause | Solution |
|---|---|---|
| `ModuleNotFoundError` when running scripts | Virtual environment not activated or dependencies missing | Activate `venv` and run `pip install -r requirements.txt`. |
| `ImportError: attempted relative import with no known parent package` | Running script directly without module flag | Run modules using `-m` flag (e.g. `python -m src.train` instead of `python src/train.py`). |
| `streamlit: command not found` | Streamlit executable not in system PATH | Execute using `python -m streamlit run app.py`. |
| PowerShell script execution error (`cannot be loaded because running scripts is disabled`) | Windows ExecutionPolicy restriction | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell. |
| Missing model artifact `models/diabetes_model.pkl` | First run on fresh clone | Run `python -m src.train` to generate model artifacts. |

---

## 11. Limitations & Medical Disclaimer

> **⚠️ Medical Disclaimer:** This application is a machine-learning-based predictive model developed strictly for educational, research, and project-based learning (PBL) demonstration purposes. It is **not** a clinical diagnostic tool and does **not** provide professional medical advice, diagnosis, or treatment. Always consult a qualified physician or healthcare provider for medical concerns.

---

## 12. Reproducibility & Model Training Workflow

To reproduce the model selection and evaluation from scratch:

1. Run dataset acquisition and validation:
   ```bash
   python -m src.data_loader
   ```
2. Clean data and handle invalid zeros:
   ```bash
   python -m src.preprocess
   ```
3. Generate EDA reports and figures:
   ```bash
   python -m src.eda
   ```
4. Execute 5-fold cross-validation, hyperparameter tuning, model comparison, and serialization:
   ```bash
   python -m src.train
   ```
5. Run automated test suite:
   ```bash
   python -m pytest
   ```
6. Launch Streamlit application:
   ```bash
   streamlit run app.py
   ```
