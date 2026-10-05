# Predictive Healthcare Diagnostic App (PHDA) - Diabetes Prediction System

A machine-learning-powered healthcare web application that estimates diabetes risk based on clinical health metrics using the PIMA Indians Diabetes Dataset. Built with Python, Pandas, Scikit-Learn, and Streamlit.

---

## 1. Project Overview

The **Predictive Healthcare Diagnostic App (PHDA)** is an end-to-end Machine Learning MVP designed to evaluate patient diabetes likelihood based on 8 clinical features. 

### Key Capabilities
- **Reproducible Data Acquisition & Schema Validation**: Automated data ingestion with strict schema enforcement.
- **Data Cleaning & Medical Preprocessing**: Handles medically invalid zero entries (in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI) as missing data using leakage-free imputation pipelines.
- **Exploratory Data Analysis (EDA)**: Statistical profiling and visualization plots.
- **Model Training & Evaluation**: Compares Logistic Regression and Random Forest using stratified 80/20 train/test splits.
- **Interactive Web UI**: A Streamlit application for entering clinical metrics and obtaining instant risk estimates with confidence scores.

---

## 2. Prerequisites

- **Python**: Recommended **Python 3.9+** (Tested on Python 3.14).
- **Package Manager**: `pip` (bundled with Python).
- **Virtual Environment**: Strongly recommended (`venv`) to isolate dependencies.

---

## 3. Installation on a Fresh PC

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
  python3 -m venv venv
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

## 4. Project Structure

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
│   ├── model_evaluation.json # Reproducible evaluation metrics
│   └── figures/              # Generated EDA visualization charts (.png)
├── src/
│   ├── __init__.py           # Package initializer
│   ├── data_loader.py        # Data ingestion & schema validation
│   ├── preprocess.py         # Data cleaning & ColumnTransformer pipeline
│   ├── eda.py                # EDA statistics & chart generation
│   ├── train.py              # Model training, evaluation, & selection
│   └── predict.py            # Reusable inference module & feature contract
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

## 5. Running the Application

After activating your virtual environment and installing dependencies:

```bash
streamlit run app.py
```
*(If `streamlit` command is not in PATH, use: `python -m streamlit run app.py`)*

### Accessing the Web App
1. Open your browser and navigate to `http://localhost:8501`.
2. Adjust the clinical measurements in the form controls.
3. Click **"🔍 Assess Diabetes Risk"** to view prediction results, probability breakdown, and confidence scores.

---

## 6. Model & Data Setup

- **Model Artifacts in Git**: Data CSVs (`data/raw/`, `data/processed/`) and model artifacts (`models/*.pkl`, `models/*.joblib`) are excluded from Git repository tracking via `.gitignore`.
- **Automatic Setup**: When running `streamlit run app.py` for the first time, if `models/diabetes_model.pkl` is missing, `load_model()` automatically triggers the data ingestion and model training pipeline. It downloads the dataset from the public source and generates `models/diabetes_model.pkl` seamlessly.

### Manual Setup (Optional Step-by-Step Execution)
If you prefer to execute each pipeline stage manually:

1. **Acquire Raw Dataset**:
   ```bash
   python -m src.data_loader
   ```
2. **Clean Dataset & Process Missing Values**:
   ```bash
   python -m src.preprocess
   ```
3. **Generate EDA Visualizations & Reports**:
   ```bash
   python -m src.eda
   ```
4. **Train & Evaluate Models**:
   ```bash
   python -m src.train
   ```

---

## 7. Basic Usage & Input Features

### Feature Inputs
The model accepts 8 clinical measurements:

| Feature | Description | Unit | Range |
|---|---|---|---|
| `Pregnancies` | Number of times pregnant | count | 0 – 17 |
| `Glucose` | Plasma glucose concentration at 2h in oral glucose test | mg/dL | 0 – 200 |
| `BloodPressure` | Diastolic blood pressure | mm Hg | 0 – 122 |
| `SkinThickness` | Triceps skin fold thickness | mm | 0 – 99 |
| `Insulin` | 2-Hour serum insulin | μU/ml | 0 – 846 |
| `BMI` | Body Mass Index ($\text{weight in kg}/(\text{height in m})^2$) | $\text{kg/m}^2$ | 0.0 – 67.1 |
| `DiabetesPedigreeFunction` | Genetic risk score based on family history | score | 0.078 – 2.420 |
| `Age` | Patient age | years | 21 – 81 |

### Prediction Results
- **Positive Outcome (`1`)**: Model estimates a high risk of diabetes.
- **Negative Outcome (`0`)**: Model estimates a low risk of diabetes.
- **Probability Scores**: Displays exact percentage probabilities for both outcomes (e.g. 94.0% Diabetic Risk).

---

## 8. Testing & Verification

Run the complete automated test suite using `pytest`:

```bash
pytest
```
*(Or run via python module: `python -m pytest`)*

The test suite consists of **31 unit and integration tests** covering:
- Schema validation & data ingestion ([tests/test_data_loader.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_data_loader.py))
- Invalid zero handling & leakage prevention ([tests/test_preprocess.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_preprocess.py))
- EDA summary statistics & plot rendering ([tests/test_eda.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_eda.py))
- Model training & metrics selection ([tests/test_models.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_models.py))
- Inference interface & `.pkl` serialization ([tests/test_predict.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_predict.py))
- Streamlit UI configuration & defaults ([tests/test_app.py](file:///d:/Programming/Data%20Science/PHDA/tests/test_app.py))

---

## 9. Troubleshooting

| Symptom / Issue | Likely Cause | Solution |
|---|---|---|
| `ModuleNotFoundError` when running scripts | Virtual environment not activated or dependencies missing | Activate `venv` and run `pip install -r requirements.txt`. |
| `ImportError: attempted relative import with no known parent package` | Running script directly without module flag | Run modules using `-m` flag (e.g. `python -m src.train` instead of `python src/train.py`). |
| `streamlit: command not found` | Streamlit executable not in system PATH | Execute using `python -m streamlit run app.py`. |
| PowerShell script execution error (`cannot be loaded because running scripts is disabled`) | Windows ExecutionPolicy restriction | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell. |
| Missing model artifact `models/diabetes_model.pkl` | First run on fresh clone | Run `python -m src.train` or start `streamlit run app.py` (which auto-trains missing artifacts). |

---

## 10. Limitations & Medical Disclaimer

> **⚠️ Medical Disclaimer:** This application is a machine-learning-based predictive model developed strictly for educational, research, and project-based learning (PBL) demonstration purposes. It is **not** a clinical diagnostic tool and does **not** provide professional medical advice, diagnosis, or treatment. Always consult a qualified physician or healthcare provider for medical concerns.

---

## 11. Reproducibility & Model Training Workflow

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
4. Train models, compare metrics, and select best pipeline:
   ```bash
   python -m src.train
   ```
   *(Model metrics will be saved to `reports/model_evaluation.json` and winning Random Forest pipeline to `models/diabetes_model.pkl`)*.
5. Run verification test suite:
   ```bash
   python -m pytest
   ```
6. Launch web interface:
   ```bash
   streamlit run app.py
   ```
