# Predictive Healthcare Diagnostic App (PHDA) - Diabetes Prediction MVP

A machine learning application for predicting diabetes risk using clinical health indicators from the PIMA Indians Diabetes dataset.

## Project Structure

```text
PHDA/
├── data/
│   ├── raw/          # Raw dataset files (git-ignored)
│   └── processed/    # Processed / cleaned datasets (git-ignored)
├── models/           # Trained model artifacts (git-ignored)
├── src/
│   ├── __init__.py
│   └── data_loader.py # Data acquisition & schema validation module
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py # Unit tests for data loading & schema validation
├── app.py            # Streamlit UI entry point (placeholder for upcoming UI stage)
├── requirements.txt  # Project dependencies
├── .gitignore        # Git ignore rules for data and model artifacts
└── README.md         # Project documentation
```

## Dataset Provenance & Metadata

- **Name:** Pima Indians Diabetes Database
- **Originating Agency:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Host / Source:** UCI Machine Learning Repository
- **Reproducible Acquisition URL:** `https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv`
- **Target Variable:** `Outcome` (Binary: `0` = non-diabetic, `1` = diabetic)

### Feature Schema & Column Definitions

| Column | Description | Data Type | Range / Type |
|---|---|---|---|
| `Pregnancies` | Number of times pregnant | Numeric (Integer) | $\ge 0$ |
| `Glucose` | Plasma glucose concentration at 2h in an oral glucose tolerance test | Numeric | $\ge 0$ |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Numeric | $\ge 0$ |
| `SkinThickness` | Triceps skin fold thickness (mm) | Numeric | $\ge 0$ |
| `Insulin` | 2-Hour serum insulin ($\mu\text{U/ml}$) | Numeric | $\ge 0$ |
| `BMI` | Body mass index ($\text{weight in kg}/(\text{height in m})^2$) | Numeric (Float) | $\ge 0.0$ |
| `DiabetesPedigreeFunction` | Genetic risk score based on family diabetes history | Numeric (Float) | $\ge 0.0$ |
| `Age` | Age in years | Numeric (Integer) | $> 0$ |
| `Outcome` | Target label for diabetes diagnosis | Categorical / Binary | `0` or `1` |

## Setup & Data Ingestion

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data acquisition & schema validation:**
   ```bash
   python -m src.data_loader
   ```
   This will download `diabetes.csv` into `data/raw/` if not present, validate schema integrity, and display summary statistics.

## Testing

Run automated schema validation and data loader tests:
```bash
pytest
```
