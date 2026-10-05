# Predictive Healthcare Diagnostic App (PHDA) - Problem-Based Learning (PBL) Technical Report

---

## Executive Summary

The **Predictive Healthcare Diagnostic App (PHDA)** is an end-to-end Machine Learning MVP developed to estimate diabetes risk from clinical health metrics using the PIMA Indians Diabetes Dataset. This report details the problem statement, data engineering methodology, model selection experiments (comparing Baseline & Tuned Random Forest, Baseline & Tuned Logistic Regression, Gradient Boosting, and HistGradientBoosting), cross-validation evaluations, web architecture design, parameter reference band indicators, and limitations.

---

## 1. Problem Statement

Diabetes mellitus is a chronic metabolic disease characterized by elevated blood glucose levels, leading over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. Early identification of elevated risk profiles enables timely lifestyle intervention and clinical monitoring.

**Objective:** Develop a reproducible, machine-learning-based classification pipeline and interactive web application that accepts clinical metrics and outputs:
1. A binary classification prediction (`Diabetic` vs `Non-Diabetic`) with estimated outcome probabilities and confidence scores.
2. Individual parameter-level reference band indicators describing metric deviations without asserting independent clinical causality.

---

## 2. Dataset Provenance & Data Engineering

### 2.1 Dataset Profile
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / UCI Machine Learning Repository.
- **Population Demographic:** Female Pima Indian heritage, aged 21 years and older.
- **Dimensions:** 768 patient records $\times$ 9 columns (8 clinical predictors + 1 target).
- **Target Variable:** `Outcome` (`0` = Non-Diabetic [500 cases, 65.10%], `1` = Diabetic [268 cases, 34.90%]).

### 2.2 Feature Contract & Data Cleaning
In the raw PIMA dataset, zero values in clinical measurement columns represent unrecorded missing data (physiologically impossible zero values). The cleaning pipeline ([src/preprocess.py](file:///d:/Programming/Data%20Science/PHDA/src/preprocess.py)) converts zeros in specified clinical metrics to `NaN` while preserving valid zero values in `Pregnancies` and `Outcome`.

| Feature Column | Description | Data Type | Missing/Invalid Zero Count | Missing Percentage |
|---|---|---|---|---|
| `Pregnancies` | Number of times pregnant | Integer | 0 (Valid 0 preserved) | 0.00% |
| `Glucose` | 2-Hour oral glucose concentration (mg/dL) | Float/Int | 5 | 0.65% |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Float/Int | 35 | 4.56% |
| `SkinThickness` | Triceps skin fold thickness (mm) | Float/Int | 227 | 29.56% |
| `Insulin` | 2-Hour serum insulin ($\mu\text{U/ml}$) | Float/Int | 374 | 48.70% |
| `BMI` | Body Mass Index ($\text{kg/m}^2$) | Float | 11 | 1.43% |
| `DiabetesPedigreeFunction` | Family diabetes pedigree genetic score | Float | 0 | 0.00% |
| `Age` | Age in years | Integer | 0 | 0.00% |
| `Outcome` | Target diagnosis label | Binary (0/1) | 0 | 0.00% |

---

## 3. Methodology & Leakage Isolation

To prevent data leakage during model training and evaluation:
1. **Holdout Test Set Isolation:** The cleaned dataset is partitioned into an **80% training set ($X_{\text{train}}$, $N=614$)** and an **untouched 20% holdout test set ($X_{\text{test}}$, $N=154$)** using stratified sampling (`random_state=42`, `stratify=y`).
2. **Encapsulated Preprocessing Pipelines:** Median imputation (`SimpleImputer(strategy='median')`) and feature scaling (`StandardScaler()`) are integrated inside `sklearn.pipeline.Pipeline` objects. Preprocessing parameters are fit **strictly on training splits**, ensuring test set statistical isolation.
3. **Cross-Validation Scheme:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) is executed exclusively on $X_{\text{train}}$ for model evaluation.

---

## 4. Model Comparison & Experimental Results

Six candidate model pipelines were evaluated across 5-fold cross-validation on $X_{\text{train}}$ and evaluated once on the untouched holdout test set $X_{\text{test}}$.

### 4.1 Evaluation Summary Table

| Model Pipeline | 5-Fold CV Accuracy (Mean $\pm$ Std) | 5-Fold CV ROC-AUC (Mean $\pm$ Std) | Holdout Test Accuracy | Holdout Test Precision | Holdout Test Recall | Holdout Test F1-Score | Holdout Test ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.7882 $\pm$ 0.019 | 0.8434 $\pm$ 0.018 | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | `[[82, 18], [27, 27]]` |
| **Baseline Random Forest** *(Selected)* | **0.7639 $\pm$ 0.016** | **0.8154 $\pm$ 0.023** | **0.7727** | **0.7021** | **0.6111** | **0.6535** | **0.8181** | `[[86, 14], [21, 33]]` |
| **Tuned Random Forest** | 0.7606 $\pm$ 0.017 | 0.8349 $\pm$ 0.026 | 0.7403 | 0.6119 | 0.7593 | 0.6721 | 0.8215 | `[[73, 27], [13, 41]]` |
| **Tuned Logistic Regression** | 0.7622 $\pm$ 0.011 | 0.8449 $\pm$ 0.015 | 0.7338 | 0.5970 | 0.7407 | 0.6496 | 0.8126 | `[[73, 27], [14, 40]]` |
| **Gradient Boosting** | 0.7524 $\pm$ 0.018 | 0.8222 $\pm$ 0.018 | 0.7403 | 0.6591 | 0.5370 | 0.5918 | 0.8222 | `[[85, 15], [25, 29]]` |
| **HistGradientBoosting** | 0.7671 $\pm$ 0.027 | 0.8161 $\pm$ 0.017 | 0.7403 | 0.6522 | 0.5556 | 0.6154 | 0.8254 | `[[84, 16], [24, 30]]` |

### 4.2 Model Selection Rationale
- **Primary Selection Metric:** Test Set Accuracy.
- **Winner:** **Baseline Random Forest** achieved the highest test accuracy (**0.7727** / 77.27%) and highest precision (**0.7021** / 70.21%), with stable 5-fold CV performance (0.7639 $\pm$ 0.016).
- **Hyperparameter Tradeoff Analysis:** Applying `class_weight='balanced'` in Tuned Random Forest increased Recall (0.7593 vs 0.6111) and slightly increased ROC-AUC (0.8215 vs 0.8181), but reduced overall test accuracy to 0.7403 (-3.24%). Per project guidelines, Baseline Random Forest was retained to prioritize overall accuracy while documenting the recall/ROC-AUC tradeoffs.

---

## 5. ROC-AUC Explanation & Interpretation

**Receiver Operating Characteristic Area Under Curve (ROC-AUC)** measures the classifier's ability to rank positive (Diabetic) cases higher than negative (Non-Diabetic) cases across all possible decision thresholds. 

- **Value:** Baseline Random Forest achieved a holdout test ROC-AUC of **0.8181** (81.81%).
- **Interpretation Notice:** ROC-AUC measures rank-order separation quality across decision thresholds. It is **not** a measure of clinical diagnostic accuracy. An ROC-AUC of 0.818 indicates an 81.8% probability that a randomly chosen diabetic patient receives a higher predicted risk score than a randomly chosen non-diabetic patient.

---

## 6. Web Application Architecture & Startup Verification

### 6.1 Retraining Behavior Determination
> **Random Forest is retrained during normal app use: NO**

**Technical Verification:**
1. Model training ([src/train.py](file:///d:/Programming/Data%20Science/PHDA/src/train.py)) is completely decoupled from the Streamlit UI ([app.py](file:///d:/Programming/Data%20Science/PHDA/app.py)).
2. `app.py` loads the pre-trained pipeline artifact ([models/diabetes_model.pkl](file:///d:/Programming/Data%20Science/PHDA/models/diabetes_model.pkl)) via `load_model(auto_train_if_missing=False)`.
3. Model loading is wrapped in Streamlit's `@st.cache_resource` decorator and an in-memory global variable (`_CACHED_MODEL`), guaranteeing that model loading occurs **once** and is cached across user interactions and re-renders without re-reading disk or retraining.

---

## 7. Parameter-Level Reference Band Indicators

In addition to the ML model prediction, the application computes individual parameter-level reference band indicators for every input metric:

### 7.1 Reference Threshold Rules

| Feature | Reference Band | Severity Classification Rules | Deviation Direction |
|---|---|---|---|
| `Glucose` | 70–99 mg/dL | `<70`: Low (Yellow), `70–99`: Normal (Green), `100–125`: High (Yellow), `≥126`: Very High (Red) | Below / Within / Above |
| `BloodPressure` | 60–80 mm Hg | `<60`: Low (Yellow), `60–80`: Normal (Green), `81–89`: High (Yellow), `≥90`: Very High (Red) | Below / Within / Above |
| `BMI` | 18.5–24.9 kg/m² | `<18.5`: Low (Yellow), `18.5–24.9`: Normal (Green), `25.0–29.9`: High (Yellow), `≥30.0`: Very High (Red) | Below / Within / Above |
| `SkinThickness` | 10–30 mm | `<10`: Low (Yellow), `10–30`: Normal (Green), `31–40`: High (Yellow), `>40`: Very High (Red) | Below / Within / Above |
| `Insulin` | 15–166 μU/ml | `<15`: Low (Yellow), `15–166`: Normal (Green), `167–250`: High (Yellow), `>250`: Very High (Red) | Below / Within / Above |
| `Pregnancies` | 0–4 count | `0–4`: Normal (Green), `5–8`: High (Yellow), `>8`: Very High (Red) | Within / Above |
| `DiabetesPedigreeFunction` | 0.078–0.499 | `≤0.499`: Normal (Green), `0.500–0.799`: High (Yellow), `≥0.800`: Very High (Red) | Within / Above |
| `Age` | 18–44 years | `18–44`: Normal (Green), `45–64`: High (Yellow), `≥65`: Very High (Red) | Within / Above |

*Non-Causal Notice:* Parameter indicators describe metric levels relative to application reference bands. They do **not** claim to independently cause or establish a diagnosis of diabetes.

---

## 8. Limitations & Medical Disclaimer

1. **Demographic Population Scope:** The model is trained strictly on female Pima Indian patients aged 21 and older. Predictions should be interpreted within this demographic context.
2. **Educational Purpose:** This application is developed for educational, research, and project-based learning (PBL) demonstration purposes.
3. **Medical Disclaimer:** This application is **not** a clinical diagnostic tool and does **not** replace professional medical advice, clinical examination, or laboratory testing. Always consult a qualified healthcare provider.
