# One-Class Malignancy Anomaly & Reconstruction Engine

A production-grade Python inference pipeline and Streamlit web application for single-class breast oncology anomaly detection and manifold alignment.

---

## 🎯 Clinical & Algorithmic Paradigm

In traditional binary classification, models require balanced positive and negative classes. However, in specialized oncology triage and biopsy validation, training datasets often consist strictly of **confirmed malignant cases ($N=616$)**.

This system trains a deep generative manifold and density estimator on confirmed malignant cases:
- **Low Autoencoder Reconstruction Error $(x - \hat{x})^2$ + High Density Score:** Patient features closely follow the confirmed malignant manifold $\to$ **High Malignancy Likelihood / Elevated Triage Level (RED/YELLOW)**.
- **High Reconstruction Error + Low Density Score:** Patient features diverge from the malignant manifold $\to$ **Low Alignment / Likely Benign or Healthy Profile (GREEN)**.

---

## 🏗️ Repository Layout

```
├── data/
│   ├── raw/
│   │   └── malignant_breast_cancer_dataset.xlsx   # Confirmed malignant dataset (N=616)
│   └── processed/
│       └── processed_malignant_features.csv       # Standardized baseline features
├── models/
│   ├── autoencoder.pth                            # PyTorch Autoencoder weights & baseline MSE
│   ├── isolation_forest.pkl                       # Single-class Isolation Forest
│   ├── ocsvm.pkl                                  # One-Class SVM
│   ├── scaler.pkl                                 # StandardScaler artifact
│   ├── imputer.pkl                                # MICE IterativeImputer artifact
│   └── feature_meta.json                          # Feature centroids and scale parameters
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                           # Feature engineering & MICE imputation
│   ├── anomaly_models.py                          # PyTorch AE, IsolationForest, OneClassSVM
│   └── inference.py                               # Single-patient assessment & attribution
├── app/
│   └── streamlit_app.py                           # Interactive medical triage dashboard
├── requirements.txt
├── main.py                                        # CLI training, artifact generation & validation
└── README.md
```

---

## 🔬 Feature Engineering & Extracted Clinical Signals

From the raw 21 clinical attributes, the engine computes:
1. **`Symptom_Severity_Index`**: Integer sum ($0-5$) of binary primary breast symptoms:
   - `Breast Lump`
   - `Breast Swelling`
   - `Breast/Nipple pain`
   - `Nipple Discharge`
   - `Nipple retraction`
2. **`Metabolic_Risk_Score`**: Integer sum ($0-3$) of metabolic comorbidities:
   - `Hypertension`
   - `Diabetes`
   - `PUD (Peptic Ulcer Disease)`
3. **`Familial_History_Score`**: Integer sum ($0-2$) of family history flags:
   - `Family Hx of Breast Cancer`
   - `Family Hx of Other Cancers`
4. **`Diagnostic_Lag_Days`**: Absolute duration in days between `DATE OF REG` and `Diagnosis Date (main)`.
5. **MICE Imputation**: Multivariable imputation chained equations (`IterativeImputer`) handling missing clinical fields.

---

## 🚀 Quick Start

### 1. Installation
Using `uv` or `pip`:
```bash
# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Train Models & Validate Pipeline
Run the CLI orchestrator to process data, train the Autoencoder ($N=616$), fit Isolation Forest / OCSVM, and run sample validation:
```bash
python main.py
```

### 3. Launch Interactive Web Interface
Start the Streamlit oncology triage dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Triage Classifications

| Triage Level | Concordance % | Autoencoder MSE | Clinical Recommended Action |
|---|---|---|---|
| 🔴 **RED** | $\ge 70\%$ | Low ($\le P_{95}$) | **Priority Diagnostic Referral**: Fast-track imaging (Mammography / Ultrasound / BI-RADS) and urgent oncology consult. |
| 🟡 **YELLOW** | $40\% - 69\%$ | Moderate | **Further Evaluation Recommended**: Correlate with physical exam, ultrasound, and short-interval clinical review. |
| 🟢 **GREEN** | $< 40\%$ | High ($> P_{95}$) | **Low Alignment**: Routine age-appropriate screening and standard primary follow-up. |

---

## 🧪 Testing & Verification
Verify inference programmatically using the Python API:
```python
from src.inference import InferenceEngine

engine = InferenceEngine()

patient = {
    "Current Age": 58,
    "Sex": "Female",
    "Breast Lump": "Yes",
    "Breast Swelling": "Yes",
    "Breast/Nipple pain": "Yes",
    "Nipple Discharge": "No",
    "Nipple retraction": "Yes",
    "Hypertension": "Yes",
    "Diabetes": "No",
    "PUD": "No",
    "Family Hx of Breast Cancer": "Yes",
    "Family Hx of Other Cancers": "No",
    "Stage (main)": "Stage III",
    "Laterality": "Left",
    "DATE OF REG": "2024-01-10",
    "Diagnosis Date (main)": "2024-01-28",
}

assessment = engine.assess_patient(patient)
print(f"Concordance: {assessment['concordance_percentage']}%")
print(f"Triage: {assessment['triage_level']['label']}")
```
