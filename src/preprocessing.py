"""
Module A: Data Alignment & Preprocessing
Handles raw clinical records (Excel/CSV), feature engineering, MICE imputation, and feature standardization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


class MalignantDataPreprocessor:
    """
    Robust clinical data preprocessor for single-class breast cancer malignancy analysis.
    Implements domain-specific clinical feature extraction, MICE imputation, and scaling.
    """

    EXPECTED_RAW_COLUMNS: List[str] = [
        "DATE OF REG",
        "Sex",
        "Current Age",
        "Family Hx of Breast Cancer",
        "Family Hx of Other Cancers",
        "Smoking Hx",
        "Alcohol History",
        "Breast Lump",
        "Breast Swelling",
        "Breast/Nipple pain",
        "Nipple Discharge",
        "Nipple retraction",
        "Diagnosis Description",
        "Diagnosis Date (main)",
        "Diagnosis Site",
        "Laterality",
        "Stage (main)",
        "Co-morbidities",
        "Hypertension",
        "Diabetes",
        "PUD",
    ]

    SYMPTOM_COLS: List[str] = [
        "Breast Lump",
        "Breast Swelling",
        "Breast/Nipple pain",
        "Nipple Discharge",
        "Nipple retraction",
    ]

    METABOLIC_COLS: List[str] = [
        "Hypertension",
        "Diabetes",
        "PUD",
    ]

    FAMILIAL_COLS: List[str] = [
        "Family Hx of Breast Cancer",
        "Family Hx of Other Cancers",
    ]

    FEATURE_NAMES: List[str] = [
        "Current Age",
        "Sex_Female",
        "Smoking_Hx",
        "Alcohol_Hx",
        "Breast_Lump",
        "Breast_Swelling",
        "Breast_Nipple_Pain",
        "Nipple_Discharge",
        "Nipple_Retraction",
        "Hypertension",
        "Diabetes",
        "PUD",
        "Family_Hx_Breast",
        "Family_Hx_Other",
        "Stage_Numeric",
        "Laterality_Bilateral",
        "Symptom_Severity_Index",
        "Metabolic_Risk_Score",
        "Familial_History_Score",
        "Diagnostic_Lag_Days",
    ]

    def __init__(self, artifacts_dir: Union[str, Path] = "models"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[IterativeImputer] = None
        self.is_fitted: bool = False
        self.feature_meta: Dict[str, Any] = {}

    @staticmethod
    def _parse_binary_value(val: Any) -> Optional[int]:
        """Convert various boolean/string representations to 1, 0, or None."""
        if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", "", "null", "unknown", "?"]:
            return None
        s = str(val).strip().lower()
        if s in ["1", "1.0", "yes", "y", "true", "t", "positive", "pos", "present"]:
            return 1
        if s in ["0", "0.0", "no", "n", "false", "f", "negative", "neg", "absent"]:
            return 0
        try:
            num = float(val)
            return 1 if num > 0.5 else 0
        except Exception:
            return None

    @staticmethod
    def _parse_sex(val: Any) -> Optional[int]:
        """1 for Female, 0 for Male, None if missing."""
        if pd.isna(val) or val is None:
            return None
        s = str(val).strip().lower()
        if s in ["f", "female", "woman", "w", "1", "1.0"]:
            return 1
        if s in ["m", "male", "man", "0", "0.0"]:
            return 0
        return None

    @staticmethod
    def _parse_stage(val: Any) -> Optional[float]:
        """Convert clinical stage string/number to ordinal numeric value (0 to 4)."""
        if pd.isna(val) or val is None:
            return None
        s = str(val).strip().upper()
        if "IV" in s or "4" in s:
            return 4.0
        if "III" in s or "3" in s:
            return 3.0
        if "II" in s or "2" in s:
            return 2.0
        if "I" in s or "1" in s:
            return 1.0
        if "0" in s or "IN SITU" in s or "CIS" in s:
            return 0.0
        return 2.0  # Default median stage for malignant presentation if unspecified

    @staticmethod
    def _parse_laterality(val: Any) -> Optional[int]:
        """1 for Bilateral, 0 for Unilateral/Left/Right."""
        if pd.isna(val) or val is None:
            return None
        s = str(val).strip().lower()
        if "bilateral" in s or "both" in s or s == "2":
            return 1
        return 0

    def parse_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer structured features from raw clinical DataFrame.
        """
        # Create a standardized copy
        clean_df = pd.DataFrame(index=df.index)

        # 1. Current Age
        if "Current Age" in df.columns:
            clean_df["Current Age"] = pd.to_numeric(df["Current Age"], errors="coerce")
        else:
            clean_df["Current Age"] = np.nan

        # 2. Demographics & Lifestyle
        clean_df["Sex_Female"] = df["Sex"].apply(self._parse_sex) if "Sex" in df.columns else 1
        clean_df["Smoking_Hx"] = df["Smoking Hx"].apply(self._parse_binary_value) if "Smoking Hx" in df.columns else 0
        clean_df["Alcohol_Hx"] = df["Alcohol History"].apply(self._parse_binary_value) if "Alcohol History" in df.columns else 0

        # 3. Primary Symptoms
        clean_df["Breast_Lump"] = df["Breast Lump"].apply(self._parse_binary_value) if "Breast Lump" in df.columns else np.nan
        clean_df["Breast_Swelling"] = df["Breast Swelling"].apply(self._parse_binary_value) if "Breast Swelling" in df.columns else np.nan
        clean_df["Breast_Nipple_Pain"] = df["Breast/Nipple pain"].apply(self._parse_binary_value) if "Breast/Nipple pain" in df.columns else np.nan
        clean_df["Nipple_Discharge"] = df["Nipple Discharge"].apply(self._parse_binary_value) if "Nipple Discharge" in df.columns else np.nan
        clean_df["Nipple_Retraction"] = df["Nipple retraction"].apply(self._parse_binary_value) if "Nipple retraction" in df.columns else np.nan

        # 4. Metabolic Comorbidities
        clean_df["Hypertension"] = df["Hypertension"].apply(self._parse_binary_value) if "Hypertension" in df.columns else np.nan
        clean_df["Diabetes"] = df["Diabetes"].apply(self._parse_binary_value) if "Diabetes" in df.columns else np.nan
        clean_df["PUD"] = df["PUD"].apply(self._parse_binary_value) if "PUD" in df.columns else np.nan

        # 5. Familial History
        clean_df["Family_Hx_Breast"] = df["Family Hx of Breast Cancer"].apply(self._parse_binary_value) if "Family Hx of Breast Cancer" in df.columns else np.nan
        clean_df["Family_Hx_Other"] = df["Family Hx of Other Cancers"].apply(self._parse_binary_value) if "Family Hx of Other Cancers" in df.columns else np.nan

        # 6. Diagnosis Characteristics
        clean_df["Stage_Numeric"] = df["Stage (main)"].apply(self._parse_stage) if "Stage (main)" in df.columns else np.nan
        clean_df["Laterality_Bilateral"] = df["Laterality"].apply(self._parse_laterality) if "Laterality" in df.columns else 0

        # 7. Domain Score Engineering
        # Symptom Severity Index: sum of 5 primary symptoms
        symp_cols = ["Breast_Lump", "Breast_Swelling", "Breast_Nipple_Pain", "Nipple_Discharge", "Nipple_Retraction"]
        clean_df["Symptom_Severity_Index"] = clean_df[symp_cols].fillna(0).sum(axis=1)

        # Metabolic Risk Score: sum of comorbidities
        met_cols = ["Hypertension", "Diabetes", "PUD"]
        clean_df["Metabolic_Risk_Score"] = clean_df[met_cols].fillna(0).sum(axis=1)

        # Familial History Score: sum of family cancer flags
        fam_cols = ["Family_Hx_Breast", "Family_Hx_Other"]
        clean_df["Familial_History_Score"] = clean_df[fam_cols].fillna(0).sum(axis=1)

        # Diagnostic Lag Days
        if "DATE OF REG" in df.columns and "Diagnosis Date (main)" in df.columns:
            date_reg = pd.to_datetime(df["DATE OF REG"], errors="coerce")
            date_diag = pd.to_datetime(df["Diagnosis Date (main)"], errors="coerce")
            lag = (date_diag - date_reg).dt.days.abs()
            clean_df["Diagnostic_Lag_Days"] = lag
        else:
            clean_df["Diagnostic_Lag_Days"] = np.nan

        # Ensure order matches FEATURE_NAMES
        return clean_df[self.FEATURE_NAMES]

    def fit(self, raw_df: pd.DataFrame) -> MalignantDataPreprocessor:
        """
        Fit MICE iterative imputer and standard scaler on raw malignant training dataset.
        """
        clean_df = self.parse_raw_dataframe(raw_df)

        # Fit MICE imputer
        self.imputer = IterativeImputer(
            max_iter=20,
            random_state=42,
            initial_strategy="median",
            min_value=0.0,
            verbose=0,
        )
        imputed_matrix = self.imputer.fit_transform(clean_df.values)

        # Fit Standard Scaler
        self.scaler = StandardScaler()
        scaled_matrix = self.scaler.fit_transform(imputed_matrix)

        self.is_fitted = True

        # Store metadata for introspection
        self.feature_meta = {
            "feature_names": self.FEATURE_NAMES,
            "feature_means": self.scaler.mean_.tolist(),
            "feature_scales": self.scaler.scale_.tolist(),
            "feature_vars": self.scaler.var_.tolist(),
            "num_samples_fitted": int(len(raw_df)),
        }

        # Save artifacts
        self.save_artifacts()
        return self

    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw patient clinical DataFrame to imputed, standardized numpy array.
        """
        if not self.is_fitted:
            self.load_artifacts()

        clean_df = self.parse_raw_dataframe(raw_df)
        imputed = self.imputer.transform(clean_df.values)
        scaled = self.scaler.transform(imputed)
        return scaled

    def transform_single_patient(self, patient_dict: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transform a single raw patient dictionary into a scaled 1xD numpy vector
        along with the extracted unscaled feature DataFrame.
        """
        df_single = pd.DataFrame([patient_dict])
        clean_df = self.parse_raw_dataframe(df_single)

        if not self.is_fitted:
            self.load_artifacts()

        imputed = self.imputer.transform(clean_df.values)
        scaled = self.scaler.transform(imputed)
        clean_imputed_df = pd.DataFrame(imputed, columns=self.FEATURE_NAMES)
        return scaled, clean_imputed_df

    def inverse_transform(self, scaled_matrix: np.ndarray) -> np.ndarray:
        """Convert scaled array back to original feature space."""
        if not self.is_fitted:
            self.load_artifacts()
        return self.scaler.inverse_transform(scaled_matrix)

    def save_artifacts(self) -> None:
        """Persist scaler, imputer, and feature metadata to models directory."""
        if self.scaler is not None:
            joblib.dump(self.scaler, self.artifacts_dir / "scaler.pkl")
        if self.imputer is not None:
            joblib.dump(self.imputer, self.artifacts_dir / "imputer.pkl")
        with open(self.artifacts_dir / "feature_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_meta, f, indent=2)

    def load_artifacts(self) -> None:
        """Load persisted preprocessing artifacts from disk."""
        scaler_path = self.artifacts_dir / "scaler.pkl"
        imputer_path = self.artifacts_dir / "imputer.pkl"
        meta_path = self.artifacts_dir / "feature_meta.json"

        if not scaler_path.exists() or not imputer_path.exists():
            raise FileNotFoundError(
                f"Preprocessing artifacts not found in {self.artifacts_dir}. "
                "Please run main.py first to fit and save the baseline models."
            )

        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.feature_meta = json.load(f)

        self.is_fitted = True


def load_raw_dataset(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load raw malignant clinical dataset from Excel (.xlsx/.xls) or CSV.
    If no path is provided, checks data/raw/ directory automatically.
    """
    if data_path is not None:
        target = Path(data_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found at {data_path}")
        if target.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(target)
        return pd.read_csv(target)

    # Search default directories
    candidates = [
        Path("data/raw/malignant_breast_cancer_dataset.xlsx"),
        Path("data/raw/malignant_breast_cancer_dataset.xls"),
        Path("data/raw/malignant_breast_cancer_dataset.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix.lower() in [".xlsx", ".xls"]:
                return pd.read_excel(cand)
            return pd.read_csv(cand)

    # Search any file in data/raw
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.suffix.lower() in [".xlsx", ".xls"]:
                return pd.read_excel(f)
            if f.suffix.lower() == ".csv":
                return pd.read_csv(f)

    raise FileNotFoundError("No dataset found in data/raw/. Please provide a clinical dataset file.")
