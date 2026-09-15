"""
Unit and Integration Tests for One-Class Malignancy Anomaly & Reconstruction Engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch

from src.anomaly_models import AutoencoderTrainer, MalignantAutoencoder, MalignantStatisticalEnsemble
from src.inference import InferenceEngine
from src.preprocessing import MalignantDataPreprocessor, load_raw_dataset


def test_dataset_loading_and_shape():
    """Verify raw dataset exists and has exactly 616 rows."""
    df = load_raw_dataset()
    assert len(df) == 616, f"Expected 616 samples, got {len(df)}"
    assert "Current Age" in df.columns
    assert "Stage (main)" in df.columns


def test_preprocessor_transformation():
    """Test feature extraction, MICE imputation, and scaling."""
    preprocessor = MalignantDataPreprocessor(artifacts_dir="models")
    preprocessor.load_artifacts()
    assert preprocessor.is_fitted

    sample_patient = {
        "Current Age": 52,
        "Sex": "Female",
        "Breast Lump": "Yes",
        "Breast Swelling": "No",
        "Breast/Nipple pain": "Yes",
        "Nipple Discharge": "No",
        "Nipple retraction": "No",
        "Hypertension": "Yes",
        "Diabetes": "No",
        "PUD": "No",
        "Family Hx of Breast Cancer": "Yes",
        "Family Hx of Other Cancers": "No",
        "Stage (main)": "Stage II",
        "Laterality": "Left",
        "DATE OF REG": "2024-01-01",
        "Diagnosis Date (main)": "2024-01-15",
    }

    scaled_vec, clean_df = preprocessor.transform_single_patient(sample_patient)
    assert scaled_vec.shape == (1, 20)
    assert clean_df["Symptom_Severity_Index"].iloc[0] == 2.0  # Lump + Pain
    assert clean_df["Metabolic_Risk_Score"].iloc[0] == 1.0   # Hypertension
    assert clean_df["Familial_History_Score"].iloc[0] == 1.0 # Family Hx Breast
    assert clean_df["Diagnostic_Lag_Days"].iloc[0] == 14.0   # 14 days


def test_autoencoder_architecture_and_forward():
    """Verify autoencoder dimensions and forward pass."""
    model = MalignantAutoencoder(input_dim=20, latent_dim=4)
    dummy_input = torch.randn(8, 20)
    recon, latent = model(dummy_input)
    assert recon.shape == (8, 20)
    assert latent.shape == (8, 4)


def test_inference_engine_assessment():
    """Verify end-to-end inference evaluation across different patient profiles."""
    engine = InferenceEngine(models_dir="models")

    # 1. High risk / confirmed malignant profile
    malignant_case = {
        "Current Age": 60,
        "Sex": "Female",
        "Breast Lump": "Yes",
        "Breast Swelling": "Yes",
        "Breast/Nipple pain": "Yes",
        "Nipple Discharge": "No",
        "Nipple retraction": "Yes",
        "Hypertension": "Yes",
        "Diabetes": "Yes",
        "PUD": "No",
        "Family Hx of Breast Cancer": "Yes",
        "Family Hx of Other Cancers": "No",
        "Stage (main)": "Stage III",
        "Laterality": "Left",
        "DATE OF REG": "2024-01-01",
        "Diagnosis Date (main)": "2024-01-20",
    }
    res_m = engine.assess_patient(malignant_case)
    assert res_m["concordance_percentage"] >= 70.0
    assert res_m["triage_level"]["code"] == "RED"
    assert len(res_m["feature_attributions"]) == 20
    assert "Nipple_Retraction" in [f["feature"] for f in res_m["feature_attributions"][:5]]

    # 2. Asymptomatic / low risk screening profile
    healthy_case = {
        "Current Age": 22,
        "Sex": "Female",
        "Breast Lump": "No",
        "Breast Swelling": "No",
        "Breast/Nipple pain": "No",
        "Nipple Discharge": "No",
        "Nipple retraction": "No",
        "Hypertension": "No",
        "Diabetes": "No",
        "PUD": "No",
        "Family Hx of Breast Cancer": "No",
        "Family Hx of Other Cancers": "No",
        "Stage (main)": "Stage 0",
        "Laterality": "Left",
        "DATE OF REG": "2024-01-01",
        "Diagnosis Date (main)": "2024-01-02",
    }
    res_h = engine.assess_patient(healthy_case)
    assert res_h["concordance_percentage"] < 40.0
    assert res_h["triage_level"]["code"] == "GREEN"
    assert res_h["metrics"]["autoencoder_mse"] > res_m["metrics"]["autoencoder_mse"]


def test_batch_assessment():
    """Verify batch cohort processing with clinical column labels."""
    engine = InferenceEngine(models_dir="models")
    raw_df = load_raw_dataset()
    sample_df = raw_df.head(10)
    batch_res = engine.assess_batch(sample_df)
    assert len(batch_res) == 10
    assert "Clinical_Match_%" in batch_res.columns
    assert "Triage_Priority" in batch_res.columns
    assert "Clinical_Assessment" in batch_res.columns
    assert "Recommended_Action" in batch_res.columns


if __name__ == "__main__":
    test_dataset_loading_and_shape()
    print("[PASS] test_dataset_loading_and_shape")
    test_preprocessor_transformation()
    print("[PASS] test_preprocessor_transformation")
    test_autoencoder_architecture_and_forward()
    print("[PASS] test_autoencoder_architecture_and_forward")
    test_inference_engine_assessment()
    print("[PASS] test_inference_engine_assessment")
    test_batch_assessment()
    print("[PASS] test_batch_assessment")
    print("\nALL TESTS PASSED SUCCESSFULLY!")

