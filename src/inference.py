"""
Module C: Generalized Inference Engine
Evaluates malignancy risk and reconstruction concordance for any future incoming patient.
Calculates reconstruction error, normalized malignancy concordance index, per-feature attribution,
and clinical triage recommendations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.anomaly_models import AutoencoderTrainer, MalignantStatisticalEnsemble
from src.preprocessing import MalignantDataPreprocessor


CLINICAL_FEATURE_LABELS: Dict[str, str] = {
    "Current Age": "Patient Age",
    "Sex_Female": "Female Biological Sex",
    "Smoking_Hx": "Smoking History",
    "Alcohol_Hx": "Alcohol Consumption",
    "Breast_Lump": "Palpable Breast Mass",
    "Breast_Swelling": "Localized Breast Swelling",
    "Breast_Nipple_Pain": "Breast / Nipple Pain",
    "Nipple_Discharge": "Nipple Discharge",
    "Nipple_Retraction": "Nipple / Skin Retraction",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes Mellitus",
    "PUD": "Peptic Ulcer History",
    "Family_Hx_Breast": "Familial Breast Cancer History",
    "Family_Hx_Other": "Familial Other Cancer History",
    "Stage_Numeric": "Tumor Clinical Stage",
    "Laterality_Bilateral": "Bilateral Presentation",
    "Symptom_Severity_Index": "Total Symptom Burden",
    "Metabolic_Risk_Score": "Metabolic Comorbidity Count",
    "Familial_History_Score": "Familial Risk Factors Count",
    "Diagnostic_Lag_Days": "Elapsed Presentation Days",
}


class InferenceEngine:
    """
    Production-grade clinical inference engine for evaluating patient alignment against
    the confirmed malignant breast cancer registry (616 reference cases).
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        device: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Preprocessor
        self.preprocessor = MalignantDataPreprocessor(artifacts_dir=self.models_dir)
        self.preprocessor.load_artifacts()

        # Load Autoencoder
        self.autoencoder_trainer = AutoencoderTrainer(
            input_dim=len(self.preprocessor.FEATURE_NAMES),
            latent_dim=4,
            device=self.device,
        )
        self.autoencoder_trainer.load_model(self.models_dir / "autoencoder.pth")

        # Load Statistical Models
        self.stats_ensemble = MalignantStatisticalEnsemble(artifacts_dir=self.models_dir)
        self.stats_ensemble.load_artifacts()

        # Baseline calibration statistics
        self.baseline_p95_mse = getattr(self.autoencoder_trainer, "baseline_p95_mse", 1.0)
        self.baseline_mean_mse = getattr(self.autoencoder_trainer, "baseline_mean_mse", 0.48)
        self.baseline_std_mse = getattr(self.autoencoder_trainer, "baseline_std_mse", 0.35)

    def _calculate_concordance(
        self,
        mse: float,
        iforest_score: float,
        ocsvm_score: float,
    ) -> Tuple[float, float, float]:
        """
        Convert pattern match metrics into a calibrated 0.00 - 1.00 Malignancy Concordance Index.
        """
        # 1. Pattern consistency score based on deviation from reference malignant registry
        z_mse = (mse - self.baseline_mean_mse) / (self.baseline_std_mse + 1e-6)
        ae_concordance = float(1.0 / (1.0 + np.exp(1.5 * (z_mse - 0.5))))
        ae_concordance = np.clip(ae_concordance, 0.01, 0.99)

        # 2. Multivariate pattern density score
        if_concordance = float(1.0 / (1.0 + np.exp(-15.0 * (iforest_score - 0.05))))
        if_concordance = np.clip(if_concordance, 0.01, 0.99)

        # 3. Clinical boundary alignment score
        oc_concordance = float(1.0 / (1.0 + np.exp(-6.0 * (ocsvm_score - 0.05))))
        oc_concordance = np.clip(oc_concordance, 0.01, 0.99)

        # Multi-model clinical ensemble (60% Deep Pattern Fit + 25% Density + 15% Boundary)
        combined_score = float(0.60 * ae_concordance + 0.25 * if_concordance + 0.15 * oc_concordance)
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        return combined_score, ae_concordance, if_concordance

    def _determine_triage_level(self, score: float) -> Tuple[str, str, str, str, str]:
        """
        Return Triage Level, clinician category, badge color, clinical summary, and next steps.
        """
        if score >= 0.70:
            return (
                "RED",
                "High Priority - High Malignancy Profile Match",
                "#EF4444",
                "Presentation strongly aligns with confirmed malignant clinical profiles in the reference registry.",
                "Fast-track diagnostic workup: Schedule diagnostic bilateral mammography, targeted ultrasound, and urgent surgical oncology consultation.",
            )
        elif score >= 0.40:
            return (
                "YELLOW",
                "Moderate Priority - Intermediate Clinical Match",
                "#F59E0B",
                "Presentation exhibits partial overlap with confirmed malignant profiles. Further clinical correlation required.",
                "Secondary diagnostic workup: Correlate with physical breast examination, breast ultrasound, and short-interval clinical review within 30 days.",
            )
        else:
            return (
                "GREEN",
                "Routine Care - Low Malignancy Profile Match",
                "#10B981",
                "Presentation departs significantly from confirmed malignant profiles and reflects low clinical alignment.",
                "Routine clinical pathway: Maintain age-appropriate screening guidelines and standard annual wellness monitoring.",
            )

    def assess_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient clinical profile and return clear, doctor-friendly triage analysis.
        """
        # Preprocess single patient
        scaled_vec, clean_df = self.preprocessor.transform_single_patient(patient_dict)

        # 1. Compute reconstruction errors and latent projection
        total_mse_arr, feat_errors_arr, latents_arr = self.autoencoder_trainer.compute_reconstruction_errors(scaled_vec)
        total_mse = float(total_mse_arr[0])
        feat_errors = feat_errors_arr[0]
        latent_4d = latents_arr[0].tolist()

        # 2. Compute statistical scores
        if_scores, oc_scores = self.stats_ensemble.score_samples(scaled_vec)
        iforest_score = float(if_scores[0])
        ocsvm_score = float(oc_scores[0])

        # 3. Malignancy Concordance & Clinical Triage Level
        concordance_score, ae_concordance, if_concordance = self._calculate_concordance(
            total_mse, iforest_score, ocsvm_score
        )
        triage_code, triage_label, triage_color, triage_summary, triage_action = self._determine_triage_level(concordance_score)

        # 4. Feature Attribution in Clear Clinical Terms
        feature_names = self.preprocessor.FEATURE_NAMES
        total_err_sum = float(np.sum(feat_errors)) if np.sum(feat_errors) > 0 else 1e-6
        feature_attributions: List[Dict[str, Any]] = []

        for idx, raw_name in enumerate(feature_names):
            err_val = float(feat_errors[idx])
            pct_contrib = float((err_val / total_err_sum) * 100.0)
            unscaled_val = clean_df.iloc[0][raw_name] if raw_name in clean_df.columns else None

            mean_baseline = self.preprocessor.feature_meta.get("feature_means", [0] * len(feature_names))[idx]
            clinical_label = CLINICAL_FEATURE_LABELS.get(raw_name, raw_name.replace("_", " "))

            # Human-readable patient value
            if raw_name in ["Breast_Lump", "Breast_Swelling", "Breast_Nipple_Pain", "Nipple_Discharge", "Nipple_Retraction", "Hypertension", "Diabetes", "PUD", "Family_Hx_Breast", "Family_Hx_Other", "Smoking_Hx", "Alcohol_Hx"]:
                val_display = "Present" if unscaled_val == 1 else "Absent"
                ref_display = f"{mean_baseline * 100:.0f}% in cohort"
            elif raw_name == "Sex_Female":
                val_display = "Female" if unscaled_val == 1 else "Male"
                ref_display = "99% Female cohort"
            elif raw_name == "Stage_Numeric":
                stage_map = {0: "Stage 0 (In Situ)", 1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}
                val_display = stage_map.get(int(round(unscaled_val or 2)), f"Stage {int(round(unscaled_val or 2))}")
                ref_display = "Median Stage II/III"
            elif raw_name == "Current Age":
                val_display = f"{int(round(unscaled_val or 50))} yrs"
                ref_display = f"{mean_baseline:.0f} yrs avg"
            elif raw_name == "Diagnostic_Lag_Days":
                val_display = f"{int(round(unscaled_val or 0))} days"
                ref_display = f"{mean_baseline:.0f} days avg"
            else:
                val_display = str(round(float(unscaled_val), 1)) if unscaled_val is not None else "N/A"
                ref_display = f"{mean_baseline:.1f} avg"

            feature_attributions.append({
                "feature": raw_name,
                "clinical_label": clinical_label,
                "reconstruction_error": round(err_val, 4),
                "pct_contribution": round(pct_contrib, 1),
                "patient_value": round(float(unscaled_val), 2) if unscaled_val is not None else None,
                "patient_display": val_display,
                "reference_cohort_mean": round(float(mean_baseline), 2),
                "reference_display": ref_display,
            })

        # Sort descending by influence
        feature_attributions.sort(key=lambda x: x["reconstruction_error"], reverse=True)

        return {
            "malignancy_concordance_score": round(concordance_score, 4),
            "concordance_percentage": round(concordance_score * 100.0, 1),
            "triage_level": {
                "code": triage_code,
                "label": triage_label,
                "color": triage_color,
                "summary": triage_summary,
                "action": triage_action,
            },
            "clinical_indices": {
                "profile_consistency_score": round((1.0 - np.clip(total_mse / (self.baseline_p95_mse * 1.5), 0, 1)) * 100, 1),
                "pattern_fit_level": "Strong Match" if total_mse <= self.baseline_mean_mse else "Moderate Match" if total_mse <= self.baseline_p95_mse else "Significant Deviation",
            },
            "metrics": {
                "autoencoder_mse": round(total_mse, 4),
                "isolation_forest_score": round(iforest_score, 4),
                "baseline_p95_mse": round(self.baseline_p95_mse, 4),
            },
            "latent_representation": [round(x, 4) for x in latent_4d],
            "feature_attributions": feature_attributions,
            "processed_patient_features": clean_df.iloc[0].to_dict(),
        }

    def assess_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a batch of patient clinical records for clinical coordinators and oncologists.
        """
        scaled_mat = self.preprocessor.transform(df_raw)
        total_mses, _, _ = self.autoencoder_trainer.compute_reconstruction_errors(scaled_mat)
        if_scores, oc_scores = self.stats_ensemble.score_samples(scaled_mat)

        results = []
        for i in range(len(df_raw)):
            score, ae_c, if_c = self._calculate_concordance(float(total_mses[i]), float(if_scores[i]), float(oc_scores[i]))
            code, label, color, summary, action = self._determine_triage_level(score)
            results.append({
                "Clinical_Match_%": round(score * 100.0, 1),
                "Triage_Priority": code,
                "Clinical_Assessment": label,
                "Recommended_Action": action,
            })

        res_df = pd.DataFrame(results, index=df_raw.index)
        return pd.concat([df_raw, res_df], axis=1)
