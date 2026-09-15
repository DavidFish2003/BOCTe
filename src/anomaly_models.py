"""
Module B: Anomaly Detection Architectures
PyTorch Autoencoder (Reconstruction Error Engine), Isolation Forest, and One-Class SVM
strictly trained on the confirmed malignant data manifold.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set random seeds across libraries for strict reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MalignantAutoencoder(nn.Module):
    """
    PyTorch Autoencoder tailored for reconstruction of malignant clinical profiles.
    Architecture:
      Encoder: Input Dim -> 16 -> 8 -> 4 (Bottleneck)
      Decoder: 4 -> 8 -> 16 -> Input Dim
    """

    def __init__(self, input_dim: int = 20, latent_dim: int = 4, dropout_rate: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder network: Input Dim -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim),
        )

        # Decoder network: 4 -> 8 -> 16 -> Input Dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent bottleneck embedding."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent bottleneck."""
        return self.decoder(z)


class AutoencoderTrainer:
    """
    Trainer and orchestrator for the PyTorch Malignant Autoencoder.
    """

    def __init__(
        self,
        input_dim: int = 20,
        latent_dim: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ):
        set_deterministic_seeds(42)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MalignantAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_history: Dict[str, list] = {"loss": []}
        self.baseline_mse_distribution: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        val_split: float = 0.1,
        patience: int = 25,
    ) -> MalignantAutoencoder:
        """
        Train the Autoencoder on the scaled malignant manifold.
        """
        set_deterministic_seeds(42)
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        val_size = int(n_samples * val_split)
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32) if val_size > 0 else None

        train_dataset = TensorDataset(X_tr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = running_loss / len(train_idx)
            self.training_history["loss"].append(epoch_train_loss)

            # Validation step
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_x = X_val.to(self.device)
                    val_recon, _ = self.model(val_x)
                    val_loss = criterion(val_recon, val_x).item()
                scheduler.step(val_loss)
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Compute baseline MSE distribution on all training malignant samples
        self.model.eval()
        with torch.no_grad():
            full_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            full_recon, _ = self.model(full_tensor)
            per_sample_mse = torch.mean((full_recon - full_tensor) ** 2, dim=1).cpu().numpy()
            self.baseline_mse_distribution = per_sample_mse
            self.baseline_mean_mse = float(np.mean(per_sample_mse))
            self.baseline_std_mse = float(np.std(per_sample_mse))
            self.baseline_p95_mse = float(np.percentile(per_sample_mse, 95))

        return self.model

    def compute_reconstruction_errors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate total MSE, per-feature squared error, and latent embeddings for inputs X.
        Returns:
          total_mse: (N,) array
          feature_errors: (N, D) array
          latents: (N, 4) array
        """
        self.model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon, latent = self.model(t_x)
            diff_sq = (recon - t_x) ** 2
            total_mse = torch.mean(diff_sq, dim=1).cpu().numpy()
            feature_errors = diff_sq.cpu().numpy()
            latents = latent.cpu().numpy()
        return total_mse, feature_errors, latents

    def save_model(self, path: Union[str, Path] = "models/autoencoder.pth") -> None:
        """Save PyTorch weights and baseline metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "baseline_mse_distribution": self.baseline_mse_distribution,
            "baseline_mean_mse": float(np.mean(self.baseline_mse_distribution)) if self.baseline_mse_distribution is not None else 0.0,
            "baseline_std_mse": float(np.std(self.baseline_mse_distribution)) if self.baseline_mse_distribution is not None else 1.0,
            "baseline_p95_mse": float(np.percentile(self.baseline_mse_distribution, 95)) if self.baseline_mse_distribution is not None else 1.0,
        }
        torch.save(save_payload, path)

    def load_model(self, path: Union[str, Path] = "models/autoencoder.pth") -> None:
        """Load PyTorch weights and metadata."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found.")
        payload = torch.load(path, map_location=self.device)
        self.input_dim = payload.get("input_dim", 20)
        self.latent_dim = payload.get("latent_dim", 4)
        self.model = MalignantAutoencoder(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.baseline_mse_distribution = payload.get("baseline_mse_distribution", None)
        self.baseline_mean_mse = payload.get("baseline_mean_mse", 0.0)
        self.baseline_std_mse = payload.get("baseline_std_mse", 1.0)
        self.baseline_p95_mse = payload.get("baseline_p95_mse", 1.0)
        self.model.eval()


class MalignantStatisticalEnsemble:
    """
    Trains and persists Isolation Forest and One-Class SVM on scaled malignant baseline features.
    """

    def __init__(self, artifacts_dir: Union[str, Path] = "models"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.iforest: Optional[IsolationForest] = None
        self.ocsvm: Optional[OneClassSVM] = None

    def fit(self, X_train: np.ndarray) -> MalignantStatisticalEnsemble:
        """Fit Isolation Forest and One-Class SVM on malignant training samples."""
        set_deterministic_seeds(42)

        # Isolation Forest (contamination is low because training set is 100% malignant)
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )
        self.iforest.fit(X_train)

        # One-Class SVM (RBF kernel, nu=0.05)
        self.ocsvm = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
        )
        self.ocsvm.fit(X_train)

        self.save_artifacts()
        return self

    def score_samples(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute decision scores for samples X.
        Higher score = deeper inlier / more concordant with malignant manifold.
        """
        if self.iforest is None or self.ocsvm is None:
            self.load_artifacts()

        # decision_function: positive indicates inlier, negative indicates outlier
        iforest_scores = self.iforest.decision_function(X)
        ocsvm_scores = self.ocsvm.decision_function(X)
        return iforest_scores, ocsvm_scores

    def save_artifacts(self) -> None:
        """Persist statistical models."""
        if self.iforest is not None:
            joblib.dump(self.iforest, self.artifacts_dir / "isolation_forest.pkl")
        if self.ocsvm is not None:
            joblib.dump(self.ocsvm, self.artifacts_dir / "ocsvm.pkl")

    def load_artifacts(self) -> None:
        """Load persisted models."""
        if_path = self.artifacts_dir / "isolation_forest.pkl"
        oc_path = self.artifacts_dir / "ocsvm.pkl"
        if not if_path.exists() or not oc_path.exists():
            raise FileNotFoundError(f"Statistical models not found in {self.artifacts_dir}.")
        self.iforest = joblib.load(if_path)
        self.ocsvm = joblib.load(oc_path)
