"""
Model Manager for Neuro-Pulse Deepfake Detection.

Handles loading and managing all trained ML and DL models from OUTPUT_DIP_FINAL.
Provides unified interface for single model and ensemble predictions.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn

from src.models.dl_architectures import (
    DL_MODEL_CLASSES,
    DeepfakeCNN1D,
    DeepfakeBiLSTM,
    DeepfakeCNNBiLSTM,
    DeepfakeTransformer,
    PhysNetMLP,
)


# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "OUTPUT_DIP_FINAL" / "features"


class ModelManager:
    """
    Unified manager for all ML and DL models.

    Loads trained models from OUTPUT_DIP_FINAL/features and provides
    methods for prediction using individual models or ensembles.

    Attributes:
        ml_models: Dict of loaded scikit-learn models
        dl_models: Dict of loaded PyTorch models
        scaler: StandardScaler for feature normalization
        device: PyTorch device (cuda/cpu)
    """

    # ML model names (scikit-learn / XGBoost / LightGBM)
    ML_MODEL_NAMES = [
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "SVM_RBF",
        "GradientBoosting",
        "ExtraTrees",
        "LogisticRegression",
        "Ensemble_Top3",
    ]

    # DL model names (PyTorch)
    DL_MODEL_NAMES = [
        "CNN_1D",
        "BiLSTM_Attention",
        "CNN_BiLSTM",
        "Transformer",
        "PhysNet_MLP",
    ]

    def __init__(self, model_dir: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None):
        """
        Initialize ModelManager.

        Args:
            model_dir: Path to directory containing trained models.
                      Defaults to OUTPUT_DIP_FINAL/features.
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.ml_models: Dict[str, object] = {}
        self.dl_models: Dict[str, nn.Module] = {}
        self.scaler = None
        self.dl_scaler = None

        self._loaded = False

    def load_all(self, verbose: bool = True) -> None:
        """
        Load all available models and scalers.

        Args:
            verbose: Print loading progress
        """
        if verbose:
            print(f"[ModelManager] Loading models from: {self.model_dir}")
            print(f"[ModelManager] Device: {self.device}")

        # Load scalers
        self._load_scalers(verbose)

        # Load ML models
        self._load_ml_models(verbose)

        # Load DL models
        self._load_dl_models(verbose)

        self._loaded = True

        if verbose:
            print(f"[ModelManager] Loaded {len(self.ml_models)} ML models, "
                  f"{len(self.dl_models)} DL models")

    def _load_scalers(self, verbose: bool) -> None:
        """Load StandardScaler for feature normalization."""
        scaler_path = self.model_dir / "scaler.joblib"
        dl_scaler_path = self.model_dir / "dl_scaler.joblib"

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            if verbose:
                print(f"  [+] Loaded scaler.joblib")
        else:
            if verbose:
                print(f"  [-] scaler.joblib not found")

        if dl_scaler_path.exists():
            self.dl_scaler = joblib.load(dl_scaler_path)
            if verbose:
                print(f"  [+] Loaded dl_scaler.joblib")
        else:
            # Fall back to main scaler
            self.dl_scaler = self.scaler

    def _load_ml_models(self, verbose: bool) -> None:
        """Load all ML models (joblib files)."""
        for name in self.ML_MODEL_NAMES:
            model_path = self.model_dir / f"{name}_best.joblib"
            if model_path.exists():
                try:
                    self.ml_models[name] = joblib.load(model_path)
                    if verbose:
                        print(f"  [+] Loaded {name}")
                except Exception as e:
                    if verbose:
                        print(f"  [-] Failed to load {name}: {e}")
            else:
                if verbose:
                    print(f"  [-] {name}_best.joblib not found")

    def _load_dl_models(self, verbose: bool) -> None:
        """Load all DL models (PyTorch checkpoints)."""
        for name in self.DL_MODEL_NAMES:
            checkpoint_path = self.model_dir / f"{name}_checkpoint.pth"
            if checkpoint_path.exists():
                try:
                    # Create model instance
                    model_class = DL_MODEL_CLASSES[name]
                    model = model_class(n_features=35)

                    # Load checkpoint
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=False
                    )

                    # Load state dict
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)

                    model.to(self.device)
                    model.eval()
                    self.dl_models[name] = model

                    if verbose:
                        print(f"  [+] Loaded {name}")
                except Exception as e:
                    if verbose:
                        print(f"  [-] Failed to load {name}: {e}")
            else:
                if verbose:
                    print(f"  [-] {name}_checkpoint.pth not found")

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get lists of available ML and DL models.

        Returns:
            Dict with 'ml' and 'dl' lists
        """
        return {
            "ml": list(self.ml_models.keys()),
            "dl": list(self.dl_models.keys()),
        }

    def _ensure_loaded(self) -> None:
        """Ensure models are loaded."""
        if not self._loaded:
            self.load_all(verbose=False)

    def scale_features(self, features: np.ndarray, use_dl_scaler: bool = False) -> np.ndarray:
        """
        Scale features using the trained scaler.

        Args:
            features: Raw features (N, 35) or (35,)
            use_dl_scaler: Use DL scaler instead of ML scaler

        Returns:
            Scaled features
        """
        scaler = self.dl_scaler if use_dl_scaler else self.scaler

        if scaler is None:
            return features

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return scaler.transform(features)

    def predict_ml(self, features: np.ndarray, model_name: str = "Ensemble_Top3",
                   return_proba: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using an ML model.

        Args:
            features: Input features (N, 35) or (35,)
            model_name: Name of ML model to use
            return_proba: Return probabilities (if False, returns binary predictions)

        Returns:
            Tuple of (predictions, probabilities)
        """
        self._ensure_loaded()

        if model_name not in self.ml_models:
            raise ValueError(f"ML model '{model_name}' not loaded. "
                           f"Available: {list(self.ml_models.keys())}")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scale_features(features, use_dl_scaler=False)

        model = self.ml_models[model_name]

        # Get predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]

        return predictions, probabilities

    def predict_dl(self, features: np.ndarray, model_name: str = "CNN_BiLSTM",
                   return_proba: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using a DL model.

        Args:
            features: Input features (N, 35) or (35,)
            model_name: Name of DL model to use
            return_proba: Return probabilities

        Returns:
            Tuple of (predictions, probabilities)
        """
        self._ensure_loaded()

        if model_name not in self.dl_models:
            raise ValueError(f"DL model '{model_name}' not loaded. "
                           f"Available: {list(self.dl_models.keys())}")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features using DL scaler
        features_scaled = self.scale_features(features, use_dl_scaler=True)

        model = self.dl_models[model_name]

        # Convert to tensor
        with torch.no_grad():
            x = torch.FloatTensor(features_scaled).to(self.device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            predictions = preds.cpu().numpy()
            probabilities = probs[:, 1].cpu().numpy()

        return predictions, probabilities

    def predict_ensemble(self, features: np.ndarray,
                        models: Optional[List[str]] = None,
                        weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble prediction by averaging probabilities from multiple models.

        Args:
            features: Input features (N, 35) or (35,)
            models: List of model names to use. If None, uses top performers.
            weights: Optional weights for each model (must match length of models)

        Returns:
            Tuple of (predictions, ensemble_probabilities)
        """
        self._ensure_loaded()

        if models is None:
            # Default ensemble: top 3 ML + all DL
            models = ["Ensemble_Top3", "LogisticRegression", "CNN_BiLSTM",
                     "CNN_1D", "PhysNet_MLP"]

        # Filter to available models
        available = self.get_available_models()
        all_available = set(available["ml"]) | set(available["dl"])
        models = [m for m in models if m in all_available]

        if not models:
            raise ValueError("No models available for ensemble prediction")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        all_probs = []
        for model_name in models:
            if model_name in self.ml_models:
                _, probs = self.predict_ml(features, model_name)
            else:
                _, probs = self.predict_dl(features, model_name)
            all_probs.append(probs)

        # Stack and average
        all_probs = np.array(all_probs)

        if weights is not None:
            weights = np.array(weights).reshape(-1, 1)
            ensemble_probs = np.average(all_probs, axis=0, weights=weights.flatten())
        else:
            ensemble_probs = np.mean(all_probs, axis=0)

        predictions = (ensemble_probs >= 0.5).astype(int)

        return predictions, ensemble_probs

    def predict(self, features: np.ndarray,
               mode: str = "best",
               model_name: Optional[str] = None) -> Dict:
        """
        High-level prediction interface.

        Args:
            features: Input features (N, 35) or (35,)
            mode: Prediction mode
                  - "best": Use best performing ML model (Ensemble_Top3)
                  - "ml": Use specified ML model
                  - "dl": Use specified DL model
                  - "ensemble": Use multi-model ensemble
            model_name: Model name (required for 'ml' and 'dl' modes)

        Returns:
            Dict with keys:
                - prediction: Binary prediction (0=real, 1=fake)
                - probability: Fake class probability
                - label: "Real" or "Deepfake"
                - confidence: Confidence percentage
                - model: Model name used
        """
        self._ensure_loaded()

        if mode == "best":
            preds, probs = self.predict_ml(features, "Ensemble_Top3")
            model_used = "Ensemble_Top3"
        elif mode == "ml":
            if model_name is None:
                model_name = "Ensemble_Top3"
            preds, probs = self.predict_ml(features, model_name)
            model_used = model_name
        elif mode == "dl":
            if model_name is None:
                model_name = "CNN_BiLSTM"
            preds, probs = self.predict_dl(features, model_name)
            model_used = model_name
        elif mode == "ensemble":
            preds, probs = self.predict_ensemble(features)
            model_used = "Multi-Model Ensemble"
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'best', 'ml', 'dl', or 'ensemble'")

        # Handle batch vs single prediction
        is_single = len(preds) == 1

        if is_single:
            pred = int(preds[0])
            prob = float(probs[0])
            return {
                "prediction": pred,
                "probability": prob,
                "label": "Deepfake" if pred == 1 else "Real",
                "confidence": max(prob, 1 - prob) * 100,
                "model": model_used,
            }
        else:
            results = []
            for pred, prob in zip(preds, probs):
                results.append({
                    "prediction": int(pred),
                    "probability": float(prob),
                    "label": "Deepfake" if pred == 1 else "Real",
                    "confidence": max(prob, 1 - prob) * 100,
                    "model": model_used,
                })
            return {"results": results, "model": model_used}

    def predict_video_features(self, features: np.ndarray,
                               use_all_models: bool = False) -> Dict:
        """
        Comprehensive prediction on video features with optional multi-model analysis.

        Args:
            features: Feature vector(s) (35,) or (N, 35)
            use_all_models: Run prediction on all available models

        Returns:
            Dict with comprehensive prediction results
        """
        self._ensure_loaded()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        result = {
            "primary": self.predict(features, mode="best"),
            "ensemble": self.predict(features, mode="ensemble"),
        }

        if use_all_models:
            result["all_models"] = {}

            # ML models
            for name in self.ml_models.keys():
                preds, probs = self.predict_ml(features, name)
                result["all_models"][name] = {
                    "prediction": int(preds[0]),
                    "probability": float(probs[0]),
                    "type": "ML"
                }

            # DL models
            for name in self.dl_models.keys():
                preds, probs = self.predict_dl(features, name)
                result["all_models"][name] = {
                    "prediction": int(preds[0]),
                    "probability": float(probs[0]),
                    "type": "DL"
                }

        return result


# Singleton instance for convenience
_model_manager: Optional[ModelManager] = None


def get_model_manager(model_dir: Optional[str] = None,
                      device: Optional[str] = None) -> ModelManager:
    """
    Get or create the global ModelManager instance.

    Args:
        model_dir: Path to models (only used on first call)
        device: PyTorch device (only used on first call)

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model_dir=model_dir, device=device)
        _model_manager.load_all(verbose=True)
    return _model_manager
