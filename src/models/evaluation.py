"""
Model Evaluation Module for Neuro-Pulse Deepfake Detection.

Provides utilities for evaluating ML/DL models on test sets,
computing metrics, and generating performance reports.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from src.models.model_manager import ModelManager


class ModelEvaluator:
    """
    Evaluates ML and DL models on test data.

    Computes comprehensive metrics and generates reports.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Initialize evaluator.

        Args:
            model_manager: ModelManager instance (or creates one)
        """
        if model_manager is None:
            model_manager = ModelManager()
            model_manager.load_all(verbose=False)
        self.model_manager = model_manager

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        model_type: str = "auto",
    ) -> Dict:
        """
        Evaluate a single model on test data.

        Args:
            X_test: Test features (N, 35)
            y_test: Test labels (N,)
            model_name: Name of the model
            model_type: "ml", "dl", or "auto" (detect automatically)

        Returns:
            Dict with evaluation metrics
        """
        if model_type == "auto":
            if model_name in self.model_manager.ml_models:
                model_type = "ml"
            elif model_name in self.model_manager.dl_models:
                model_type = "dl"
            else:
                raise ValueError(f"Model '{model_name}' not found")

        # Get predictions
        if model_type == "ml":
            preds, probs = self.model_manager.predict_ml(X_test, model_name)
        else:
            preds, probs = self.model_manager.predict_dl(X_test, model_name)

        return self._compute_metrics(y_test, preds, probs, model_name)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
    ) -> Dict:
        """Compute all evaluation metrics."""
        return {
            "model": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            "average_precision": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(np.sum(y_true == 0)),
        }

    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluate all available models on test data.

        Args:
            X_test: Test features (N, 35)
            y_test: Test labels (N,)

        Returns:
            DataFrame with metrics for all models
        """
        results = []

        # Evaluate ML models
        for name in self.model_manager.ml_models.keys():
            try:
                metrics = self.evaluate_model(X_test, y_test, name, "ml")
                metrics["type"] = "ML"
                results.append(metrics)
            except Exception as e:
                print(f"[WARN] Failed to evaluate {name}: {e}")

        # Evaluate DL models
        for name in self.model_manager.dl_models.keys():
            try:
                metrics = self.evaluate_model(X_test, y_test, name, "dl")
                metrics["type"] = "DL"
                results.append(metrics)
            except Exception as e:
                print(f"[WARN] Failed to evaluate {name}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("auc", ascending=False)
        return df

    def print_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        model_type: str = "auto",
    ) -> str:
        """
        Print detailed classification report for a model.

        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Model name
            model_type: "ml", "dl", or "auto"

        Returns:
            Classification report string
        """
        if model_type == "auto":
            if model_name in self.model_manager.ml_models:
                model_type = "ml"
            else:
                model_type = "dl"

        if model_type == "ml":
            preds, probs = self.model_manager.predict_ml(X_test, model_name)
        else:
            preds, probs = self.model_manager.predict_dl(X_test, model_name)

        report = classification_report(
            y_test, preds, target_names=["Real", "Deepfake"]
        )

        print(f"\n{'=' * 60}")
        print(f"CLASSIFICATION REPORT: {model_name}")
        print(f"{'=' * 60}")
        print(report)
        print(f"AUC: {roc_auc_score(y_test, probs):.4f}")

        return report


def load_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from saved numpy files.

    Args:
        data_dir: Directory containing X_test.npy and y_test.npy

    Returns:
        Tuple of (X_test, y_test)
    """
    data_dir = Path(data_dir)
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    return X_test, y_test


def evaluate_models_from_files(
    data_dir: str,
    model_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to evaluate all models from saved data.

    Args:
        data_dir: Directory containing test data
        model_dir: Directory containing trained models

    Returns:
        DataFrame with evaluation results
    """
    X_test, y_test = load_test_data(data_dir)

    manager = ModelManager(model_dir=model_dir)
    manager.load_all(verbose=True)

    evaluator = ModelEvaluator(manager)
    results = evaluator.evaluate_all_models(X_test, y_test)

    return results


def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON — SORTED BY AUC")
    print("=" * 80)

    display_df = results_df[["model", "type", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print(display_df.to_string(index=False))
