"""
ML/DL Models package for Neuro-Pulse deepfake detection.
"""
from src.models.dl_architectures import (
    DeepfakeCNN1D,
    DeepfakeBiLSTM,
    DeepfakeCNNBiLSTM,
    DeepfakeTransformer,
    PhysNetMLP,
    DL_MODEL_CLASSES,
    get_dl_model,
)
from src.models.model_manager import ModelManager, get_model_manager
from src.models.evaluation import ModelEvaluator, load_test_data

__all__ = [
    # DL Architectures
    "DeepfakeCNN1D",
    "DeepfakeBiLSTM",
    "DeepfakeCNNBiLSTM",
    "DeepfakeTransformer",
    "PhysNetMLP",
    "DL_MODEL_CLASSES",
    "get_dl_model",
    # Model Management
    "ModelManager",
    "get_model_manager",
    # Evaluation
    "ModelEvaluator",
    "load_test_data",
]
