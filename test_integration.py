#!/usr/bin/env python3
"""
Test script for verifying the Neuro-Pulse ML/DL integration.

Run this after installing dependencies:
    pip install -r requirements.txt
    python test_integration.py
"""
import os
import sys
import numpy as np


def test_file_structure():
    """Verify all required files exist."""
    print("\n" + "=" * 60)
    print("TEST 1: File Structure")
    print("=" * 60)

    files = [
        ("src/models/__init__.py", "Models package"),
        ("src/models/dl_architectures.py", "DL architectures"),
        ("src/models/model_manager.py", "Model manager"),
        ("src/models/evaluation.py", "Evaluation module"),
        ("src/deepfake_detector.py", "Deepfake detector"),
        ("src/feature_extractor.py", "Feature extractor"),
        ("src/rppg_extractor.py", "rPPG extractor"),
        ("src/signal_processor.py", "Signal processor"),
        ("main.py", "Main entry point"),
    ]

    all_ok = True
    for path, desc in files:
        if os.path.exists(path):
            print(f"  [OK] {desc}: {path}")
        else:
            print(f"  [FAIL] {desc}: {path}")
            all_ok = False

    return all_ok


def test_model_artifacts():
    """Verify trained model files exist."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Artifacts")
    print("=" * 60)

    model_dir = "OUTPUT_DIP_FINAL/features"

    # Expected ML models
    ml_models = [
        "RandomForest_best.joblib",
        "XGBoost_best.joblib",
        "LightGBM_best.joblib",
        "SVM_RBF_best.joblib",
        "GradientBoosting_best.joblib",
        "ExtraTrees_best.joblib",
        "LogisticRegression_best.joblib",
        "Ensemble_Top3_best.joblib",
    ]

    # Expected DL models
    dl_models = [
        "CNN_1D_checkpoint.pth",
        "BiLSTM_Attention_checkpoint.pth",
        "CNN_BiLSTM_checkpoint.pth",
        "Transformer_checkpoint.pth",
        "PhysNet_MLP_checkpoint.pth",
    ]

    # Scalers
    scalers = ["scaler.joblib", "dl_scaler.joblib"]

    all_ok = True

    print("\n  ML Models:")
    for model in ml_models:
        path = os.path.join(model_dir, model)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"    [OK] {model} ({size:.1f} KB)")
        else:
            print(f"    [FAIL] {model}")
            all_ok = False

    print("\n  DL Models:")
    for model in dl_models:
        path = os.path.join(model_dir, model)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"    [OK] {model} ({size:.1f} KB)")
        else:
            print(f"    [FAIL] {model}")
            all_ok = False

    print("\n  Scalers:")
    for scaler in scalers:
        path = os.path.join(model_dir, scaler)
        if os.path.exists(path):
            print(f"    [OK] {scaler}")
        else:
            print(f"    [FAIL] {scaler}")
            all_ok = False

    return all_ok


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 60)
    print("TEST 3: Module Imports")
    print("=" * 60)

    modules = [
        ("src.models.dl_architectures", "DL Architectures"),
        ("src.models.model_manager", "Model Manager"),
        ("src.models.evaluation", "Evaluation"),
        ("src.feature_extractor", "Feature Extractor"),
        ("src.rppg_extractor", "rPPG Extractor"),
    ]

    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    return all_ok


def test_dl_models():
    """Test DL model architectures."""
    print("\n" + "=" * 60)
    print("TEST 4: DL Model Architectures")
    print("=" * 60)

    try:
        import torch
        from src.models.dl_architectures import (
            DeepfakeCNN1D,
            DeepfakeBiLSTM,
            DeepfakeCNNBiLSTM,
            DeepfakeTransformer,
            PhysNetMLP,
        )

        models = [
            ("CNN_1D", DeepfakeCNN1D),
            ("BiLSTM_Attention", DeepfakeBiLSTM),
            ("CNN_BiLSTM", DeepfakeCNNBiLSTM),
            ("Transformer", DeepfakeTransformer),
            ("PhysNet_MLP", PhysNetMLP),
        ]

        # Test input
        x = torch.randn(4, 35)

        all_ok = True
        for name, model_class in models:
            try:
                model = model_class(n_features=35)
                output = model(x)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  [OK] {name}: output={output.shape}, params={n_params:,}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_ok = False

        return all_ok

    except ImportError as e:
        print(f"  [SKIP] PyTorch not installed: {e}")
        return True


def test_model_loading():
    """Test loading trained models."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Loading")
    print("=" * 60)

    try:
        from src.models.model_manager import ModelManager

        manager = ModelManager()
        manager.load_all(verbose=False)

        available = manager.get_available_models()
        print(f"  ML Models loaded: {len(available['ml'])}")
        print(f"  DL Models loaded: {len(available['dl'])}")

        if available['ml']:
            print(f"    ML: {', '.join(available['ml'])}")
        if available['dl']:
            print(f"    DL: {', '.join(available['dl'])}")

        return len(available['ml']) > 0 or len(available['dl']) > 0

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_prediction():
    """Test making predictions."""
    print("\n" + "=" * 60)
    print("TEST 6: Predictions")
    print("=" * 60)

    try:
        from src.models.model_manager import ModelManager

        manager = ModelManager()
        manager.load_all(verbose=False)

        # Create dummy features
        features = np.random.randn(35)

        # Test ML prediction
        if manager.ml_models:
            model_name = list(manager.ml_models.keys())[0]
            result = manager.predict(features, mode="ml", model_name=model_name)
            print(f"  [OK] ML prediction ({model_name}):")
            print(f"       Label: {result['label']}, Confidence: {result['confidence']:.1f}%")

        # Test DL prediction
        if manager.dl_models:
            model_name = list(manager.dl_models.keys())[0]
            result = manager.predict(features, mode="dl", model_name=model_name)
            print(f"  [OK] DL prediction ({model_name}):")
            print(f"       Label: {result['label']}, Confidence: {result['confidence']:.1f}%")

        # Test ensemble prediction
        result = manager.predict(features, mode="ensemble")
        print(f"  [OK] Ensemble prediction:")
        print(f"       Label: {result['label']}, Confidence: {result['confidence']:.1f}%")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test model evaluation on saved test data."""
    print("\n" + "=" * 60)
    print("TEST 7: Model Evaluation (against saved test set)")
    print("=" * 60)

    try:
        data_dir = "OUTPUT_DIP_FINAL/features"

        # Check if test data exists
        x_test_path = os.path.join(data_dir, "X_test.npy")
        y_test_path = os.path.join(data_dir, "y_test.npy")

        if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
            print("  [SKIP] Test data not found")
            return True

        from src.models.evaluation import ModelEvaluator, load_test_data
        from src.models.model_manager import ModelManager

        X_test, y_test = load_test_data(data_dir)
        print(f"  Test data: X={X_test.shape}, y={y_test.shape}")
        print(f"  Class distribution: Real={np.sum(y_test==0)}, Fake={np.sum(y_test==1)}")

        manager = ModelManager()
        manager.load_all(verbose=False)

        evaluator = ModelEvaluator(manager)
        results = evaluator.evaluate_all_models(X_test, y_test)

        print(f"\n  Top 5 Models by AUC:")
        for _, row in results.head(5).iterrows():
            print(f"    {row['model']:<25} AUC={row['auc']:.4f} F1={row['f1']:.4f}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEURO-PULSE ML/DL INTEGRATION TEST")
    print("=" * 60)

    results = []

    results.append(("File Structure", test_file_structure()))
    results.append(("Model Artifacts", test_model_artifacts()))
    results.append(("Module Imports", test_imports()))
    results.append(("DL Architectures", test_dl_models()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Predictions", test_prediction()))
    results.append(("Evaluation", test_evaluation()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[-]"
        print(f"  {symbol} {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Check output above")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
