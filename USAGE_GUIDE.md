# Usage Guide - Neuro-Pulse Deepfake Detection

## Table of Contents
1. [Installation](#installation)
2. [Command-Line Usage](#command-line-usage)
3. [Python API](#python-api)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Clone and Setup
```bash
cd /Users/likhith./pyVHR_rrpg_2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_integration.py
```

Expected output:
```
============================================================
ALL TESTS PASSED!
============================================================
```

---

## Command-Line Usage

### Check System Info
```bash
python main.py info
```

Output shows:
- Available ML models (8)
- Available DL models (5)
- Feature extraction details
- System capabilities

### Detect Deepfakes - Single Video

#### Mode 1: Best Model (Default)
Uses the highest-performing model (LogisticRegression, AUC: 0.7423):
```bash
python main.py detect video.mp4
```

Output:
```
--- Processing: video.mp4 ---

  RESULT: Real
  Confidence: 75.6%
  Probability (fake): 0.244
  Model: Ensemble_Top3

  VIDEO INFO:
    File: video.mp4
    FPS: 30
    Resolution: 1280x720
    Face detection rate: 98.3%
```

#### Mode 2: Specific ML Model
Choose any of: RandomForest, XGBoost, LightGBM, SVM_RBF, GradientBoosting, ExtraTrees, LogisticRegression

```bash
python main.py detect video.mp4 --mode ml --model XGBoost
```

#### Mode 3: Specific DL Model
Choose any of: CNN_1D, BiLSTM_Attention, CNN_BiLSTM, Transformer, PhysNet_MLP

```bash
python main.py detect video.mp4 --mode dl --model CNN_BiLSTM
```

#### Mode 4: Multi-Model Ensemble
Averages predictions from multiple models for robust consensus:
```bash
python main.py detect video.mp4 --mode ensemble
```

#### Mode 5: All Models Analysis
Run all models and show detailed breakdown:
```bash
python main.py detect video.mp4 --mode all
```

Output includes:
- Primary prediction (best model)
- Ensemble prediction
- All 13 model predictions with probabilities

### Batch Processing

Detect deepfakes in multiple videos:
```bash
python main.py detect video1.mp4 video2.mp4 video3.mp4 --output results.json
```

This:
- Processes videos sequentially
- Shows progress bar
- Saves all results to `results.json`

### Advanced Options

Extract features for inspection:
```bash
python main.py detect video.mp4 --features
```

Increase verbosity:
```bash
python main.py detect video.mp4 -v
```

Custom rPPG method (GREEN, CHROM, or POS):
```bash
python main.py detect video.mp4 --method CHROM
```

Specify custom model directory:
```bash
python main.py detect video.mp4 --model-dir /path/to/models
```

---

## Python API

### Basic Usage

```python
from src.deepfake_detector import DeepfakeDetector
import json

# Initialize detector
detector = DeepfakeDetector()

# Detect single video
result = detector.detect("video.mp4")

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Probability (Fake): {result['probability']:.4f}")

# Clean up
detector.close()
```

### Advanced: Using ModelManager Directly

```python
from src.models.model_manager import ModelManager
import numpy as np

# Initialize manager
manager = ModelManager()
manager.load_all(verbose=True)

# Feature extraction (for pre-extracted features)
features = np.random.randn(35)  # Example: 35-dim feature vector

# Single model prediction (ML)
predictions, probs = manager.predict_ml(
    features,
    model_name="XGBoost"
)
print(f"ML Prediction: {predictions[0]}, Probability: {probs[0]:.4f}")

# Single model prediction (DL)
predictions, probs = manager.predict_dl(
    features,
    model_name="CNN_BiLSTM"
)
print(f"DL Prediction: {predictions[0]}, Probability: {probs[0]:.4f}")

# Ensemble prediction
predictions, probs = manager.predict_ensemble(features)
print(f"Ensemble: {predictions[0]}, Confidence: {max(probs[0], 1-probs[0])*100:.1f}%")

# Get available models
available = manager.get_available_models()
print(f"ML Models: {available['ml']}")
print(f"DL Models: {available['dl']}")
```

### Advanced: Video Batch Processing

```python
from src.deepfake_detector import DeepfakeDetector
import json

detector = DeepfakeDetector(rppg_method="CHROM")

# Process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = detector.detect_batch(video_paths, mode="ensemble")

# Save results
with open("detections.json", "w") as f:
    json.dump(results, f, indent=2)

# Analyze results
for result in results:
    video = result["video_path"]
    label = result["label"]
    confidence = result["confidence"]
    print(f"{video}: {label} ({confidence:.1f}%)")

detector.close()
```

### Advanced: Model Evaluation

```python
from src.models.evaluation import ModelEvaluator, load_test_data
from src.models.model_manager import ModelManager
import pandas as pd

# Load test data
X_test, y_test = load_test_data("OUTPUT_DIP_FINAL/features")

# Create evaluator
manager = ModelManager()
manager.load_all(verbose=False)
evaluator = ModelEvaluator(manager)

# Evaluate all models
results = evaluator.evaluate_all_models(X_test, y_test)

# Display results sorted by AUC
print(results[["model", "accuracy", "auc", "f1"]].sort_values("auc", ascending=False))

# Detailed report for best model
best_model = results.iloc[0]["model"]
report = evaluator.print_classification_report(X_test, y_test, best_model)
```

---

## Examples

### Example 1: Quick Deepfake Check
```bash
source venv/bin/activate
python main.py detect suspect_video.mp4
```

### Example 2: Detailed Analysis with All Models
```bash
python main.py detect video.mp4 --mode all -v --output results.json
```

### Example 3: Using Specific High-Performance Model
```bash
# CNN_BiLSTM is one of the best DL models (AUC: 0.7397)
python main.py detect video.mp4 --mode dl --model CNN_BiLSTM
```

### Example 4: Batch Processing Dataset
```bash
# Process all videos in a directory
for video in dataset/*.mp4; do
    python main.py detect "$video" --mode ensemble
done
```

### Example 5: Python Script for Automated Detection
```python
#!/usr/bin/env python3
"""
Automated deepfake detection on video dataset.
"""
from src.deepfake_detector import DeepfakeDetector
from pathlib import Path
import json

def process_dataset(video_dir, output_file="results.json"):
    detector = DeepfakeDetector()

    videos = list(Path(video_dir).glob("*.mp4"))
    results = {}

    for video_path in videos:
        print(f"Processing {video_path.name}...", end=" ")

        result = detector.detect(str(video_path), mode="ensemble")

        if "error" not in result:
            results[video_path.name] = {
                "label": result["label"],
                "confidence": result["confidence"],
                "probability": result["probability"],
            }
            print(f"✓ {result['label']}")
        else:
            print(f"✗ {result['error']}")

    detector.close()

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    n_deepfakes = sum(1 for r in results.values() if r["label"] == "Deepfake")
    print(f"Summary: {n_deepfakes} deepfakes out of {len(results)} videos")

if __name__ == "__main__":
    process_dataset("./videos")
```

### Example 6: Real-Time Webcam Liveness Detection
```bash
python main.py liveness
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Problem**: Missing dependencies
**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Model not found
**Problem**: "Model 'xyz' not loaded"
**Solution**:
```bash
# Check available models
python main.py info

# Verify OUTPUT_DIP_FINAL/features/ directory exists
ls OUTPUT_DIP_FINAL/features/*.joblib
```

### Issue: Video processing fails
**Problem**: "Cannot open video" or "Insufficient face frames"
**Causes & Solutions**:
- Video format not supported → Convert to MP4 (H.264)
- Face not detected → Ensure clear, frontal face
- Too short video → Need at least 2 seconds of face data

```bash
# Adjust max frames if video is very long
python main.py detect video.mp4 --max-frames 600

# Try different rPPG method if GREEN doesn't work
python main.py detect video.mp4 --method CHROM
```

### Issue: Slow predictions
**Problem**: Inference takes too long
**Solution**:
- Use faster ML model (LogisticRegression is very fast)
- Use single DL model instead of ensemble
- Reduce video frame count:
```bash
python main.py detect video.mp4 --max-frames 150
```

### Issue: Memory issues
**Problem**: "CUDA out of memory" or system slowdown
**Solution**:
- Process videos one at a time (not batch)
- Use ML models instead of DL (lower memory)
- Reduce feature extraction frames

### Issue: scikit-learn version warning
**Problem**: InconsistentVersionWarning during model loading
**Status**: ⚠️ **This is safe to ignore**
- Models load and work correctly
- Risk is minimal for inference
- Can be fixed by retraining models with current sklearn version

---

## Output Interpretation

### Prediction Output
```
RESULT: Deepfake
Confidence: 87.5%
Probability (fake): 0.875
```

Interpretation:
- **Prediction**: 0 = Real, 1 = Deepfake
- **Confidence**: 0-100%, higher = more certain
- **Probability**: Raw probability from model (0-1)

### Video Quality Metrics
```
Face detection rate: 98.3%
```

Interpretation:
- Face was detected in 98.3% of processed frames
- Higher is better (min 50% for valid detection)
- Low rate indicates poor video quality or face angle

---

## Performance Benchmarks

On test set (79 videos, ~10 seconds each):

| Model | Speed | AUC | Memory |
|-------|-------|-----|--------|
| LogisticRegression (ML) | <100ms | 0.7423 | 1 MB |
| CNN_BiLSTM (DL) | ~500ms | 0.7397 | 50 MB |
| XGBoost (ML) | ~200ms | 0.7115 | 10 MB |
| Ensemble | ~2000ms | ~0.74 | 100 MB |

---

**For more help**: `python main.py --help` or `python main.py detect --help`
