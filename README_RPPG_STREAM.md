# Stream 1: rPPG Physiological Deepfake Detection

## Complete Technical Documentation for Research Publication

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Flowchart](#3-pipeline-flowchart)
4. [Feature Extraction Engine](#4-feature-extraction-engine)
5. [All 111 Features Explained](#5-all-111-features-explained)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Deep Learning Pipeline](#7-deep-learning-pipeline)
8. [Hybrid Ensemble System](#8-hybrid-ensemble-system)
9. [Training Protocol](#9-training-protocol)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Hardware Requirements](#11-hardware-requirements)
12. [Research Paper Checklist](#12-research-paper-checklist)
13. [Known Limitations](#13-known-limitations)
14. [References](#14-references)

---

## 1. Executive Summary

### What This System Does

This notebook implements **Stream 1** of a dual-stream deepfake detection system using **Remote Photoplethysmography (rPPG)**. It exploits the fundamental limitation of deepfake generation:

> **Key Insight**: Real human faces exhibit subtle color variations caused by blood flow beneath the skin. Deepfake generators (GANs, Diffusion Models) cannot replicate these physiological signals because they have no concept of human circulatory systems.

### Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Features Extracted | 111 | Per video |
| ML Classifiers | 7 | + 4 ensemble methods |
| DL Architectures | 8 | + ensemble |
| ROIs Analyzed | 9 | Facial regions |
| rPPG Methods | 3 | CHROM, POS, GREEN |

### Key Innovations

1. **111-Dimensional Feature Vector**: Most comprehensive rPPG feature set in literature
2. **Cross-ROI Correlation Analysis**: Detects inconsistent pulse patterns across facial regions
3. **Phase Synchronization Features**: Exploits pulse wave propagation physics
4. **Hybrid ML+DL Ensemble**: Combines tree-based and neural approaches
5. **HRV Analysis**: Uses heart rate variability as deepfake indicator

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        rPPG DEEPFAKE DETECTION SYSTEM                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT                                                                          │
│   ┌─────────────────┐                                                           │
│   │  Video Dataset  │  400 Real + 400 Fake Videos                               │
│   │  (MP4/AVI/MOV)  │  Variable duration (5-30 seconds)                         │
│   └────────┬────────┘                                                           │
│            │                                                                     │
│            ▼                                                                     │
│   PREPROCESSING                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Frame Sampling (N=180 balanced frames)                                  │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │  MediaPipe FaceMesh (468 landmarks per frame)                           │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │  9 ROI Extraction (Forehead, Cheeks, Chin, Nose, Jaw regions)           │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │  RGB Signal Extraction (Mean R, G, B per ROI per frame)                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   rPPG SIGNAL PROCESSING                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  CHROM/POS/GREEN Algorithm                                               │   │
│   │  (Separates pulse signal from illumination/motion noise)                │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │  Bandpass Filter (0.7 - 4.0 Hz = 42-240 BPM)                            │   │
│   │  3rd order Butterworth, zero-phase (filtfilt)                           │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │  Signal Detrending (λ=300 regularization)                               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   FEATURE EXTRACTION (111 Features)                                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Signal Quality (24)     │  Cross-ROI Correlation (20)                  │   │
│   │  HRV Features (8)        │  Geometry Features (20)                      │   │
│   │  Spectral Features (12)  │  Temporal Stability (5)                      │   │
│   │  Phase Sync (3)          │  Skin Reflection (4)                         │   │
│   │  Multi-Band Power (9)    │  RGB Correlation (2)                         │   │
│   │  Spatial Pulse Var (5)   │                                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   CLASSIFICATION                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │  ┌─────────────────────┐    ┌─────────────────────┐                     │   │
│   │  │   ML PIPELINE       │    │   DL PIPELINE       │                     │   │
│   │  │                     │    │                     │                     │   │
│   │  │  RobustScaler       │    │  Feature Augment    │                     │   │
│   │  │  SelectKBest(40)    │    │  DataLoader         │                     │   │
│   │  │  PolynomialFeatures │    │                     │                     │   │
│   │  │  → ~820 features    │    │  8 Neural Networks  │                     │   │
│   │  │                     │    │  (CNN, LSTM, Trans- │                     │   │
│   │  │  7 ML Classifiers   │    │   former, MLP...)   │                     │   │
│   │  │  + 4 Ensembles      │    │                     │                     │   │
│   │  └──────────┬──────────┘    └──────────┬──────────┘                     │   │
│   │             │                          │                                │   │
│   │             └──────────┬───────────────┘                                │   │
│   │                        │                                                │   │
│   │                        ▼                                                │   │
│   │            ┌───────────────────────┐                                    │   │
│   │            │   HYBRID ENSEMBLE     │                                    │   │
│   │            │   (4 fusion methods)  │                                    │   │
│   │            └───────────┬───────────┘                                    │   │
│   │                        │                                                │   │
│   └────────────────────────┼────────────────────────────────────────────────┘   │
│                            │                                                     │
│                            ▼                                                     │
│   OUTPUT                                                                         │
│   ┌─────────────────┐                                                           │
│   │   P_rPPG        │  Probability score [0, 1]                                 │
│   │   per video     │  0 = Real, 1 = Deepfake                                   │
│   └─────────────────┘                                                           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DETAILED DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────────────────────────┐
│ 1. VIDEO INPUT                      │
│    • Read video file (cv2)          │
│    • Get metadata (fps, frames)     │
│    • Balanced frame sampling        │
│      N = min(180, total_frames)     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 2. FACE DETECTION (per frame)       │
│    • MediaPipe FaceMesh             │
│    • 468 landmark extraction        │
│    • Skip frame if detection fails  │
│    • Require all 9 ROIs valid       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 3. ROI EXTRACTION                   │
│                                     │
│    ┌─────────┐ ┌─────────┐          │
│    │FOREHEAD │ │L.CHEEK  │          │
│    │36 marks │ │15 marks │          │
│    └────┬────┘ └────┬────┘          │
│         │           │               │
│    ┌────┴───────────┴────┐          │
│    │  + R.CHEEK, CHIN,   │          │
│    │    NOSE, L.JAW,     │          │
│    │    R.JAW, L.FH,     │          │
│    │    R.FH             │          │
│    └─────────────────────┘          │
│                                     │
│    Output: 9 ROIs × T frames        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 4. COLOR SIGNAL EXTRACTION          │
│                                     │
│    For each ROI:                    │
│    • Extract pixel coordinates      │
│    • Compute mean RGB per frame     │
│    • Create time series: (T, 3)     │
│                                     │
│    Output: 9 × (T, 3) RGB signals   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 5. rPPG SIGNAL RECOVERY             │
│                                     │
│    ┌──────────────────────────────┐ │
│    │ CHROM Algorithm (default)   │ │
│    │                              │ │
│    │ 1. Normalize RGB by mean    │ │
│    │ 2. Xs = 3R - 2G             │ │
│    │ 3. Ys = 1.5R + G - 1.5B     │ │
│    │ 4. α = std(Xs) / std(Ys)    │ │
│    │ 5. BVP = Xs - α × Ys        │ │
│    │                              │ │
│    │ Window: 1.6 × fps frames    │ │
│    └──────────────────────────────┘ │
│                                     │
│    Output: 9 × BVP signals          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 6. SIGNAL FILTERING                 │
│                                     │
│    • Detrend (λ=300 regularized)    │
│    • Bandpass: 0.7 - 4.0 Hz         │
│    • Butterworth order 3            │
│    • Zero-phase (filtfilt)          │
│                                     │
│    effective_fps computed after     │
│    frame subsampling for accuracy   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 7. FEATURE EXTRACTION               │
│    (111 features per video)         │
│                                     │
│    Per-ROI Signal Quality ──────┐   │
│    Cross-ROI Correlation ───────┤   │
│    HRV Features ────────────────┤   │
│    Geometry Features ───────────┼──▶ 111-D Vector
│    Spectral Features ───────────┤   │
│    Temporal Stability ──────────┤   │
│    Phase Synchronization ───────┤   │
│    Skin Reflection ─────────────┤   │
│    RGB Correlation ─────────────┘   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 8. ML PREPROCESSING                 │
│                                     │
│    Raw 111 features                 │
│         │                           │
│         ▼                           │
│    RobustScaler (IQR-based)         │
│         │                           │
│         ▼                           │
│    SelectKBest (top 40 by XGBoost)  │
│         │                           │
│         ▼                           │
│    PolynomialFeatures (degree=2)    │
│         │                           │
│         ▼                           │
│    ~820 interaction features        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 9. MODEL TRAINING                   │
│                                     │
│    PARALLEL:                        │
│    ┌───────────┐  ┌───────────┐     │
│    │ ML Models │  │ DL Models │     │
│    │           │  │           │     │
│    │ XGBoost   │  │ CNN-1D    │     │
│    │ LightGBM  │  │ BiLSTM    │     │
│    │ RF        │  │ CNN-LSTM  │     │
│    │ ExtraTrees│  │ Transformer│    │
│    │ GB        │  │ PhysNet   │     │
│    │ AdaBoost  │  │ MultiScale│     │
│    │ SVM       │  │ TempAttn  │     │
│    │ Stacking  │  │ Wide&Deep │     │
│    │ Voting    │  │           │     │
│    └─────┬─────┘  └─────┬─────┘     │
│          │              │           │
│          └──────┬───────┘           │
│                 │                   │
│                 ▼                   │
│          HYBRID ENSEMBLE            │
│          (4 fusion strategies)      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 10. OUTPUT                          │
│                                     │
│    • rppg_predictions.csv           │
│      (video_id, P_rPPG, true_label) │
│    • Model weights (.pth)           │
│    • Feature importances            │
│    • Training curves                │
└─────────────────────────────────────┘
                 │
                 ▼
                END
```

---

## 4. Feature Extraction Engine

### 4.1 ROI Definitions

The 9 Regions of Interest are defined using MediaPipe FaceMesh landmark indices:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FACIAL ROI MAP                            │
│                                                                  │
│                    ┌──────────────────┐                         │
│                    │    FOREHEAD      │  36 landmarks           │
│                    │  [10,338,297...] │  Strongest pulse signal │
│                    └──────────────────┘                         │
│                                                                  │
│     ┌──────────┐                        ┌──────────┐            │
│     │ L.FORE-  │                        │ R.FORE-  │            │
│     │   HEAD   │                        │   HEAD   │            │
│     │ 11 marks │                        │ 11 marks │            │
│     └──────────┘                        └──────────┘            │
│                                                                  │
│     ┌──────────┐    ┌──────────┐        ┌──────────┐            │
│     │ L.CHEEK  │    │   NOSE   │        │ R.CHEEK  │            │
│     │ 15 marks │    │ 12 marks │        │ 15 marks │            │
│     └──────────┘    └──────────┘        └──────────┘            │
│                                                                  │
│     ┌──────────┐    ┌──────────┐        ┌──────────┐            │
│     │  L.JAW   │    │   CHIN   │        │  R.JAW   │            │
│     │ 11 marks │    │ 12 marks │        │ 11 marks │            │
│     └──────────┘    └──────────┘        └──────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 rPPG Methods

| Method | Formula | Rationale |
|--------|---------|-----------|
| **GREEN** | `BVP = G(t)` | Hemoglobin absorption peaks in green wavelength |
| **CHROM** | `BVP = Xs - α×Ys` where `Xs=3R-2G`, `Ys=1.5R+G-1.5B` | Chrominance-based separation of pulse from illumination |
| **POS** | `BVP = S1 + α×S2` where `S1=Gn-Bn`, `S2=Gn+Bn-2Rn` | Plane Orthogonal to Skin tone vector |

### 4.3 Bandpass Filter Specifications

```python
BANDPASS_LOW  = 0.7 Hz   # 42 BPM minimum
BANDPASS_HIGH = 4.0 Hz   # 240 BPM maximum
BANDPASS_ORDER = 3       # Butterworth order
```

---

## 5. All 111 Features Explained

### Feature Index Table

| Index | Feature Name | Category | Description |
|-------|-------------|----------|-------------|
| 0-11 | `fh_*` | Forehead Signal | SNR, spectral purity, peak prominence, dominant freq, harmonic ratio, spectral entropy, spectral centroid, MAD, STD, ZCR, kurtosis, skewness |
| 12-23 | `lc_*` | Left Cheek Signal | Same 12 metrics for left cheek ROI |
| 24-31 | `corr_*`, `coherence_*`, `phase_diff_*` | Cross-ROI Primary | Pearson correlation, spectral coherence, phase difference between FH-LC, FH-RC, LC-RC |
| 32-34 | `bpm_*`, `signal_*` | BPM & Quality | BPM estimate, stationarity, consistency across ROIs |
| 35-42 | `hrv_*` | Heart Rate Variability | RMSSD, SDNN, pNN50, pNN20, LF power, HF power, LF/HF ratio, total power |
| 43-47 | `signal_*` | Signal Quality | Energy, entropy, spectral flatness, crest factor, complexity |
| 48-67 | `geo_*` | Facial Geometry | Eye distance, width, height ratios; nose, mouth, face aspect; jaw symmetry; nose position; facial thirds; symmetry scores |
| 68-79 | `corr_*`, `coherence_*` | Extended Cross-ROI | Correlations for FH-chin, FH-nose, chin-nose, cheeks-chin, cheeks-nose, LF-RF, LJ-RJ; coherence for additional pairs |
| 80-88 | `band_power_*`, `band_ratio_*` | Multi-Band Frequency | Low (0.7-1.5Hz), mid (1.5-3Hz), high (3-4Hz) band powers for FH and LC; ratios; variance |
| 89-93 | `bpm_variance_*`, `spatial_*` | Spatial Pulse Variance | BPM variance, STD, range, IQR across all 9 ROIs; consistency score |
| 94-98 | `temporal_*` | Temporal Stability | BPM STD over time windows, BPM range, SNR STD, stability score, consistency index |
| 99-102 | `skin_reflection_*` | Skin Reflection | HSV V-channel variance for FH and LC, mean difference, specular score |
| 103-105 | `snr_*`, `patch_*` | Patch Quality | SNR STD and range across ROIs, quality consistency |
| 106-108 | `phase_sync_*` | Phase Synchronization | Mean phase difference, STD, consistency score |
| 109-110 | `rgb_corr_*` | RGB Correlation | Green-Red and Green-Blue channel correlation |

### Feature Category Breakdown

```
┌──────────────────────────────────────────────────────────────┐
│              111 FEATURES BY CATEGORY                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Per-ROI Signal Quality (24) ████████████████████████        │
│  ├── Forehead (12): SNR, purity, prominence, freq...        │
│  └── Left Cheek (12): Same metrics                          │
│                                                              │
│  Cross-ROI Correlation (20) ████████████████████             │
│  ├── Primary (8): fh-lc, fh-rc, lc-rc correlations         │
│  └── Extended (12): All other ROI pairs                     │
│                                                              │
│  Facial Geometry (20) ████████████████████                   │
│  ├── Eye metrics (5)                                        │
│  ├── Face proportions (8)                                   │
│  └── Symmetry scores (7)                                    │
│                                                              │
│  Multi-Band Frequency (9) █████████                          │
│  ├── Band powers (6)                                        │
│  └── Ratios (3)                                             │
│                                                              │
│  HRV Features (8) ████████                                   │
│  ├── Time-domain (4): RMSSD, SDNN, pNN50, pNN20            │
│  └── Frequency-domain (4): LF, HF, ratio, total            │
│                                                              │
│  Temporal Stability (5) █████                                │
│                                                              │
│  Spatial Pulse Variance (5) █████                            │
│                                                              │
│  Signal Quality (5) █████                                    │
│                                                              │
│  Skin Reflection (4) ████                                    │
│                                                              │
│  Phase Synchronization (3) ███                               │
│                                                              │
│  BPM & Overall Quality (3) ███                               │
│                                                              │
│  RGB Correlation (2) ██                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Machine Learning Pipeline

### 6.1 Preprocessing Pipeline

```
111 Raw Features
       │
       ▼
┌──────────────────────────────────────┐
│  RobustScaler                        │
│  • Centers by median                 │
│  • Scales by IQR (Q3 - Q1)          │
│  • Robust to outliers               │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  SelectKBest (40 features)           │
│  • Quick XGBoost importance ranking │
│  • Keep top 40 most predictive      │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  PolynomialFeatures (degree=2)       │
│  • interaction_only=True            │
│  • include_bias=False               │
│  • Creates 40 + C(40,2) = 820       │
└──────────────────────────────────────┘
       │
       ▼
~820 Expanded Features
```

### 6.2 ML Classifiers

| Model | Type | Hyperparameter Search | Key Settings |
|-------|------|----------------------|--------------|
| **XGBoost** | Gradient Boosting | RandomizedSearchCV (80 iter) | GPU-accelerated, reg_alpha/lambda |
| **LightGBM** | Gradient Boosting | RandomizedSearchCV (80 iter) | GPU-accelerated, leaf-wise growth |
| **RandomForest** | Bagging | RandomizedSearchCV (50 iter) | 200-700 estimators |
| **ExtraTrees** | Bagging | RandomizedSearchCV (50 iter) | More randomization |
| **GradientBoosting** | Boosting | RandomizedSearchCV (50 iter) | sklearn implementation |
| **AdaBoost** | Boosting | GridSearchCV | DecisionTree base |
| **SVM (RBF)** | SVM | RandomizedSearchCV (40 iter) | RBF kernel |

### 6.3 Ensemble Methods

| Ensemble | Strategy | Description |
|----------|----------|-------------|
| **Soft Voting** | Average | Top 7 models, average probabilities |
| **Weighted Voting** | Weighted Average | Weight by individual AUC |
| **Stacking (LR)** | Meta-Learning | LogisticRegression meta-learner |
| **Stacking (MLP)** | Meta-Learning | MLP(32,16) meta-learner |
| **Calibrated Stacking** | Calibration | Isotonic calibration + stacking |

---

## 7. Deep Learning Pipeline

### 7.1 Architecture Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    8 DEEP LEARNING ARCHITECTURES                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   1. CNN-1D         │    │   2. BiLSTM         │             │
│  │   + SE + ResConv    │    │   + Multi-Head Attn │             │
│  │                     │    │                     │             │
│  │   Two parallel      │    │   3-layer BiLSTM    │             │
│  │   branches (k=3,5)  │    │   4-head attention  │             │
│  │   SE attention      │    │   hidden=128        │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   3. CNN-BiLSTM     │    │   4. Transformer    │             │
│  │   Hybrid           │    │   Pre-LayerNorm     │             │
│  │                     │    │                     │             │
│  │   Conv extraction   │    │   CLS token         │             │
│  │   + LSTM temporal   │    │   4 encoder layers  │             │
│  │   + MH attention    │    │   d_model=64        │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   5. PhysNet MLP    │    │   6. MultiScale CNN │             │
│  │   Dense Residual    │    │   + CBAM            │             │
│  │                     │    │                     │             │
│  │   5 bottleneck      │    │   4 parallel k-sizes│             │
│  │   blocks + SE       │    │   (3,5,7,11)        │             │
│  │   hidden=256        │    │   Channel+Spatial   │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   7. Temporal Attn  │    │   8. Wide & Deep    │             │
│  │   Self-Attention    │    │   Dual Path         │             │
│  │                     │    │                     │             │
│  │   TransformerEnc    │    │   Wide: Direct      │             │
│  │   + SE block        │    │   Deep: 3 blocks    │             │
│  │   d=128, 4 heads    │    │   Concatenate       │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 (max) | With early stopping |
| Early Stopping | Patience 15 | On validation AUC |
| Learning Rate | Model-specific | 1e-3 to 5e-4 |
| Optimizer | AdamW | With weight decay |
| Scheduler | CosineAnnealing | With warm restarts |
| Batch Size | 64 | For P100 memory |
| Mixed Precision | FP16 | AMP enabled |
| Gradient Clipping | Max norm 1.0 | Stability |
| Label Smoothing | 0.05 | Regularization |

### 7.3 Data Augmentation

| Augmentation | Probability | Range |
|--------------|-------------|-------|
| Gaussian Noise | Always | std=0.02 |
| Feature Dropout | p=0.1 | Random features masked |
| Scale Jitter | Always | 0.95-1.05 |

---

## 8. Hybrid Ensemble System

### 8.1 Fusion Strategies

```
┌──────────────────────────────────────────────────────────────────┐
│                    HYBRID ENSEMBLE FUSION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ML Models                      DL Models                       │
│   ┌───────────┐                  ┌───────────┐                  │
│   │ XGBoost   │                  │ CNN-1D    │                  │
│   │ LightGBM  │                  │ BiLSTM    │                  │
│   │ Stacking  │                  │ PhysNet   │                  │
│   │ Voting    │                  │ ...       │                  │
│   └─────┬─────┘                  └─────┬─────┘                  │
│         │                              │                        │
│         └──────────────┬───────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │      FUSION METHODS          │                        │
│         ├──────────────────────────────┤                        │
│         │                              │                        │
│         │  1. Simple Average           │                        │
│         │     P = (ΣPᵢ) / N            │                        │
│         │                              │                        │
│         │  2. Weighted Average         │                        │
│         │     P = Σ(wᵢ × Pᵢ)          │                        │
│         │     wᵢ = AUCᵢ / ΣAUC        │                        │
│         │                              │                        │
│         │  3. Meta-Learner Stacking    │                        │
│         │     LogReg on predictions    │                        │
│         │                              │                        │
│         │  4. Rank-Based Fusion        │                        │
│         │     Convert to ranks first   │                        │
│         │                              │                        │
│         └──────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│                    P_rPPG                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Training Protocol

### 9.1 Data Split

```
Total Videos: 800 (400 Real + 400 Fake)
         │
         ▼
┌─────────────────────────────────────┐
│  train_test_split                   │
│  • test_size = 0.2                  │
│  • stratify = labels                │
│  • random_state = 42                │
└─────────────────────────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
   Train: 640           Test: 160
   (320 Real)          (80 Real)
   (320 Fake)          (80 Fake)
```

### 9.2 Cross-Validation

- **ML**: 5-Fold StratifiedKFold for hyperparameter tuning
- **DL**: Train/Val split with early stopping

---

## 10. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **AUC-ROC** | Area under ROC curve | Primary ranking metric |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **F1-Score** | 2×(P×R)/(P+R) | Balance precision/recall |
| **Precision** | TP/(TP+FP) | False positive control |
| **Recall** | TP/(TP+FN) | Detection rate |

---

## 11. Hardware Requirements

### Kaggle P100 Configuration

| Resource | Required | Available | Status |
|----------|----------|-----------|--------|
| GPU VRAM | ~12 GB | 16 GB | OK |
| RAM | ~16 GB | 16 GB | Tight |
| Disk | ~5 GB | 5 GB | OK |
| Time | ~3-4 hours | 9 hours | OK |

### Optimization Strategies

1. **Batch Size 64** for DL models
2. **AMP (FP16)** mixed precision
3. **Sequential model training** (not parallel)
4. **Checkpoint saving** every 25 videos
5. **Garbage collection** between models

---

## 12. Research Paper Checklist

### Essential Elements

| Element | Status | Notes |
|---------|--------|-------|
| Dataset description | Needed | Add video counts, durations, sources |
| Statistical significance | Needed | Add p-values, 95% CI |
| Ablation study | Needed | Feature group importance |
| Cross-dataset validation | Needed | Test on FF++, DFDC |
| SOTA comparison | Needed | Table vs published methods |
| Failure case analysis | Needed | When/why model fails |
| Full citations | Needed | Add DOI, page numbers |

### Recommended Additions

1. **K-Fold Cross-Validation**: Replace single split with 5-fold CV
2. **Bootstrap Confidence Intervals**: 1000+ samples
3. **McNemar Test**: For model comparison significance
4. **Subject-level splitting**: Ensure no person overlap in train/test

---

## 13. Known Limitations

1. **Single Dataset**: Only tested on one deepfake dataset
2. **Limited Generalization**: May not work on new deepfake methods
3. **Computational Cost**: Feature extraction is slow (~1-2 min/video)
4. **Face Requirement**: Requires clear frontal face with good lighting
5. **Video Quality**: Needs sufficient fps (>15) for rPPG extraction

---

## 14. References

1. **FakeCatcher**: Ciftci, U. A., Demir, I., & Yin, L. (2020). "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals." IEEE TPAMI.

2. **DeepFakesON-Phys**: Hernandez-Ortega, J., et al. (2020). "DeepFakesON-Phys: DeepFakes Detection based on Heart Rate Estimation." AAAI Workshop.

3. **CHROM Method**: de Haan, G., & Jeanne, V. (2013). "Robust Pulse Rate From Chrominance-Based rPPG." IEEE TBE.

4. **POS Method**: Wang, W., et al. (2017). "Algorithmic Principles of Remote PPG." IEEE TBE.

5. **MediaPipe FaceMesh**: Lugaresi, C., et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." CVPR Workshop.

---

*Document Version: 1.0*
*Last Updated: March 2026*
*Notebook: final_MODEL.ipynb*
