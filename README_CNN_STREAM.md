# Stream 2: CNN Spatio-Temporal Deepfake Detection

## Complete Technical Documentation for Research Publication

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Flowchart](#3-pipeline-flowchart)
4. [Face Detection & Extraction](#4-face-detection--extraction)
5. [Spatio-Temporal Model Architecture](#5-spatio-temporal-model-architecture)
6. [Temporal Aggregation (BiLSTM + Attention)](#6-temporal-aggregation-bilstm--attention)
7. [Data Augmentation](#7-data-augmentation)
8. [Training Protocol](#8-training-protocol)
9. [Grad-CAM Visualization](#9-grad-cam-visualization)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Hardware Requirements](#11-hardware-requirements)
12. [Late Fusion Integration](#12-late-fusion-integration)
13. [Research Paper Checklist](#13-research-paper-checklist)
14. [Known Limitations](#14-known-limitations)
15. [References](#15-references)

---

## 1. Executive Summary

### What This System Does

This notebook implements **Stream 2** of a dual-stream deepfake detection system using **Spatio-Temporal CNN** with **BiLSTM Temporal Aggregation**. Unlike naive frame averaging approaches, this architecture actively detects inter-frame inconsistencies:

> **Key Insight**: Modern deepfakes (GANs, Diffusion Models) often generate visually convincing individual frames but fail to maintain temporal consistency. Our BiLSTM models these inter-frame dependencies to catch artifacts like flickering, blending boundary shifts, and unnatural micro-motions.

### Performance Summary

| Component | Specification |
|-----------|--------------|
| Backbone | EfficientNet-B4 (1792-dim features) |
| Temporal Model | BiLSTM (2-layer, bidirectional, 256 hidden) |
| Attention | Multi-Head Self-Attention (4 heads) |
| Frames/Video | 15 (evenly sampled) |
| Interpretability | Grad-CAM heatmaps |

### Key Innovations

1. **Temporal Aggregation**: BiLSTM replaces naive frame averaging
2. **Multi-Head Self-Attention**: Learns which frames are most informative
3. **Grad-CAM Visualization**: Proves model learns meaningful artifacts
4. **Video-Level Training**: Each batch contains full videos, not random frames
5. **Research-Grade Architecture**: Proper sequence modeling for publication

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    CNN SPATIO-TEMPORAL DEEPFAKE DETECTION                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Video Dataset (400 Real + 400 Fake)                                     │   │
│   │  → Extract 15 evenly spaced frames per video                            │   │
│   │  → MTCNN face detection per frame                                        │   │
│   │  → 224×224×3 face crops                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   SPATIAL FEATURE EXTRACTION                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   Frame 1  ──┐                                                          │   │
│   │   Frame 2  ──┤                                                          │   │
│   │   Frame 3  ──┤     ┌────────────────────────┐                           │   │
│   │      ...    ──┼───▶│   EfficientNet-B4      │───▶  T × 1792-dim        │   │
│   │   Frame 14 ──┤     │   (pretrained backbone) │      feature vectors     │   │
│   │   Frame 15 ──┘     └────────────────────────┘                           │   │
│   │                                                                          │   │
│   │   Each frame processed independently through CNN backbone               │   │
│   │   Output: Sequence of T spatial feature vectors                         │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   TEMPORAL AGGREGATION                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   Feature Sequence (15 × 1792)                                          │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │   ┌──────────────────────────────────────────────────────────┐          │   │
│   │   │               BiLSTM (2-layer, bidirectional)            │          │   │
│   │   │                                                          │          │   │
│   │   │   ←── Forward LSTM (256 hidden) ───→                    │          │   │
│   │   │   ←── Backward LSTM (256 hidden) ───→                   │          │   │
│   │   │                                                          │          │   │
│   │   │   Output: 15 × 512-dim (bidirectional concatenated)     │          │   │
│   │   └──────────────────────────────────────────────────────────┘          │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │   ┌──────────────────────────────────────────────────────────┐          │   │
│   │   │           Multi-Head Self-Attention (4 heads)            │          │   │
│   │   │                                                          │          │   │
│   │   │   • Learns which frames contribute most to detection    │          │   │
│   │   │   • Attention weights show temporal focus               │          │   │
│   │   │   • Residual connection + LayerNorm                     │          │   │
│   │   │                                                          │          │   │
│   │   │   Output: Weighted pooled vector (512-dim)              │          │   │
│   │   └──────────────────────────────────────────────────────────┘          │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   CLASSIFICATION HEAD                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   512-dim ──→ Linear(256) ──→ BN ──→ GELU ──→ Dropout(0.4)             │   │
│   │           ──→ Linear(128) ──→ BN ──→ GELU ──→ Dropout(0.2)             │   │
│   │           ──→ Linear(1) ──→ Sigmoid                                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   OUTPUT                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │   P_CNN: Deepfake probability [0, 1] per video                          │   │
│   │   Attention weights: Which frames the model focused on                  │   │
│   │   Grad-CAM: Spatial attention heatmaps                                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DETAILED PIPELINE FLOW                              │
└─────────────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────────────────────────┐
│ 1. VIDEO INPUT                      │
│    • Load video (cv2.VideoCapture)  │
│    • Get total frame count          │
│    • Compute evenly spaced indices  │
│      np.linspace(0, total-1, 15)    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 2. FRAME EXTRACTION                 │
│    For each of 15 indices:          │
│    • Seek to frame position         │
│    • Read BGR frame                 │
│    • Store in frames list           │
│                                     │
│    Output: 15 frames (BGR)          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 3. FACE DETECTION (MTCNN)           │
│    For each frame:                  │
│    • Convert BGR → RGB              │
│    • Run MTCNN detector             │
│    ┌───────────────────────────┐    │
│    │ MTCNN Config:             │    │
│    │ • image_size: 224         │    │
│    │ • margin: 40              │    │
│    │ • min_face_size: 60       │    │
│    │ • thresholds: [0.6,0.7,0.7]│   │
│    │ • post_process: False     │    │
│    │ • keep_all: False         │    │
│    └───────────────────────────┘    │
│    • If face detected: crop         │
│    • If not: center crop fallback   │
│                                     │
│    Output: 15 × (224, 224, 3) faces │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 4. DATA AUGMENTATION (Train only)   │
│                                     │
│    ┌─────────────────────────────┐  │
│    │ Albumentations Pipeline:    │  │
│    │ • HorizontalFlip (p=0.5)    │  │
│    │ • ShiftScaleRotate (p=0.3)  │  │
│    │ • BrightnessContrast (p=0.5)│  │
│    │ • HueSaturationValue (p=0.3)│  │
│    │ • ImageCompression (p=0.5)  │  │
│    │ • GaussNoise (p=0.3)        │  │
│    │ • GaussianBlur (p=0.2)      │  │
│    │ • CoarseDropout (p=0.2)     │  │
│    │ • ImageNet Normalize        │  │
│    └─────────────────────────────┘  │
│                                     │
│    Output: Augmented tensors        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 5. VIDEO-LEVEL BATCHING             │
│                                     │
│    DeepfakeVideoDataset:            │
│    • Each sample = 1 video          │
│    • Returns (T, C, H, W) tensor    │
│    • Includes padding mask          │
│                                     │
│    Batch Shape:                     │
│    • frames: (B, T, C, H, W)        │
│    • labels: (B,)                   │
│    • mask: (B, T)                   │
│                                     │
│    B=8, T=15, C=3, H=W=224          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 6. SPATIAL FEATURE EXTRACTION       │
│                                     │
│    Reshape: (B, T, C, H, W)         │
│          → (B×T, C, H, W)           │
│                                     │
│    EfficientNet-B4 forward:         │
│    • Conv layers extract features   │
│    • Global average pooling         │
│    • Output: (B×T, 1792)            │
│                                     │
│    Reshape: (B×T, 1792)             │
│          → (B, T, 1792)             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 7. TEMPORAL MODELING (BiLSTM)       │
│                                     │
│    Input: (B, T, 1792)              │
│         ↓                           │
│    BiLSTM Layer 1:                  │
│    • Forward: h₁→h₂→...→h₁₅        │
│    • Backward: h₁←h₂←...←h₁₅       │
│         ↓                           │
│    BiLSTM Layer 2:                  │
│    • Same bidirectional processing  │
│         ↓                           │
│    Output: (B, T, 512)              │
│    (256 forward + 256 backward)     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 8. ATTENTION POOLING                │
│                                     │
│    Multi-Head Self-Attention:       │
│    • Q = K = V from BiLSTM output   │
│    • 4 attention heads              │
│    • Scaled dot-product attention   │
│                                     │
│    Attention(Q,K,V) = softmax(QK^T/√d)V
│                                     │
│    Residual + LayerNorm:            │
│    • out = LN(x + Dropout(Attn(x))) │
│                                     │
│    Masked Mean Pooling:             │
│    • Aggregate across time dim      │
│    • Respect padding mask           │
│                                     │
│    Output: (B, 512)                 │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 9. CLASSIFICATION                   │
│                                     │
│    (B, 512)                         │
│         ↓                           │
│    Linear(512↠256) → BN → GELU     │
│         ↓                           │
│    Dropout(0.4)                     │
│         ↓                           │
│    Linear(256↠128) → BN → GELU     │
│         ↓                           │
│    Dropout(0.2)                     │
│         ↓                           │
│    Linear(128↠1)                   │
│         ↓                           │
│    BCEWithLogitsLoss (training)     │
│    Sigmoid (inference)              │
│                                     │
│    Output: P_CNN ∈ [0, 1]           │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 10. OUTPUT / EXPORT                 │
│                                     │
│    • cnn_predictions.csv            │
│      (video_id, P_CNN, true_label)  │
│                                     │
│    • gradcam_gallery.png            │
│      (visual interpretability)      │
│                                     │
│    • best_cnn_model.pth             │
│      (model weights)                │
└─────────────────────────────────────┘
                 │
                 ▼
                END
```

---

## 4. Face Detection & Extraction

### 4.1 MTCNN Configuration

```python
MTCNN(
    image_size=224,        # Output face crop size
    margin=40,             # Pixels around face
    min_face_size=60,      # Skip tiny faces
    thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
    factor=0.709,          # Scale pyramid factor
    post_process=False,    # Raw [0,255] values, no normalization
    keep_all=False,        # Only largest face
)
```

### 4.2 Fallback Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE DETECTION FALLBACK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Frame                                                    │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────┐                                        │
│   │   MTCNN Detection   │                                        │
│   └──────────┬──────────┘                                        │
│              │                                                   │
│        Face found?                                               │
│        /        \                                                │
│      YES         NO                                              │
│       │          │                                               │
│       ▼          ▼                                               │
│   Use MTCNN   Center Crop                                        │
│   crop        (min(H,W) square)                                  │
│       │          │                                               │
│       └────┬─────┘                                               │
│            │                                                     │
│            ▼                                                     │
│   Resize to 224×224                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Spatio-Temporal Model Architecture

### 5.1 Model Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SpatioTemporalDeepfakeCNN                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   COMPONENT 1: EfficientNet-B4 Backbone                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   │   Input: (B×T, 3, 224, 224)                                       │    │
│   │                                                                    │    │
│   │   Stem:                                                           │    │
│   │   • Conv2d(3→48, k=3, s=2) → BN → Swish                          │    │
│   │                                                                    │    │
│   │   MBConv Blocks (7 stages):                                       │    │
│   │   • Stage 1: MBConv1 (24 channels)                               │    │
│   │   • Stage 2: MBConv6 (32 channels)                               │    │
│   │   • Stage 3: MBConv6 (56 channels)                               │    │
│   │   • Stage 4: MBConv6 (112 channels)                              │    │
│   │   • Stage 5: MBConv6 (160 channels)                              │    │
│   │   • Stage 6: MBConv6 (272 channels)                              │    │
│   │   • Stage 7: MBConv6 (448 channels)                              │    │
│   │                                                                    │    │
│   │   Head:                                                           │    │
│   │   • Conv2d(448→1792, k=1) → BN → Swish                           │    │
│   │   • GlobalAvgPool2d → 1792-dim                                   │    │
│   │                                                                    │    │
│   │   Output: (B×T, 1792)                                             │    │
│   │                                                                    │    │
│   │   Parameters: ~17.5M                                              │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   COMPONENT 2: BiLSTM Temporal Aggregator                                    │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   │   Input: (B, T, 1792)                                             │    │
│   │                                                                    │    │
│   │   nn.LSTM(                                                        │    │
│   │       input_size=1792,                                            │    │
│   │       hidden_size=256,                                            │    │
│   │       num_layers=2,                                               │    │
│   │       bidirectional=True,                                         │    │
│   │       dropout=0.4,                                                │    │
│   │       batch_first=True                                            │    │
│   │   )                                                               │    │
│   │                                                                    │    │
│   │   Output: (B, T, 512)  # 256×2 bidirectional                     │    │
│   │                                                                    │    │
│   │   Parameters: ~4.7M                                               │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   COMPONENT 3: Multi-Head Self-Attention                                     │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   │   Input: (B, T, 512)                                              │    │
│   │                                                                    │    │
│   │   nn.MultiheadAttention(                                          │    │
│   │       embed_dim=512,                                              │    │
│   │       num_heads=4,                                                │    │
│   │       dropout=0.1,                                                │    │
│   │       batch_first=True                                            │    │
│   │   )                                                               │    │
│   │                                                                    │    │
│   │   + LayerNorm + Residual Connection                              │    │
│   │                                                                    │    │
│   │   Masked Mean Pooling → (B, 512)                                 │    │
│   │                                                                    │    │
│   │   Parameters: ~1M                                                 │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   COMPONENT 4: Classification Head                                           │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   │   nn.Sequential(                                                  │    │
│   │       Linear(512, 256),                                           │    │
│   │       BatchNorm1d(256),                                           │    │
│   │       GELU(),                                                     │    │
│   │       Dropout(0.4),                                               │    │
│   │       Linear(256, 128),                                           │    │
│   │       BatchNorm1d(128),                                           │    │
│   │       GELU(),                                                     │    │
│   │       Dropout(0.2),                                               │    │
│   │       Linear(128, 1)                                              │    │
│   │   )                                                               │    │
│   │                                                                    │    │
│   │   Parameters: ~166K                                               │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   TOTAL PARAMETERS: ~23.4M                                                   │
│   TRAINABLE PARAMETERS: ~23.4M (or ~5.9M if backbone frozen)                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Temporal Aggregation (BiLSTM + Attention)

### 6.1 Why Temporal Modeling?

```
┌──────────────────────────────────────────────────────────────────┐
│               NAIVE AVERAGING vs TEMPORAL MODELING               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   NAIVE APPROACH (Previous):                                     │
│   ┌────────────────────────────────────────────────────────┐    │
│   │   Frame 1 → CNN → P₁  ┐                                │    │
│   │   Frame 2 → CNN → P₂  │                                │    │
│   │   Frame 3 → CNN → P₃  │───→ P_final = mean(P₁...P₁₅)  │    │
│   │   ...                 │                                │    │
│   │   Frame 15 → CNN → P₁₅┘                                │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   PROBLEM: Cannot detect inter-frame artifacts:                  │
│   • Temporal flickering between frames                          │
│   • Blending boundary shifts over time                          │
│   • Unnatural micro-motion patterns                             │
│   • Phase inconsistencies across frames                         │
│                                                                  │
│   ─────────────────────────────────────────────────────────────  │
│                                                                  │
│   TEMPORAL MODELING (Current):                                   │
│   ┌────────────────────────────────────────────────────────┐    │
│   │   Frame 1 → CNN → f₁  ┐                                │    │
│   │   Frame 2 → CNN → f₂  │                                │    │
│   │   Frame 3 → CNN → f₃  ├──→ BiLSTM ──→ Attention ──→ P │    │
│   │   ...                 │                                │    │
│   │   Frame 15 → CNN → f₁₅┘                                │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   DETECTS:                                                       │
│   ✓ Temporal flickering (sudden feature changes)                │
│   ✓ Blending boundary evolution (gradual shifts)                │
│   ✓ Motion coherence violations                                 │
│   ✓ Phase synchronization breaks                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 BiLSTM Operation

```
Time →  t=1    t=2    t=3    ...    t=15

Input:   f₁     f₂     f₃    ...    f₁₅     (1792-dim each)
         │      │      │             │
         ▼      ▼      ▼             ▼
       ┌────────────────────────────────┐
       │     Forward LSTM (256 hidden)  │
       │    h₁ → h₂ → h₃ → ... → h₁₅   │
       └────────────────────────────────┘
                        +
       ┌────────────────────────────────┐
       │    Backward LSTM (256 hidden)  │
       │    h₁ ← h₂ ← h₃ ← ... ← h₁₅   │
       └────────────────────────────────┘
         │      │      │             │
         ▼      ▼      ▼             ▼
Output: [h→₁;h←₁] [h→₂;h←₂] ...  [h→₁₅;h←₁₅]   (512-dim each)
```

### 6.3 Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                   TEMPORAL SELF-ATTENTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: H = [h₁, h₂, ..., h₁₅]  ∈ ℝ^(B×T×512)                 │
│                                                                  │
│   Step 1: Compute Q, K, V for each head (4 heads)               │
│                                                                  │
│      Q = H × W_Q    K = H × W_K    V = H × W_V                  │
│           (512×128)     (512×128)     (512×128)                 │
│                                                                  │
│   Step 2: Scaled Dot-Product Attention                          │
│                                                                  │
│      Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V             │
│                                                                  │
│      d_k = 128 (per-head dimension)                             │
│                                                                  │
│   Step 3: Concatenate heads + Linear projection                  │
│                                                                  │
│      MultiHead = Concat(head₁, ..., head₄) × W_O               │
│                         (4×128=512)        (512×512)            │
│                                                                  │
│   Step 4: Residual + LayerNorm                                  │
│                                                                  │
│      Out = LayerNorm(H + Dropout(MultiHead))                    │
│                                                                  │
│   Step 5: Masked Mean Pooling                                   │
│                                                                  │
│      pooled = Σ(Out × mask) / Σ(mask)                          │
│                                                                  │
│      Output: (B, 512) aggregated video representation           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Augmentation

### 7.1 Training Augmentations

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|------------|---------|
| **HorizontalFlip** | 0.5 | - | Handle left/right face orientations |
| **ShiftScaleRotate** | 0.3 | shift=0.05, scale=0.05, rotate=10° | Simulate camera movement |
| **BrightnessContrast** | 0.5 | brightness=±0.2, contrast=±0.2 | Handle lighting variation |
| **HueSaturationValue** | 0.3 | hue=±10, sat=±20, val=±15 | Handle color calibration differences |
| **ImageCompression** | 0.5 | quality=50-100 | Simulate compression artifacts (critical for deepfakes) |
| **GaussNoise** | 0.3 | var=10-50 | Handle sensor noise |
| **GaussianBlur** | 0.2 | kernel=3-5 | Handle focus variation |
| **CoarseDropout** | 0.2 | holes=4, size=20×20 | Regularization (occlusion robustness) |

### 7.2 Validation/Test Transforms

```python
A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 8. Training Protocol

### 8.1 Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 8 | Video-level batching (8 videos × 15 frames = 120 images/batch) |
| **Epochs** | 20 (max) | With early stopping |
| **Learning Rate** | 1e-4 | AdamW with weight decay |
| **Weight Decay** | 1e-2 | L2 regularization |
| **Optimizer** | AdamW | Better generalization than Adam |
| **Scheduler** | CosineAnnealingLR | Smooth decay to 1e-6 |
| **Early Stopping** | Patience 5 | Stop if val AUC doesn't improve |
| **Gradient Clipping** | Max norm 1.0 | Training stability |
| **Mixed Precision** | FP16 (AMP) | Memory efficiency |
| **Loss** | BCEWithLogitsLoss | Numerically stable binary CE |

### 8.2 Training Loop

```
For each epoch:
    │
    ├──→ Training Phase
    │    │
    │    ├── For each batch (B videos × T frames):
    │    │   ├── Forward pass (all frames through backbone + temporal)
    │    │   ├── Compute BCEWithLogitsLoss
    │    │   ├── Backward pass (with AMP scaler)
    │    │   ├── Gradient clipping
    │    │   └── Optimizer step
    │    │
    │    └── Compute train metrics (loss, acc, AUC)
    │
    ├──→ Validation Phase
    │    │
    │    └── Same forward pass, no gradients
    │
    ├──→ Scheduler Step (CosineAnnealing)
    │
    ├──→ Early Stopping Check
    │    │
    │    └── If val_AUC > best: save model, reset patience
    │        Else: increment patience counter
    │
    └──→ If patience == 5: stop training
```

---

## 9. Grad-CAM Visualization

### 9.1 What is Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations showing which spatial regions of the input image contributed most to the classification decision.

```
┌──────────────────────────────────────────────────────────────────┐
│                     GRAD-CAM COMPUTATION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: Forward pass to target layer                          │
│                                                                  │
│      Input Image ──→ ... ──→ [conv_head] ──→ GAP ──→ logit     │
│                               ↑                                  │
│                        Activations A^k                           │
│                        (feature maps)                            │
│                                                                  │
│   Step 2: Backward pass to get gradients                        │
│                                                                  │
│      ∂y^c / ∂A^k  (gradient of class score w.r.t. activations)  │
│                                                                  │
│   Step 3: Compute importance weights                             │
│                                                                  │
│      α^c_k = (1/Z) Σᵢ Σⱼ (∂y^c / ∂A^k_ij)                       │
│                                                                  │
│      (Global average pool of gradients)                         │
│                                                                  │
│   Step 4: Weighted combination + ReLU                           │
│                                                                  │
│      L^c_Grad-CAM = ReLU(Σ_k α^c_k × A^k)                       │
│                                                                  │
│   Step 5: Upsample to input resolution                          │
│                                                                  │
│      224×224 heatmap showing "where the model looks"            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 Interpretation for Deepfake Detection

```
┌──────────────────────────────────────────────────────────────────┐
│               EXPECTED GRAD-CAM PATTERNS                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   REAL FACE                      DEEPFAKE FACE                   │
│   ┌──────────────┐              ┌──────────────┐                │
│   │              │              │    ████████  │                │
│   │   ░░░░░░░░   │              │   █        █ │ ← Jawline      │
│   │  ░        ░  │              │  █  ████    █│   blending     │
│   │  ░  ░░░░  ░  │              │  █  ████    █│ ← Eye region   │
│   │  ░  ░░░░  ░  │              │  █          █│                │
│   │  ░░░░░░░░░░  │              │   ██████████ │                │
│   │    ░░░░░░    │              │              │                │
│   └──────────────┘              └──────────────┘                │
│                                                                  │
│   Diffuse, low attention       Focused on manipulation          │
│                                boundaries                        │
│                                                                  │
│   Expected attention areas for deepfakes:                        │
│   • Jawline (blending boundary)                                 │
│   • Hairline (face swap boundary)                               │
│   • Eyes (GAN artifacts, unnatural reflections)                 │
│   • Mouth edges (lip sync artifacts)                            │
│   • Skin texture (pore-level anomalies)                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 10. Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **AUC-ROC** | Area under Receiver Operating Characteristic curve | Probability that model ranks random deepfake higher than random real |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **Precision** | TP / (TP + FP) | Of predicted deepfakes, how many are actually deepfakes |
| **Recall** | TP / (TP + FN) | Of actual deepfakes, how many were detected |

---

## 11. Hardware Requirements

### 11.1 Kaggle P100 Memory Budget

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU MEMORY BREAKDOWN (P100 - 16GB)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Model Parameters (FP16):                                       │
│   • EfficientNet-B4: ~35 MB                                     │
│   • BiLSTM: ~19 MB                                              │
│   • Attention + Classifier: ~2 MB                               │
│   Total: ~56 MB                                                 │
│                                                                  │
│   Activations (per batch B=8, T=15):                            │
│   • Input frames: 8 × 15 × 3 × 224 × 224 × 2 = ~1.1 GB         │
│   • Backbone features: 8 × 15 × 1792 × 2 = ~430 MB             │
│   • LSTM hidden states: 8 × 15 × 512 × 2 = ~123 MB             │
│   • Attention: ~50 MB                                           │
│   Total: ~1.7 GB                                                │
│                                                                  │
│   Gradients (backprop):                                         │
│   • Same as activations: ~1.7 GB                                │
│                                                                  │
│   Optimizer States (AdamW):                                     │
│   • 2 × parameters: ~112 MB                                     │
│                                                                  │
│   PyTorch Allocator Overhead: ~500 MB                           │
│                                                                  │
│   ═══════════════════════════════════════════════════           │
│   TOTAL ESTIMATED: ~4 GB                                        │
│   ═══════════════════════════════════════════════════           │
│                                                                  │
│   P100 Available: 16 GB                                         │
│   Safety Margin: 12 GB                                          │
│   Status: ✓ SAFE                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Late Fusion Integration

### 12.1 Export Format

```csv
video_id,P_CNN,true_label
video_001.mp4,0.823,1
video_002.mp4,0.127,0
video_003.mp4,0.956,1
...
```

### 12.2 Fusion with rPPG Stream

```python
# Load predictions from both streams
cnn_df = pd.read_csv('cnn_predictions.csv')
rppg_df = pd.read_csv('rppg_predictions.csv')

# Merge on video_id
merged = cnn_df.merge(rppg_df, on='video_id', suffixes=('_cnn', '_rppg'))

# Fusion strategies:

# 1. Simple Average
merged['P_final'] = (merged['P_CNN'] + merged['P_rPPG']) / 2

# 2. Weighted Average (based on individual AUCs)
w_cnn, w_rppg = 0.55, 0.45  # Tune based on validation
merged['P_final'] = w_cnn * merged['P_CNN'] + w_rppg * merged['P_rPPG']

# 3. Learned Fusion (train LogReg on validation set)
from sklearn.linear_model import LogisticRegression
X_fusion = merged[['P_CNN', 'P_rPPG']].values
y_fusion = merged['true_label'].values
fusion_clf = LogisticRegression().fit(X_fusion, y_fusion)
merged['P_final'] = fusion_clf.predict_proba(X_fusion)[:, 1]
```

---

## 13. Research Paper Checklist

### Essential Elements for Publication

| Element | Status | Notes |
|---------|--------|-------|
| **K-Fold Cross-Validation** | Recommended | Add 5-fold CV with reported std |
| **Ablation Studies** | Recommended | Compare BiLSTM vs averaging, frame counts |
| **Baseline Comparison** | Recommended | Compare with XceptionNet, F3-Net |
| **Confidence Intervals** | Recommended | 95% CI via bootstrap |
| **Statistical Significance** | Recommended | McNemar test vs baselines |
| **Grad-CAM Analysis** | Implemented | Gallery shows model focus areas |
| **Cross-Dataset Eval** | Recommended | Test on FF++, DFDC, Celeb-DF |
| **Failure Case Analysis** | Recommended | When/why model fails |

### Suggested Ablation Experiments

1. **Temporal Model Ablation**
   - No temporal (frame averaging): Baseline
   - BiLSTM only (no attention)
   - BiLSTM + Attention (current)
   - Pure Transformer

2. **Frame Count Ablation**
   - T = 5, 10, 15, 20, 25 frames

3. **Backbone Ablation**
   - EfficientNet-B0, B2, B4, B7
   - ResNet-50, ResNet-101

---

## 14. Known Limitations

1. **Single Dataset**: Trained/tested on one dataset only
2. **Face Requirement**: Assumes clear frontal face in video
3. **Fixed Frame Count**: Uses exactly 15 frames regardless of video length
4. **No Audio**: Does not utilize audio track for lip-sync detection
5. **Memory Bound**: Batch size limited by GPU memory
6. **Detection Only**: Cannot localize manipulated region (only video-level)

---

## 15. References

1. **EfficientNet**: Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML.

2. **MTCNN**: Zhang, K., et al. (2016). "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." IEEE Signal Processing Letters.

3. **BiLSTM**: Schuster, M., & Paliwal, K. K. (1997). "Bidirectional Recurrent Neural Networks." IEEE Transactions on Signal Processing.

4. **Attention**: Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.

5. **Grad-CAM**: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

6. **Albumentations**: Buslaev, A., et al. (2020). "Albumentations: Fast and Flexible Image Augmentations." Information.

7. **FaceForensics++**: Rossler, A., et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV.

8. **Deepfake Detection Survey**: Mirsky, Y., & Lee, W. (2021). "The Creation and Detection of Deepfakes: A Survey." ACM Computing Surveys.

---

*Document Version: 1.0*
*Last Updated: March 2026*
*Notebook: CNN_SPATIAL_STREAM.ipynb*
