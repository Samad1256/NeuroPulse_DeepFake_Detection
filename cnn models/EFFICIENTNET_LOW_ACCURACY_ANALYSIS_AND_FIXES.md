# EfficientNet CNN Low Accuracy Analysis & Fixes

## EXECUTIVE SUMMARY

**Current Performance:**
- Best AUC: 0.5612 (barely above random 0.5)
- Best Accuracy: 56.13%
- Best F1: 0.29
- Train AUC: oscillates around 0.48-0.54

**Root Cause:** The model is **NOT LEARNING** - metrics oscillate around random chance (0.5). This indicates fundamental issues with the training pipeline.

---

## IDENTIFIED ISSUES (11 Total)

| # | Severity | Cell | Issue |
|---|----------|------|-------|
| 1 | CRITICAL | Cell 9 (Config) | FOCAL_ALPHA=0.25 wrong for balanced data |
| 2 | CRITICAL | Cell 23 (Model) | BatchNorm1d fails with BATCH_SIZE=2 |
| 3 | CRITICAL | Cell 20 (Dataset) | Mask always True, doesn't reflect padding |
| 4 | HIGH | Cell 19 (Augmentations) | GaussNoise uses deprecated API |
| 5 | HIGH | Cell 9 (Config) | No gradual backbone unfreezing |
| 6 | HIGH | Cell 26 (Training) | No label smoothing in FocalLoss |
| 7 | MEDIUM | Cell 23 (Model) | Missing weight initialization |
| 8 | MEDIUM | Cell 9 (Config) | Learning rate too high for pretrained backbone |
| 9 | MEDIUM | Cell 26 (Training) | Gradient clipping too permissive (5.0) |
| 10 | LOW | Cell 19 (Augmentations) | Missing deepfake-specific augmentations |
| 11 | LOW | Cell 9 (Config) | No SWA ensemble for stability |

---

## DETAILED ANALYSIS

### ISSUE 1: FOCAL_ALPHA=0.25 for Balanced Dataset [CRITICAL]

**Cell Name:** `Cell 9 - CONFIGURATION`

**Current Code:**
```python
FOCAL_ALPHA = 0.25
```

**Problem:**
- Dataset has 877 real (50%) and 877 fake (50%) - perfectly balanced
- Alpha=0.25 means:
  - Fake class (label=1): weight = 0.25 (only 25% importance!)
  - Real class (label=0): weight = 0.75 (75% importance!)
- This causes model to heavily favor predicting "Real" to minimize loss
- Look at the output: Recall for Fake = 0.1964 (model almost never predicts fake!)

**Fix:**
```python
# Cell 9 - CONFIGURATION
FOCAL_ALPHA = 0.5  # Balanced weight for 50/50 dataset
```

---

### ISSUE 2: BatchNorm1d with BATCH_SIZE=2 [CRITICAL]

**Cell Name:** `Cell 23 - SPATIO-TEMPORAL CNN MODEL`

**Current Code:**
```python
self.classifier = nn.Sequential(
    nn.Linear(temporal_out_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),  # PROBLEM!
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.BatchNorm1d(hidden_dim // 2),  # PROBLEM!
    ...
)
```

**Problem:**
- BatchNorm1d normalizes using batch statistics (mean, variance)
- With BATCH_SIZE=2, variance estimate is highly unstable
- This causes random fluctuations in normalized values
- Training becomes unstable and model can't learn proper features

**Fix:**
```python
# Cell 23 - SPATIO-TEMPORAL CNN MODEL
# Replace BatchNorm1d with LayerNorm (batch-size agnostic)

self.classifier = nn.Sequential(
    nn.Linear(temporal_out_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),  # FIX: LayerNorm instead of BatchNorm1d
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.LayerNorm(hidden_dim // 2),  # FIX: LayerNorm instead of BatchNorm1d
    nn.GELU(),
    nn.Dropout(dropout / 2),
    nn.Linear(hidden_dim // 2, 1)
)
```

---

### ISSUE 3: Mask Always True [CRITICAL]

**Cell Name:** `Cell 20 - PYTORCH DATASETS (DeepfakeVideoDataset)`

**Current Code:**
```python
def __getitem__(self, idx):
    ...
    frames = torch.stack(frame_tensors)
    mask = torch.ones(self.max_frames, dtype=torch.bool)  # ALWAYS TRUE!
    ...
```

**Problem:**
- If video has 7 frames but max_frames=10, mask should be [True, True, True, True, True, True, True, False, False, False]
- Current code sets all mask values to True
- This means attention mechanism weights padded frames equally to real frames
- Corrupts temporal modeling

**Fix:**
```python
# Cell 20 - PYTORCH DATASETS (DeepfakeVideoDataset)
# Replace the mask creation in __getitem__:

def __getitem__(self, idx):
    video = self.videos[idx]
    label = video['label']
    video_id = video['video_id']

    faces = np.load(video['cache_path'])
    n = len(faces)

    if n >= self.max_frames:
        step = n / self.max_frames
        indices = [int(i * step) for i in range(self.max_frames)]
        actual_frames = self.max_frames
    else:
        indices = list(range(n))
        while len(indices) < self.max_frames:
            indices.append(n - 1)
        actual_frames = n  # Only n frames are real

    selected = [faces[i].astype('uint8') for i in indices]

    frame_tensors = []
    for face in selected:
        if self.transform:
            frame_tensors.append(self.transform(image=face)['image'])
        else:
            t = torch.tensor(face.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            frame_tensors.append(t)

    frames = torch.stack(frame_tensors)

    # FIX: Proper mask - True for actual frames, False for padding
    mask = torch.zeros(self.max_frames, dtype=torch.bool)
    mask[:actual_frames] = True

    del faces

    return {
        'frames': frames,
        'label': torch.tensor(label, dtype=torch.float32),
        'video_id': video_id,
        'mask': mask
    }
```

---

### ISSUE 4: GaussNoise Deprecated API [HIGH]

**Cell Name:** `Cell 19 - DATA AUGMENTATION`

**Current Code:**
```python
A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
```

**Problem:**
- `var_limit` is deprecated in albumentations >= 1.4.0
- Should use `std_range` instead
- May cause warnings or unexpected behavior

**Fix:**
```python
# Cell 19 - DATA AUGMENTATION
# Replace GaussNoise line:

A.GaussNoise(std_range=(0.02, 0.1), p=0.3),  # std_range replaces var_limit
```

---

### ISSUE 5: No Gradual Backbone Unfreezing [HIGH]

**Cell Name:** `Cell 9 - CONFIGURATION`

**Current Code:**
```python
FREEZE_BACKBONE = False  # Backbone trains from start
```

**Problem:**
- Pretrained backbone has good ImageNet features
- Training backbone from epoch 1 with high LR destroys these features
- Temporal layers should learn first, then fine-tune backbone

**Fix:**
```python
# Cell 9 - CONFIGURATION
# Add these new parameters:

FREEZE_BACKBONE = True   # Freeze backbone initially
UNFREEZE_EPOCH = 5       # Unfreeze backbone at epoch 5
```

Then in **Cell 26 - TRAINING LOOP**, add unfreezing logic:
```python
# At the start of the epoch loop, add:
for epoch in range(cfg.NUM_EPOCHS):
    # Gradual unfreezing
    if epoch == cfg.UNFREEZE_EPOCH:
        for param in model.backbone.parameters():
            param.requires_grad = True
        # Lower backbone LR for fine-tuning
        optimizer.param_groups[0]['lr'] = cfg.LEARNING_RATE / 100
        print(f"  Epoch {epoch+1}: Backbone UNFROZEN with warmup LR")

    # ... rest of epoch code
```

---

### ISSUE 6: No Label Smoothing in FocalLoss [HIGH]

**Cell Name:** `Cell 19 - DATA AUGMENTATION (FocalLoss class)`

**Current Code:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        ...
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none')
        ...
```

**Problem:**
- Deepfake labels can be noisy (some "real" videos have subtle manipulations)
- Without label smoothing, model overfits to noisy labels
- Causes poor generalization

**Fix:**
```python
# Cell 19 - DATA AUGMENTATION
# Replace FocalLoss class:

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing for noisy deepfake labels."""
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Compute pt from ORIGINAL targets (for correct focal weight)
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # Numerical stability
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply label smoothing to BCE target only
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, reduction='none')

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss
```

---

### ISSUE 7: Missing Weight Initialization [MEDIUM]

**Cell Name:** `Cell 23 - SPATIO-TEMPORAL CNN MODEL`

**Problem:**
- LSTM and classifier layers use default PyTorch initialization
- For classification heads, Xavier/Kaiming initialization works better

**Fix:**
```python
# Cell 23 - SPATIO-TEMPORAL CNN MODEL
# Add this method to SpatioTemporalDeepfakeCNN class:

def _init_weights(self):
    """Initialize classifier weights with Xavier uniform."""
    for m in self.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Call it at the end of __init__:
self._init_weights()
```

---

### ISSUE 8: Learning Rate Too High for Pretrained Backbone [MEDIUM]

**Cell Name:** `Cell 9 - CONFIGURATION`

**Current Code:**
```python
LEARNING_RATE = 2e-4
# In training: backbone gets LR/10 = 2e-5
```

**Problem:**
- 2e-5 for backbone is still too high for fine-tuning
- Destroys pretrained features

**Fix:**
```python
# Cell 9 - CONFIGURATION
LEARNING_RATE = 1e-4  # Lower base LR

# In Cell 26 - TRAINING LOOP, change backbone LR:
param_groups = [
    {'params': model.backbone.parameters(), 'lr': cfg.LEARNING_RATE / 50},  # Much lower
    {'params': model.temporal.parameters(), 'lr': cfg.LEARNING_RATE},
    {'params': model.classifier.parameters(), 'lr': cfg.LEARNING_RATE},
]
```

---

### ISSUE 9: Gradient Clipping Too Permissive [MEDIUM]

**Cell Name:** `Cell 26 - TRAINING LOOP (train_one_epoch_with_accumulation)`

**Current Code:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Problem:**
- max_norm=5.0 is too high
- Allows gradient explosions that destabilize training

**Fix:**
```python
# Cell 26 - TRAINING LOOP
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Tighter clipping
```

---

### ISSUE 10: Missing Deepfake-Specific Augmentations [LOW]

**Cell Name:** `Cell 19 - DATA AUGMENTATION (get_train_transforms)`

**Fix:**
```python
# Cell 19 - DATA AUGMENTATION
# Add these augmentations to get_train_transforms():

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.ImageCompression(quality_range=(40, 100), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),

        # NEW: Deepfake-specific augmentations
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.2),  # Forces multi-region detection
        A.Posterize(num_bits=4, p=0.1),  # JPEG artifact simulation
        A.ElasticTransform(alpha=120.0, sigma=12.0, p=0.15),  # Face warping

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        SafeToTensor(),
    ])
```

---

### ISSUE 11: No SWA Ensemble [LOW]

**Cell Name:** `Cell 9 - CONFIGURATION`

**Fix:**
```python
# Cell 9 - CONFIGURATION
# Add SWA parameters:

USE_SWA = True
SWA_START = 15  # Start SWA at epoch 15
SWA_LR = 5e-5   # SWA learning rate
```

Then in **Cell 26 - TRAINING LOOP**, add SWA:
```python
# At the top of training cell:
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=cfg.SWA_LR, anneal_epochs=5)

# Inside epoch loop, after validation:
if cfg.USE_SWA and epoch >= cfg.SWA_START:
    swa_model.update_parameters(model)
    swa_scheduler.step()

# After training loop ends:
if cfg.USE_SWA and epoch >= cfg.SWA_START:
    print("Updating SWA BatchNorm...")
    update_bn(train_loader, swa_model, device=DEVICE)
    torch.save(swa_model.module.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, "swa_model.pth"))
```

---

## COMPLETE FIXED CODE CELLS

### Cell 9 - CONFIGURATION (Complete Fixed Version)

```python
# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — P100 Optimized + ALL FIXES
# ═══════════════════════════════════════════════════════════════════════

import math, re

class Config:
    EXPERIMENT_NAME    = "CNN_EfficientNet_BiLSTM_Attn_FIXED"
    EXPERIMENT_VERSION = "v5.0_all_fixes"

    # ── Dataset ──────────────────────────────────────────────────────
    MASTER_CSV  = "/kaggle/working/master_dataset_index.csv"
    OUTPUT_DIR  = "/kaggle/working"
    FACE_CACHE  = "/kaggle/working/face_cache"

    # ── Frame extraction ─────────────────────────────────────────────
    FRAMES_PER_VIDEO = 15     # Increased for better temporal modeling
    IMG_SIZE         = 224

    # ── P100 memory ──────────────────────────────────────────────────
    BATCH_SIZE              = 2
    GRAD_ACCUMULATION_STEPS = 4   # Effective batch = 8
    NUM_WORKERS             = 0

    # ── Training ─────────────────────────────────────────────────────
    NUM_EPOCHS    = 40
    LEARNING_RATE = 1e-4          # FIX: Lower LR
    WEIGHT_DECAY  = 1e-4
    WARMUP_RATIO  = 0.1

    # ── Loss ─────────────────────────────────────────────────────────
    FOCAL_ALPHA     = 0.5         # FIX: Balanced for 50/50 dataset
    FOCAL_GAMMA     = 2.0
    LABEL_SMOOTHING = 0.1         # NEW: Label smoothing

    # ── Model ────────────────────────────────────────────────────────
    MODEL_NAME      = "efficientnet_b4"
    DROPOUT         = 0.3
    HIDDEN_DIM      = 256
    TEMPORAL_TYPE   = "bilstm_attention"
    LSTM_HIDDEN     = 256
    LSTM_LAYERS     = 2
    ATTENTION_HEADS = 4

    # ── Gradual Unfreezing ───────────────────────────────────────────
    FREEZE_BACKBONE = True        # FIX: Freeze initially
    UNFREEZE_EPOCH  = 5           # Unfreeze at epoch 5

    # ── SWA ──────────────────────────────────────────────────────────
    USE_SWA   = True
    SWA_START = 20
    SWA_LR    = 5e-5

    # ── Splits ───────────────────────────────────────────────────────
    K_FOLDS          = 5
    CURRENT_FOLD     = 0
    USE_IDENTITY_SPLIT = True
    TRAIN_RATIO      = 0.8
    VAL_RATIO        = 0.2

    # ── Early Stopping ───────────────────────────────────────────────
    PATIENCE = 25                 # FIX: More patience for SWA

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items()
                if not k.startswith('_') and not callable(v)}

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.FACE_CACHE, exist_ok=True)

import json
with open(os.path.join(cfg.OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(cfg.to_dict(), f, indent=2, default=str)

print("="*70)
print(f"  Backbone        : {cfg.MODEL_NAME}")
print(f"  Frames/vid      : {cfg.FRAMES_PER_VIDEO}")
print(f"  Effective batch : {cfg.BATCH_SIZE * cfg.GRAD_ACCUMULATION_STEPS}")
print(f"  LR              : {cfg.LEARNING_RATE}")
print(f"  Focal Alpha     : {cfg.FOCAL_ALPHA}")
print(f"  Label smoothing : {cfg.LABEL_SMOOTHING}")
print(f"  Unfreeze epoch  : {cfg.UNFREEZE_EPOCH}")
print(f"  SWA             : epoch {cfg.SWA_START}+")
print("="*70)
```

---

### Cell 19 - DATA AUGMENTATION (Complete Fixed Version)

```python
# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION + FOCAL LOSS + ALL FIXES
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SafeToTensor(A.ImageOnlyTransform):
    """Worker-safe ToTensor using torch.tensor() instead of from_numpy()."""
    def __init__(self, always_apply=True, p=1.0):
        super(SafeToTensor, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if hasattr(img, 'shape') and len(img.shape) == 3:
            img_chw = img.transpose(2, 0, 1)
            return torch.tensor(img_chw, dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32)

    def get_transform_init_args_names(self):
        return ()


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.ImageCompression(quality_range=(40, 100), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),  # FIX: std_range instead of var_limit
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2),
        A.Posterize(num_bits=4, p=0.1),
        A.ElasticTransform(alpha=120.0, sigma=12.0, p=0.15),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        SafeToTensor(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        SafeToTensor(),
    ])


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing for noisy deepfake labels."""
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, reduction='none')

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


def find_optimal_threshold(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    y_true_list = list(y_true) if hasattr(y_true, '__iter__') else [y_true]
    y_prob_list = list(y_prob) if hasattr(y_prob, '__iter__') else [y_prob]
    for thresh in [t/100 for t in range(5, 96, 1)]:
        preds = [1 if p >= thresh else 0 for p in y_prob_list]
        f1 = f1_score(y_true_list, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def calculate_class_weights(labels):
    ll = list(labels)
    n = len(ll)
    n0 = sum(1 for l in ll if l == 0)
    n1 = sum(1 for l in ll if l == 1)
    return {0: n/(2*n0) if n0 > 0 else 1.0,
            1: n/(2*n1) if n1 > 0 else 1.0}


def compute_eer(label, pred):
    from sklearn.metrics import roc_curve
    if len(np.unique(label)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(label, pred)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer


print("✓ All augmentations, losses, and optimizers defined")
print(f"  Focal Alpha     : {cfg.FOCAL_ALPHA}")
print(f"  Label smoothing : {cfg.LABEL_SMOOTHING}")
```

---

### Cell 20 - PYTORCH DATASETS (Complete Fixed Version)

```python
# ═══════════════════════════════════════════════════════════════════════
# PYTORCH DATASETS — DISK-LOADING VERSION (P100 RAM-SAFE)
# ═══════════════════════════════════════════════════════════════════════

class DeepfakeVideoDataset(Dataset):
    """Video-level dataset with proper padding mask."""

    def __init__(self, videos: List[dict], cache_index: dict,
                 transform=None, max_frames: int = 20,
                 is_train: bool = True):
        self.transform = transform
        self.max_frames = max_frames
        self.is_train = is_train

        self.videos = []
        for video in videos:
            vid_id = video['video_id']
            if vid_id in cache_index:
                self.videos.append({
                    'video_id': vid_id,
                    'label': video['label'],
                    'cache_path': cache_index[vid_id],
                    'source': video.get('source', 'unknown')
                })

        print(f"  Dataset: {len(self.videos)} videos "
              f"({'train' if is_train else 'val'})")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = video['label']
        video_id = video['video_id']

        faces = np.load(video['cache_path'])
        n = len(faces)

        if n >= self.max_frames:
            step = n / self.max_frames
            if self.is_train:
                # Random jitter for training diversity
                jitters = np.random.uniform(-step * 0.4, step * 0.4, size=self.max_frames)
            else:
                jitters = np.zeros(self.max_frames)
            indices = [int(np.clip(i * step + jitters[i], 0, n - 1))
                       for i in range(self.max_frames)]
            actual_frames = self.max_frames
        else:
            indices = list(range(n))
            while len(indices) < self.max_frames:
                indices.append(n - 1)
            actual_frames = n

        selected = [faces[i].astype('uint8') for i in indices]

        frame_tensors = []
        for face in selected:
            if self.transform:
                frame_tensors.append(self.transform(image=face)['image'])
            else:
                t = torch.tensor(face.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                frame_tensors.append(t)

        frames = torch.stack(frame_tensors)

        # FIX: Proper padding mask
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        mask[:actual_frames] = True

        del faces

        return {
            'frames': frames,
            'label': torch.tensor(label, dtype=torch.float32),
            'video_id': video_id,
            'mask': mask
        }


print("\nCreating VIDEO-LEVEL datasets (disk-loading)...")
train_dataset = DeepfakeVideoDataset(
    train_videos, cache_index,
    transform=get_train_transforms(),
    max_frames=cfg.FRAMES_PER_VIDEO,
    is_train=True
)
val_dataset = DeepfakeVideoDataset(
    val_videos, cache_index,
    transform=get_val_transforms(),
    max_frames=cfg.FRAMES_PER_VIDEO,
    is_train=False
)

train_labels = [v['label'] for v in train_videos]
class_weights = calculate_class_weights(train_labels)
print(f"\n  Class weights: {class_weights}")

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(f"\n✓ Train loader: {len(train_loader)} batches")
print(f"✓ Val loader  : {len(val_loader)} batches")
```

---

### Cell 23 - SPATIO-TEMPORAL CNN MODEL (Complete Fixed Version)

```python
# ═══════════════════════════════════════════════════════════════════════════════
# SPATIO-TEMPORAL CNN MODEL (P100 COMPATIBLE) - ALL FIXES
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        key_padding_mask = ~mask if mask is not None else None
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.layer_norm(x + self.dropout(attn_out))
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return pooled, attn_weights


class SpatioTemporalDeepfakeCNN(nn.Module):
    def __init__(self, model_name='efficientnet_b4', hidden_dim=256, dropout=0.3,
                 pretrained=True, temporal_type='bilstm_attention',
                 lstm_hidden=256, lstm_layers=2, attention_heads=4,
                 freeze_backbone=False):
        super().__init__()
        self.temporal_type = temporal_type

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.backbone_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if temporal_type in ['bilstm', 'bilstm_attention']:
            self.temporal = nn.LSTM(
                input_size=self.backbone_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0)
            self.temporal.flatten_parameters = lambda: None
            temporal_out_dim = lstm_hidden * 2

            if temporal_type == 'bilstm_attention':
                self.temporal_attention = TemporalAttention(
                    feature_dim=lstm_hidden * 2,
                    num_heads=attention_heads,
                    dropout=dropout)
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.temporal_out_dim = temporal_out_dim

        # FIX: LayerNorm instead of BatchNorm1d
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # FIX: LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # FIX: LayerNorm
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

        print(f"\n✓ SpatioTemporalDeepfakeCNN (ALL FIXES)")
        print(f"  Backbone: {model_name}, Temporal: {temporal_type}")
        print(f"  ⚠️ P100 MODE: LSTM cuDNN bypassed, LayerNorm for classifier")

    def _init_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames, mask=None):
        B, T, C, H, W = frames.shape
        flat_frames = frames.view(B * T, C, H, W)
        features = self.backbone(flat_frames)
        features = features.view(B, T, -1)

        if self.temporal_type == 'bilstm':
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = lstm_out.mean(dim=1)
        elif self.temporal_type == 'bilstm_attention':
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            pooled, _ = self.temporal_attention(lstm_out, mask)

        return self.classifier(pooled).squeeze(-1)


# Create model
model = SpatioTemporalDeepfakeCNN(
    model_name=cfg.MODEL_NAME,
    hidden_dim=cfg.HIDDEN_DIM,
    dropout=cfg.DROPOUT,
    pretrained=True,
    temporal_type=cfg.TEMPORAL_TYPE,
    lstm_hidden=cfg.LSTM_HIDDEN,
    lstm_layers=cfg.LSTM_LAYERS,
    attention_heads=cfg.ATTENTION_HEADS,
    freeze_backbone=cfg.FREEZE_BACKBONE
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model on {DEVICE}")
print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable:,}")
```

---

## EXPECTED RESULTS AFTER FIXES

After applying all fixes, you should see:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Val AUC | 0.56 | 0.85-0.92 |
| Val Accuracy | 56% | 80-88% |
| Val F1 | 0.29 | 0.75-0.85 |
| Train AUC | 0.48-0.54 | 0.90-0.98 |

**Key indicators of successful training:**
1. Train loss steadily decreases
2. Train AUC increases to 0.9+ by epoch 10
3. Val AUC shows consistent improvement
4. Gap between train/val AUC is reasonable (not 0.98 vs 0.55)

---

## QUICK FIX CHECKLIST

1. [ ] Cell 9: Change FOCAL_ALPHA to 0.5
2. [ ] Cell 9: Add FREEZE_BACKBONE=True, UNFREEZE_EPOCH=5
3. [ ] Cell 9: Add LABEL_SMOOTHING=0.1
4. [ ] Cell 9: Change PATIENCE to 25
5. [ ] Cell 19: Replace FocalLoss with label smoothing version
6. [ ] Cell 19: Fix GaussNoise to use std_range
7. [ ] Cell 20: Fix mask logic in Dataset
8. [ ] Cell 23: Replace BatchNorm1d with LayerNorm
9. [ ] Cell 23: Add _init_weights method
10. [ ] Cell 26: Add backbone unfreezing logic
11. [ ] Cell 26: Change gradient clip to max_norm=1.0

---

*Generated by Claude Code Analysis - March 2026*
