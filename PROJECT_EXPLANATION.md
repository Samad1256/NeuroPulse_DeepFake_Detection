# Neuro-Pulse: Project Explanation (Simple Summary)

> This document explains every feature, every component, and the full workflow of our deepfake detection system in **simple English** — suitable for presenting to your professor.

---

## Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [How Does It Work? (The Big Picture)](#2-how-does-it-work-the-big-picture)
3. [Stream 1: rPPG — Reading the Heartbeat from Video](#3-stream-1-rppg--reading-the-heartbeat-from-video)
4. [What is an ROI?](#4-what-is-an-roi)
5. [The 3 rPPG Methods — How We Extract the Pulse](#5-the-3-rppg-methods--how-we-extract-the-pulse)
6. [All 111 Features Explained Simply](#6-all-111-features-explained-simply)
7. [The ML Pipeline — Traditional Machine Learning](#7-the-ml-pipeline--traditional-machine-learning)
8. [The DL Pipeline — 8 Deep Learning Models](#8-the-dl-pipeline--8-deep-learning-models)
9. [Hybrid Ensemble — Combining ML and DL](#9-hybrid-ensemble--combining-ml-and-dl)
10. [Stream 2: CNN — Looking at the Face Visually](#10-stream-2-cnn--looking-at-the-face-visually)
11. [Late Fusion — Merging Both Streams](#11-late-fusion--merging-both-streams)
12. [Training Techniques and Optimizations](#12-training-techniques-and-optimizations)
13. [Complete Workflow Step-by-Step](#13-complete-workflow-step-by-step)
14. [Why This Approach is Strong](#14-why-this-approach-is-strong)

---

## 1. What Does This Project Do?

This project **detects deepfake videos** — videos where a person's face has been artificially swapped or manipulated using AI.

We use **two completely different approaches** running in parallel:
- **Stream 1 (rPPG):** Checks if the person in the video has a real heartbeat visible through their skin
- **Stream 2 (CNN):** Checks if the face *looks* visually real or has subtle manipulation artifacts

Then we **combine both results** (Late Fusion) to make a final decision: **Real or Deepfake**.

---

## 2. How Does It Work? (The Big Picture)

```
                        Video Input (400 Real + 400 Fake)
                                    |
                    ┌───────────────┴───────────────┐
                    |                               |
            Stream 1: rPPG                  Stream 2: CNN
        (Heartbeat Analysis)            (Visual Appearance)
                    |                               |
          Extract 111 features            Detect face with MTCNN
          from skin color changes         Feed into EfficientNet-B4
                    |                               |
          ML Models + DL Models           Classify each frame
          + Hybrid Ensemble               Average across frames
                    |                               |
              P_rPPG (score)                P_CNN (score)
                    |                               |
                    └───────────────┬───────────────┘
                                    |
                              Late Fusion
                         (Combine both scores)
                                    |
                        Final Decision: Real or Fake
```

**Why two streams?** A deepfake can fool the eye (it looks real) but it cannot fake a heartbeat. Conversely, a low-quality video might have a weak heartbeat signal but clearly show visual artifacts. By combining both, we catch deepfakes that either stream alone might miss.

---

## 3. Stream 1: rPPG — Reading the Heartbeat from Video

### What is rPPG?

**rPPG** stands for **Remote Photoplethysmography**. In simple terms:

> When your heart beats, blood rushes to your face. This causes **tiny color changes** in your skin that are invisible to the naked eye but can be measured by a camera.

Think of it like this: every time your heart pumps, your skin gets slightly redder for a fraction of a second (because more blood flows to the surface), then returns to normal. A camera recording at 30 frames per second can capture these tiny fluctuations.

**Real faces** have these rhythmic color changes because they have real blood flow.
**Deepfake faces** do NOT have these changes because the AI that generated them doesn't know anything about blood circulation — it only copies visual appearance.

### How We Extract the Pulse Signal

1. **Read the video** frame by frame
2. **Detect the face** using MediaPipe FaceMesh (which maps 468 points on the face)
3. **Select 9 skin regions** (ROIs) on the face
4. **Measure the average color** (Red, Green, Blue) of each region in every frame
5. **Apply an rPPG algorithm** (CHROM, POS, or GREEN) to separate the heartbeat signal from noise
6. **Filter the signal** to keep only frequencies between 0.7 Hz and 4.0 Hz (which corresponds to 42–240 beats per minute — the range of human heart rates)
7. **Extract 111 features** from this cleaned signal

---

## 4. What is an ROI?

**ROI = Region of Interest**

An ROI is simply a **specific area of the face** where we measure the skin color. We use 9 ROIs:

| ROI | Where on the Face | Why This Area? |
|-----|-------------------|----------------|
| **Forehead** | Full forehead above the eyebrows | Large, flat skin area — strongest pulse signal |
| **Left Cheek** | Left cheek below the eye | Rich blood supply, clearly visible |
| **Right Cheek** | Right cheek below the eye | Mirror of left cheek — should match in real faces |
| **Left Forehead** | Left half of forehead | Lets us check if left and right sides have the same pulse |
| **Right Forehead** | Right half of forehead | Mirror of left forehead |
| **Chin** | Area below the lower lip | Different skin thickness — tests if pulse travels there |
| **Nose** | Nose bridge and tip | Good blood supply but thin skin |
| **Left Jaw** | Left jawline | Farthest from forehead — tests pulse wave travel time |
| **Right Jaw** | Right jawline | Mirror of left jaw |

**Key insight:** In a real face, ALL of these regions show the same heartbeat, just with tiny timing differences (because blood takes time to travel from one area to another). In a deepfake, the color changes in each region are random and unrelated — there is no underlying heartbeat connecting them.

We identify these regions using **MediaPipe FaceMesh**, which detects **468 landmark points** on the face. Each ROI is defined by a specific set of these landmarks (for example, the forehead uses landmarks like #10, #338, #297, etc.).

---

## 5. The 3 rPPG Methods — How We Extract the Pulse

We have 3 different algorithms to extract the heartbeat signal from the skin color data. Each works differently:

### GREEN Method (Simplest)
- **How it works:** Just takes the **Green color channel** over time
- **Why Green?** Hemoglobin (the molecule that carries oxygen in blood) absorbs green light more than red or blue. So the green channel shows the strongest blood volume pulse
- **Pros:** Simple, fast
- **Cons:** Sensitive to lighting changes and head movement

### CHROM Method (Chrominance-Based) — *Our Default*
- **How it works:** Separates color changes caused by **blood flow** from color changes caused by **lighting and motion**
- **In simple terms:** It creates special color combinations (`3R - 2G` and `1.5R + G - 1.5B`) that mathematically cancel out lighting effects, leaving only the heartbeat signal
- **Pros:** Much more robust to lighting changes than GREEN
- **Based on:** Research by de Haan & Jeanne (2013)

### POS Method (Plane Orthogonal to Skin)
- **How it works:** Projects the color signal onto a mathematical plane that is **perpendicular to the skin tone direction**
- **In simple terms:** It finds the "angle" in color space where the heartbeat signal lives and isolates it from everything else
- **Pros:** Most robust of the three methods
- **Based on:** Research by Wang et al. (2017)

**We use CHROM as the default** because it offers the best balance of accuracy and speed.

---

## 6. All 111 Features Explained Simply

After extracting the pulse signal from each ROI, we compute **111 numbers** (features) that describe the video. These features are what the machine learning models use to decide "real or fake."

### A. Forehead Signal Quality (12 features)

These 12 features describe how clean and strong the heartbeat signal is from the forehead:

| Feature | Simple Explanation |
|---------|-------------------|
| `fh_snr` | **Signal-to-Noise Ratio** — How strong is the heartbeat compared to random noise? Higher = cleaner signal. Real faces have high SNR. |
| `fh_spectral_purity` | **Spectral Purity** — Is the signal dominated by ONE frequency (the heartbeat)? Real pulses are pure; fake signals are messy. |
| `fh_peak_prominence` | **Peak Prominence** — How much does the heartbeat frequency "stick out" from surrounding frequencies? Like a mountain peak vs. rolling hills. |
| `fh_dominant_freq` | **Dominant Frequency** — The main heartbeat frequency in Hz. For example, 1.2 Hz = 72 beats per minute. |
| `fh_harmonic_ratio` | **Harmonic Ratio** — Real heartbeats produce "echoes" at double the frequency (harmonics). This measures if those echoes exist. |
| `fh_spectral_entropy` | **Spectral Entropy** — How messy or disordered is the frequency content? Low entropy = clean heartbeat. High entropy = noise. |
| `fh_spectral_centroid` | **Spectral Centroid** — The "center of gravity" of the frequency spectrum. Tells where the energy is concentrated. |
| `fh_mad` | **Mean Absolute Deviation** — How much does the signal fluctuate on average? Captures pulse amplitude. |
| `fh_std` | **Standard Deviation** — How spread out are the signal values? Measures overall variability. |
| `fh_zcr` | **Zero-Crossing Rate** — How often does the signal cross zero? Captures the oscillatory (wave-like) nature of the pulse. |
| `fh_kurtosis` | **Kurtosis** — How "peaky" is the signal? Real pulses have sharp, characteristic peaks. |
| `fh_skewness` | **Skewness** — Is the signal asymmetric? Real pulse waves rise sharply (blood pumps in) then fall slowly (blood drains). |

### B. Left Cheek Signal Quality (12 features)

Exactly the same 12 measurements as forehead, but taken from the **left cheek**:
`lc_snr`, `lc_spectral_purity`, `lc_peak_prominence`, `lc_dominant_freq`, `lc_harmonic_ratio`, `lc_spectral_entropy`, `lc_spectral_centroid`, `lc_mad`, `lc_std`, `lc_zcr`, `lc_kurtosis`, `lc_skewness`

**Why both regions?** If the forehead and cheek both show similar strong signals, it's probably real. If one is strong and the other is weak or they disagree, something is wrong.

### C. Cross-ROI Correlation (8 features)

These measure **how similar the pulse signals are between different face regions**:

| Feature | Simple Explanation |
|---------|-------------------|
| `corr_fh_lc` | Correlation between forehead and left cheek pulse. Real: HIGH (same heart). Fake: LOW (random). |
| `corr_fh_rc` | Correlation between forehead and right cheek. |
| `corr_lc_rc` | Correlation between left and right cheek. Should be very high in real faces (mirror regions). |
| `coherence_fh_lc` | Do the forehead and left cheek share the same **frequency** content? Measured in the heartbeat band (0.7–4 Hz). |
| `coherence_fh_rc` | Same for forehead vs. right cheek. |
| `coherence_lc_rc` | Same for left vs. right cheek. |
| `phase_diff_fh_lc` | **Timing difference** between forehead and left cheek. Blood takes a few milliseconds longer to reach the cheek. Real faces show a small, consistent delay. |
| `phase_diff_fh_rc` | Timing difference between forehead and right cheek. |

**Key insight:** In a real face, the heart pumps blood that reaches ALL regions — so all regions should show correlated, coherent signals with small phase delays. In deepfakes, these relationships are broken.

### D. BPM and Signal Quality (3 features)

| Feature | Simple Explanation |
|---------|-------------------|
| `bpm_estimate` | The estimated **heart rate** in beats per minute. Derived from the dominant frequency of the forehead signal. |
| `signal_stationarity` | Does the signal stay consistent over time, or does it wildly change? Real heartbeats are steady; noise fluctuates. |
| `bpm_consistency` | Is the heart rate the **same** across forehead, left cheek, and right cheek? In a real person, yes. In a deepfake, each region shows a different "fake" heart rate. |

### E. Heart Rate Variability — HRV (8 features)

These capture the **natural variation** in time between heartbeats. Your heart doesn't beat like a metronome — it naturally speeds up and slows down slightly. This pattern is very hard to fake.

| Feature | Simple Explanation |
|---------|-------------------|
| `hrv_rmssd` | Beat-to-beat variability. How much does the time between consecutive beats change? |
| `hrv_sdnn` | Overall variability in beat timing across the entire signal. |
| `hrv_pnn50` | What percentage of consecutive beats differ by more than 50 milliseconds? |
| `hrv_pnn20` | Same but with a 20ms threshold (more sensitive). |
| `hrv_lf_power` | Energy in the **low-frequency** band (0.04–0.15 Hz) of heart rate variability. Related to the body's stress response system. |
| `hrv_hf_power` | Energy in the **high-frequency** band (0.15–0.4 Hz). Related to breathing (your heart speeds up slightly when you inhale). |
| `hrv_lf_hf_ratio` | Ratio of low to high frequency. A well-known medical metric for autonomic nervous system balance. |
| `hrv_total_power` | Total energy in heart rate variability. |

**Why this matters:** Deepfakes have no real nervous system, so they cannot reproduce realistic HRV patterns.

### F. Signal Quality Metrics (5 features)

| Feature | Simple Explanation |
|---------|-------------------|
| `signal_energy` | How strong is the pulse signal overall? (Average squared amplitude) |
| `signal_entropy` | How random/unpredictable is the signal? Real pulses are structured; noise is random. |
| `spectral_flatness` | Is the frequency content "flat" like noise (value near 1) or "peaked" like a heartbeat (value near 0)? |
| `crest_factor` | Ratio of the tallest peak to the average signal. Real pulses have sharp, characteristic peaks. |
| `signal_complexity` | How complex is the signal structure? Measures the number of unique patterns in the waveform. |

### G. Facial Geometry (20 features)

These don't use the heartbeat at all — they measure the **shape and proportions of the face** using the 468 landmark points:

| Feature | Simple Explanation |
|---------|-------------------|
| `geo_eye_distance_ratio` | How far apart are the eyes relative to face width? |
| `geo_eye_width_ratio` | Is the left eye the same width as the right eye? (Should be ~1.0) |
| `geo_eye_height_ratio` | Is the left eye the same height as the right? |
| `geo_left_eye_aspect` | Shape of the left eye (width/height). |
| `geo_right_eye_aspect` | Shape of the right eye. |
| `geo_nose_width_ratio` | How wide is the nose compared to the face? |
| `geo_mouth_width_ratio` | How wide is the mouth compared to the face? |
| `geo_mouth_aspect` | Shape of the mouth (width/height). |
| `geo_face_aspect` | Is the face long and narrow or short and wide? (height/width) |
| `geo_jaw_symmetry` | Is the jawline symmetric? Left side vs. right side. |
| `geo_nose_position` | Where is the nose positioned vertically on the face? |
| `geo_jaw_angle` | How sharp or wide is the chin angle? |
| `geo_eye_nose_angle` | The angle formed between the eyes and nose. |
| `geo_upper_face_ratio` | What proportion of the face height is the upper third (forehead to eyes)? |
| `geo_middle_face_ratio` | Proportion of the middle third (eyes to nose). |
| `geo_lower_face_ratio` | Proportion of the lower third (nose to chin). |
| `geo_left_symmetry` | Left cheek distance from the nose relative to face width. |
| `geo_right_symmetry` | Right cheek distance from the nose. |
| `geo_overall_symmetry_mean` | Average facial symmetry score (1.0 = perfect). |
| `geo_overall_symmetry_std` | How much does symmetry vary? Higher = more asymmetric. |

**Why this matters:** Deepfake algorithms sometimes produce faces with slightly wrong proportions — eyes slightly too close, jaw a bit off, etc. These are invisible to the human eye but measurable with geometry.

### H. Extended Spatial Correlations (12 features)

Similar to the Cross-ROI features (Section C), but covering MORE region pairs:

| Feature | Simple Explanation |
|---------|-------------------|
| `corr_fh_chin` | Pulse correlation: forehead vs. chin |
| `corr_fh_nose` | Pulse correlation: forehead vs. nose |
| `corr_chin_nose` | Pulse correlation: chin vs. nose |
| `corr_lc_chin` | Pulse correlation: left cheek vs. chin |
| `corr_rc_chin` | Pulse correlation: right cheek vs. chin |
| `corr_lc_nose` | Pulse correlation: left cheek vs. nose |
| `corr_rc_nose` | Pulse correlation: right cheek vs. nose |
| `corr_lf_rf` | Pulse correlation: left forehead vs. right forehead |
| `corr_lj_rj` | Pulse correlation: left jaw vs. right jaw |
| `coherence_fh_chin` | Frequency coherence: forehead vs. chin |
| `coherence_chin_nose` | Frequency coherence: chin vs. nose |
| `coherence_lj_rj` | Frequency coherence: left jaw vs. right jaw |

**The idea:** The more region pairs we check, the harder it is for a deepfake to fool us. Even if one pair happens to look real by chance, having 20 different pairs makes it nearly impossible for ALL of them to look real simultaneously in a fake video.

### I. Multi-Band Frequency Analysis (9 features)

Instead of looking at the whole frequency range at once, we split it into 3 bands:

| Feature | Simple Explanation |
|---------|-------------------|
| `band_power_low_fh` | Energy in 0.7–1.5 Hz band (42–90 BPM, resting heart rate) for forehead |
| `band_power_mid_fh` | Energy in 1.5–3.0 Hz band (90–180 BPM, active heart rate) for forehead |
| `band_power_high_fh` | Energy in 3.0–4.0 Hz band (180–240 BPM, mostly noise/artifacts) for forehead |
| `band_power_low_lc` | Same low band for left cheek |
| `band_power_mid_lc` | Same mid band for left cheek |
| `band_power_high_lc` | Same high band for left cheek |
| `band_ratio_low_high` | Ratio of low to high band. Real faces: most energy in low band. Fakes: elevated high-frequency artifacts. |
| `band_ratio_mid_high` | Ratio of mid to high band. |
| `band_power_variance` | How unevenly is energy spread across the 3 bands? |

**Why this matters:** A real heartbeat concentrates energy in the low band (resting humans have ~60–90 BPM). Deepfakes often show energy scattered randomly across all bands.

### J. Spatial Pulse Variance (5 features)

These check if the heart rate is **consistent across ALL 9 face regions**:

| Feature | Simple Explanation |
|---------|-------------------|
| `bpm_variance_all_regions` | How different are the BPM estimates across all 9 ROIs? Real: nearly identical. Fake: all different. |
| `bpm_std_all_regions` | Standard deviation of BPM across regions. |
| `bpm_range_all_regions` | Biggest BPM minus smallest BPM. A range of 5 BPM is normal; a range of 50 BPM screams "fake." |
| `bpm_iqr_all_regions` | The middle 50% spread of BPMs (robust to outliers). |
| `spatial_pulse_consistency` | A score from 0 to 1. 1 = perfectly consistent heartbeat everywhere. 0 = completely inconsistent. |

### K. Temporal Stability (5 features)

These check if the heartbeat stays **stable over time** (the signal is split into 5 time windows):

| Feature | Simple Explanation |
|---------|-------------------|
| `temporal_bpm_std` | Does the heart rate jump around between time windows? Real: stable. Fake: erratic. |
| `temporal_bpm_range` | Maximum heart rate swing over the video duration. |
| `temporal_snr_std` | Does the signal quality stay consistent or fluctuate wildly? |
| `temporal_stability_score` | Overall stability score (0 to 1). Higher = more stable. |
| `temporal_consistency_index` | Another stability metric. Higher = more consistent. |

### L. Skin Reflection (4 features)

These analyze the **brightness patterns** on the skin:

| Feature | Simple Explanation |
|---------|-------------------|
| `skin_reflection_variance_fh` | How much does the brightness of forehead skin vary over time? Real skin has natural specular (shiny) reflections. |
| `skin_reflection_variance_lc` | Same for left cheek. |
| `skin_reflection_mean_diff` | Difference in average brightness between forehead and cheek. |
| `specular_reflection_score` | How the relative brightness between regions changes over time. Real faces have natural, correlated brightness changes. |

### M. Patch-Level Signal Quality (3 features)

| Feature | Simple Explanation |
|---------|-------------------|
| `snr_std_all_regions` | Is the signal quality uniformly good or bad across the face? Real faces: consistent quality everywhere. |
| `snr_range_all_regions` | Biggest quality difference between any two regions. |
| `patch_quality_consistency` | Consistency score (0 to 1). Real faces: high. Fakes: low (some patches accidentally look good, others don't). |

### N. Phase Synchronization (3 features)

| Feature | Simple Explanation |
|---------|-------------------|
| `phase_sync_mean` | Average timing offset of the pulse wave across all region pairs. |
| `phase_sync_std` | How consistent are those timing offsets? Real: consistent (blood flows predictably). Fake: random. |
| `phase_sync_consistency` | Score from 0 to 1. Higher = more synchronized = more likely real. |

### O. RGB Channel Correlation (2 features)

| Feature | Simple Explanation |
|---------|-------------------|
| `rgb_corr_green_red` | How correlated are the Green and Red color channels over time? Real skin has specific biological relationships between R, G, B changes (due to hemoglobin absorption). |
| `rgb_corr_green_blue` | Green vs. Blue correlation. Deepfakes break these biological color relationships because GANs manipulate pixels for visual look, not for biological accuracy. |

---

## 7. The ML Pipeline — Traditional Machine Learning

After extracting 111 features per video, we transform them and feed them into machine learning classifiers.

### Feature Preprocessing Pipeline

```
111 raw features
      |
      v
RobustScaler (normalize values, robust to outliers)
      |
      v
SelectKBest (keep the best 40 features using ANOVA F-test)
      |
      v
PolynomialFeatures (degree=2, creates feature interactions & squares)
      |
      v
~820 expanded features (ready for classification)
```

### ML Classifiers (11 models)

| Model | What It Does (Simple) |
|-------|----------------------|
| **XGBoost** | Builds many small decision trees one after another, where each new tree corrects the mistakes of the previous ones. State-of-the-art for tabular data. Uses GPU. |
| **LightGBM** | Similar to XGBoost but faster — it builds trees by choosing the best leaf to split (leaf-wise growth). Uses GPU. |
| **Random Forest** | Builds hundreds of independent decision trees in parallel and lets them "vote" on the answer. Robust and hard to overfit. |
| **Gradient Boosting** | Like XGBoost but uses scikit-learn's implementation. Reliable baseline. |
| **Extra Trees** | Like Random Forest but with random split points — adds more randomness, which can improve generalization. |
| **HistGradientBoosting** | Scikit-learn's fast histogram-based gradient boosting. Fastest for large datasets. |
| **SVM (RBF)** | Finds the best boundary in high-dimensional space using a "radial basis function" kernel. Good for complex patterns. |
| **SVM (Linear)** | Finds a straight-line boundary. Simpler but fast. |
| **AdaBoost** | Trains weak classifiers (small trees) and gives more weight to hard-to-classify samples. |
| **KNN** | Classifies a video by looking at its K nearest neighbors in feature space and picking the majority label. |
| **Logistic Regression** | Fits a linear model and uses sigmoid to output probabilities. Simple but interpretable. |

### ML Ensemble Methods (5 ensembles)

| Ensemble | How It Combines Models |
|----------|----------------------|
| **Soft Voting (Top 7)** | Takes the 7 best individual models and averages their probability outputs. |
| **Weighted Voting** | Same as above but gives higher weight to models with better AUC scores. |
| **Stacking Classifier** | Trains a meta-learner (Logistic Regression) that learns HOW to combine the base models' outputs optimally. |
| **Stacking with MLP** | Same stacking idea but uses a small neural network as the meta-learner. |
| **Calibrated Stacking** | Stacking + probability calibration so the output probabilities are more reliable. |

**Hyperparameter tuning:** Every model uses **RandomizedSearchCV** or **GridSearchCV** with 5-fold cross-validation to find the best settings. This means we try many combinations of settings and pick the one that performs best on held-out data.

---

## 8. The DL Pipeline — 8 Deep Learning Models

The same 111 features (after scaling) are also fed into 8 different neural network architectures:

### Model 1: CNN-1D (with Squeeze-Excitation)
- **Plain English:** Treats the 111 features like a 1D signal (similar to an audio waveform) and slides filters across it to find patterns. The "Squeeze-Excitation" block lets the model learn which features are most important and pay more attention to them.
- **Analogy:** Like reading a sentence and highlighting the important words.

### Model 2: BiLSTM with Multi-Head Attention
- **Plain English:** Groups the 111 features into sequences of 5, then reads them forward AND backward (bidirectional) using an LSTM (a type of neural network designed for sequences). Multi-Head Attention lets it focus on the most relevant parts of the sequence.
- **Analogy:** Like reading a paragraph forwards and backwards, then highlighting the key sentences.

### Model 3: CNN-BiLSTM Hybrid
- **Plain English:** First uses CNN to find local patterns, then uses BiLSTM to understand how those patterns relate to each other across the feature sequence.
- **Analogy:** First zoom in on individual words (CNN), then understand the sentence meaning (BiLSTM).

### Model 4: Transformer (Pre-LayerNorm)
- **Plain English:** Uses the same attention mechanism as GPT and BERT. Every feature can "look at" every other feature to understand relationships. A special CLS token collects all the information for the final classification.
- **Analogy:** Like a meeting where every team member can talk to every other member simultaneously.

### Model 5: PhysNet MLP (Dense Residual)
- **Plain English:** A deep stack of fully-connected layers with "skip connections" (residual connections) so information can flow directly from early layers to later layers without getting lost.
- **Analogy:** Like a highway with express lanes that bypass traffic jams.

### Model 6: MultiScale CNN with CBAM
- **Plain English:** Uses 4 different filter sizes (3, 5, 7, 11) simultaneously to capture patterns at different scales — small local patterns AND large global patterns. CBAM (Convolutional Block Attention Module) adds both spatial and channel attention.
- **Analogy:** Like looking at a painting from up close AND far away at the same time.

### Model 7: Temporal Attention Network
- **Plain English:** Uses a standard Transformer encoder (like in NLP) with Squeeze-Excitation on top. Treats features as a sequence and learns which parts of the sequence matter most.

### Model 8: Wide & Deep
- **Plain English:** Has two parallel paths — a "Wide" path that memorizes direct feature-to-label relationships (like a lookup table), and a "Deep" path that learns complex non-linear patterns. Combined, they get the best of both worlds.
- **Analogy:** Combining a cheat sheet (wide) with deep understanding (deep).

### DL Ensemble
After training all 8 models, we combine them:
- **Average Ensemble:** Average all predictions
- **Weighted Ensemble:** Weight each model by its individual performance
- **Selective Ensemble:** Only use models that performed above a threshold

---

## 9. Hybrid Ensemble — Combining ML and DL

The final step of Stream 1 combines the **best ML models** and **all DL models** into one "super-ensemble":

| Method | How It Works |
|--------|-------------|
| **Simple Average** | Average the probability outputs from all ML and DL models equally. |
| **Weighted Average** | Weight each model by its AUC score. Better models get more influence. |
| **Meta-Learner Stacking** | Train a Logistic Regression that learns the optimal way to combine all models. |
| **Rank-Based** | Convert each model's outputs to ranks (1st, 2nd, 3rd...) and average the ranks. Robust to different probability scales. |

The best-performing method is automatically selected as the final **P_rPPG** score.

---

## 10. Stream 2: CNN — Looking at the Face Visually

While Stream 1 analyzes invisible heartbeat signals, Stream 2 looks at the **visual appearance** of the face.

### Step 1: Face Detection (MTCNN)

**MTCNN** (Multi-task Cascaded Convolutional Networks) is a face detection model that:
- Finds the face in each video frame
- Outputs a cropped 224x224 pixel face image
- Uses a 40-pixel margin around the face (to include some context)
- Falls back to a simple center crop if no face is found

### Step 2: Frame Sampling

We don't process every frame (too slow). Instead:
- We extract **15 evenly spaced frames** from each video
- For each frame, we extract one face crop

### Step 3: Data Augmentation (Training Only)

To prevent the model from memorizing the training data, we apply random transformations:

| Augmentation | What It Does |
|-------------|--------------|
| **HorizontalFlip** | Randomly mirrors the face left-to-right |
| **ShiftScaleRotate** | Slightly moves, resizes, or rotates the face |
| **RandomBrightnessContrast** | Changes brightness and contrast randomly |
| **HueSaturationValue** | Slightly shifts colors |
| **ImageCompression** | Adds JPEG compression artifacts (relevant because real deepfakes are often compressed) |
| **GaussNoise** | Adds random noise |
| **GaussianBlur** | Slightly blurs the image |
| **CoarseDropout** | Randomly blacks out small rectangular patches (forces model to not rely on any single area) |

### Step 4: EfficientNet-B4 (The Neural Network)

```
Input: 224 x 224 x 3 face image
         |
         v
EfficientNet-B4 Backbone (pretrained on ImageNet)
  - 19.3 million parameters
  - Learns to extract visual features (edges, textures, patterns)
  - Output: 1792-dimensional feature vector
         |
         v
Classifier Head:
  Linear (1792 -> 256)
  BatchNorm
  ReLU activation
  Dropout (40%)
  Linear (256 -> 1 output logit)
         |
         v
Sigmoid -> probability (0 = real, 1 = fake)
```

**Why EfficientNet-B4?** It is one of the most efficient and accurate image classification models. It was trained on millions of images (ImageNet), so it already understands visual patterns. We just fine-tune it for our deepfake detection task.

### Step 5: Video-Level Prediction

1. Get the per-frame probability for each of the 15 frames
2. Average them all → **P_CNN** (one number per video)

### Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (learning rate=0.0001, weight decay=0.01) |
| Loss Function | BCEWithLogitsLoss (binary cross-entropy on logits) |
| Scheduler | CosineAnnealingLR (goes from 0.0001 down to 0.000001 over 15 epochs) |
| Mixed Precision | AMP enabled (trains faster on GPU, uses less memory) |
| Gradient Clipping | Max norm 1.0 (prevents exploding gradients) |
| Early Stopping | Patience=5 (stops if no improvement for 5 epochs) |
| Batch Size | 32 |
| Max Epochs | 15 |

---

## 11. Late Fusion — Merging Both Streams

After both notebooks have run, we have two CSV files:
- `rppg_predictions.csv` → contains `video_id` and `P_rPPG` (score from Stream 1)
- `cnn_predictions.csv` → contains `video_id` and `P_CNN` (score from Stream 2)

We merge them on `video_id` and combine the scores:

### Fusion Strategies

| Strategy | Formula | When It Works Best |
|----------|---------|-------------------|
| **Simple Average** | P_final = (P_rPPG + P_CNN) / 2 | When both streams are equally reliable |
| **Weighted Average** | P_final = 0.6 * P_rPPG + 0.4 * P_CNN | When one stream is more accurate |
| **Learned (LogReg)** | P_final = LogisticRegression(P_rPPG, P_CNN) | Learns the optimal combination automatically |
| **Rank-Based** | Average the ranks of both scores | Robust when probability scales differ |

### Final Decision
- If P_final > 0.5 → **Deepfake**
- If P_final <= 0.5 → **Real**

---

## 12. Training Techniques and Optimizations

### Techniques Used Across Both Streams

| Technique | What It Does | Why We Use It |
|-----------|-------------|---------------|
| **Mixed Precision (AMP)** | Uses 16-bit floats for most computation instead of 32-bit | Trains 2x faster, uses less GPU memory |
| **CosineAnnealingLR** | Gradually decreases the learning rate following a cosine curve | Helps the model converge to a better solution |
| **Early Stopping** | Stops training if performance hasn't improved for 5 epochs | Prevents overfitting |
| **Gradient Clipping** | Caps gradient magnitudes at 1.0 | Prevents training instability |
| **Label Smoothing** | Softens labels from 0/1 to 0.05/0.95 | Reduces overconfidence, improves generalization |
| **Focal Loss** | Gives more weight to hard-to-classify samples | Helps with difficult examples |
| **Dropout** | Randomly disables neurons during training | Prevents overfitting |
| **BatchNorm** | Normalizes layer outputs to have mean=0, std=1 | Stabilizes training, enables faster learning |
| **RobustScaler** | Scales features using median and IQR (robust to outliers) | Better than standard scaling when data has outliers |
| **Stratified Split** | Splits data keeping 50/50 real/fake ratio in both train and test | Prevents imbalanced splits |
| **SEED=42** | Fixed random seed everywhere | Ensures reproducibility |

### Kaggle P100 Optimizations

| Optimization | Detail |
|-------------|--------|
| GPU-accelerated XGBoost/LightGBM | Uses CUDA for tree-based models |
| Batch size 32 for CNN | Optimal for P100's 16GB VRAM |
| AMP mixed precision | Halves memory usage, enables larger batches |
| Auto-save checkpoints every 25 videos | Can resume if session times out |

---

## 13. Complete Workflow Step-by-Step

### Phase 1: Data Processing (Stream 1 — `final_MODEL.ipynb`)

1. Load 400 real + 400 fake videos
2. For each video:
   a. Sample N frames evenly
   b. Detect face using MediaPipe FaceMesh (468 landmarks)
   c. Extract 9 ROIs from the face
   d. Compute mean RGB color per ROI per frame
   e. Apply CHROM algorithm to extract pulse signal
   f. Bandpass filter the signal (0.7–4.0 Hz)
   g. Compute 111 features (signal quality, cross-ROI, HRV, geometry, etc.)
3. Save features as NumPy arrays and CSV

### Phase 2: ML Training (Stream 1 — `final_MODEL.ipynb`)

4. Preprocess: RobustScaler → SelectKBest(40) → PolynomialFeatures(degree=2)
5. Train 11 ML classifiers with hyperparameter tuning
6. Build 5 ensemble methods
7. Evaluate all models on held-out test set

### Phase 3: DL Training (Stream 1 — `final_MODEL.ipynb`)

8. Optuna hyperparameter optimization (50 trials)
9. Train 8 DL architectures
10. Build DL ensemble (average, weighted, selective)
11. Build Hybrid ML+DL ensemble
12. Export P_rPPG per video

### Phase 4: CNN Training (Stream 2 — `CNN_SPATIAL_STREAM.ipynb`)

13. Extract faces from all videos using MTCNN (15 frames per video)
14. Split into train/validation (same seed, same strategy as Stream 1)
15. Train EfficientNet-B4 with augmentations
16. Export P_CNN per video

### Phase 5: Late Fusion

17. Merge P_rPPG and P_CNN on video_id
18. Apply fusion strategies (average, weighted, learned, rank-based)
19. Final prediction: Real or Deepfake

---

## 14. Why This Approach is Strong

### Multi-Modal Detection
We don't rely on just one signal. We use **physiological signals (heartbeat)** AND **visual appearance**. A deepfake would need to fool BOTH systems simultaneously, which is extremely difficult.

### Biological Signals Can't Be Faked
Current deepfake generators (like GANs) focus on making faces look realistic. They have no concept of blood circulation, heart rate variability, or pulse wave propagation. Our rPPG features exploit this fundamental gap.

### Redundancy Through Ensemble
We don't trust any single model. We train **11 ML classifiers + 8 DL models + multiple ensembles** and combine their outputs. Even if some models are wrong, the majority vote is usually correct.

### Spatial Consistency Checks
We check pulse signals across **9 different face regions** and measure their correlations. In a real face, all 9 regions show the same heartbeat. In a deepfake, they don't. This gives us (9 choose 2) = 36 pairwise relationships to check.

### Temporal Consistency Checks
We verify that the heartbeat stays stable over time. A real heart doesn't randomly jump from 60 to 150 BPM. We measure this stability with 5 temporal features.

### Feature Diversity
Our 111 features cover **14 different categories** — from signal processing (SNR, spectral purity) to physiology (HRV, phase synchronization) to geometry (facial proportions). This diversity makes our system robust against different types of deepfakes.

### Cross-Notebook Alignment
Both notebooks use:
- Same random seed (42)
- Same data splitting strategy (single stratified train_test_split)
- Same sorted video ordering
- Same export format (video_id-based CSV)

This ensures perfect alignment when combining results in Late Fusion.

---

*This document covers every component of the Neuro-Pulse deepfake detection system. Each feature, each model, and each design decision serves a specific purpose in achieving accurate, robust deepfake detection.*
