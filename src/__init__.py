"""
Neuro-Pulse rPPG processing modules.

Modules:
    - face_detector: Face detection utilities
    - rppg_extractor: rPPG signal extraction (GREEN, CHROM, POS)
    - signal_processor: Signal processing utilities
    - feature_extractor: 35-dimensional physiological feature extraction
    - video_pipeline: Video batch processing for feature extraction
    - liveness_detector: Real-time threshold-based liveness detection
    - deepfake_detector: End-to-end ML/DL deepfake detection
    - models: ML and DL model architectures and management
"""
from src.rppg_extractor import (
    extract_green,
    extract_chrom,
    extract_pos,
    RPPG_METHODS,
    bandpass_filter,
    detrend_signal,
    get_roi_mean_rgb,
    SignalBuffer,
)
from src.feature_extractor import (
    extract_features,
    FEATURE_NAMES,
)
from src.signal_processor import (
    estimate_bpm,
    compute_snr,
    compute_spectral_purity,
    compute_peak_prominence,
    compute_roi_correlation,
    compute_signal_stationarity,
)

__all__ = [
    # rPPG extraction
    "extract_green",
    "extract_chrom",
    "extract_pos",
    "RPPG_METHODS",
    "bandpass_filter",
    "detrend_signal",
    "get_roi_mean_rgb",
    "SignalBuffer",
    # Feature extraction
    "extract_features",
    "FEATURE_NAMES",
    # Signal processing
    "estimate_bpm",
    "compute_snr",
    "compute_spectral_purity",
    "compute_peak_prominence",
    "compute_roi_correlation",
    "compute_signal_stationarity",
]
