"""
Deepfake Detector - End-to-End Pipeline for Video Deepfake Detection.

This module provides a unified interface for detecting deepfakes in videos
using rPPG-based physiological signal analysis and ML/DL classifiers.

Pipeline:
    1. Video Input -> Face Detection (MediaPipe)
    2. ROI Extraction (Forehead, Left Cheek, Right Cheek)
    3. rPPG Signal Extraction (GREEN/CHROM/POS methods)
    4. Feature Extraction (35 physiological features)
    5. Model Inference (ML/DL classifiers)
    6. Prediction Output

Usage:
    from src.deepfake_detector import DeepfakeDetector

    detector = DeepfakeDetector()
    result = detector.detect("video.mp4")
    print(result)
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp

from src.rppg_extractor import (
    bandpass_filter,
    detrend_signal,
    get_roi_mean_rgb,
    RPPG_METHODS,
)
from src.feature_extractor import extract_features, FEATURE_NAMES
from src.models.model_manager import ModelManager
from configs.config import (
    ROI_FOREHEAD,
    ROI_LEFT_CHEEK,
    ROI_RIGHT_CHEEK,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BANDPASS_ORDER,
)


class DeepfakeDetector:
    """
    End-to-end deepfake detector using rPPG physiological signals.

    Integrates video processing, feature extraction, and ML/DL inference
    into a single unified pipeline.

    Attributes:
        model_manager: ModelManager instance for predictions
        face_mesh: MediaPipe FaceMesh for face detection
        rppg_method: rPPG extraction method (GREEN, CHROM, POS)
        max_frames: Maximum frames to process per video
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        rppg_method: str = "GREEN",
        max_frames: int = 300,
    ):
        """
        Initialize the DeepfakeDetector.

        Args:
            model_dir: Path to trained models directory.
                      Defaults to OUTPUT_DIP_FINAL/features.
            device: PyTorch device ('cuda', 'cpu', or None for auto)
            rppg_method: rPPG extraction method (GREEN, CHROM, POS)
            max_frames: Max frames per video (300 = ~10s at 30fps)
        """
        self.rppg_method = rppg_method
        self.max_frames = max_frames

        # Initialize model manager
        self.model_manager = ModelManager(model_dir=model_dir, device=device)
        self.model_manager.load_all(verbose=True)

        # Initialize face mesh (shared instance)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._initialized = True

    def extract_features_from_video(
        self, video_path: str
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract 35-dimensional rPPG features from a video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (features, metadata)
            - features: (35,) numpy array or None on failure
            - metadata: Dict with video processing info
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, {"error": f"Cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rgb_forehead = []
        rgb_left_cheek = []
        rgb_right_cheek = []
        frames_with_face = 0
        frame_count = 0

        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                rgb_fh = get_roi_mean_rgb(
                    frame, landmarks, ROI_FOREHEAD, frame_h, frame_w
                )
                rgb_lc = get_roi_mean_rgb(
                    frame, landmarks, ROI_LEFT_CHEEK, frame_h, frame_w
                )
                rgb_rc = get_roi_mean_rgb(
                    frame, landmarks, ROI_RIGHT_CHEEK, frame_h, frame_w
                )

                if rgb_fh is not None and rgb_lc is not None and rgb_rc is not None:
                    rgb_forehead.append(rgb_fh)
                    rgb_left_cheek.append(rgb_lc)
                    rgb_right_cheek.append(rgb_rc)
                    frames_with_face += 1

        cap.release()

        metadata = {
            "video": os.path.basename(video_path),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "face_frames": frames_with_face,
            "face_detection_rate": frames_with_face / max(frame_count, 1),
        }

        # Need at least 2 seconds of face data
        min_frames = int(2.0 * fps)
        if frames_with_face < min_frames:
            metadata["error"] = f"Insufficient face frames: {frames_with_face}/{min_frames}"
            return None, metadata

        # Convert to arrays
        rgb_fh_arr = np.array(rgb_forehead)
        rgb_lc_arr = np.array(rgb_left_cheek)
        rgb_rc_arr = np.array(rgb_right_cheek)

        # Extract rPPG using chosen method
        extract_fn = RPPG_METHODS.get(self.rppg_method, RPPG_METHODS["GREEN"])
        bvp_fh = bandpass_filter(
            detrend_signal(extract_fn(rgb_fh_arr, fps)),
            fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )
        bvp_lc = bandpass_filter(
            detrend_signal(extract_fn(rgb_lc_arr, fps)),
            fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )
        bvp_rc = bandpass_filter(
            detrend_signal(extract_fn(rgb_rc_arr, fps)),
            fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )

        # Extract 35 features
        features, _ = extract_features(
            bvp_fh, bvp_lc, bvp_rc, fps, BANDPASS_LOW, BANDPASS_HIGH
        )

        return features, metadata

    def detect(
        self,
        video_path: str,
        mode: str = "best",
        model_name: Optional[str] = None,
        return_features: bool = False,
    ) -> Dict:
        """
        Detect if a video is real or deepfake.

        Args:
            video_path: Path to video file
            mode: Prediction mode
                  - "best": Use best ML model (Ensemble_Top3)
                  - "ml": Use specific ML model
                  - "dl": Use specific DL model
                  - "ensemble": Use multi-model ensemble
                  - "all": Run all models and return consensus
            model_name: Model name (for 'ml' and 'dl' modes)
            return_features: Include extracted features in result

        Returns:
            Dict with detection results:
                - prediction: 0 (real) or 1 (deepfake)
                - label: "Real" or "Deepfake"
                - probability: Fake class probability
                - confidence: Confidence percentage
                - model: Model used for prediction
                - video_info: Video metadata
                - features: (optional) 35-dim feature vector
        """
        # Extract features
        features, metadata = self.extract_features_from_video(video_path)

        if features is None:
            return {
                "error": metadata.get("error", "Feature extraction failed"),
                "video_info": metadata,
                "prediction": None,
            }

        # Get prediction
        if mode == "all":
            prediction_result = self.model_manager.predict_video_features(
                features, use_all_models=True
            )
        else:
            prediction_result = self.model_manager.predict(
                features, mode=mode, model_name=model_name
            )

        result = {
            **prediction_result,
            "video_info": metadata,
        }

        if return_features:
            result["features"] = features.tolist()
            result["feature_names"] = FEATURE_NAMES

        return result

    def detect_batch(
        self,
        video_paths: List[str],
        mode: str = "best",
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Detect deepfakes in multiple videos.

        Args:
            video_paths: List of video file paths
            mode: Prediction mode
            verbose: Print progress

        Returns:
            List of detection results
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(video_paths, desc="Processing") if verbose else video_paths

        for video_path in iterator:
            result = self.detect(video_path, mode=mode)
            result["video_path"] = video_path
            results.append(result)

        return results

    def detect_webcam_frame(
        self,
        frame: np.ndarray,
        signal_buffer: "SignalBufferForDetection",
        mode: str = "best",
    ) -> Optional[Dict]:
        """
        Process a single webcam frame for real-time detection.

        Args:
            frame: BGR frame from webcam
            signal_buffer: Buffer collecting RGB signals
            mode: Prediction mode

        Returns:
            Detection result if buffer is ready, else None
        """
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            rgb_fh = get_roi_mean_rgb(
                frame, landmarks, ROI_FOREHEAD, frame_h, frame_w
            )
            rgb_lc = get_roi_mean_rgb(
                frame, landmarks, ROI_LEFT_CHEEK, frame_h, frame_w
            )
            rgb_rc = get_roi_mean_rgb(
                frame, landmarks, ROI_RIGHT_CHEEK, frame_h, frame_w
            )

            if all(x is not None for x in [rgb_fh, rgb_lc, rgb_rc]):
                signal_buffer.add_frame(rgb_fh, rgb_lc, rgb_rc)

        # Check if we have enough frames
        if signal_buffer.is_ready():
            features = signal_buffer.extract_features()
            if features is not None:
                return self.model_manager.predict(features, mode=mode)

        return None

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get lists of available ML and DL models."""
        return self.model_manager.get_available_models()

    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()


class SignalBufferForDetection:
    """
    Signal buffer for real-time deepfake detection.

    Collects RGB signals from video frames and extracts features
    when enough data is accumulated.
    """

    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 5.0,
        rppg_method: str = "GREEN",
    ):
        """
        Initialize signal buffer.

        Args:
            fps: Video frame rate
            buffer_seconds: Seconds of data to collect
            rppg_method: rPPG extraction method
        """
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        self.rppg_method = rppg_method

        self.rgb_forehead: List[np.ndarray] = []
        self.rgb_left_cheek: List[np.ndarray] = []
        self.rgb_right_cheek: List[np.ndarray] = []

    def add_frame(self, rgb_fh: np.ndarray, rgb_lc: np.ndarray, rgb_rc: np.ndarray):
        """Add RGB values from a single frame."""
        self.rgb_forehead.append(rgb_fh)
        self.rgb_left_cheek.append(rgb_lc)
        self.rgb_right_cheek.append(rgb_rc)

        # Maintain buffer size
        if len(self.rgb_forehead) > self.max_frames:
            self.rgb_forehead.pop(0)
            self.rgb_left_cheek.pop(0)
            self.rgb_right_cheek.pop(0)

    def is_ready(self) -> bool:
        """Check if buffer has enough data."""
        return len(self.rgb_forehead) >= int(2.0 * self.fps)

    def extract_features(self) -> Optional[np.ndarray]:
        """Extract features from buffered signals."""
        if not self.is_ready():
            return None

        rgb_fh = np.array(self.rgb_forehead)
        rgb_lc = np.array(self.rgb_left_cheek)
        rgb_rc = np.array(self.rgb_right_cheek)

        extract_fn = RPPG_METHODS.get(self.rppg_method, RPPG_METHODS["GREEN"])

        bvp_fh = bandpass_filter(
            detrend_signal(extract_fn(rgb_fh, self.fps)),
            self.fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )
        bvp_lc = bandpass_filter(
            detrend_signal(extract_fn(rgb_lc, self.fps)),
            self.fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )
        bvp_rc = bandpass_filter(
            detrend_signal(extract_fn(rgb_rc, self.fps)),
            self.fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER
        )

        features, _ = extract_features(
            bvp_fh, bvp_lc, bvp_rc, self.fps, BANDPASS_LOW, BANDPASS_HIGH
        )

        return features

    def clear(self):
        """Clear the buffer."""
        self.rgb_forehead.clear()
        self.rgb_left_cheek.clear()
        self.rgb_right_cheek.clear()


def detect_deepfake(
    video_path: str,
    mode: str = "best",
    model_name: Optional[str] = None,
) -> Dict:
    """
    Convenience function for single video detection.

    Args:
        video_path: Path to video file
        mode: Prediction mode (best, ml, dl, ensemble, all)
        model_name: Model name for ml/dl modes

    Returns:
        Detection result dict
    """
    detector = DeepfakeDetector()
    return detector.detect(video_path, mode=mode, model_name=model_name)
