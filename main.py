"""
Neuro-Pulse: rPPG-Based Liveness & Deepfake Detection System

Three main use cases:
  1. Webcam Liveness Detection (threshold-based, real-time)
  2. Video Deepfake Detection (ML/DL classifier on 35 rPPG features)
  3. Feature Extraction Pipeline (for training)

Usage:
  python main.py liveness          # Real-time webcam liveness
  python main.py detect <video>    # Detect deepfake in video
  python main.py extract           # Extract features from dataset
  python main.py info              # Show system info and models
"""
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Pulse: rPPG-Based Liveness & Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time webcam liveness detection (threshold-based)
  python main.py liveness

  # Detect deepfake in a single video
  python main.py detect video.mp4

  # Detect with specific model
  python main.py detect video.mp4 --mode ml --model RandomForest
  python main.py detect video.mp4 --mode dl --model CNN_BiLSTM
  python main.py detect video.mp4 --mode ensemble

  # Detect with all models (shows consensus)
  python main.py detect video.mp4 --mode all

  # Batch process multiple videos
  python main.py detect video1.mp4 video2.mp4 video3.mp4

  # Extract features from dataset for ML training
  python main.py extract --real-dir /path/to/real --fake-dir /path/to/fake

  # Show available models and system info
  python main.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ─── Liveness Detection ───────────────────────────────────────
    live_parser = subparsers.add_parser(
        "liveness", help="Real-time webcam liveness detection (Use Case 1)"
    )
    live_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # ─── Deepfake Detection ───────────────────────────────────────
    detect_parser = subparsers.add_parser(
        "detect", help="Detect deepfakes in video files (Use Case 2)"
    )
    detect_parser.add_argument(
        "videos", nargs="+", help="Video file(s) to analyze"
    )
    detect_parser.add_argument(
        "--mode", type=str, default="best",
        choices=["best", "ml", "dl", "ensemble", "all"],
        help="Prediction mode: best (default), ml, dl, ensemble, or all"
    )
    detect_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (for ml/dl modes). See 'info' command for available models."
    )
    detect_parser.add_argument(
        "--method", type=str, default="GREEN",
        choices=["GREEN", "CHROM", "POS"],
        help="rPPG extraction method (default: GREEN)"
    )
    detect_parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Max frames per video (300 = ~10s at 30fps)"
    )
    detect_parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to trained models directory"
    )
    detect_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for batch results (JSON)"
    )
    detect_parser.add_argument(
        "--features", action="store_true",
        help="Include extracted features in output"
    )
    detect_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    # ─── Feature Extraction ───────────────────────────────────────
    extract_parser = subparsers.add_parser(
        "extract", help="Extract rPPG features from video dataset (Use Case 2 training)"
    )
    extract_parser.add_argument("--real-dir", type=str, default=None,
                                help="Directory of real videos")
    extract_parser.add_argument("--fake-dir", type=str, default=None,
                                help="Directory of deepfake videos")
    extract_parser.add_argument("--output", type=str, default="./output",
                                help="Output directory for features")
    extract_parser.add_argument("--method", type=str, default="GREEN",
                                choices=["GREEN", "CHROM", "POS"],
                                help="rPPG extraction method")
    extract_parser.add_argument("--max-frames", type=int, default=300,
                                help="Max frames per video (300 = ~10s at 30fps)")

    # ─── Info ─────────────────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info", help="Show system info and available models"
    )
    info_parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to trained models directory"
    )

    args = parser.parse_args()

    if args.command == "liveness":
        from src.liveness_detector import run_liveness_detection
        run_liveness_detection()

    elif args.command == "detect":
        run_detection(args)

    elif args.command == "extract":
        from src.video_pipeline import process_dataset
        process_dataset(
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            method=args.method,
            max_frames=args.max_frames,
            output_dir=args.output,
        )

    elif args.command == "info":
        show_info(args)

    else:
        parser.print_help()


def run_detection(args):
    """Run deepfake detection on video(s)."""
    import json
    from src.deepfake_detector import DeepfakeDetector

    print("=" * 60)
    print("NEURO-PULSE DEEPFAKE DETECTOR")
    print("=" * 60)
    print()

    # Initialize detector
    detector = DeepfakeDetector(
        model_dir=args.model_dir,
        rppg_method=args.method,
        max_frames=args.max_frames,
    )
    print()

    results = []

    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            results.append({"video": video_path, "error": "File not found"})
            continue

        print(f"\n--- Processing: {video_path} ---")

        result = detector.detect(
            video_path,
            mode=args.mode,
            model_name=args.model,
            return_features=args.features,
        )

        # Print result
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print_detection_result(result, args.verbose)

        result["video_path"] = video_path
        results.append(result)

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[INFO] Results saved to: {args.output}")

    detector.close()


def print_detection_result(result, verbose=False):
    """Print detection result in a formatted way."""
    if "all_models" in result:
        # Mode: all - show all model results
        print("\n  PRIMARY RESULT (Best Model):")
        primary = result["primary"]
        color = "\033[91m" if primary["prediction"] == 1 else "\033[92m"
        reset = "\033[0m"
        print(f"    {color}Prediction: {primary['label']}{reset}")
        print(f"    Confidence: {primary['confidence']:.1f}%")
        print(f"    Probability (fake): {primary['probability']:.4f}")

        print("\n  ENSEMBLE RESULT:")
        ens = result["ensemble"]
        color = "\033[91m" if ens["prediction"] == 1 else "\033[92m"
        print(f"    {color}Prediction: {ens['label']}{reset}")
        print(f"    Confidence: {ens['confidence']:.1f}%")

        print("\n  ALL MODEL PREDICTIONS:")
        print("  " + "-" * 50)
        print(f"  {'Model':<25} {'Type':<6} {'Prediction':<10} {'Prob':>8}")
        print("  " + "-" * 50)

        for name, info in result["all_models"].items():
            label = "Deepfake" if info["prediction"] == 1 else "Real"
            print(f"  {name:<25} {info['type']:<6} {label:<10} {info['probability']:>8.4f}")

    else:
        # Single model result
        color = "\033[91m" if result["prediction"] == 1 else "\033[92m"
        reset = "\033[0m"
        print(f"\n  {color}RESULT: {result['label']}{reset}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Probability (fake): {result['probability']:.4f}")
        print(f"  Model: {result['model']}")

    if verbose and "video_info" in result:
        info = result["video_info"]
        print(f"\n  VIDEO INFO:")
        print(f"    File: {info.get('video', 'N/A')}")
        print(f"    FPS: {info.get('fps', 'N/A')}")
        print(f"    Resolution: {info.get('resolution', 'N/A')}")
        print(f"    Face detection rate: {info.get('face_detection_rate', 0)*100:.1f}%")


def show_info(args):
    """Show system info and available models."""
    from src.models.model_manager import ModelManager
    import torch

    print("=" * 60)
    print("NEURO-PULSE SYSTEM INFO")
    print("=" * 60)

    # System info
    print("\n[SYSTEM]")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model manager
    print("\n[MODELS]")
    manager = ModelManager(model_dir=args.model_dir)
    manager.load_all(verbose=False)

    models = manager.get_available_models()

    print("\n  ML Models:")
    for name in models["ml"]:
        print(f"    - {name}")

    print("\n  DL Models:")
    for name in models["dl"]:
        print(f"    - {name}")

    print("\n[FEATURE EXTRACTION]")
    from src.feature_extractor import FEATURE_NAMES
    print(f"  Feature dimension: {len(FEATURE_NAMES)}")
    print(f"  Feature groups:")
    print(f"    - Forehead spectral: 7 features")
    print(f"    - Forehead temporal: 5 features")
    print(f"    - Left cheek spectral: 7 features")
    print(f"    - Left cheek temporal: 5 features")
    print(f"    - Cross-ROI: 8 features")
    print(f"    - Global quality: 3 features")

    print("\n[USAGE]")
    print("  Detect single video:  python main.py detect video.mp4")
    print("  Use specific model:   python main.py detect video.mp4 --mode ml --model XGBoost")
    print("  Use all models:       python main.py detect video.mp4 --mode all")
    print("  Real-time liveness:   python main.py liveness")


if __name__ == "__main__":
    main()
