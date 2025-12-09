"""
RVC Training Pipeline
A seamless pipeline for training RVC voice conversion models.

Usage:
    python pipeline.py train --model-name "my_voice" --audio-file "vocals.wav" --epochs 200
    python pipeline.py infer --model-name "my_voice" --input "input.wav" --output "output.wav"
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

from config import PipelineConfig, TrainingConfig, InferenceConfig, PLATFORM
from setup import RVCSetup
from dataset import DatasetPreparer
from train import RVCTrainer
from inference import RVCInference

logger = logging.getLogger(__name__)


def print_platform_info():
    """Print detected platform information."""
    logger.info("=" * 50)
    logger.info("Platform Configuration (Auto-Detected)")
    logger.info("=" * 50)
    logger.info(f"System: {PLATFORM.system}")
    logger.info(f"CPU Cores: {PLATFORM.cpu_count}")
    logger.info(f"CUDA Available: {PLATFORM.has_cuda}" + (f" ({PLATFORM.cuda_device_count} GPU(s))" if PLATFORM.has_cuda else ""))
    logger.info(f"Tools: aria2c={'Yes' if PLATFORM.has_aria2c else 'No'}, git={'Yes' if PLATFORM.has_git else 'No'}, ffmpeg={'Yes' if PLATFORM.has_ffmpeg else 'No'}")
    logger.info("=" * 50)


class RVCPipeline:
    """
    Complete RVC training and inference pipeline.

    This class orchestrates the entire process from setup to inference:
    1. Setup: Clone RVC repo, download pretrained models
    2. Dataset Preparation: Preprocess audio, extract F0, extract features
    3. Training: Train the voice conversion model
    4. Inference: Convert audio using the trained model
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the RVC pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.setup = RVCSetup(self.config)
        self.dataset = DatasetPreparer(self.config)
        self.trainer = RVCTrainer(self.config)
        self.inference = RVCInference(self.config)

    def run_setup(self, force: bool = False) -> bool:
        """
        Run the setup process.

        Args:
            force: Force reinstallation

        Returns:
            True if setup successful
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Setting up RVC environment")
        logger.info("=" * 60)
        return self.setup.setup_all(force_reinstall=force)

    def run_full_training(
        self,
        model_name: str,
        audio_files: List[Path],
        epochs: int = 200,
        batch_size: int = 8,
        save_frequency: int = 50,
        skip_setup: bool = False
    ) -> bool:
        """
        Run the complete training pipeline.

        Args:
            model_name: Name for the trained model
            audio_files: List of audio files for training
            epochs: Number of training epochs
            batch_size: Training batch size
            save_frequency: Save checkpoint every N epochs
            skip_setup: Skip the setup step if already done

        Returns:
            True if training completed successfully
        """
        # Update config
        self.config.training.model_name = model_name
        self.config.training.epochs = epochs
        self.config.training.batch_size = batch_size
        self.config.training.save_frequency = save_frequency

        # Step 1: Setup (if needed)
        if not skip_setup:
            if not self.run_setup():
                logger.error("Setup failed!")
                return False

        # Step 2: Prepare dataset
        logger.info("=" * 60)
        logger.info("STEP 2: Preparing dataset")
        logger.info("=" * 60)

        if not self.dataset.prepare_dataset(audio_files=audio_files, model_name=model_name):
            logger.error("Dataset preparation failed!")
            return False

        # Step 3: Train model
        logger.info("=" * 60)
        logger.info("STEP 3: Training model")
        logger.info("=" * 60)

        if not self.trainer.train(model_name=model_name):
            logger.error("Training failed!")
            return False

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {self.config.paths.get_model_weights_path(model_name)}")

        return True

    def run_inference(
        self,
        input_audio: Path,
        output_audio: Optional[Path] = None,
        model_name: Optional[str] = None,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe"
    ) -> Optional[Path]:
        """
        Run voice conversion inference.

        Args:
            input_audio: Path to input audio file
            output_audio: Path for output audio
            model_name: Model to use
            pitch_shift: Pitch adjustment (-12 to 12)
            f0_method: F0 extraction method

        Returns:
            Path to converted audio or None if failed
        """
        logger.info("=" * 60)
        logger.info("Running Voice Conversion")
        logger.info("=" * 60)

        return self.inference.convert(
            input_audio=input_audio,
            output_audio=output_audio,
            model_name=model_name,
            pitch_shift=pitch_shift,
            f0_method=f0_method
        )

    def list_models(self) -> List[str]:
        """List available trained models."""
        return self.inference.list_available_models()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RVC Voice Conversion Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup RVC environment
  python pipeline.py setup

  # Train a new model
  python pipeline.py train --model-name "my_voice" --audio-file "vocals.wav" --epochs 200

  # Train with multiple audio files
  python pipeline.py train --model-name "my_voice" --audio-file "file1.wav" "file2.wav" "file3.wav"

  # Convert audio using a trained model
  python pipeline.py infer --model-name "my_voice" --input "song.wav" --output "converted.wav"

  # Convert with pitch adjustment (male to female: +12, female to male: -12)
  python pipeline.py infer --model-name "my_voice" --input "song.wav" --pitch 12

  # List available models
  python pipeline.py list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup RVC environment")
    setup_parser.add_argument("--force", action="store_true", help="Force reinstallation")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new RVC model")
    train_parser.add_argument("--model-name", "-m", required=True, help="Name for the model")
    train_parser.add_argument("--audio-file", "-a", nargs="+", required=True,
                              help="Audio file(s) for training (3-7 minutes of clean vocals recommended)")
    train_parser.add_argument("--epochs", "-e", type=int, default=200,
                              help="Number of training epochs (default: 200)")
    train_parser.add_argument("--batch-size", "-b", type=int, default=8,
                              help="Training batch size (default: 8)")
    train_parser.add_argument("--save-frequency", "-s", type=int, default=50,
                              help="Save checkpoint every N epochs (default: 50)")
    train_parser.add_argument("--skip-setup", action="store_true",
                              help="Skip setup if already done")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Convert audio using a trained model")
    infer_parser.add_argument("--model-name", "-m", required=True, help="Model to use")
    infer_parser.add_argument("--input", "-i", required=True, help="Input audio file")
    infer_parser.add_argument("--output", "-o", help="Output audio file")
    infer_parser.add_argument("--pitch", "-p", type=int, default=0,
                              help="Pitch shift (-12 to 12, default: 0)")
    infer_parser.add_argument("--f0-method", default="rmvpe",
                              choices=["rmvpe", "pm", "harvest"],
                              help="F0 extraction method (default: rmvpe)")

    # List command
    subparsers.add_parser("list", help="List available trained models")

    # Info command
    subparsers.add_parser("info", help="Show platform and system information")

    return parser


def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Show platform info at startup
    print_platform_info()

    # Handle info command early (before pipeline init)
    if args.command == "info":
        return 0

    # Initialize pipeline
    pipeline = RVCPipeline()

    if args.command == "setup":
        success = pipeline.run_setup(force=args.force)
        return 0 if success else 1

    elif args.command == "train":
        audio_files = [Path(f) for f in args.audio_file]

        # Validate audio files exist
        for f in audio_files:
            if not f.exists():
                logger.error(f"Audio file not found: {f}")
                return 1

        success = pipeline.run_full_training(
            model_name=args.model_name,
            audio_files=audio_files,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            skip_setup=args.skip_setup
        )
        return 0 if success else 1

    elif args.command == "infer":
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None

        result = pipeline.run_inference(
            input_audio=input_path,
            output_audio=output_path,
            model_name=args.model_name,
            pitch_shift=args.pitch,
            f0_method=args.f0_method
        )
        return 0 if result else 1

    elif args.command == "list":
        models = pipeline.list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No trained models found.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
