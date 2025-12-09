"""
Example usage of the RVC Training Pipeline.

This script demonstrates both programmatic and simplified usage patterns.
"""
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# EXAMPLE 1: Simple Training Pipeline (Recommended)
# =============================================================================

def train_voice_model_simple():
    """
    Simplest way to train a voice model.
    Just provide audio files and a model name.
    """
    from pipeline import RVCPipeline

    # Initialize the pipeline
    pipeline = RVCPipeline()

    # Train with default settings
    # Audio should be 3-7 minutes of clean vocals
    success = pipeline.run_full_training(
        model_name="my_voice",
        audio_files=[Path("path/to/your/vocals.wav")],
        epochs=200,
        batch_size=8,
        save_frequency=50
    )

    if success:
        print("Training completed successfully!")


# =============================================================================
# EXAMPLE 2: Custom Configuration
# =============================================================================

def train_voice_model_custom():
    """
    Training with custom configuration for advanced users.
    """
    from config import PipelineConfig, TrainingConfig
    from pipeline import RVCPipeline

    # Create custom configuration
    config = PipelineConfig()

    # Customize training settings
    config.training.epochs = 300
    config.training.batch_size = 4  # Reduce if running out of GPU memory
    config.training.save_frequency = 25
    config.training.use_ov2_pretrained = True  # Use better pretrained models

    # Customize F0 extraction
    config.f0.method = "rmvpe_gpu"  # Options: pm, harvest, rmvpe, rmvpe_gpu
    config.f0.gpu_id = 0

    # Initialize pipeline with custom config
    pipeline = RVCPipeline(config)

    # Run training
    success = pipeline.run_full_training(
        model_name="custom_voice",
        audio_files=[
            Path("vocals1.wav"),
            Path("vocals2.wav"),
            Path("vocals3.wav"),
        ],
        skip_setup=True  # Skip setup if already done
    )

    return success


# =============================================================================
# EXAMPLE 3: Voice Conversion Inference
# =============================================================================

def convert_voice():
    """
    Convert audio using a trained model.
    """
    from pipeline import RVCPipeline

    pipeline = RVCPipeline()

    # Basic conversion
    result = pipeline.run_inference(
        input_audio=Path("song_vocals.wav"),
        output_audio=Path("converted_output.wav"),
        model_name="my_voice",
        pitch_shift=0,  # 0 for same gender, +12 male→female, -12 female→male
        f0_method="rmvpe"
    )

    if result:
        print(f"Converted audio saved to: {result}")


# =============================================================================
# EXAMPLE 4: Using Individual Components
# =============================================================================

def step_by_step_training():
    """
    Use individual components for fine-grained control.
    """
    from config import PipelineConfig
    from setup import RVCSetup
    from dataset import DatasetPreparer
    from train import RVCTrainer

    config = PipelineConfig()
    config.training.model_name = "step_by_step_model"

    # Step 1: Setup
    setup = RVCSetup(config)
    setup.setup_all()

    # Step 2: Prepare dataset
    preparer = DatasetPreparer(config)
    preparer.prepare_dataset(
        audio_files=[Path("vocals.wav")],
        model_name="step_by_step_model"
    )

    # Step 3: Train
    trainer = RVCTrainer(config)
    trainer.train(
        model_name="step_by_step_model",
        epochs=100,
        batch_size=8
    )


# =============================================================================
# EXAMPLE 5: Batch Inference
# =============================================================================

def batch_convert():
    """
    Convert multiple audio files at once.
    """
    from config import PipelineConfig
    from inference import RVCInference

    config = PipelineConfig()
    inference = RVCInference(config)

    # List of files to convert
    input_files = [
        Path("song1.wav"),
        Path("song2.wav"),
        Path("song3.wav"),
    ]

    results = inference.batch_convert(
        input_files=input_files,
        output_dir=Path("converted_outputs"),
        model_name="my_voice",
        pitch_shift=0
    )

    print(f"Successfully converted {len(results)} files")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
RVC Training Pipeline Examples
==============================

This file contains example code showing how to use the pipeline.

For command-line usage, run:
    python pipeline.py --help

Examples:
    # Setup RVC environment
    python pipeline.py setup

    # Train a model
    python pipeline.py train -m "my_voice" -a "vocals.wav" -e 200

    # Convert audio
    python pipeline.py infer -m "my_voice" -i "input.wav" -o "output.wav"

    # List available models
    python pipeline.py list

Edit this file to customize the examples for your use case.
    """)
