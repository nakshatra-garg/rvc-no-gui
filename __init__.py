"""
RVC Training Pipeline - Headless voice conversion model training.
"""
from config import (
    PipelineConfig,
    PathConfig,
    TrainingConfig,
    InferenceConfig,
    PreprocessConfig,
    F0Config,
    PretrainedConfig,
    PlatformConfig,
    PLATFORM,
)
from setup import RVCSetup, setup_rvc
from dataset import DatasetPreparer, prepare_dataset
from train import RVCTrainer, train_model
from inference import RVCInference, convert_audio
from pipeline import RVCPipeline

__version__ = "1.0.0"
__all__ = [
    # Config classes
    "PipelineConfig",
    "PathConfig",
    "TrainingConfig",
    "InferenceConfig",
    "PreprocessConfig",
    "F0Config",
    "PretrainedConfig",
    "PlatformConfig",
    "PLATFORM",
    # Main classes
    "RVCPipeline",
    "RVCSetup",
    "DatasetPreparer",
    "RVCTrainer",
    "RVCInference",
    # Convenience functions
    "setup_rvc",
    "prepare_dataset",
    "train_model",
    "convert_audio",
]
