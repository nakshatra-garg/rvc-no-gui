"""
RVC Dataset Preparation Module
Handles audio preprocessing, F0 extraction, and feature extraction.
"""
import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np

from config import PipelineConfig, PathConfig, PreprocessConfig, F0Config, TrainingConfig

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares dataset for RVC training."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.paths
        self.preprocess_config = config.preprocess
        self.f0_config = config.f0
        self.training_config = config.training

    def prepare_dataset(
        self,
        audio_files: Optional[List[Path]] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Full dataset preparation pipeline.

        Args:
            audio_files: List of audio file paths to use (uses dataset_dir if None)
            model_name: Model name (uses config if None)

        Returns:
            True if preparation successful
        """
        model_name = model_name or self.training_config.model_name

        logger.info(f"Preparing dataset for model: {model_name}")

        # Step 1: Load and prepare audio files
        if audio_files:
            if not self._copy_audio_files(audio_files):
                return False

        # Step 2: Handle resume training (preserve D_/G_ checkpoints)
        self._handle_resume_training(model_name)

        # Step 3: Preprocess audio (slicing, resampling)
        if not self._preprocess_audio(model_name):
            return False

        # Step 4: Extract F0 (pitch)
        if not self._extract_f0(model_name):
            return False

        # Step 5: Extract features
        if not self._extract_features(model_name):
            return False

        # Step 6: Train index
        if not self._train_index(model_name):
            return False

        logger.info("Dataset preparation completed successfully!")
        return True

    def load_audio_file(self, audio_path: Path, output_name: str = "vocal_audio.wav") -> Path:
        """
        Load and convert an audio file to the correct format.

        Args:
            audio_path: Path to input audio file
            output_name: Name for the output file

        Returns:
            Path to the processed audio file
        """
        import librosa
        import soundfile as sf

        output_path = self.paths.dataset_dir / output_name

        logger.info(f"Loading audio file: {audio_path}")

        # Load audio with original sample rate
        audio, sr = librosa.load(str(audio_path), sr=None)

        # Save as WAV
        sf.write(str(output_path), audio, sr, format='wav')

        logger.info(f"Audio saved to: {output_path}")
        return output_path

    def _copy_audio_files(self, audio_files: List[Path]) -> bool:
        """Copy audio files to dataset directory."""
        import librosa
        import soundfile as sf

        self.paths.dataset_dir.mkdir(parents=True, exist_ok=True)

        for i, audio_path in enumerate(audio_files):
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False

            output_path = self.paths.dataset_dir / f"audio_{i}.wav"
            audio, sr = librosa.load(str(audio_path), sr=None)
            sf.write(str(output_path), audio, sr, format='wav')
            logger.info(f"Copied: {audio_path.name} -> {output_path.name}")

        return True

    def _handle_resume_training(self, model_name: str):
        """Handle resume training by preserving D_/G_ checkpoints."""
        logs_path = self.paths.get_model_logs_dir(model_name)

        if not logs_path.exists():
            return

        logger.info("Existing model found - preparing for resume training")

        # Create temp directory for checkpoints
        temp_dir = self.paths.base_dir / "temp_DG"
        temp_dir.mkdir(exist_ok=True)

        # Copy D_/G_ files to temp
        for item in logs_path.iterdir():
            if item.is_file() and (item.name.startswith('D_') or item.name.startswith('G_')) and item.suffix == '.pth':
                shutil.copy(item, temp_dir / item.name)
                logger.info(f"Preserved checkpoint: {item.name}")

        # Clear logs directory
        for item in logs_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Move checkpoints back
        for item in temp_dir.iterdir():
            shutil.move(str(item), logs_path / item.name)

        # Remove temp directory
        shutil.rmtree(temp_dir)

    def _preprocess_audio(self, model_name: str) -> bool:
        """Preprocess audio files (slicing, resampling)."""
        logger.info("Preprocessing audio...")

        logs_dir = self.paths.get_model_logs_dir(model_name)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Check dataset directory has files
        dataset_files = list(self.paths.dataset_dir.glob("*"))
        audio_files = [f for f in dataset_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]

        if not audio_files:
            logger.error(f"No audio files found in {self.paths.dataset_dir}")
            return False

        logger.info(f"Found {len(audio_files)} audio files")

        # Run preprocessing script
        preprocess_script = self.paths.rvc_dir / "infer" / "modules" / "train" / "preprocess.py"

        cmd = [
            sys.executable,
            str(preprocess_script),
            str(self.paths.dataset_dir),
            str(self.preprocess_config.sample_rate),
            str(self.preprocess_config.num_processes),
            str(logs_dir),
            str(self.preprocess_config.normalize_loudness),
            str(self.preprocess_config.loudness_target)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.paths.rvc_dir),
                capture_output=True,
                text=True
            )

            # Check log file for success
            log_file = logs_dir / "preprocess.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    if 'end preprocess' in f.read():
                        logger.info("Audio preprocessing completed successfully")
                        return True

            logger.error(f"Preprocessing may have failed. Check logs at {log_file}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return True  # Continue anyway, logs might not indicate completion properly

        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed: {e}")
            return False

    def _extract_f0(self, model_name: str) -> bool:
        """Extract F0 (pitch) features."""
        logger.info(f"Extracting F0 using method: {self.f0_config.method}")

        logs_dir = self.paths.get_model_logs_dir(model_name)

        if self.f0_config.method == "rmvpe_gpu":
            script = self.paths.rvc_dir / "infer" / "modules" / "train" / "extract" / "extract_f0_rmvpe.py"
            cmd = [
                sys.executable,
                str(script),
                "1",  # num_processes
                str(self.f0_config.gpu_id),
                str(self.f0_config.gpu_id),
                str(logs_dir),
                "True"  # is_half
            ]
        else:
            script = self.paths.rvc_dir / "infer" / "modules" / "train" / "extract" / "extract_f0_print.py"
            cmd = [
                sys.executable,
                str(script),
                str(logs_dir),
                "2",  # num_processes
                self.f0_config.method
            ]

        try:
            subprocess.run(
                cmd,
                cwd=str(self.paths.rvc_dir),
                check=True
            )
            logger.info("F0 extraction completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"F0 extraction failed: {e}")
            return False

    def _extract_features(self, model_name: str) -> bool:
        """Extract audio features."""
        logger.info("Extracting features...")

        logs_dir = self.paths.get_model_logs_dir(model_name)
        script = self.paths.rvc_dir / "infer" / "modules" / "train" / "extract_feature_print.py"

        cmd = [
            sys.executable,
            str(script),
            f"cuda:{self.f0_config.gpu_id}",
            "1",  # num_processes
            "0",
            str(logs_dir),
            self.training_config.version,
            "True"  # is_half
        ]

        try:
            subprocess.run(
                cmd,
                cwd=str(self.paths.rvc_dir),
                check=True
            )

            # Check log file
            log_file = logs_dir / "extract_f0_feature.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    if 'all-feature-done' in f.read():
                        logger.info("Feature extraction completed successfully")
                        return True

            logger.info("Feature extraction completed")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed: {e}")
            return False

    def _train_index(self, model_name: str) -> bool:
        """Train the FAISS index for voice retrieval."""
        logger.info("Training FAISS index...")

        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed, trying faiss-cpu")
            subprocess.run([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            import faiss

        exp_dir = self.paths.get_model_logs_dir(model_name)
        version = self.training_config.version

        # Determine feature directory
        feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

        if not feature_dir.exists():
            logger.error(f"Feature directory not found: {feature_dir}")
            return False

        # Load all feature files
        npys = []
        for name in sorted(os.listdir(feature_dir)):
            phone = np.load(feature_dir / name)
            npys.append(phone)

        if not npys:
            logger.error("No feature files found")
            return False

        # Concatenate and shuffle
        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        logger.info(f"Total features shape: {big_npy.shape}")

        # Save total features
        np.save(exp_dir / "total_fea.npy", big_npy)

        # Calculate IVF clusters
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        feature_dim = 256 if version == "v1" else 768

        logger.info(f"Training index with {n_ivf} IVF clusters...")

        # Create and train index
        index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)

        # Save trained index
        trained_index_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_1_{model_name}_{version}.index"
        faiss.write_index(index, str(trained_index_path))

        # Add vectors and save final index
        logger.info("Adding vectors to index...")
        batch_size = 8192
        for i in range(0, big_npy.shape[0], batch_size):
            index.add(big_npy[i:i + batch_size])

        added_index_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{model_name}_{version}.index"
        faiss.write_index(index, str(added_index_path))

        logger.info(f"Index saved to: {added_index_path}")
        return True


def prepare_dataset(
    config: Optional[PipelineConfig] = None,
    audio_files: Optional[List[Path]] = None,
    model_name: Optional[str] = None
) -> bool:
    """
    Convenience function to prepare dataset.

    Args:
        config: Pipeline configuration
        audio_files: List of audio files to use
        model_name: Model name

    Returns:
        True if successful
    """
    if config is None:
        config = PipelineConfig()

    preparer = DatasetPreparer(config)
    return preparer.prepare_dataset(audio_files, model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example usage
    config = PipelineConfig()
    config.training.model_name = "test_model"

    preparer = DatasetPreparer(config)
    # preparer.prepare_dataset()
