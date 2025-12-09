"""
RVC Model Training Module
Handles model training with configurable parameters.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from random import shuffle
from typing import Optional
import logging

from config import PipelineConfig, TrainingConfig, PathConfig

logger = logging.getLogger(__name__)


class RVCTrainer:
    """Handles RVC model training."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.paths
        self.training_config = config.training

    def train(
        self,
        model_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_frequency: Optional[int] = None,
        resume: bool = True
    ) -> bool:
        """
        Train the RVC model.

        Args:
            model_name: Name of the model
            epochs: Number of training epochs
            batch_size: Training batch size
            save_frequency: Save checkpoint every N epochs
            resume: Whether to resume from existing checkpoints

        Returns:
            True if training completed successfully
        """
        # Use provided values or fall back to config
        model_name = model_name or self.training_config.model_name
        epochs = epochs or self.training_config.epochs
        batch_size = batch_size or self.training_config.batch_size
        save_frequency = save_frequency or self.training_config.save_frequency

        logger.info(f"Starting training for model: {model_name}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Save frequency: {save_frequency}")

        # Generate filelist
        if not self._generate_filelist(model_name):
            return False

        # Copy config file
        if not self._copy_config(model_name):
            return False

        # Run training
        return self._run_training(model_name, epochs, batch_size, save_frequency)

    def _generate_filelist(self, model_name: str) -> bool:
        """Generate the training filelist."""
        logger.info("Generating training filelist...")

        exp_dir = self.paths.get_model_logs_dir(model_name)
        version = self.training_config.version
        sample_rate = self.training_config.sample_rate
        speaker_id = self.training_config.speaker_id
        use_f0 = self.training_config.use_f0

        # Directory paths
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"

        if not gt_wavs_dir.exists() or not feature_dir.exists():
            logger.error("Required directories not found. Run dataset preparation first.")
            return False

        # Get common file names across all directories
        if use_f0:
            names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            )
        else:
            names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            )

        if not names:
            logger.error("No matching files found in directories")
            return False

        logger.info(f"Found {len(names)} matching audio segments")

        # Generate filelist entries
        opt = []
        for name in names:
            gt_path = str(gt_wavs_dir / f"{name}.wav").replace("\\", "\\\\")
            feature_path = str(feature_dir / f"{name}.npy").replace("\\", "\\\\")

            if use_f0:
                f0_path = str(f0_dir / f"{name}.wav.npy").replace("\\", "\\\\")
                f0nsf_path = str(f0nsf_dir / f"{name}.wav.npy").replace("\\", "\\\\")
                opt.append(f"{gt_path}|{feature_path}|{f0_path}|{f0nsf_path}|{speaker_id}")
            else:
                opt.append(f"{gt_path}|{feature_path}|{speaker_id}")

        # Add mute entries
        fea_dim = 256 if version == "v1" else 768
        mute_dir = self.paths.logs_dir / "mute"

        for _ in range(2):
            mute_gt = str(mute_dir / "0_gt_wavs" / f"mute{sample_rate}.wav").replace("\\", "\\\\")
            mute_feature = str(mute_dir / f"3_feature{fea_dim}" / "mute.npy").replace("\\", "\\\\")

            if use_f0:
                mute_f0 = str(mute_dir / "2a_f0" / "mute.wav.npy").replace("\\", "\\\\")
                mute_f0nsf = str(mute_dir / "2b-f0nsf" / "mute.wav.npy").replace("\\", "\\\\")
                opt.append(f"{mute_gt}|{mute_feature}|{mute_f0}|{mute_f0nsf}|{speaker_id}")
            else:
                opt.append(f"{mute_gt}|{mute_feature}|{speaker_id}")

        # Shuffle and write filelist
        shuffle(opt)
        filelist_path = exp_dir / "filelist.txt"
        with open(filelist_path, "w") as f:
            f.write("\n".join(opt))

        logger.info(f"Filelist written to: {filelist_path}")
        return True

    def _copy_config(self, model_name: str) -> bool:
        """Copy model configuration file."""
        exp_dir = self.paths.get_model_logs_dir(model_name)
        version = self.training_config.version
        sample_rate = self.training_config.sample_rate

        # Determine config path
        if version == "v1" or sample_rate == "40k":
            config_path = self.paths.rvc_dir / "configs" / "v1" / f"{sample_rate}.json"
        else:
            config_path = self.paths.rvc_dir / "configs" / "v2" / f"{sample_rate}.json"

        config_save_path = exp_dir / "config.json"

        if config_save_path.exists():
            logger.info("Config file already exists")
            return True

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        with open(config_path, "r") as f:
            config_data = json.load(f)

        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")

        logger.info(f"Config copied to: {config_save_path}")
        return True

    def _run_training(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        save_frequency: int
    ) -> bool:
        """Run the training process."""
        logger.info("Starting training process...")

        training_script = self.paths.rvc_dir / "infer" / "modules" / "train" / "train.py"

        # Get pretrained model paths
        pretrained_g = self.training_config.get_pretrained_g_path(self.paths)
        pretrained_d = self.training_config.get_pretrained_d_path(self.paths)

        # Build command
        cmd = [
            sys.executable,
            str(training_script),
            "-e", model_name,
            "-sr", self.training_config.sample_rate,
            "-f0", "1" if self.training_config.use_f0 else "0",
            "-bs", str(batch_size),
            "-g", self.training_config.gpu_ids,
            "-te", str(epochs),
            "-se", str(save_frequency),
            "-l", "1" if self.training_config.save_latest_only else "0",
            "-c", "1" if self.training_config.cache_data_in_gpu else "0",
            "-sw", "1" if self.training_config.save_every_weights else "0",
            "-v", self.training_config.version,
        ]

        # Add pretrained models if they exist
        if pretrained_g.exists():
            cmd.extend(["-pg", str(pretrained_g)])
            logger.info(f"Using pretrained G: {pretrained_g}")
        else:
            logger.warning(f"Pretrained G not found: {pretrained_g}")

        if pretrained_d.exists():
            cmd.extend(["-pd", str(pretrained_d)])
            logger.info(f"Using pretrained D: {pretrained_d}")
        else:
            logger.warning(f"Pretrained D not found: {pretrained_d}")

        logger.info("Training command: " + " ".join(cmd))

        try:
            # Run training with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=str(self.paths.rvc_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Print output in real-time
            for line in process.stdout:
                print(line.strip())

            process.wait()

            if process.returncode == 0:
                logger.info("Training completed successfully!")
                return True
            else:
                logger.error(f"Training failed with return code: {process.returncode}")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False

    def get_latest_checkpoint(self, model_name: str) -> Optional[Path]:
        """Get the latest checkpoint for a model."""
        exp_dir = self.paths.get_model_logs_dir(model_name)

        g_files = list(exp_dir.glob("G_*.pth"))
        if not g_files:
            return None

        # Sort by epoch number
        g_files.sort(key=lambda x: int(x.stem.split("_")[1]))
        return g_files[-1]

    def get_model_weights(self, model_name: str) -> Optional[Path]:
        """Get the final model weights file."""
        weights_path = self.paths.get_model_weights_path(model_name)
        return weights_path if weights_path.exists() else None


def train_model(
    config: Optional[PipelineConfig] = None,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    save_frequency: Optional[int] = None
) -> bool:
    """
    Convenience function to train a model.

    Args:
        config: Pipeline configuration
        model_name: Model name
        epochs: Number of epochs
        batch_size: Batch size
        save_frequency: Save frequency

    Returns:
        True if successful
    """
    if config is None:
        config = PipelineConfig()

    trainer = RVCTrainer(config)
    return trainer.train(model_name, epochs, batch_size, save_frequency)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example usage
    config = PipelineConfig()
    config.training.model_name = "test_model"
    config.training.epochs = 200

    trainer = RVCTrainer(config)
    # trainer.train()
