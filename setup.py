"""
RVC Setup and Installation Module
Handles cloning RVC repository, downloading pretrained models, and installing dependencies.
"""
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional
import logging

from config import PipelineConfig, PathConfig, PretrainedConfig, PLATFORM

logger = logging.getLogger(__name__)


class RVCSetup:
    """Handles RVC environment setup and installation."""

    RVC_REPO_URL = "https://github.com/nakshatra-garg/rvc-deps"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.paths
        self.pretrained = config.pretrained

    def setup_all(self, force_reinstall: bool = False) -> bool:
        """
        Run complete setup process.

        Args:
            force_reinstall: If True, remove existing installation and reinstall

        Returns:
            True if setup completed successfully
        """
        logger.info("Starting RVC setup...")

        # Check required tools
        if not PLATFORM.has_git:
            logger.error("Git is not installed. Please install Git first.")
            logger.error("  Ubuntu/Debian: sudo apt install git")
            logger.error("  Windows: https://git-scm.com/download/win")
            return False

        if not PLATFORM.has_ffmpeg:
            logger.warning("FFmpeg not found. Some audio operations may fail.")
            logger.warning("  Ubuntu/Debian: sudo apt install ffmpeg")
            logger.warning("  Windows: https://ffmpeg.org/download.html")

        if not PLATFORM.has_aria2c:
            logger.info("aria2c not found, will use slower urllib for downloads")

        # Create directories
        self.paths.ensure_directories()

        # Clone RVC repository
        if not self._clone_rvc_repo(force=force_reinstall):
            return False

        # Patch RVC for PyTorch 2.6+ compatibility
        self._patch_torch_load_compatibility()

        # Download pretrained models
        if not self._download_pretrained_models():
            return False

        # Download additional files
        if not self._download_additional_files():
            return False

        # Install Python dependencies
        if not self._install_dependencies():
            return False

        logger.info("RVC setup completed successfully!")
        return True

    def _patch_torch_load_compatibility(self):
        """
        Patch RVC files for PyTorch 2.6+ compatibility.
        PyTorch 2.6 changed torch.load to use weights_only=True by default,
        which breaks loading fairseq/hubert models.
        """
        logger.info("Patching RVC for PyTorch 2.6+ compatibility...")

        rvc_dir = self.paths.rvc_dir

        # Files that need patching for torch.load
        files_to_patch = [
            rvc_dir / "infer" / "modules" / "train" / "extract_feature_print.py",
            rvc_dir / "infer" / "lib" / "rmvpe.py",
            rvc_dir / "infer" / "modules" / "vc" / "modules.py",
            rvc_dir / "infer" / "modules" / "train" / "train.py",
        ]

        # Fix matplotlib compatibility in utils.py (tostring_rgb -> buffer_rgba)
        utils_file = rvc_dir / "infer" / "lib" / "train" / "utils.py"
        if utils_file.exists():
            try:
                content = utils_file.read_text(encoding='utf-8')
                if 'tostring_rgb' in content:
                    # Simple line replacement for matplotlib 3.8+ compatibility
                    content = content.replace(
                        'data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")',
                        'data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]'
                    )
                    content = content.replace(
                        'data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))',
                        '# data already in correct shape from buffer_rgba'
                    )
                    utils_file.write_text(content, encoding='utf-8')
                    logger.info("  Patched: utils.py (matplotlib compatibility)")
            except Exception as e:
                logger.warning(f"  Failed to patch utils.py: {e}")

        for filepath in files_to_patch:
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding='utf-8')
                original_content = content

                # Pattern 1: torch.load(xxx) -> torch.load(xxx, weights_only=False)
                # But avoid double-patching if already has weights_only
                import re

                # Find torch.load calls that don't already have weights_only
                # This regex finds torch.load(...) that doesn't contain weights_only
                def patch_torch_load(match):
                    full_match = match.group(0)
                    if 'weights_only' in full_match:
                        return full_match  # Already patched
                    # Insert weights_only=False before the closing paren
                    return full_match[:-1] + ', weights_only=False)'

                # Match torch.load(...) calls - simple cases
                content = re.sub(
                    r'torch\.load\([^)]+\)',
                    patch_torch_load,
                    content
                )

                if content != original_content:
                    filepath.write_text(content, encoding='utf-8')
                    logger.info(f"  Patched: {filepath.name}")

            except Exception as e:
                logger.warning(f"  Failed to patch {filepath.name}: {e}")

        # Also patch fairseq if installed (common issue)
        try:
            import fairseq
            fairseq_path = Path(fairseq.__file__).parent / "checkpoint_utils.py"
            if fairseq_path.exists():
                content = fairseq_path.read_text(encoding='utf-8')
                if 'weights_only=False' not in content:
                    # Add weights_only=False to the torch.load call
                    content = content.replace(
                        'torch.load(f, map_location=torch.device("cpu"))',
                        'torch.load(f, map_location=torch.device("cpu"), weights_only=False)'
                    )
                    fairseq_path.write_text(content, encoding='utf-8')
                    logger.info("  Patched: fairseq/checkpoint_utils.py")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"  Could not patch fairseq: {e}")

        logger.info("Patching complete")

    def _is_valid_rvc_repo(self, rvc_dir: Path) -> bool:
        """Check if the RVC directory contains a valid clone."""
        required_files = [
            "requirements.txt",
            "infer/modules/train/train.py",
        ]
        required_dirs = [
            "infer",
            "configs",
        ]

        for f in required_files:
            if not (rvc_dir / f).exists():
                return False
        for d in required_dirs:
            if not (rvc_dir / d).is_dir():
                return False
        return True

    def _clone_rvc_repo(self, force: bool = False) -> bool:
        """Clone the RVC repository."""
        rvc_dir = self.paths.rvc_dir

        if rvc_dir.exists():
            if force:
                logger.info(f"Removing existing RVC directory: {rvc_dir}")
                shutil.rmtree(rvc_dir)
            elif self._is_valid_rvc_repo(rvc_dir):
                logger.info(f"RVC directory already exists and is valid: {rvc_dir}")
                return True
            else:
                logger.warning(f"RVC directory exists but is incomplete/invalid. Re-cloning...")
                shutil.rmtree(rvc_dir)

        logger.info(f"Cloning RVC repository to {rvc_dir}...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.RVC_REPO_URL, str(rvc_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("RVC repository cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone RVC repository: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Git is not installed. Please install Git first.")
            return False

    def _download_pretrained_models(self) -> bool:
        """Download pretrained models using aria2c or urllib."""
        pretrained_dir = self.paths.pretrained_dir
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        # Download standard pretrained models
        for filename in self.pretrained.standard_pretrains:
            filepath = pretrained_dir / filename
            if not filepath.exists():
                url = f"{self.pretrained.standard_base_url}/{filename}"
                if not self._download_file(url, filepath):
                    logger.warning(f"Failed to download {filename}")

        # Download OV2 Super pretrained models
        for filename in self.pretrained.ov2_pretrains:
            filepath = pretrained_dir / filename
            if not filepath.exists():
                url = f"{self.pretrained.ov2_base_url}/{filename}"
                if not self._download_file(url, filepath):
                    logger.warning(f"Failed to download {filename}")

        return True

    def _download_file(self, url: str, filepath: Path) -> bool:
        """Download a file using aria2c (fast) or urllib (fallback)."""
        logger.info(f"Downloading {filepath.name}...")

        # Try aria2c first (faster, supports resuming)
        if self._try_aria2c_download(url, filepath):
            return True

        # Fallback to urllib
        return self._try_urllib_download(url, filepath)

    def _try_aria2c_download(self, url: str, filepath: Path) -> bool:
        """Try downloading with aria2c."""
        try:
            result = subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",  # Continue downloading
                    "-x", "16",  # Max connections per server
                    "-s", "16",  # Split file into parts
                    "-k", "1M",  # Min split size
                    url,
                    "-d", str(filepath.parent),
                    "-o", filepath.name
                ],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Downloaded {filepath.name} successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _try_urllib_download(self, url: str, filepath: Path) -> bool:
        """Download using urllib."""
        import urllib.request
        try:
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Downloaded {filepath.name} successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download {filepath.name}: {e}")
            return False

    def _download_additional_files(self) -> bool:
        """Download additional required files."""
        rvc_dir = self.paths.rvc_dir

        # Base URL for RVC models on HuggingFace
        hf_base_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

        # Required model files for training
        model_files = [
            # RMVPE model for F0 extraction
            (f"{hf_base_url}/rmvpe.pt",
             rvc_dir / "assets" / "rmvpe" / "rmvpe.pt"),

            # Hubert model for feature extraction
            (f"{hf_base_url}/hubert_base.pt",
             rvc_dir / "assets" / "hubert" / "hubert_base.pt"),
        ]

        # Additional utility files
        additional_files = [
            ("https://raw.githubusercontent.com/RejektsAI/EasyTools/main/easyfuncs.py",
             rvc_dir / "easyfuncs.py"),
        ]

        all_files = model_files + additional_files

        for url, filepath in all_files:
            if not filepath.exists():
                # Ensure parent directory exists
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if not self._download_file(url, filepath):
                    logger.error(f"Failed to download required file: {filepath.name}")
                    # rmvpe and hubert are critical - fail if they can't be downloaded
                    if "rmvpe" in str(filepath) or "hubert" in str(filepath):
                        return False

        return True

    def _install_dependencies(self) -> bool:
        """Install Python dependencies."""
        requirements_file = self.paths.rvc_dir / "requirements.txt"

        if not requirements_file.exists():
            logger.warning("requirements.txt not found in RVC directory")
            return True

        logger.info("Installing Python dependencies...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )

            # Install additional packages
            additional_packages = ["librosa", "soundfile", "pydub", "faiss-cpu"]
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + additional_packages,
                check=True
            )

            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def verify_installation(self) -> bool:
        """Verify that RVC is properly installed."""
        checks = [
            (self.paths.rvc_dir.exists(), "RVC directory exists"),
            (self.paths.pretrained_dir.exists(), "Pretrained directory exists"),
            ((self.paths.pretrained_dir / "f0G32k.pth").exists(), "Standard pretrained models exist"),
        ]

        all_passed = True
        for check, description in checks:
            status = "OK" if check else "FAILED"
            logger.info(f"  [{status}] {description}")
            if not check:
                all_passed = False

        return all_passed


def setup_rvc(config: Optional[PipelineConfig] = None, force: bool = False) -> bool:
    """
    Convenience function to setup RVC.

    Args:
        config: Pipeline configuration (uses default if None)
        force: Force reinstallation

    Returns:
        True if setup successful
    """
    if config is None:
        config = PipelineConfig()

    setup = RVCSetup(config)
    return setup.setup_all(force_reinstall=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    config = PipelineConfig()
    setup_rvc(config)
