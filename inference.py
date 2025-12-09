"""
RVC Inference Module
Handles voice conversion using trained models.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import logging

from config import PipelineConfig, InferenceConfig, PathConfig

logger = logging.getLogger(__name__)


class RVCInference:
    """Handles RVC voice conversion inference."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.paths
        self.inference_config = config.inference

    def convert(
        self,
        input_audio: Path,
        output_audio: Optional[Path] = None,
        model_name: Optional[str] = None,
        pitch_shift: Optional[int] = None,
        f0_method: Optional[str] = None,
        index_rate: Optional[float] = None
    ) -> Optional[Path]:
        """
        Convert audio using a trained RVC model.

        Args:
            input_audio: Path to input audio file
            output_audio: Path for output audio (auto-generated if None)
            model_name: Model to use (uses config if None)
            pitch_shift: Pitch adjustment (-12 to 12)
            f0_method: F0 extraction method
            index_rate: Index influence rate (0-1)

        Returns:
            Path to output audio file, or None if failed
        """
        # Use provided values or fall back to config
        model_name = model_name or self.inference_config.model_name
        pitch_shift = pitch_shift if pitch_shift is not None else self.inference_config.pitch_shift
        f0_method = f0_method or self.inference_config.f0_method
        index_rate = index_rate if index_rate is not None else self.inference_config.index_rate

        # Validate input
        input_audio = Path(input_audio)
        if not input_audio.exists():
            logger.error(f"Input audio not found: {input_audio}")
            return None

        # Set default output path
        if output_audio is None:
            output_audio = self.paths.output_dir / f"{input_audio.stem}_{model_name}_converted.wav"

        output_audio = Path(output_audio)
        output_audio.parent.mkdir(parents=True, exist_ok=True)

        # Get model and index paths
        model_path = self._get_model_path(model_name)
        index_path = self._get_index_path(model_name)

        if model_path is None:
            logger.error(f"Model weights not found for: {model_name}")
            return None

        logger.info(f"Converting audio with model: {model_name}")
        logger.info(f"  Input: {input_audio}")
        logger.info(f"  Output: {output_audio}")
        logger.info(f"  Pitch shift: {pitch_shift}")
        logger.info(f"  F0 method: {f0_method}")

        # Run inference
        success = self._run_inference(
            input_audio=input_audio,
            output_audio=output_audio,
            model_path=model_path,
            index_path=index_path,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=index_rate
        )

        if success and output_audio.exists():
            logger.info(f"Conversion successful: {output_audio}")
            return output_audio
        else:
            logger.error("Conversion failed")
            return None

    def batch_convert(
        self,
        input_files: List[Path],
        output_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[Path]:
        """
        Convert multiple audio files.

        Args:
            input_files: List of input audio paths
            output_dir: Directory for output files
            model_name: Model to use
            **kwargs: Additional arguments for convert()

        Returns:
            List of successfully converted file paths
        """
        output_dir = output_dir or self.paths.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for input_file in input_files:
            output_file = output_dir / f"{input_file.stem}_converted.wav"
            result = self.convert(
                input_audio=input_file,
                output_audio=output_file,
                model_name=model_name,
                **kwargs
            )
            if result:
                results.append(result)

        logger.info(f"Batch conversion complete: {len(results)}/{len(input_files)} successful")
        return results

    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the model weights path."""
        model_path = self.paths.get_model_weights_path(model_name)
        if model_path.exists():
            return model_path

        # Try without .pth extension
        alt_path = self.paths.weights_dir / model_name
        if alt_path.exists():
            return alt_path

        return None

    def _get_index_path(self, model_name: str) -> Optional[Path]:
        """Get the index file path."""
        logs_dir = self.paths.get_model_logs_dir(model_name)

        if not logs_dir.exists():
            return None

        # Find index file matching pattern
        for file in logs_dir.iterdir():
            if file.name.startswith("added_") and file.name.endswith(".index"):
                return file

        return None

    def _run_inference(
        self,
        input_audio: Path,
        output_audio: Path,
        model_path: Path,
        index_path: Optional[Path],
        pitch_shift: int,
        f0_method: str,
        index_rate: float
    ) -> bool:
        """Run the inference script."""
        inference_script = self.paths.rvc_dir / "tools" / "cmd" / "infer_cli.py"

        if not inference_script.exists():
            logger.error(f"Inference script not found: {inference_script}")
            return False

        # Set environment variables
        env = os.environ.copy()
        env["weight_root"] = str(model_path.parent)
        if index_path:
            env["index_root"] = str(index_path.parent)

        # Build command
        cmd = [
            sys.executable,
            str(inference_script),
            "--f0up_key", str(pitch_shift),
            "--input_path", str(input_audio),
            "--f0method", f0_method,
            "--opt_path", str(output_audio),
            "--model_name", model_path.name,
            "--index_rate", str(index_rate),
            "--device", self.inference_config.device,
            "--is_half", str(self.inference_config.is_half),
            "--filter_radius", str(self.inference_config.filter_radius),
            "--resample_sr", str(self.inference_config.resample_sr),
            "--rms_mix_rate", str(self.inference_config.rms_mix_rate),
            "--protect", str(self.inference_config.consonant_protection),
        ]

        if index_path:
            cmd.extend(["--index_path", index_path.name])

        logger.debug("Inference command: " + " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.paths.rvc_dir),
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Inference stderr: {result.stderr}")
                return False

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Inference failed: {e}")
            return False

    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        models = []
        if self.paths.weights_dir.exists():
            for file in self.paths.weights_dir.glob("*.pth"):
                models.append(file.stem)
        return models


def convert_audio(
    input_audio: Path,
    output_audio: Optional[Path] = None,
    config: Optional[PipelineConfig] = None,
    model_name: Optional[str] = None,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe"
) -> Optional[Path]:
    """
    Convenience function to convert audio.

    Args:
        input_audio: Input audio path
        output_audio: Output audio path
        config: Pipeline configuration
        model_name: Model name
        pitch_shift: Pitch adjustment
        f0_method: F0 extraction method

    Returns:
        Path to converted audio or None
    """
    if config is None:
        config = PipelineConfig()

    inference = RVCInference(config)
    return inference.convert(
        input_audio=input_audio,
        output_audio=output_audio,
        model_name=model_name,
        pitch_shift=pitch_shift,
        f0_method=f0_method
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example usage
    config = PipelineConfig()
    inference = RVCInference(config)

    # List available models
    models = inference.list_available_models()
    print(f"Available models: {models}")
