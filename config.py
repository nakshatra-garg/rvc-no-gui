"""
RVC Training Pipeline Configuration
"""
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class PlatformConfig:
    """Auto-detected platform configuration."""
    # Platform info
    system: str = field(init=False)
    is_windows: bool = field(init=False)
    is_linux: bool = field(init=False)
    is_macos: bool = field(init=False)

    # Hardware detection
    cpu_count: int = field(init=False)
    has_cuda: bool = field(init=False)
    cuda_device_count: int = field(init=False)

    # Tool availability
    has_aria2c: bool = field(init=False)
    has_git: bool = field(init=False)
    has_ffmpeg: bool = field(init=False)

    def __post_init__(self):
        # Platform detection
        self.system = platform.system()
        self.is_windows = self.system == "Windows"
        self.is_linux = self.system == "Linux"
        self.is_macos = self.system == "Darwin"

        # CPU count
        self.cpu_count = os.cpu_count() or 1

        # CUDA detection
        self.has_cuda, self.cuda_device_count = self._detect_cuda()

        # Tool availability
        self.has_aria2c = shutil.which("aria2c") is not None
        self.has_git = shutil.which("git") is not None
        self.has_ffmpeg = shutil.which("ffmpeg") is not None

    def _detect_cuda(self) -> tuple:
        """Detect CUDA availability and device count."""
        try:
            import torch
            if torch.cuda.is_available():
                return True, torch.cuda.device_count()
        except ImportError:
            # Try nvidia-smi as fallback
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    count = len(result.stdout.strip().split("\n"))
                    return True, count
            except (FileNotFoundError, Exception):
                pass
        return False, 0

    def get_optimal_num_processes(self) -> int:
        """Get optimal number of processes for parallel operations."""
        # Use half of CPU cores, minimum 1, maximum 8
        return max(1, min(self.cpu_count // 2, 8))

    def get_device(self) -> str:
        """Get the best available device."""
        if self.has_cuda:
            return "cuda:0"
        return "cpu"

    def summary(self) -> str:
        """Return a summary of detected platform configuration."""
        lines = [
            f"Platform: {self.system}",
            f"CPU Cores: {self.cpu_count}",
            f"CUDA Available: {self.has_cuda}" + (f" ({self.cuda_device_count} GPU(s))" if self.has_cuda else ""),
            f"Tools: aria2c={'Yes' if self.has_aria2c else 'No'}, git={'Yes' if self.has_git else 'No'}, ffmpeg={'Yes' if self.has_ffmpeg else 'No'}",
        ]
        return "\n".join(lines)


# Global platform config (detected once at import)
PLATFORM = PlatformConfig()


@dataclass
class PathConfig:
    """Path configuration for RVC pipeline."""
    # Base directories
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    rvc_dir: Path = field(default_factory=lambda: Path.cwd() / "RVC")
    dataset_dir: Path = field(default_factory=lambda: Path.cwd() / "dataset")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "output")

    # RVC subdirectories (set after init)
    logs_dir: Path = field(init=False)
    weights_dir: Path = field(init=False)
    pretrained_dir: Path = field(init=False)
    audios_dir: Path = field(init=False)

    def __post_init__(self):
        self.logs_dir = self.rvc_dir / "logs"
        self.weights_dir = self.rvc_dir / "assets" / "weights"
        self.pretrained_dir = self.rvc_dir / "assets" / "pretrained_v2"
        self.audios_dir = self.rvc_dir / "audios"

    def get_model_logs_dir(self, model_name: str) -> Path:
        return self.logs_dir / model_name

    def get_model_weights_path(self, model_name: str) -> Path:
        return self.weights_dir / f"{model_name}.pth"

    def ensure_directories(self):
        """Create all necessary directories."""
        dirs = [
            self.dataset_dir,
            self.output_dir,
            self.logs_dir,
            self.weights_dir,
            self.audios_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PretrainedConfig:
    """Configuration for pretrained models."""
    # Standard pretrained models
    standard_pretrains: list = field(default_factory=lambda: [
        "f0D32k.pth", "f0G32k.pth"
    ])

    # OV2 Super pretrained models (better quality)
    ov2_pretrains: list = field(default_factory=lambda: [
        "f0Ov2Super32kD.pth", "f0Ov2Super32kG.pth"
    ])

    # Download URLs
    standard_base_url: str = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2"
    ov2_base_url: str = "https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main"


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing."""
    sample_rate: int = 32000
    num_processes: int = field(default_factory=lambda: PLATFORM.get_optimal_num_processes())
    normalize_loudness: bool = False
    loudness_target: float = 3.0


@dataclass
class F0Config:
    """Configuration for F0 (pitch) extraction."""
    method: Literal["pm", "harvest", "rmvpe", "rmvpe_gpu"] = field(
        default_factory=lambda: "rmvpe_gpu" if PLATFORM.has_cuda else "rmvpe"
    )
    gpu_id: int = 0


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model settings
    model_name: str = "my_model"
    version: Literal["v1", "v2"] = "v2"
    sample_rate: Literal["32k", "40k", "48k"] = "32k"

    # Training hyperparameters
    epochs: int = 200
    batch_size: int = 8
    save_frequency: int = 50

    # Pretrained model settings
    use_ov2_pretrained: bool = True

    # GPU settings
    gpu_ids: str = "0"
    cache_data_in_gpu: bool = False

    # Saving options
    save_latest_only: bool = True
    save_every_weights: bool = True

    # F0 (pitch) settings
    use_f0: bool = True
    speaker_id: int = 0

    def get_pretrained_g_path(self, paths: PathConfig) -> Path:
        if self.use_ov2_pretrained:
            return paths.pretrained_dir / f"f0Ov2Super{self.sample_rate}G.pth"
        return paths.pretrained_dir / f"f0G{self.sample_rate}.pth"

    def get_pretrained_d_path(self, paths: PathConfig) -> Path:
        if self.use_ov2_pretrained:
            return paths.pretrained_dir / f"f0Ov2Super{self.sample_rate}D.pth"
        return paths.pretrained_dir / f"f0D{self.sample_rate}.pth"


@dataclass
class InferenceConfig:
    """Configuration for voice conversion inference."""
    model_name: str = "my_model"

    # Pitch adjustment
    # Male to Male / Female to Female: 0
    # Female to Male: -12
    # Male to Female: 12
    pitch_shift: int = 0

    # F0 extraction method
    f0_method: Literal["rmvpe", "pm", "harvest"] = "rmvpe"

    # Index settings
    index_rate: float = 0.5

    # Audio processing
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 0.0
    consonant_protection: float = 0.5

    # Device settings (auto-detected)
    device: str = field(default_factory=lambda: PLATFORM.get_device())
    is_half: bool = field(default_factory=lambda: PLATFORM.has_cuda)  # FP16 only on CUDA


@dataclass
class PipelineConfig:
    """Main configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    f0: F0Config = field(default_factory=F0Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Platform config is global but accessible here for convenience
    @property
    def platform(self) -> PlatformConfig:
        return PLATFORM

    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> "PipelineConfig":
        """Create config with a specific model name."""
        config = cls(**kwargs)
        config.training.model_name = model_name
        config.inference.model_name = model_name
        return config

    def print_platform_info(self):
        """Print detected platform information."""
        print("=" * 50)
        print("Platform Configuration (Auto-Detected)")
        print("=" * 50)
        print(PLATFORM.summary())
        print("=" * 50)
