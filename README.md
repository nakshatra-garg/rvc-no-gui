# RVC Headless Training Pipeline

A headless Python pipeline for training [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models without the WebUI.

## Features

- **Headless operation** - No GUI required, perfect for servers/cloud
- **Auto platform detection** - Detects OS, CUDA, CPU cores automatically
- **PyTorch 2.6+ compatible** - Automatic patches for latest PyTorch versions
- **Simple CLI** - Train and convert with single commands

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git
- FFmpeg (for audio processing)
- ~10GB disk space for models and dependencies

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/rvc_headless.git
cd rvc_headless

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup RVC (clones RVC repo + downloads pretrained models)
python pipeline.py setup
```

## Quick Start

### Check System Info

```bash
python pipeline.py info
```

Output:
```
==================================================
Platform Configuration (Auto-Detected)
==================================================
System: Linux
CPU Cores: 16
CUDA Available: True (1 GPU(s))
Tools: aria2c=Yes, git=Yes, ffmpeg=Yes
==================================================
```

### Train a Model

```bash
# Basic training (200 epochs)
python pipeline.py train -m "my_voice" -a "vocals.wav" -e 200

# Multiple audio files
python pipeline.py train -m "my_voice" -a "file1.wav" "file2.wav" -e 300

# Skip setup on subsequent runs (faster)
python pipeline.py train -m "my_voice" -a "vocals.wav" -e 200 --skip-setup
```

**Audio requirements:** 3-7 minutes of clean isolated vocals (no background music/noise).

### Convert Audio

```bash
# Basic conversion
python pipeline.py infer -m "my_voice" -i "input.wav" -o "output.wav"

# With pitch shift
python pipeline.py infer -m "my_voice" -i "input.wav" --pitch 12   # Male to Female
python pipeline.py infer -m "my_voice" -i "input.wav" --pitch -12  # Female to Male
```

### List Models

```bash
python pipeline.py list
```

## Python API

```python
from pipeline import RVCPipeline
from pathlib import Path

pipeline = RVCPipeline()

# Train
pipeline.run_full_training(
    model_name="my_voice",
    audio_files=[Path("vocals.wav")],
    epochs=200,
    batch_size=8,
    skip_setup=True  # Skip setup if already done
)

# Convert
pipeline.run_inference(
    input_audio=Path("input.wav"),
    output_audio=Path("output.wav"),
    model_name="my_voice",
    pitch_shift=0
)
```

## Configuration

Settings are auto-detected based on your platform. Override via `config.py`:

```python
from config import PipelineConfig, PLATFORM
from pipeline import RVCPipeline

# Check detected platform
print(PLATFORM.summary())

# Custom config
config = PipelineConfig()
config.training.epochs = 300
config.training.batch_size = 4  # Reduce if low on GPU memory
config.f0.method = "rmvpe_gpu"  # pm, harvest, rmvpe, rmvpe_gpu

pipeline = RVCPipeline(config)
```

### Auto-detected Settings

| Setting | Auto-detection |
|---------|---------------|
| `device` | `cuda:0` if GPU available, else `cpu` |
| `is_half` | `True` on CUDA (FP16), `False` on CPU |
| `f0_method` | `rmvpe_gpu` if CUDA, else `rmvpe` |
| `num_processes` | Half of CPU cores (max 8) |

## Project Structure

```
rvc_headless/
├── pipeline.py      # Main CLI entry point (orchestrator)
├── config.py        # Configuration + platform detection
├── setup.py         # RVC installation + patching
├── dataset.py       # Audio preprocessing
├── train.py         # Model training
├── inference.py     # Voice conversion
├── example.py       # Usage examples
└── requirements.txt
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `setup` | Setup RVC environment (clone repo, download models) |
| `train` | Train a new model |
| `infer` | Convert audio using trained model |
| `list` | List available trained models |
| `info` | Show platform and system information |

### Setup Options

| Flag | Description |
|------|-------------|
| `--force` | Force reinstall (re-clone repo, re-download models) |

### Train Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model-name` | Model name (required) | - |
| `-a, --audio-file` | Audio file(s) (required) | - |
| `-e, --epochs` | Training epochs | 200 |
| `-b, --batch-size` | Batch size | 8 |
| `-s, --save-frequency` | Checkpoint frequency | 50 |
| `--skip-setup` | Skip setup step (faster for subsequent runs) | False |

### Infer Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model-name` | Model to use (required) | - |
| `-i, --input` | Input audio (required) | - |
| `-o, --output` | Output path | auto-generated |
| `-p, --pitch` | Pitch shift (-12 to 12) | 0 |
| `--f0-method` | F0 method (rmvpe/pm/harvest) | rmvpe |

## Troubleshooting

### PyTorch 2.6+ Errors
The setup automatically patches RVC for PyTorch 2.6+ compatibility. If you see `weights_only` errors, run:
```bash
python pipeline.py setup --force
```

### Missing Models
If training fails with missing `rmvpe.pt` or `hubert_base.pt`, run setup again:
```bash
python pipeline.py setup
```

### Out of GPU Memory
Reduce batch size:
```bash
python pipeline.py train -m "my_voice" -a "vocals.wav" -b 4
```

## License

MIT
