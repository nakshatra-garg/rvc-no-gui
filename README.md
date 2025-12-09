# RVC No GUI - Headless Voice Cloning & Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://github.com/nakshatra-garg/rvc-no-gui)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/nakshatragarg)

> **Train and run RVC voice conversion models without a GUI** - Perfect for servers, cloud instances, automation, and headless environments.

A lightweight, production-ready Python pipeline for [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) that eliminates the need for a web interface. Train custom voice models, clone voices, and convert audio - all from the command line or Python API.

## Why RVC No GUI?

- **No WebUI Required** - Run on headless servers, Docker containers, or cloud GPUs (AWS, GCP, Colab, RunPod)
- **Simple CLI & Python API** - Train and infer with single commands or integrate into your projects
- **Auto Platform Detection** - Automatically configures for your OS, CUDA version, and hardware
- **PyTorch 2.6+ Compatible** - Works with latest PyTorch versions out of the box
- **Lightweight** - Only the essentials, no bloated dependencies

## Key Features

| Feature | Description |
|---------|-------------|
| Voice Cloning | Clone any voice with 3-7 minutes of audio |
| Voice Conversion | Transform vocals while preserving emotion & tone |
| Batch Processing | Process multiple audio files automatically |
| GPU Acceleration | Full CUDA support for fast training & inference |
| Cross-Platform | Works on Linux, Windows, and macOS |
| Python API | Easy integration with existing projects |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nakshatra-garg/rvc-no-gui.git
cd rvc-no-gui

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup RVC (downloads pretrained models)
python pipeline.py setup
```

### Train a Voice Model

```bash
# Train with your audio samples (3-7 min of clean vocals)
python pipeline.py train -m "my_voice" -a "vocals.wav" -e 200

# Multiple audio files
python pipeline.py train -m "my_voice" -a "sample1.wav" "sample2.wav" -e 300
```

### Convert Audio (Voice Cloning)

```bash
# Basic voice conversion
python pipeline.py infer -m "my_voice" -i "input.wav" -o "output.wav"

# With pitch adjustment
python pipeline.py infer -m "my_voice" -i "input.wav" --pitch 12   # Higher pitch
python pipeline.py infer -m "my_voice" -i "input.wav" --pitch -12  # Lower pitch
```

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

## Python API

```python
from pipeline import RVCPipeline
from pathlib import Path

pipeline = RVCPipeline()

# Train a model
pipeline.run_full_training(
    model_name="my_voice",
    audio_files=[Path("vocals.wav")],
    epochs=200,
    batch_size=8,
    skip_setup=True
)

# Convert audio
pipeline.run_inference(
    input_audio=Path("input.wav"),
    output_audio=Path("output.wav"),
    model_name="my_voice",
    pitch_shift=0
)
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, CPU works but slower)
- FFmpeg (for audio processing)
- ~10GB disk space for models

## Configuration

Settings are auto-detected. Override via Python:

```python
from config import PipelineConfig
from pipeline import RVCPipeline

config = PipelineConfig()
config.training.epochs = 300
config.training.batch_size = 4  # Lower if GPU memory is limited
config.f0.method = "rmvpe_gpu"  # Options: pm, harvest, rmvpe, rmvpe_gpu

pipeline = RVCPipeline(config)
```

### Auto-detected Settings

| Setting | Detection Logic |
|---------|----------------|
| `device` | `cuda:0` if GPU available, else `cpu` |
| `is_half` | `True` on CUDA (FP16), `False` on CPU |
| `f0_method` | `rmvpe_gpu` if CUDA, else `rmvpe` |
| `num_processes` | Half of CPU cores (max 8) |

## CLI Reference

| Command | Description |
|---------|-------------|
| `setup` | Initialize RVC environment and download models |
| `train` | Train a new voice model |
| `infer` | Convert audio using a trained model |
| `list` | List available trained models |
| `info` | Display system information |

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model-name` | Model name (required) | - |
| `-a, --audio-file` | Audio file(s) (required) | - |
| `-e, --epochs` | Training epochs | 200 |
| `-b, --batch-size` | Batch size | 8 |
| `-s, --save-frequency` | Checkpoint save frequency | 50 |
| `--skip-setup` | Skip setup for faster runs | False |

### Inference Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model-name` | Model to use (required) | - |
| `-i, --input` | Input audio file (required) | - |
| `-o, --output` | Output file path | auto |
| `-p, --pitch` | Pitch shift (-12 to 12) | 0 |
| `--f0-method` | F0 method (rmvpe/pm/harvest) | rmvpe |

## Project Structure

```
rvc-no-gui/
├── pipeline.py      # Main CLI & orchestrator
├── config.py        # Configuration & platform detection
├── setup.py         # RVC installation & patching
├── dataset.py       # Audio preprocessing
├── train.py         # Model training
├── inference.py     # Voice conversion
├── example.py       # Usage examples
└── requirements.txt
```

## Use Cases

- **Content Creators** - Create consistent AI voices for videos/podcasts
- **Game Developers** - Generate character voices
- **Music Production** - Voice conversion for vocals
- **Accessibility** - Text-to-speech with custom voices
- **Research** - Voice cloning experiments

## Troubleshooting

### PyTorch 2.6+ Errors
```bash
python pipeline.py setup --force
```

### Missing Models
```bash
python pipeline.py setup
```

### Out of GPU Memory
```bash
python pipeline.py train -m "my_voice" -a "vocals.wav" -b 4
```

## Related Projects

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Original RVC with web interface
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) - Alternative voice conversion

## Contributing

Contributions welcome! Please open an issue or PR.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

**Keywords:** rvc, voice cloning, voice conversion, ai voice, text to speech, tts, voice changer, voice model, deep learning, pytorch, machine learning, audio processing, speech synthesis, voice synthesis, headless, no gui, cli, python, cuda, gpu
