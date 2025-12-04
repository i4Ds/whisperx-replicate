# WhisperX Replicate

A [Cog](https://github.com/replicate/cog)-based deployment of WhisperX for German speech-to-text transcription using the `faster-whisper-large-v3-turbo` model.

## Overview

This repository packages WhisperX as a Replicate-compatible model, enabling easy deployment and inference via Cog. It uses:

- **WhisperX** (i4ds fork) for transcription with VAD (Voice Activity Detection)
- **faster-whisper-large-v3-turbo** model for fast, accurate German transcription
- **Cog** for containerization and deployment to Replicate

## Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA 12.1 support
- Docker

## Setup

### 1. Download the Model

First, download the model to your Hugging Face cache. You can use the helper script:

```bash
python get_models.py
```

This will download the model (`i4ds/daily-brook-134`) to your local Hugging Face cache.

### 2. Copy Model to Repository

Copy the cached model to the `models/` directory:

```bash
./copy_models.sh
```

This creates the following structure:

```
models/
└── faster-whisper-large-v3-turbo/
    ├── config.json
    ├── tokenizer.json
    ├── vocabulary.json
    └── ...
```

### 3. Build the Cog Image

```bash
cog build
```

## Usage

### Run a Prediction

```bash
cog predict -i audio_file=@your_audio.mp3
```

### Input Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_file` | Audio file to transcribe | (required) |
| `language` | Language (fixed to German) | `de` |
| `batch_size` | Parallelization for transcription | `8` |
| `temperature` | Sampling temperature | `0` |
| `vad_onset` | VAD onset threshold | `0.500` |
| `vad_offset` | VAD offset threshold | `0.363` |
| `align_output` | Enable word-level timestamps | `False` |
| `debug` | Print timing/memory info | `True` |

### Example

```bash
cog predict -i audio_file=@interview.mp3
```

### Output

The prediction returns:

- **segments**: Transcription in SRT subtitle format
- **detected_language**: The detected language code (e.g., `de`)

## File Structure

```
├── cog.yaml              # Cog configuration (CUDA, Python, dependencies)
├── predict.py            # Main prediction class for Cog
├── requirements.txt      # Python dependencies
├── copy_models.sh        # Script to copy model from HF cache
├── get_vad_model_url.py  # Helper to download model
└── models/               # Local model directory
    └── faster-whisper-large-v3-turbo/
```

## Deployment to Replicate

```bash
cog login
cog push r8.im/your-username/whisperx-german
```

## Notes

- The model is hardcoded to German (`de`) transcription
- Uses `float16` compute type for GPU efficiency
- VAD is enabled by default for better handling of speech segments
