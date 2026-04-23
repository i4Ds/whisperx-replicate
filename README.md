# STT4SG Replicate

This repo is now a Cog/Replicate wrapper around [`i4Ds/stt4sg-transcribe`](https://github.com/i4Ds/stt4sg-transcribe), with the pipeline vendored locally and a repo-local model/cache layout so the same assets are available both on the host and inside the built image.

## What Changed

- Replaced the old WhisperX-only predictor with the STT4SG transcription pipeline.
- Added a `uv` project (`pyproject.toml`, `uv.lock` once generated) for local development.
- Added repo-local cache helpers so Whisper, alignment, Silero, and pyannote assets can live under `model-store/`.
- Exposed the upstream pipeline parameters through Cog inputs instead of hardcoding a narrow set.

## Local Layout

Downloaded assets are stored locally in this repository:

```text
model-store/
├── whisper/        # local faster-whisper model directories
├── alignment/      # torchaudio / HF alignment models
├── huggingface/    # pyannote and other HF caches
├── torch/          # torch / torchaudio cache
├── xdg/            # xdg cache used by some backends
├── speechbrain/    # optional speechbrain cached models
└── nemo/           # optional nemo assets
```

That directory is copied into the Cog build context, so pre-downloaded assets are available inside the container without extra runtime downloads.

## Setup

Create the local environment:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Prefetch the default local assets:

```bash
uv run python scripts/download_assets.py
```

Notes:

- Default Whisper model: `i4ds/daily-brook-134`
- Default alignment model: `VOXPOPULI_ASR_BASE_10K_DE`
- Silero is prefetched automatically.
- Pyannote downloads require a Hugging Face token or an existing local HF login. The script will use your saved local login if one exists.

You can prefetch additional Whisper models too:

```bash
uv run python scripts/download_assets.py \
  --whisper-model i4ds/daily-brook-134 \
  --whisper-model Systran/faster-whisper-large-v3
```

## Local CLI Usage

Run the vendored pipeline directly:

```bash
uv run main.py 97_Brugg.mp3
```

Example with explicit parameters:

```bash
uv run main.py 97_Brugg.mp3 \
  --model i4ds/daily-brook-134 \
  --vad-method silero \
  --batch-size 8 \
  --no-alignment
```

## Cog Usage

Build the image:

```bash
cog build
```

Run the included test file:

```bash
cog predict \
  -i audio_file=@97_Brugg.mp3 \
  -i model=i4ds/daily-brook-134 \
  -i use_vad=true \
  -i vad_method=silero \
  -i use_alignment=true
```

The predictor returns:

- `srt_file`
- `srt_content`
- `detected_language`
- `duration_seconds`
- `num_segments`
- `logs_zip` when log saving is enabled

## Cog Inputs

The Cog predictor surfaces the upstream runtime controls:

- `audio_file`
- `model`
- `language`
- `task`
- `log_progress`
- `use_vad`
- `vad_method`
- `vad_params`
- `diarization`
- `diarization_method`
- `diarization_params`
- `num_speakers`
- `min_speakers`
- `max_speakers`
- `use_alignment`
- `alignment_model`
- `batch_size`
- `device`
- `compute_type`
- `include_speaker_labels`
- `save_logs`
- `hf_token`

`vad_params` and `diarization_params` are JSON strings so backend-specific parameters can be passed through unchanged.

## Replicate Push

```bash
cog login
cog push r8.im/your-username/stt4sg-replicate
```

## Verification Target

The intended smoke test for this repo is:

```bash
cog predict -i audio_file=@97_Brugg.mp3
```

If pyannote assets are not available locally, keep `diarization=false` and use `vad_method=silero` for the basic transcription path.
