import json
import logging
import shutil
from pathlib import Path as SysPath
from typing import Optional
from uuid import uuid4

from cog import BaseModel, BasePredictor, Input, Path

from local_cache import (
    OUTPUTS_DIR,
    configure_local_caches,
    get_default_hf_token,
    resolve_alignment_model_spec,
    resolve_whisper_model_spec,
)
from pipeline import TranscriptionConfig, TranscriptionPipeline

configure_local_caches()

logger = logging.getLogger(__name__)


class Output(BaseModel):
    srt_file: Path
    srt_content: str
    detected_language: str
    duration_seconds: float
    num_segments: int
    logs_zip: Optional[Path] = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        configure_local_caches()
        self.default_hf_token = get_default_hf_token()

    def predict(
        self,
        audio_file: Path = Input(description="Audio file to transcribe"),
        model: str = Input(
            description="Whisper model repo id, local path, or repo-local cached alias",
            default="i4ds/daily-brook-134",
        ),
        language: str = Input(
            description="Language code. Leave empty for auto-detect.",
            default="",
        ),
        task: str = Input(
            description="Whisper task",
            default="transcribe",
            choices=["transcribe", "translate"],
        ),
        log_progress: bool = Input(
            description="Log transcription progress from faster-whisper",
            default=False,
        ),
        use_vad: bool = Input(
            description="Enable voice activity detection before transcription",
            default=True,
        ),
        vad_method: str = Input(
            description="VAD backend",
            default="silero",
            choices=["silero", "pyannote", "speechbrain", "nemo"],
        ),
        vad_params: str = Input(
            description="JSON dictionary passed to the selected VAD backend",
            default="",
        ),
        diarization: bool = Input(
            description="Enable speaker diarization",
            default=False,
        ),
        diarization_method: str = Input(
            description="Speaker diarization backend",
            default="pyannote",
            choices=["pyannote", "speechbrain", "nemo"],
        ),
        diarization_params: str = Input(
            description="JSON dictionary passed to the selected diarization backend",
            default="",
        ),
        num_speakers: int = Input(
            description="Exact number of speakers. Use 0 to auto-detect.",
            default=0,
        ),
        min_speakers: int = Input(
            description="Minimum speakers for diarization. Use 0 for backend default.",
            default=1,
        ),
        max_speakers: int = Input(
            description="Maximum speakers for diarization. Use 0 for backend default.",
            default=0,
        ),
        use_alignment: bool = Input(
            description="Run CTC word-level alignment",
            default=True,
        ),
        alignment_model: str = Input(
            description="Alignment model repo id, torchaudio bundle name, or local path",
            default="",
        ),
        batch_size: int = Input(
            description="Batched inference size. Use 0 to disable batched mode.",
            default=0,
        ),
        device: str = Input(
            description="Execution device",
            default="auto",
            choices=["auto", "cuda", "cpu"],
        ),
        compute_type: str = Input(
            description="Whisper compute type",
            default="auto",
            choices=["auto", "float16", "float32", "int8"],
        ),
        include_speaker_labels: bool = Input(
            description="Include speaker labels in SRT output when diarization is enabled",
            default=True,
        ),
        save_logs: bool = Input(
            description="Write JSON logs and return them as a zip archive",
            default=True,
        ),
        hf_token: str = Input(
            description="Optional Hugging Face token. Leave empty to use your saved local login.",
            default="",
        ),
    ) -> Output:
        configure_local_caches()

        parsed_vad_params = json.loads(vad_params) if vad_params.strip() else None
        parsed_diarization_params = (
            json.loads(diarization_params) if diarization_params.strip() else None
        )

        config = TranscriptionConfig(
            whisper_model=resolve_whisper_model_spec(model),
            language=language or None,
            task=task,
            use_vad=use_vad or diarization,
            use_diarization=diarization,
            vad_method=vad_method,
            vad_params=parsed_vad_params,
            diarization_method=diarization_method,
            diarization_params=parsed_diarization_params,
            num_speakers=num_speakers or None,
            min_speakers=min_speakers or None,
            max_speakers=max_speakers or None,
            use_alignment=use_alignment,
            alignment_model=resolve_alignment_model_spec(alignment_model or None),
            include_speaker_labels=include_speaker_labels,
            hf_token=hf_token or self.default_hf_token,
            log_progress=log_progress,
            batch_size=batch_size or None,
        )

        if device != "auto":
            config.device = device
        if compute_type != "auto":
            config.compute_type = compute_type
        elif config.device == "cpu":
            config.compute_type = "float32"

        run_dir = OUTPUTS_DIR / "replicate" / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / f"{SysPath(str(audio_file)).stem}.srt"

        pipeline = TranscriptionPipeline(config)
        result = pipeline.transcribe(audio_file, output_path=output_path, save_logs=save_logs)

        logs_zip = None
        if save_logs and result.get("log_dir"):
            log_dir = SysPath(result["log_dir"])
            if log_dir.exists():
                archive_base = run_dir / "logs"
                archive_path = shutil.make_archive(
                    str(archive_base), "zip", root_dir=str(log_dir.parent), base_dir=log_dir.name
                )
                logs_zip = Path(archive_path)

        return Output(
            srt_file=Path(str(output_path)),
            srt_content=result["srt_content"],
            detected_language=result["transcription"]["language"],
            duration_seconds=float(result["transcription"]["duration"]),
            num_segments=len(result["final_segments"]),
            logs_zip=logs_zip,
        )
