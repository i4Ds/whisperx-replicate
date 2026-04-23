"""
Main Transcription Pipeline.

Combines:
1. VAD
2. Faster-Whisper transcription
3. CTC forced alignment
4. Optional diarization
5. SRT output generation
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from pydub import AudioSegment

from ctc_alignment import CTCAligner, save_alignment_log
from local_cache import (
    ALIGNMENT_MODELS_DIR,
    OUTPUTS_DIR,
    configure_local_caches,
    resolve_alignment_model_spec,
    resolve_whisper_model_spec,
)
from srt_formatter import save_transcription_log, segments_to_srt
from vad_diarization import CombinedVADDiarization, DiarizationResult, generate_chunk_timestamps

configure_local_caches()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_16k_wav(
    audio_path: Union[str, Path], target_sr: int = 16000
) -> Optional[Path]:
    """
    Ensure audio is 16kHz mono WAV for pyannote using pydub.

    Returns a temp WAV path if conversion is needed, otherwise None.
    """
    audio_path = Path(audio_path)
    audio = AudioSegment.from_file(audio_path)

    needs_resample = audio.frame_rate != target_sr
    needs_mono = audio.channels != 1
    is_wav = audio_path.suffix.lower() == ".wav"

    if not (needs_resample or needs_mono or not is_wav):
        logger.info("Audio already %sHz mono WAV; no resample needed", target_sr)
        return None

    temp_dir = Path(tempfile.gettempdir()) / "stt4sg_transcribe"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{audio_path.stem}_16khz.wav"

    logger.info("Resampling to %sHz mono WAV: %s -> %s", target_sr, audio_path, temp_path)
    if needs_resample:
        audio = audio.set_frame_rate(target_sr)
    if needs_mono:
        audio = audio.set_channels(1)

    audio.export(temp_path, format="wav")
    return temp_path


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription pipeline."""

    whisper_model: str = "i4ds/daily-brook-134"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"

    use_vad: bool = True
    use_diarization: bool = False
    vad_method: str = "silero"
    vad_params: Optional[Dict[str, Any]] = None
    diarization_method: str = "pyannote"
    diarization_params: Optional[Dict[str, Any]] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    vad_min_duration: float = 0.5
    vad_merge_threshold: float = 0.3

    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    batch_size: Optional[int] = None
    word_timestamps: bool = True
    log_progress: bool = False

    use_alignment: bool = True
    alignment_model: Optional[str] = None

    include_speaker_labels: bool = True
    hf_token: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptionPipeline:
    """Main transcription pipeline combining all components."""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._whisper_model = None
        self._vad_diarization = None
        self._aligner = None
        self.output_dir = OUTPUTS_DIR / "srt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def whisper_model(self) -> WhisperModel:
        if self._whisper_model is None:
            whisper_model_spec = resolve_whisper_model_spec(self.config.whisper_model)
            logger.info("Loading Whisper model: %s", whisper_model_spec)
            self._whisper_model = WhisperModel(
                whisper_model_spec,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            logger.info("Whisper model loaded")
        return self._whisper_model

    @property
    def batched_pipeline(self) -> BatchedInferencePipeline:
        return BatchedInferencePipeline(model=self.whisper_model)

    @property
    def vad_diarization(self) -> CombinedVADDiarization:
        if self._vad_diarization is None:
            self._vad_diarization = CombinedVADDiarization(
                device=self.config.device,
                use_auth_token=self.config.hf_token,
                vad_method=self.config.vad_method,
                vad_params=self.config.vad_params,
                diarization_method=self.config.diarization_method,
                diarization_params=self.config.diarization_params,
            )
        return self._vad_diarization

    def get_aligner(self, language: str) -> CTCAligner:
        model_name = resolve_alignment_model_spec(self.config.alignment_model)
        if self._aligner is None or self._aligner.language != language:
            self._aligner = CTCAligner(
                language=language,
                device=self.config.device,
                model_name=model_name,
                model_dir=str(ALIGNMENT_MODELS_DIR),
            )
        return self._aligner

    def transcribe(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_logs: bool = True,
    ) -> Dict:
        """
        Run the full transcription pipeline.
        """
        audio_path = Path(audio_path)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = audio_path.stem

        output_path = Path(output_path) if output_path else None

        if output_path:
            output_root = output_path.parent
            run_log_dir = output_root / "logs" / f"{base_name}_{run_id}"
        else:
            run_log_dir = OUTPUTS_DIR / "logs" / f"{base_name}_{run_id}"
        if save_logs:
            run_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting transcription: %s", audio_path)
        results = {
            "audio_file": str(audio_path),
            "run_id": run_id,
            "config": self.config.to_dict(),
        }

        vad_segments = None
        diarization_result = None
        temp_audio_path = None

        if self.config.use_vad:
            logger.info("Step 1: Voice Activity Detection...")
            temp_audio_path = ensure_16k_wav(audio_path, target_sr=16000)
            working_audio_path = temp_audio_path if temp_audio_path else audio_path

            vad_provider = self.vad_diarization._get_vad_provider()
            vad_segments = vad_provider.detect_speech(
                working_audio_path,
                min_duration=self.config.vad_min_duration,
                merge_threshold=self.config.vad_merge_threshold,
            )
            results["vad_segments"] = [
                {"start": s.start, "end": s.end} for s in vad_segments
            ]
            logger.info("VAD detected %s speech segments", len(vad_segments))
            if save_logs:
                self._save_vad_log(vad_segments, run_log_dir / "vad.json", str(audio_path))

        logger.info("Step 2: Transcription...")
        transcribe_kwargs = {
            "language": self.config.language,
            "task": self.config.task,
            "beam_size": self.config.beam_size,
            "word_timestamps": self.config.word_timestamps,
            "log_progress": self.config.log_progress,
        }

        use_batched = self.config.batch_size is not None

        if vad_segments and len(vad_segments) > 0:
            if use_batched:
                clip_timestamps = [{"start": s.start, "end": s.end} for s in vad_segments]
            else:
                clip_timestamps = []
                for segment in vad_segments:
                    clip_timestamps.extend([segment.start, segment.end])
            transcribe_kwargs["clip_timestamps"] = clip_timestamps
            logger.info("Using %s VAD segments as clip_timestamps", len(vad_segments))
        elif use_batched:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            clip_timestamps = generate_chunk_timestamps(duration)
            clip_timestamps[-1]["end"] = min(
                duration,
                max(clip_timestamps[-1]["start"] + 0.1, clip_timestamps[-1]["end"] - 0.01),
            )
            transcribe_kwargs["clip_timestamps"] = clip_timestamps
            logger.info("Using %s fixed 30s chunks (no VAD)", len(clip_timestamps))
        else:
            transcribe_kwargs["vad_filter"] = False

        if use_batched:
            logger.info("Using batched inference with batch_size=%s", self.config.batch_size)
            transcribe_kwargs["batch_size"] = self.config.batch_size
            pipeline = self.batched_pipeline
        else:
            pipeline = self.whisper_model

        segments_gen, info = pipeline.transcribe(str(audio_path), **transcribe_kwargs)

        segments = []
        for seg in segments_gen:
            seg_dict = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
            }
            if seg.words:
                seg_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in seg.words
                ]
            segments.append(seg_dict)

        transcription_result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }
        results["transcription"] = transcription_result
        if save_logs:
            save_transcription_log(
                segments,
                run_log_dir / "transcription.json",
                str(audio_path),
                info.language,
            )

        alignment_result = None
        if self.config.use_alignment and info.language:
            logger.info("Step 3: CTC alignment...")
            try:
                aligner = self.get_aligner(info.language)
                alignment_result = aligner.align(
                    [
                        {
                            "text": segment["text"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "avg_logprob": segment.get("avg_logprob"),
                        }
                        for segment in segments
                    ],
                    audio_path,
                )
                results["alignment"] = alignment_result.to_dict()
                if save_logs:
                    save_alignment_log(
                        alignment_result,
                        run_log_dir / "alignment.json",
                        str(audio_path),
                    )
            except Exception as exc:
                logger.warning("Alignment failed: %s", exc)

        if self.config.use_diarization:
            logger.info("Step 4: Speaker Diarization...")
            if temp_audio_path is None:
                temp_audio_path = ensure_16k_wav(audio_path, target_sr=16000)
            working_audio_path = temp_audio_path if temp_audio_path else audio_path

            diar_provider = self.vad_diarization._get_diarization_provider()
            diarization_result = diar_provider.diarize(
                working_audio_path,
                num_speakers=self.config.num_speakers,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            results["diarization"] = diarization_result.to_dict()
            if save_logs:
                self._save_diarization_log(
                    diarization_result,
                    run_log_dir / "diarization.json",
                    str(audio_path),
                )

        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.debug("Cleaned up temp audio: %s", temp_audio_path)

        if diarization_result and self.config.use_diarization:
            logger.info("Step 5: Speaker assignment and purity calculation...")
            final_segments = self._assign_speakers_with_purity(
                alignment_result.segments if alignment_result else segments,
                diarization_result,
            )
            if save_logs:
                self._save_speaker_log(final_segments, run_log_dir / "speaker_alignment.json")
        else:
            final_segments = (
                [segment.to_dict() for segment in alignment_result.segments]
                if alignment_result
                else segments
            )

        results["final_segments"] = final_segments

        logger.info("Step 6: Generating SRT...")
        output_path = output_path if output_path else self.output_dir / f"{base_name}.srt"
        srt_content = segments_to_srt(
            final_segments,
            output_path,
            include_speaker=self.config.include_speaker_labels and self.config.use_diarization,
        )
        results["srt_path"] = str(output_path)
        results["srt_content"] = srt_content
        results["log_dir"] = str(run_log_dir) if save_logs else None

        logger.info("Done! SRT: %s", output_path)
        if save_logs:
            logger.info("Logs: %s", run_log_dir)

        return results

    def _assign_speakers_with_purity(
        self, segments, diarization_result: DiarizationResult
    ) -> List[Dict]:
        if not diarization_result.segments:
            return segments if isinstance(segments[0], dict) else [segment.to_dict() for segment in segments]

        final_segments = []
        for segment in segments:
            seg_dict = segment.to_dict() if hasattr(segment, "to_dict") else dict(segment)
            seg_start, seg_end = seg_dict.get("start", 0), seg_dict.get("end", 0)
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                seg_dict["speaker"] = None
                seg_dict["purity"] = 0.0
                seg_dict["speaker_overlaps"] = {}
                final_segments.append(seg_dict)
                continue

            speaker_overlaps = {}
            for diar_seg in diarization_result.segments:
                overlap = max(0, min(seg_end, diar_seg.end) - max(seg_start, diar_seg.start))
                if overlap > 0:
                    speaker = diar_seg.speaker or "UNKNOWN"
                    speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap

            if not speaker_overlaps:
                seg_dict["speaker"] = None
                seg_dict["purity"] = 0.0
                seg_dict["speaker_overlaps"] = {}
            else:
                dominant_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                dominant_overlap = speaker_overlaps[dominant_speaker]
                total_speaker_time = sum(speaker_overlaps.values())
                purity = dominant_overlap / total_speaker_time if total_speaker_time > 0 else 0.0
                coverage = min(total_speaker_time / seg_duration, 1.0) if seg_duration > 0 else 0.0

                seg_dict["speaker"] = dominant_speaker
                seg_dict["purity"] = round(purity, 4)
                seg_dict["coverage"] = round(coverage, 4)
                seg_dict["speaker_overlaps"] = {
                    key: round(value, 4) for key, value in speaker_overlaps.items()
                }

            final_segments.append(seg_dict)
        return final_segments

    def _save_vad_log(self, segments: List, output_path: Path, audio_path: str) -> None:
        log_data = {
            "audio_file": audio_path,
            "num_segments": len(segments),
            "total_speech_duration": sum(segment.end - segment.start for segment in segments),
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.end - segment.start,
                }
                for segment in segments
            ],
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(log_data, handle, indent=2, ensure_ascii=False)
        logger.info("Saved VAD log to %s", output_path)

    def _save_diarization_log(
        self, result: DiarizationResult, output_path: Path, audio_path: str
    ) -> None:
        speakers = {}
        for segment in result.segments:
            speaker = segment.speaker or "UNKNOWN"
            duration = segment.end - segment.start
            speakers.setdefault(speaker, {"total_duration": 0, "segment_count": 0})
            speakers[speaker]["total_duration"] += duration
            speakers[speaker]["segment_count"] += 1

        log_data = {
            "audio_file": audio_path,
            "num_speakers": result.num_speakers,
            "speaker_statistics": speakers,
            "segments": [
                {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
                for segment in result.segments
            ],
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(log_data, handle, indent=2, ensure_ascii=False)
        logger.info("Saved diarization log to %s", output_path)

    def _save_speaker_log(self, final_segments: List[Dict], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(final_segments, handle, indent=2, ensure_ascii=False)
        logger.info("Saved speaker alignment log to %s", output_path)
