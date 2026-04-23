"""
Utility functions for VAD and diarization.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch

from .base import DiarizationResult, SpeechSegment

logger = logging.getLogger(__name__)


def merge_close_segments(segments: List[SpeechSegment], threshold: float) -> List[SpeechSegment]:
    if not segments:
        return segments

    merged = [segments[0]]
    for segment in segments[1:]:
        if segment.start - merged[-1].end <= threshold:
            merged[-1] = SpeechSegment(
                start=merged[-1].start,
                end=segment.end,
                speaker=merged[-1].speaker,
            )
        else:
            merged.append(segment)
    return merged


def split_long_segments(
    segments: List[SpeechSegment],
    max_duration: float = 30.0,
) -> List[SpeechSegment]:
    if not segments or max_duration <= 0:
        return segments

    result = []
    for segment in segments:
        if segment.duration <= max_duration:
            result.append(segment)
        else:
            current_start = segment.start
            while current_start < segment.end:
                chunk_end = min(current_start + max_duration, segment.end)
                result.append(
                    SpeechSegment(
                        start=current_start,
                        end=chunk_end,
                        speaker=segment.speaker,
                    )
                )
                current_start = chunk_end
    return result


def generate_chunk_timestamps(duration: float, chunk_length: float = 30.0) -> list[dict]:
    timestamps = []
    start = 0.01
    while start < duration:
        end = min(start + chunk_length, duration)
        timestamps.append({"start": start, "end": end})
        start = end
    return timestamps


def load_audio_for_pyannote(audio_path: Union[str, Path]) -> dict:
    import soundfile as sf
    import torch

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(waveform.T)
    return {"waveform": waveform, "sample_rate": sample_rate}


def get_device(requested_device: str) -> str:
    if requested_device == "cuda":
        try:
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return requested_device


def parse_rttm(rttm_path: Union[str, Path], uri: Optional[str] = None) -> List[SpeechSegment]:
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            rec_id = parts[1]
            if uri and rec_id != uri:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append(SpeechSegment(start=start, end=start + duration, speaker=speaker))
    segments.sort(key=lambda segment: segment.start)
    return segments


def save_vad_diarization_log(
    result: DiarizationResult,
    output_path: Union[str, Path],
    audio_path: Optional[str] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = result.to_dict()
    if audio_path:
        log_data["audio_file"] = str(audio_path)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(log_data, handle, indent=2, ensure_ascii=False)

    logger.info("Saved VAD/diarization log to %s", output_path)
