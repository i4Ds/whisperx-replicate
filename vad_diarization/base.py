"""
Base classes and data structures for VAD and diarization.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SpeechSegment:
    start: float
    end: float
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiarizationResult:
    segments: List[SpeechSegment]
    num_speakers: int

    def get_clip_timestamps(self) -> List[float]:
        timestamps = []
        for segment in self.segments:
            timestamps.extend([segment.start, segment.end])
        return timestamps

    def get_clip_timestamps_dict(self) -> List[Dict[str, float]]:
        return [{"start": segment.start, "end": segment.end} for segment in self.segments]

    def to_dict(self) -> dict:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "num_speakers": self.num_speakers,
        }


class VADProvider(ABC):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        self.device = device
        self.params = params or {}
        self.use_auth_token = use_auth_token

    @abstractmethod
    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        raise NotImplementedError

    def _post_process_segments(
        self,
        segments: List[SpeechSegment],
        min_duration: float,
        merge_threshold: float,
        max_duration: float = 30.0,
    ) -> List[SpeechSegment]:
        from .utils import merge_close_segments, split_long_segments

        if merge_threshold > 0 and len(segments) > 1:
            segments = merge_close_segments(segments, merge_threshold)
        segments = [segment for segment in segments if segment.duration >= min_duration]
        if max_duration > 0 and len(segments) > 0:
            segments = split_long_segments(segments, max_duration)
        return segments


class DiarizationProvider(ABC):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        self.device = device
        self.params = params or {}
        self.use_auth_token = use_auth_token

    @abstractmethod
    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        raise NotImplementedError
