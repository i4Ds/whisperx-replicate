"""
Combined VAD and diarization pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import DiarizationProvider, DiarizationResult, SpeechSegment, VADProvider
from .diarization import DiarizationFactory
from .vad import VADFactory


class CombinedVADDiarization:
    def __init__(
        self,
        device: str = "cuda",
        use_auth_token: Optional[str] = None,
        vad_method: str = "pyannote",
        vad_params: Optional[Dict[str, Any]] = None,
        diarization_method: str = "pyannote",
        diarization_params: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.use_auth_token = use_auth_token
        self.vad_method = vad_method
        self.vad_params = vad_params or {}
        self.diarization_method = diarization_method
        self.diarization_params = diarization_params or {}
        self._vad_provider: Optional[VADProvider] = None
        self._diarization_provider: Optional[DiarizationProvider] = None

    def process(
        self,
        audio_path: Union[str, Path],
        use_vad: bool = True,
        use_diarization: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        vad_min_duration: float = 0.5,
        vad_merge_threshold: float = 0.3,
    ) -> DiarizationResult:
        vad_segments: List[SpeechSegment] = []
        diar_segments: List[SpeechSegment] = []
        num_found_speakers = 0

        if use_vad:
            vad_provider = self._get_vad_provider()
            vad_segments = vad_provider.detect_speech(
                audio_path,
                min_duration=vad_min_duration,
                merge_threshold=vad_merge_threshold,
            )

        if use_diarization:
            provider = self._get_diarization_provider()
            diarization = provider.diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diar_segments = diarization.segments
            num_found_speakers = diarization.num_speakers

        if use_vad and use_diarization:
            segments = self._assign_speakers_to_segments(vad_segments, diar_segments)
            return DiarizationResult(segments=segments, num_speakers=num_found_speakers)

        if use_vad:
            return DiarizationResult(segments=vad_segments, num_speakers=0)

        return DiarizationResult(segments=diar_segments, num_speakers=num_found_speakers)

    def _get_vad_provider(self) -> VADProvider:
        if self._vad_provider is None:
            self._vad_provider = VADFactory.create(
                method=self.vad_method,
                device=self.device,
                params=self.vad_params,
                use_auth_token=self.use_auth_token,
            )
        return self._vad_provider

    def _get_diarization_provider(self) -> DiarizationProvider:
        if self._diarization_provider is None:
            self._diarization_provider = DiarizationFactory.create(
                method=self.diarization_method,
                device=self.device,
                params=self.diarization_params,
                use_auth_token=self.use_auth_token,
            )
        return self._diarization_provider

    @staticmethod
    def _assign_speakers_to_segments(
        vad_segments: List[SpeechSegment],
        diar_segments: List[SpeechSegment],
    ) -> List[SpeechSegment]:
        if not vad_segments:
            return vad_segments

        for segment in vad_segments:
            best_speaker, best_overlap = None, 0.0
            for diar_seg in diar_segments:
                overlap = max(0.0, min(segment.end, diar_seg.end) - max(segment.start, diar_seg.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.speaker
            segment.speaker = best_speaker
        return vad_segments
