"""
Voice Activity Detection and Speaker Diarization package.
"""

from .base import DiarizationProvider, DiarizationResult, SpeechSegment, VADProvider
from .combined import CombinedVADDiarization
from .diarization import (
    DiarizationFactory,
    NemoClusteringDiarization,
    PyAnnoteDiarization,
    SpeechBrainDiarization,
)
from .utils import generate_chunk_timestamps, save_vad_diarization_log
from .vad import NemoVAD, PyAnnoteVAD, SileroVAD, SpeechBrainVAD, VADFactory

__all__ = [
    "SpeechSegment",
    "DiarizationResult",
    "VADProvider",
    "DiarizationProvider",
    "save_vad_diarization_log",
    "generate_chunk_timestamps",
    "PyAnnoteVAD",
    "SileroVAD",
    "SpeechBrainVAD",
    "NemoVAD",
    "VADFactory",
    "PyAnnoteDiarization",
    "NemoClusteringDiarization",
    "SpeechBrainDiarization",
    "DiarizationFactory",
    "CombinedVADDiarization",
]
