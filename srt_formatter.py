"""
SRT file generation using pysubs2.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pysubs2

logger = logging.getLogger(__name__)


def segments_to_srt(
    segments: List[dict],
    output_path: Optional[Union[str, Path]] = None,
    include_speaker: bool = True,
) -> str:
    subs = pysubs2.SSAFile()

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        start_ms = int(segment.get("start", 0) * 1000)
        end_ms = int(segment.get("end", 0) * 1000)
        speaker = segment.get("speaker")

        if include_speaker and speaker:
            text = f"[{speaker}] {text}"

        subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))

    srt_content = subs.to_string("srt")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path))
        logger.info("Saved SRT file to %s", output_path)

    return srt_content


def save_transcription_log(
    segments: List[dict],
    output_path: Union[str, Path],
    audio_path: Optional[str] = None,
    language: Optional[str] = None,
) -> None:
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = {
        "audio_file": str(audio_path) if audio_path else None,
        "language": language,
        "segments": [
            {
                "text": segment.get("text", ""),
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "compression_ratio": segment.get("compression_ratio"),
                "speaker": segment.get("speaker"),
            }
            for segment in segments
        ],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(log_data, handle, indent=2, ensure_ascii=False)

    logger.info("Saved transcription log to %s", output_path)
