import gc
import os
import time
from typing import Any

import pysubs2
import torch
from faster_whisper.audio import decode_audio
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from cog import BaseModel, BasePredictor, Input, Path

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
# Set Env variables
os.environ["HF_HOME"] = "hf_home"
# Get Model
whisper_model = f'{os.environ["HF_HOME"]}/hub/models--i4ds--whisper4sg-srg-v2-full-mc-de-sg-corpus-v2/snapshots/40d87fed2e282d9cb9843b9d1c9b04dee3725cde'


class ModelOutput(BaseModel):
    transcription: str
    segments: Any
    load_audio_ms: float
    transcribe_ms: float


class Predictor(BasePredictor):
    def setup(self):

        start_time = time.time_ns() / 1e6
        self.model = WhisperModel(whisper_model, device="cuda", compute_type="float16")

        print(f"Model loaded in {(time.time_ns() / 1e6) - start_time} ms")

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        output_format: str = Input(
            choices=["text", "srt", "vtt"],
            default="srt",
            description="Choose the format for the format",
        ),
        language: str = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
            default="de",
        ),
        initial_prompt: str = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0.0
        ),
        vad_filter: bool = Input(
            description="Filter out non-speech audio with Silero VAD.",
            default=True,
        ),
        threshold: float = Input(
            description="Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.",
            default=0.5,
        ),
        min_speech_duration_ms: int = Input(
            description="Minimum speech duration in milliseconds", default=250
        ),
        min_silence_duration_ms: int = Input(
            description="Minimum silence duration in milliseconds", default=2000
        ),
        speech_pad_ms: int = Input(
            description="Speech padding in milliseconds", default=400
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=True,
        ),
    ) -> ModelOutput:
        with torch.inference_mode():
            asr_options = {
                "language": language,
                "temperature": [temperature],
                "initial_prompt": initial_prompt,
            }

            vad_parameters = {
                "threshold": threshold,
                "min_speech_duration_ms": min_speech_duration_ms,
                "min_silence_duration_ms": min_silence_duration_ms,
                "speech_pad_ms": speech_pad_ms,
            }

            start_time = time.time_ns() / 1e6

            audio_arr = decode_audio(str(audio_file))

            load_audio_ms = time.time_ns() / 1e6 - start_time

            if debug:
                print(f"Duration to load audio: {load_audio_ms:.2f} ms")


            start_time = time.time_ns() / 1e6

            segments, info  = self.model.transcribe(
                audio_arr,
                **asr_options,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
            )
            segments = self._fw_segments_to_whisper_output(segments)

            transcribe_ms = time.time_ns() / 1e6 - start_time
            if debug:
                print(f"Duration to transcribe: {transcribe_ms:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )
        subs = pysubs2.load_from_whisper(segments)

        if output_format == "text":
            transcription = " ".join([sub.text.strip() for sub in subs])
        elif output_format == "srt":
            transcription = subs.to_string(format_="srt")
        else:
            transcription = subs.to_string(format_="vtt")

        return ModelOutput(transcription=transcription, segments=segments, load_audio_ms=load_audio_ms, transcribe_ms=transcribe_ms)

    @staticmethod
    def _fw_segments_to_whisper_output(segments: list[Segment]) -> list[dict]:
        # to use pysubs2, the argument must be a segment list-of-dicts
        results= []
        for s in segments:
            segment_dict = {'start':s.start,'end':s.end,'text':s.text}
            results.append(segment_dict)

        return results