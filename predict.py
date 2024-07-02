import gc
import os
import shutil
import time
from typing import Any

import pysubs2
import torch
import whisperx
from cog import BaseModel, BasePredictor, Input, Path
from pydub import AudioSegment

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_model = 'i4ds/whisper4sg-srg-v2-full-mc-de-sg-corpus-v2'

# HF Variables
os.environ['HF_HOME'] = './hf'


class ModelOutput(BaseModel):
    transcription: str
    language: str
    segments: Any


class Predictor(BasePredictor):
    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        output_format: str = Input(
            choices=["text", "srt", "vtt"],
            default="plain text",
            description="Choose the format for the format",
        ),
        initial_prompt: str = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        language: str = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
            default="de",
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
    ) -> ModelOutput:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
            }

            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(
                whisper_model,
                device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if (
                detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
                or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
            ):
                result = align(audio, result, debug)
            else:
                print(
                    f"Cannot align output as language {detected_language} is not supported for alignment"
                )
            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )

        subs = pysubs2.load_from_whisper(result["segments"])
        if output_format == "text":
            transcription = " ".join([sub.text.strip() for sub in subs])
        elif output_format == "srt":
            transcription = subs.to_string(format_="srt")
        else:
            transcription = subs.to_string(format_="vtt")

        return ModelOutput(
            segments=result["segments"],
            transcription=transcription,
            language=detected_language,
        )


def get_audio_duration(file_path):
    return len(AudioSegment.from_file(file_path))


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result
